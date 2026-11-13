/* probe/npu_run.c — plan-driven generic runner for the self-built submitter.
 *
 * WHY: probe/npu_binop.c hard-wires every operator as a C struct (g_plan_*). Changing the
 * element count, the shape, or adding an operator therefore needs a recompile. This runner
 * reads the same parameters from a tiny key=value text file, so "new operator / new shape"
 * is a data change only -- which is what tools/npu_lib (Graph.compile) emits per artifact.
 *
 * It is a *copy* of the proven submission core in probe/npu_binop.c (handshake -> ALLOC ->
 * mmap -> MEM MAP -> write -> relocate address words -> descriptor assembly -> CMD RUN ->
 * wait -> DMA_BUF_SYNC -> read back). npu_binop.c is untouched; nothing here calls
 * libneuron / neuronrt.
 *
 *   ./npu_run <plan.txt>            # everything (paths included) comes from the plan
 *
 * Plan keys (all optional except asset_dir / n_elems / out):
 *   asset_dir=<device dir with cmdN.bin, subcmd.bin, cmdin.bin>
 *   old_a/old_b/old_c=0x...         captured IOVAs (matching key for relocation; 0 = unused slot)
 *   alloc_a/alloc_b/alloc_c=<bytes> allocation size per buffer (default 4096)
 *   dir0/dir1/dir2=1|2              cmdbuf direction flags, order is semantic
 *   patch_idx=2                     which cmdbuf carries the address words
 *   expect_patches=3                how many address words must be found
 *   n_elems=<n>                     int8 elements per submission (== bytes)
 *   n_inputs=1|2                    default 2
 *   cmdbuf0/cmdbuf1/cmdbuf2=<file>  default cmd0.bin/cmd1.bin/cmd2.bin
 *   extra_file=<file>               optional 3rd buffer (LUT / const weight) loaded from asset
 *   in_a/in_b/out=<device path>     data files
 *
 * Exit: 0 ok; 2 plan/arg error; 3 asset error; 4 patch mismatch; 5 ioctl/run error.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define APUSYS_MAGIC 0x41
typedef struct { uint32_t op, pad; uint64_t flags; uint64_t w0, w1, w2; } hs_args_t;
typedef struct { uint32_t op, pad; uint64_t flags; uint64_t w0, w1, w2, w3; } mem_args_t;
typedef struct { uint32_t op, pad; uint64_t flags; uint64_t w[17]; } cmd_args_t;
#define APU_HANDSHAKE _IOWR(APUSYS_MAGIC, 32, hs_args_t)
#define APU_MEM _IOWR(APUSYS_MAGIC, 33, mem_args_t)
#define APU_CMD _IOWR(APUSYS_MAGIC, 34, cmd_args_t)

_Static_assert(sizeof(hs_args_t) == 40, "hs");
_Static_assert(sizeof(mem_args_t) == 48, "mem");
_Static_assert(sizeof(cmd_args_t) == 152, "cmd");

#define CMDBUF_ALIGN 0x100u
#define MAXSTRING 512

typedef struct { uint64_t handle; void *host; uint64_t iova; uint32_t size; } buf_t;

/* ------------------------------------------------------------------ plan */
typedef struct {
  char asset_dir[MAXSTRING];
  char cmdbuf_file[3][64];
  char extra_file[64];
  char in_a[MAXSTRING], in_b[MAXSTRING], out[MAXSTRING];
  uint64_t old_iova[3];
  uint32_t alloc[3];
  int dir[3];
  int patch_idx;
  int expect_patches;
  uint32_t n_elems;
  int n_inputs;
} plan_t;

static int g_fd = -1;

static void put32(void *b, unsigned o, uint32_t v) { memcpy((char *)b + o, &v, 4); }
static void put64(void *b, unsigned o, uint64_t v) { memcpy((char *)b + o, &v, 8); }
static uint64_t get64(const void *b, unsigned o) { uint64_t v; memcpy(&v, (const char *)b + o, 8); return v; }

static void plan_defaults(plan_t *p) {
  memset(p, 0, sizeof(*p));
  snprintf(p->cmdbuf_file[0], 64, "cmd0.bin");
  snprintf(p->cmdbuf_file[1], 64, "cmd1.bin");
  snprintf(p->cmdbuf_file[2], 64, "cmd2.bin");
  p->alloc[0] = p->alloc[1] = p->alloc[2] = 4096;
  p->dir[0] = 1; p->dir[1] = 2; p->dir[2] = 1;
  p->patch_idx = 2;
  p->expect_patches = 3;
  p->n_inputs = 2;
}

static int parse_plan(const char *path, plan_t *p) {
  FILE *f = fopen(path, "r");
  char line[1024];
  if (!f) { perror(path); return -1; }
  plan_defaults(p);
  while (fgets(line, sizeof(line), f)) {
    char *s = line;
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '#' || *s == '\n' || *s == '\r' || *s == 0) continue;
    char *eq = strchr(s, '=');
    if (!eq) { fprintf(stderr, "plan: bad line: %s", line); fclose(f); return -1; }
    *eq = 0;
    char *k = s;
    char *v = eq + 1;
    char *e = v + strlen(v);
    while (e > v && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ' || e[-1] == '\t')) *--e = 0;
    while (*k == ' ' || *k == '\t') k++;
    { char *ke = k + strlen(k); while (ke > k && (ke[-1] == ' ' || ke[-1] == '\t')) *--ke = 0; }

    if (!strcmp(k, "asset_dir")) snprintf(p->asset_dir, MAXSTRING, "%s", v);
    else if (!strcmp(k, "cmdbuf0")) snprintf(p->cmdbuf_file[0], 64, "%s", v);
    else if (!strcmp(k, "cmdbuf1")) snprintf(p->cmdbuf_file[1], 64, "%s", v);
    else if (!strcmp(k, "cmdbuf2")) snprintf(p->cmdbuf_file[2], 64, "%s", v);
    else if (!strcmp(k, "extra_file")) snprintf(p->extra_file, 64, "%s", v);
    else if (!strcmp(k, "in_a")) snprintf(p->in_a, MAXSTRING, "%s", v);
    else if (!strcmp(k, "in_b")) snprintf(p->in_b, MAXSTRING, "%s", v);
    else if (!strcmp(k, "out")) snprintf(p->out, MAXSTRING, "%s", v);
    else if (!strcmp(k, "old_a")) p->old_iova[0] = strtoull(v, NULL, 0);
    else if (!strcmp(k, "old_b")) p->old_iova[1] = strtoull(v, NULL, 0);
    else if (!strcmp(k, "old_c")) p->old_iova[2] = strtoull(v, NULL, 0);
    else if (!strcmp(k, "alloc_a")) p->alloc[0] = (uint32_t)strtoul(v, NULL, 0);
    else if (!strcmp(k, "alloc_b")) p->alloc[1] = (uint32_t)strtoul(v, NULL, 0);
    else if (!strcmp(k, "alloc_c")) p->alloc[2] = (uint32_t)strtoul(v, NULL, 0);
    else if (!strcmp(k, "dir0")) p->dir[0] = atoi(v);
    else if (!strcmp(k, "dir1")) p->dir[1] = atoi(v);
    else if (!strcmp(k, "dir2")) p->dir[2] = atoi(v);
    else if (!strcmp(k, "patch_idx")) p->patch_idx = atoi(v);
    else if (!strcmp(k, "expect_patches")) p->expect_patches = atoi(v);
    else if (!strcmp(k, "n_elems")) p->n_elems = (uint32_t)strtoul(v, NULL, 0);
    else if (!strcmp(k, "n_inputs")) p->n_inputs = atoi(v);
    else fprintf(stderr, "plan: unknown key '%s' (ignored)\n", k);
  }
  fclose(f);
  if (!p->asset_dir[0] || !p->n_elems || !p->out[0]) {
    fprintf(stderr, "plan: asset_dir / n_elems / out are required\n");
    return -1;
  }
  if (p->n_inputs == 2 && !p->in_b[0]) { fprintf(stderr, "plan: n_inputs=2 needs in_b\n"); return -1; }
  return 0;
}

/* ------------------------------------------------------------ ioctl 基础 */
static int apu_open_handshake(void) {
  static const uint32_t devs[] = {1, 2, 4, 6, 7}, mems[] = {1, 2, 4, 5};
  hs_args_t hs;

  g_fd = open("/dev/apusys", O_RDWR | O_SYNC);
  if (g_fd < 0) { perror("open /dev/apusys"); return -1; }

  /* E-106B-3: BASIC x2 -> DEV(1,2,4,6,7) -> MEM(1,2,4,5) */
  for (int i = 0; i < 2; i++) {
    memset(&hs, 0, sizeof(hs)); put32(&hs, 0, 0);
    if (ioctl(g_fd, APU_HANDSHAKE, &hs) < 0) { perror("hs basic"); return -1; }
  }
  for (unsigned i = 0; i < sizeof(devs) / sizeof(devs[0]); i++) {
    memset(&hs, 0, sizeof(hs)); put32(&hs, 0, 1); put32(&hs, 16, devs[i]);
    if (ioctl(g_fd, APU_HANDSHAKE, &hs) < 0) { perror("hs dev"); return -1; }
  }
  for (unsigned i = 0; i < sizeof(mems) / sizeof(mems[0]); i++) {
    memset(&hs, 0, sizeof(hs)); put32(&hs, 0, 2); put32(&hs, 16, mems[i]);
    if (ioctl(g_fd, APU_HANDSHAKE, &hs) < 0) { perror("hs mem"); return -1; }
  }
  return 0;
}

static int apu_alloc_map(buf_t *b, uint32_t size, uint32_t align, int do_map) {
  mem_args_t m;
  memset(&m, 0, sizeof(m));
  put32(&m, 0, 0); put32(&m, 16, 0); put32(&m, 20, size); put32(&m, 24, align);
  if (ioctl(g_fd, APU_MEM, &m) < 0) { perror("mem alloc"); return -1; }
  memset(b, 0, sizeof(*b));
  b->handle = get64(&m, 0);
  b->size = size;
  b->host = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, (int)b->handle, 0);
  if (b->host == MAP_FAILED) { perror("mmap(handle)"); b->host = NULL; return -1; }
  if (do_map) {
    memset(&m, 0, sizeof(m));
    put32(&m, 0, 2); put64(&m, 16, b->handle); put32(&m, 24, size); put64(&m, 28, 2);
    if (ioctl(g_fd, APU_MEM, &m) < 0) { perror("mem map"); return -1; }
    b->iova = get64(&m, 8);
  }
  return 0;
}

static int apu_map_only(buf_t *b) {
  mem_args_t m;
  memset(&m, 0, sizeof(m));
  put32(&m, 0, 2); put64(&m, 16, b->handle); put32(&m, 24, b->size); put64(&m, 28, 2);
  if (ioctl(g_fd, APU_MEM, &m) < 0) { perror("mem map"); return -1; }
  b->iova = get64(&m, 8);
  return 0;
}

static void apu_release(buf_t *b) {
  mem_args_t m;
  if (!b->handle) return;
  memset(&m, 0, sizeof(m)); put32(&m, 0, 3); put64(&m, 16, b->handle);   /* UNMAP */
  ioctl(g_fd, APU_MEM, &m);
  if (b->host) munmap(b->host, b->size);
  memset(b, 0, sizeof(*b));
}

/* 1<<30 = START, 1<<31 = END; low bits 1=READ 2=WRITE */
static void dma_buf_sync(uint64_t handle, uint64_t flags) {
  struct dma_sync { uint64_t flags; } sy;
  sy.flags = flags;
  ioctl((int)handle, _IOW(0x62, 1, struct dma_sync), &sy);
}

/* --------------------------------------------------------------- 文件工具 */
static int file_size(const char *path, uint32_t *out) {
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
  long n = ftell(f);
  fclose(f);
  if (n <= 0) return -1;
  *out = (uint32_t)n;
  return 0;
}

static int read_file(const char *path, void *dst, size_t n) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); return -1; }
  size_t got = fread(dst, 1, n, f);
  fclose(f);
  if (got != n) { fprintf(stderr, "%s: short read %zu/%zu\n", path, got, n); return -1; }
  return 0;
}

static int write_file(const char *path, const void *src, size_t n) {
  FILE *f = fopen(path, "wb");
  if (!f) { perror(path); return -1; }
  size_t put = fwrite(src, 1, n, f);
  fclose(f);
  if (put != n) { fprintf(stderr, "%s: short write %zu/%zu\n", path, put, n); return -1; }
  return 0;
}

/* 命令流里的地址 = iova >> 4（E-106B-3），按 32 位字比较替换。 */
static int patch_addrs(unsigned char *cmd, unsigned size, const uint64_t *oldv,
                       const uint64_t *newv, unsigned *hits) {
  *hits = 0;
  for (unsigned o = 0; o + 4 <= size; o += 4) {
    uint32_t w;
    memcpy(&w, cmd + o, 4);
    for (int k = 0; k < 3; k++) {
      if (oldv[k] == 0) continue;   /* 空槽不匹配（否则 0 会匹配满屏零字） */
      uint32_t olo = (uint32_t)((oldv[k] >> 4) & 0xffffffffu);
      if (w == olo) {
        uint32_t nlo = (uint32_t)((newv[k] >> 4) & 0xffffffffu);
        memcpy(cmd + o, &nlo, 4);
        printf("patch off=0x%03x 0x%08x -> 0x%08x (buf%d)\n", o, olo, nlo, k);
        (*hits)++;
        break;
      }
    }
  }
  return 0;
}

/* ------------------------------------------------------------------ main */
int main(int argc, char **argv) {
  plan_t pl;
  char p[1024];
  buf_t cb[3] = {{0}}, a1 = {0}, b1 = {0}, c = {0}, einfo = {0};
  uint32_t cb_size[3] = {0, 0, 0};
  static unsigned char cbs[3 * 24];
  static unsigned char subcmd[256];
  static unsigned char adj[64];
  cmd_args_t cmd;
  uint32_t subcmd_size = 0;
  int deferred_map, rc = 0;
  unsigned hits = 0;

  if (argc != 2) { fprintf(stderr, "usage: %s <plan.txt>\n", argv[0]); return 2; }
  if (parse_plan(argv[1], &pl)) return 2;
  if (pl.n_elems > pl.alloc[0] || pl.n_elems > pl.alloc[2]) {
    fprintf(stderr, "n_elems=%u exceeds alloc_a/alloc_c\n", pl.n_elems);
    return 2;
  }

  for (int k = 0; k < 3; k++) {
    snprintf(p, sizeof(p), "%s/%s", pl.asset_dir, pl.cmdbuf_file[k]);
    if (file_size(p, &cb_size[k])) { fprintf(stderr, "%s: missing cmdbuf\n", p); return 3; }
  }
  snprintf(p, sizeof(p), "%s/subcmd.bin", pl.asset_dir);
  if (file_size(p, &subcmd_size) || subcmd_size > sizeof(subcmd)) return 3;

  if (apu_open_handshake()) return 5;

  for (int k = 0; k < 3; k++)
    if (apu_alloc_map(&cb[k], cb_size[k], CMDBUF_ALIGN, 1)) { rc = 5; goto done; }
  for (int k = 0; k < 3; k++) {
    snprintf(p, sizeof(p), "%s/%s", pl.asset_dir, pl.cmdbuf_file[k]);
    if (read_file(p, cb[k].host, cb_size[k])) { rc = 3; goto done; }
  }

  if (apu_alloc_map(&a1, pl.alloc[0], CMDBUF_ALIGN, 1)) { rc = 5; goto done; }
  deferred_map = (pl.extra_file[0] != 0);
  if (pl.n_inputs == 2 &&
      apu_alloc_map(&b1, pl.alloc[1], CMDBUF_ALIGN, deferred_map ? 0 : 1)) { rc = 5; goto done; }
  if (apu_alloc_map(&c, pl.alloc[2], CMDBUF_ALIGN, 1)) { rc = 5; goto done; }
  if (apu_alloc_map(&einfo, 76, 0, 1)) { rc = 5; goto done; }   /* exec_infos 必须是 MEM handle */
  memset(einfo.host, 0, 76);

  if (read_file(pl.in_a, a1.host, pl.n_elems)) { rc = 3; goto done; }
  if (pl.n_inputs == 2) {
    if (deferred_map) {
      snprintf(p, sizeof(p), "%s/%s", pl.asset_dir, pl.extra_file);
      if (read_file(p, b1.host, pl.alloc[1])) { rc = 3; goto done; }
      if (apu_map_only(&b1)) { rc = 5; goto done; }
    } else if (read_file(pl.in_b, b1.host, pl.n_elems)) { rc = 3; goto done; }
  }

  /* 命令缓冲也要 flush（漏了会偶发读到陈旧命令流） */
  for (int k = 0; k < 3; k++) {
    dma_buf_sync(cb[k].handle, (1u << 30) | 2u);
    dma_buf_sync(cb[k].handle, (1u << 31) | 2u);
  }
  dma_buf_sync(a1.handle, (1u << 30) | 2u);
  dma_buf_sync(a1.handle, (1u << 31) | 2u);
  if (pl.n_inputs == 2) {
    dma_buf_sync(b1.handle, (1u << 30) | 2u);
    dma_buf_sync(b1.handle, (1u << 31) | 2u);
  }

  {
    uint64_t newv[3] = { a1.iova, pl.n_inputs == 2 ? b1.iova : 0, c.iova };
    int k = pl.patch_idx;
    if (k < 0 || k > 2) { rc = 2; goto done; }
    patch_addrs((unsigned char *)cb[k].host, cb_size[k], pl.old_iova, newv, &hits);
    if (hits != (unsigned)pl.expect_patches) {
      fprintf(stderr, "patched %u/%d addresses (command stream mismatch?)\n",
              hits, pl.expect_patches);
      rc = 4; goto done;
    }
  }
  printf("iova A=0x%llx B=0x%llx C=0x%llx elems=%u\n",
         (unsigned long long)a1.iova, (unsigned long long)b1.iova,
         (unsigned long long)c.iova, pl.n_elems);

  memset(cbs, 0, sizeof(cbs));
  for (int k = 0; k < 3; k++) {
    uint64_t h = cb[k].handle;
    uint32_t sz = cb_size[k], al = CMDBUF_ALIGN, dir = (uint32_t)pl.dir[k];
    memcpy(cbs + k * 24 + 0, &h, 8);
    memcpy(cbs + k * 24 + 8, &sz, 4);
    memcpy(cbs + k * 24 + 12, &al, 4);
    memcpy(cbs + k * 24 + 16, &dir, 4);
  }

  memset(subcmd, 0, sizeof(subcmd));
  snprintf(p, sizeof(p), "%s/subcmd.bin", pl.asset_dir);
  if (read_file(p, subcmd, subcmd_size)) { rc = 3; goto done; }
  put32(subcmd, subcmd_size - 16, 3);
  put64(subcmd, subcmd_size - 8, (uint64_t)(uintptr_t)cbs);

  memset(adj, 0, sizeof(adj));   /* adj_matrix 全零（E-106B-2 已验证） */

  memset(&cmd, 0, sizeof(cmd));
  snprintf(p, sizeof(p), "%s/cmdin.bin", pl.asset_dir);
  if (read_file(p, &cmd, sizeof(cmd))) { rc = 3; goto done; }
  put64(&cmd, 0x10, 0);
  put64(&cmd, 0x18, 0);
  put32(&cmd, 0x48, 1);
  put64(&cmd, 0x50, (uint64_t)(uintptr_t)subcmd);
  put64(&cmd, 0x58, (uint64_t)(uintptr_t)adj);
  put64(&cmd, 0x60, 0xffffffffffffffffULL);
  put64(&cmd, 0x68, einfo.handle);

  if (ioctl(g_fd, APU_CMD, &cmd) < 0) { perror("CMD RUN"); rc = 5; goto done; }

  usleep(200000);   /* rc=0 只代表提交成功 */
  dma_buf_sync(c.handle, (1u << 30) | 1u);
  dma_buf_sync(c.handle, (1u << 31) | 1u);
  if (write_file(pl.out, c.host, pl.n_elems)) { rc = 3; goto done; }
  printf("RC=0 patches=%u out=%s\n", hits, pl.out);

  /* 证据用：让进程在提交完成后多活一会儿，方便外部采样 /proc/<pid>/maps
     （docs/OWN-SUBMITTER-LIB.md「怎么证明 run() 没碰厂商库」）。 */
  {
    const char *hold = getenv("NPU_RUN_HOLD_MS");
    if (hold && atoi(hold) > 0) { printf("HOLD pid=%d ms=%d\n", (int)getpid(), atoi(hold)); fflush(stdout); usleep((useconds_t)atoi(hold) * 1000); }
  }

done:
  apu_release(&c); apu_release(&b1); apu_release(&a1); apu_release(&einfo);
  apu_release(&cb[2]); apu_release(&cb[1]); apu_release(&cb[0]);
  if (g_fd >= 0) { close(g_fd); g_fd = -1; }
  return rc;
}
