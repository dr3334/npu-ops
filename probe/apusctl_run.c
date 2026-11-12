// apusctl v2 — 自己的 NPU 提交器（E-106B）。
//
//   apusctl add <dir> <oldA> <oldB> <oldC>
//
// <dir> 内需有：cmd368.bin / cmd32.bin / cmd16.bin（从一次合法 ADD 抓到的命令缓冲内容）
//               a.bin / b.bin（本次要算的输入，int8×64）
// <oldA>/<oldB>/<oldC> 是抓取那次运行的三个 4 KiB buffer 的 IOVA（用来在命令流里定位地址字）。
//
// 流程完全走我们自己的 ioctl：HANDSHAKE → ALLOC → mmap(host) → MEM MAP → 写内容
// → 打补丁（命令流里的旧 IOVA 换成我们新分配的 IOVA）→ 组装 mdw_cmd_in / subcmd_info /
// cmdbuf 描述符 → CMD RUN → 读回输出。
//
// 仍然 UNKNOWN 的两处（见 docs/EXPERIMENTS.md E-106B）：
//   * adj_matrix（cmd_in@0x58）的内容格式 —— 这里先传一块全零 64 B
//   * 完成等待语义 —— 先假设 CMD RUN 阻塞到完成
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

static int g_fd = -1;

static void put32(void *b, unsigned o, uint32_t v) { memcpy((char *)b + o, &v, 4); }
static void put64(void *b, unsigned o, uint64_t v) { memcpy((char *)b + o, &v, 8); }
static uint64_t get64(const void *b, unsigned o) { uint64_t v; memcpy(&v, (const char *)b + o, 8); return v; }

typedef struct { uint64_t handle; void *host; uint64_t iova; uint32_t size; } buf_t;

static int apu_open_handshake(void) {
  g_fd = open("/dev/apusys", O_RDWR | O_SYNC);
  if (g_fd < 0) { perror("open"); return -1; }
  hs_args_t hs; memset(&hs, 0, sizeof(hs)); put32(&hs, 0, 0);
  if (ioctl(g_fd, APU_HANDSHAKE, &hs) < 0) { perror("handshake"); return -1; }
  printf("handshake version=0x%llx dev=0x%llx mem=0x%llx\n",
         (unsigned long long)get64(&hs, 0), (unsigned long long)get64(&hs, 8),
         (unsigned long long)get64(&hs, 16));
  return 0;
}

static int apu_alloc_map(buf_t *b, uint32_t size, uint32_t align, int do_map) {
  mem_args_t m; memset(&m, 0, sizeof(m));
  put32(&m, 0, 0); put32(&m, 16, 0); put32(&m, 20, size); put32(&m, 24, align);
  if (ioctl(g_fd, APU_MEM, &m) < 0) { perror("alloc"); return -1; }
  b->handle = get64(&m, 0);
  b->size = size;
  b->host = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, (int)b->handle, 0);
  if (b->host == MAP_FAILED) { perror("mmap(handle)"); b->host = NULL; return -1; }
  if (do_map) {
    memset(&m, 0, sizeof(m));
    put32(&m, 0, 2); put64(&m, 16, b->handle); put32(&m, 24, size); put64(&m, 28, 2);
    if (ioctl(g_fd, APU_MEM, &m) < 0) { perror("map"); return -1; }
    b->iova = get64(&m, 8);
  } else {
    b->iova = 0;
  }
  printf("buf handle=%-3llu size=%-5u iova=0x%llx host=%p\n",
         (unsigned long long)b->handle, size, (unsigned long long)b->iova, b->host);
  return 0;
}

static int apu_release(buf_t *b, int was_mapped) {
  mem_args_t m;
  if (was_mapped) { memset(&m, 0, sizeof(m)); put32(&m, 0, 3); put64(&m, 16, b->handle); ioctl(g_fd, APU_MEM, &m); }
  memset(&m, 0, sizeof(m)); put32(&m, 0, 1); put64(&m, 16, b->handle); ioctl(g_fd, APU_MEM, &m);
  if (b->host) munmap(b->host, b->size);
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

/* 把命令流里出现的旧 IOVA 换成新的（4 字节对齐扫描，含 64 位打包形式的高 32 位） */
static int patch_addrs(unsigned char *cmd, unsigned size,
                       uint64_t oldv[3], uint64_t newv[3], int *hits) {
  *hits = 0;
  for (unsigned o = 0; o + 4 <= size; o += 4) {
    uint32_t w;
    memcpy(&w, cmd + o, 4);
    for (int k = 0; k < 3; k++) {
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

int main(int argc, char **argv) {
  if (argc < 6 || strcmp(argv[1], "add")) {
    fprintf(stderr, "usage: apusctl add <dir> <oldA> <oldB> <oldC>\n");
    return 1;
  }
  const char *dir = argv[2];
  uint64_t oldv[3] = { strtoull(argv[3], NULL, 0), strtoull(argv[4], NULL, 0),
                       strtoull(argv[5], NULL, 0) };
  char p[512];

  if (apu_open_handshake()) return 2;

  buf_t c0 = {0}, c1 = {0}, c2 = {0}, a = {0}, b = {0}, c = {0};
  if (apu_alloc_map(&c0, 368, 0x100, 0)) return 3;
  if (apu_alloc_map(&c1, 32, 0x100, 0)) return 3;
  if (apu_alloc_map(&c2, 16, 0x100, 0)) return 3;
  if (apu_alloc_map(&a, 4096, 0x100, 1)) return 3;
  if (apu_alloc_map(&b, 4096, 0x100, 1)) return 3;
  if (apu_alloc_map(&c, 4096, 0x100, 1)) return 3;

  snprintf(p, sizeof(p), "%s/cmd368.bin", dir);
  if (read_file(p, c0.host, 368)) return 4;
  snprintf(p, sizeof(p), "%s/cmd32.bin", dir);
  if (read_file(p, c1.host, 32)) return 4;
  snprintf(p, sizeof(p), "%s/cmd16.bin", dir);
  if (read_file(p, c2.host, 16)) return 4;
  snprintf(p, sizeof(p), "%s/a.bin", dir);
  if (read_file(p, a.host, 64)) return 4;
  snprintf(p, sizeof(p), "%s/b.bin", dir);
  if (read_file(p, b.host, 64)) return 4;
  memset(c.host, 0xAA, 64);            /* 预填，便于识别"没写" */

  uint64_t newv[3] = { a.iova, b.iova, c.iova };
  int hits = 0;
  patch_addrs((unsigned char *)c0.host, 368, oldv, newv, &hits);
  printf("patched %d address words\n", hits);

  /* 用户态结构：cmdbuf 描述符数组 + subcmd_info + cmd_in */
  static unsigned char cbs[3 * 24];
  memset(cbs, 0, sizeof(cbs));
  struct { uint64_t h; uint32_t sz, al, dir; } e[3] = {
      { c0.handle, 368, 0x100, 1 }, { c1.handle, 32, 0x100, 1 }, { c2.handle, 16, 0x100, 2 },
  };
  for (int k = 0; k < 3; k++) {
    memcpy(cbs + k * 24 + 0, &e[k].h, 8);
    memcpy(cbs + k * 24 + 8, &e[k].sz, 4);
    memcpy(cbs + k * 24 + 12, &e[k].al, 4);
    memcpy(cbs + k * 24 + 16, &e[k].dir, 4);
  }

  static unsigned char subcmd[80];
  snprintf(p, sizeof(p), "%s/subcmd.bin", dir);
  if (read_file(p, subcmd, 80)) return 4;
  put32(subcmd, 64, 3);                       /* num_cmdbufs */
  put64(subcmd, 72, (uint64_t)(uintptr_t)cbs);

  static unsigned char adj[64];
  memset(adj, 0, sizeof(adj));  /* UNKNOWN：先给全零，看驱动是否接受 */

  cmd_args_t cmd;
  memset(&cmd, 0, sizeof(cmd));
  snprintf(p, sizeof(p), "%s/cmdin.bin", dir);
  if (read_file(p, &cmd, 152)) return 4;
  put64(&cmd, 0x10, 0);                       /* id */
  {   /* uid：默认 0，可用 addin/uid.txt 覆盖（文本 hex） */
    uint64_t uid = 0;
    char uf[512]; snprintf(uf, sizeof(uf), "%s/uid.txt", dir);
    FILE *ufp = fopen(uf, "r");
    if (ufp) { if (fscanf(ufp, "%llx", (unsigned long long *)&uid) != 1) uid = 0; fclose(ufp); }
    put64(&cmd, 0x18, uid);
    printf("uid=0x%llx\n", (unsigned long long)uid);
  }
  put32(&cmd, 0x48, 1);                       /* num_subcmds */
  put64(&cmd, 0x50, (uint64_t)(uintptr_t)subcmd);
  put64(&cmd, 0x58, (uint64_t)(uintptr_t)adj);
  put64(&cmd, 0x60, 0xffffffffffffffffULL);
  put64(&cmd, 0x68, 9);

  printf("submit: cmd_in=%p subcmd=%p cmdbufs=%p\n", (void *)&cmd, (void *)subcmd, (void *)cbs);
  errno = 0;
  int rc = ioctl(g_fd, APU_CMD, &cmd);
  printf("CMD RUN -> rc=%d errno=%d (%s)\n", rc, errno, strerror(errno));
  if (rc < 0) { printf("out id=0x%llx fence=0x%llx\n",
                       (unsigned long long)get64(&cmd, 0),
                       (unsigned long long)get64(&cmd, 8)); }

  /* 完成等待：先假设 ioctl 阻塞；再直接读回输出 */
  unsigned char out[64];
  memcpy(out, c.host, 64);
  snprintf(p, sizeof(p), "%s/out.bin", dir);
  FILE *f = fopen(p, "wb");
  if (f) { fwrite(out, 1, 64, f); fclose(f); }
  printf("out:");
  for (int i = 0; i < 16; i++) printf(" %02x", out[i]);
  printf("\n");

  apu_release(&c, 1); apu_release(&b, 1); apu_release(&a, 1);
  apu_release(&c2, 0); apu_release(&c1, 0); apu_release(&c0, 0);
  close(g_fd);
  return 0;
}
