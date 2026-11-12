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
  hs_args_t hs;
  /* E-106B-3 实验①: 复制 runtime 的完整握手序列 BASIC x2 -> DEV(1,2,4,6,7) -> MEM(1,2,4,5) */
  for (int i = 0; i < 2; i++) {
    memset(&hs, 0, sizeof(hs)); put32(&hs, 0, 0);
    if (ioctl(g_fd, APU_HANDSHAKE, &hs) < 0) { perror("hs basic"); return -1; }
  }
  printf("handshake version=0x%llx dev=0x%llx mem=0x%llx\n",
         (unsigned long long)get64(&hs, 0), (unsigned long long)get64(&hs, 8),
         (unsigned long long)get64(&hs, 16));
  static const uint32_t devs[] = {1, 2, 4, 6, 7}, mems[] = {1, 2, 4, 5};
  for (unsigned i = 0; i < sizeof(devs) / sizeof(devs[0]); i++) {
    memset(&hs, 0, sizeof(hs)); put32(&hs, 0, 1); put32(&hs, 16, devs[i]);
    if (ioctl(g_fd, APU_HANDSHAKE, &hs) < 0) { perror("hs dev"); return -1; }
  }
  for (unsigned i = 0; i < sizeof(mems) / sizeof(mems[0]); i++) {
    memset(&hs, 0, sizeof(hs)); put32(&hs, 0, 2); put32(&hs, 16, mems[i]);
    if (ioctl(g_fd, APU_HANDSHAKE, &hs) < 0) { perror("hs mem"); return -1; }
  }
  printf("handshake sequence: 2 basic + 5 dev + 4 mem done\n");
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


/* ---------------- 自建 ADD 函数库（E-106B-3 的封装） ----------------
   固定形状：64 个 int8 元素逐位相加（命令流来自 ncc 编译产物，这里只做重定位）。
   返回 0 成功，负数失败；失败原因打印到 stderr。 */
#define NPUADD_OLD_A 0x25969000ULL
#define NPUADD_OLD_B 0x25968000ULL
#define NPUADD_OLD_C 0x2596c000ULL
#define NPUADD_DIR   "/data/local/tmp/npuops/addin"

int npu_add64(const int8_t *a, const int8_t *b, int8_t *out) {
  const char *dir = NPUADD_DIR;
  uint64_t oldv[3] = { NPUADD_OLD_A, NPUADD_OLD_B, NPUADD_OLD_C };
  char p[512];

  if (apu_open_handshake()) return -1;

  buf_t c0 = {0}, c1 = {0}, c2 = {0}, a1 = {0}, b1 = {0}, c = {0}, einfo = {0};
  if (apu_alloc_map(&c0, 368, 0x100, 1)) return -2;
  if (apu_alloc_map(&c1, 32, 0x100, 1)) return -2;
  if (apu_alloc_map(&c2, 16, 0x100, 1)) return -2;
  if (apu_alloc_map(&a1, 4096, 0x100, 1)) return -2;
  if (apu_alloc_map(&b1, 4096, 0x100, 1)) return -2;
  if (apu_alloc_map(&c, 4096, 0x100, 1)) return -2;
  if (apu_alloc_map(&einfo, 76, 0x0, 1)) return -2;
  memset(einfo.host, 0, 76);

  snprintf(p, sizeof(p), "%s/cmd368.bin", dir); if (read_file(p, c0.host, 368)) return -3;
  snprintf(p, sizeof(p), "%s/cmd32.bin", dir);  if (read_file(p, c1.host, 32))  return -3;
  snprintf(p, sizeof(p), "%s/cmd16.bin", dir);  if (read_file(p, c2.host, 16))  return -3;
  memcpy(a1.host, a, 64);
  memcpy(b1.host, b, 64);

  {   /* 写回主机 → 设备前同步 A/B */
    struct { uint64_t flags; } sy;
    int fd;
    fd = (int)a1.handle; sy.flags = (1u << 30) | 2u; ioctl(fd, _IOW(0x62,1,struct { uint64_t flags; }), &sy);
    sy.flags = (1u << 31) | 2u; ioctl(fd, _IOW(0x62,1,struct { uint64_t flags; }), &sy);
    fd = (int)b1.handle; sy.flags = (1u << 30) | 2u; ioctl(fd, _IOW(0x62,1,struct { uint64_t flags; }), &sy);
    sy.flags = (1u << 31) | 2u; ioctl(fd, _IOW(0x62,1,struct { uint64_t flags; }), &sy);
  }

  uint64_t newv[3] = { a1.iova, b1.iova, c.iova };
  int hits = 0;
  patch_addrs((unsigned char *)c0.host, 368, oldv, newv, &hits);
  if (hits != 3) { fprintf(stderr, "npu_add64: patched %d/3 addresses\n", hits); return -4; }

  static unsigned char cbs[3 * 24];
  memset(cbs, 0, sizeof(cbs));
  struct { uint64_t h; uint32_t sz, al, dir; } e[3] = {
      { c1.handle, 32, 0x100, 1 }, { c2.handle, 16, 0x100, 2 }, { c0.handle, 368, 0x100, 1 },
  };
  for (int k = 0; k < 3; k++) {
    memcpy(cbs + k*24 + 0, &e[k].h, 8);
    memcpy(cbs + k*24 + 8, &e[k].sz, 4);
    memcpy(cbs + k*24 + 12, &e[k].al, 4);
    memcpy(cbs + k*24 + 16, &e[k].dir, 4);
  }

  static unsigned char subcmd[80];
  snprintf(p, sizeof(p), "%s/subcmd.bin", dir); if (read_file(p, subcmd, 80)) return -3;
  put32(subcmd, 64, 3);
  put64(subcmd, 72, (uint64_t)(uintptr_t)cbs);

  static unsigned char adj[64];
  memset(adj, 0, sizeof(adj));

  cmd_args_t cmd;
  memset(&cmd, 0, sizeof(cmd));
  snprintf(p, sizeof(p), "%s/cmdin.bin", dir); if (read_file(p, &cmd, 152)) return -3;
  put64(&cmd, 0x10, 0);
  put64(&cmd, 0x18, 0);
  put32(&cmd, 0x48, 1);
  put64(&cmd, 0x50, (uint64_t)(uintptr_t)subcmd);
  put64(&cmd, 0x58, (uint64_t)(uintptr_t)adj);
  put64(&cmd, 0x60, 0xffffffffffffffffULL);
  put64(&cmd, 0x68, einfo.handle);          /* exec_infos = MEM handle（关键） */

  if (ioctl(g_fd, APU_CMD, &cmd) < 0) { perror("CMD RUN"); return -5; }
  usleep(200000);
  {
    struct { uint64_t flags; } sy;
    int fd = (int)c.handle;
    sy.flags = (1u << 30) | 1u; ioctl(fd, _IOW(0x62,1,struct { uint64_t flags; }), &sy);
    sy.flags = (1u << 31) | 1u; ioctl(fd, _IOW(0x62,1,struct { uint64_t flags; }), &sy);
  }
  memcpy(out, c.host, 64);

  /* 驱动不支持 FREE(op=1)，只 UNMAP，避免刷警告 */
  apu_release(&c, 1); apu_release(&b1, 1); apu_release(&a1, 1); apu_release(&einfo, 1);
  apu_release(&c2, 1); apu_release(&c1, 1); apu_release(&c0, 1);
  close(g_fd);
  return 0;
}
