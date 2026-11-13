/* probe/npu_binop.c — 通用二元逐元素提交器（E-108E）
 *
 * 泛化自 probe/npuadd.c（E-106B-3）。ioctl 协议、握手序列、cmdbuf 描述符顺序、
 * 地址字（iova>>4）重定位规则、完成等待语义全部沿用 E-106B-3 的实测结论；
 * 这里只把"每个算子不同"的部分抽成 npu_binop_plan_t（见 npu_binop.h）：
 *
 *     asset_dir        命令流资产目录（三个 cmdbuf + subcmd + cmdin，均为 ncc 编译产物）
 *     old_iova[3]      抓取那次运行的 A/B/C IOVA（重定位时的匹配键）
 *     bytes            一次提交的元素数（int8）
 *     alloc_size       A/B/C 分配大小（与抓取时一致，4 KiB）
 *     cmdbuf_dir[3]    cmdbuf 方向；顺序有语义：dir1 / dir2 / dir1（E-106B-3）
 *     patch_idx        哪个 cmdbuf 里含三处地址字
 *     expect_patches   应命中 3 处（A/B/C）
 *
 * 为什么要泛化：四个逐元素算子（ADD/MUL/SUB/MAXIMUM/MINIMUM）编译出来的 .dla 都是
 * "单层、双输入、同形状、输出同形状"，提交路径完全一致；不同之处只有命令流内容和
 * 元素数 —— 也就是 plan 里的字段。这样"自建提交器"不再是一个只服务 ADD 的特例。
 *
 * 证据状态：
 *   * ADD：plan 指向 E-106B-3 抓的 /data/local/tmp/npuops/addin，实测与 npu_add64() 等价
 *          （npubinop demo 自检 C==A+B，见 docs/OP-elementwise.md §7）。
 *   * MUL/SUB/MAXIMUM/MINIMUM：命令流资产**未抓取**，npu_binop() 返回 ENOASSET。
 *          抓取步骤（每算子一次，不改本文件）：
 *            1) 用 probe/run_shim.sh 跑 ops/<op>_1x64_int8/model.dla（LD_PRELOAD=libiolog2.so）；
 *            2) 从 dump 里取出 3 个 cmdbuf、subcmd(80B)、cmdin(152B) 与 A/B/C 的 IOVA；
 *            3) 放到 /data/local/tmp/npuops/binop/<op>/，文件名见下面 g_plan_* 的
 *               cmdbuf_file 字段，再加 subcmd.bin / cmdin.bin；
 *            4) 把该算子的 plan 从 NULL 改成实际参数即可。
 *
 * 仍然 UNKNOWN（沿用 E-106B 的标注）：adj_matrix（cmd_in@0x58）的内容格式（本文件传全零
 * 64 B，E-106B-2 已验证零缓冲可用）；完成等待的真实语义（这里假设 CMD RUN 阻塞 + 200 ms）。
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

#include "npu_binop.h"

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

/* A/B/C 的通用上限：资产目录里 cmdbuf 大小由文件决定，但数据 buffer 沿用 4 KiB 分配
   （E-106B-3 抓取时 runtime 就是 4 KiB；分配更大不会改变命令流语义）。 */
#define NPU_BINOP_ALLOC 4096u
#define NPU_BINOP_CMDBUF_ALIGN 0x100u

static int g_fd = -1;

static void put32(void *b, unsigned o, uint32_t v) { memcpy((char *)b + o, &v, 4); }
static void put64(void *b, unsigned o, uint64_t v) { memcpy((char *)b + o, &v, 8); }
static uint64_t get64(const void *b, unsigned o) { uint64_t v; memcpy(&v, (const char *)b + o, 8); return v; }

typedef struct { uint64_t handle; void *host; uint64_t iova; uint32_t size; } buf_t;

/* ---------------------------------------------------------------- plan 表 */
/* ADD：E-106B-3 抓取的那次运行。三个旧 IOVA 是那次 runtime 分配的地址。 */
static const npu_binop_plan_t g_plan_add = {
    .name = "ADD",
    .asset_dir = "/data/local/tmp/npuops/addin",
    .old_iova = { 0x25969000ULL, 0x25968000ULL, 0x2596c000ULL }, /* A, B, C */
    .bytes = 64,
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 2,
    .cmdbuf_dir = { 1, 2, 1 },   /* 32 B(dir1) / 16 B(dir2) / 368 B(dir1)，顺序有语义 */
    .patch_idx = 2,              /* 地址字在 368 B 那个 cmdbuf 里 */
    .expect_patches = 3,
    .cmdbuf_file = { "cmd32.bin", "cmd16.bin", "cmd368.bin" },
};

/* MUL/SUB/MAXIMUM/MINIMUM：接口就位、资产未抓取 -> NULL（npu_binop 返回 ENOASSET）。
   抓取后把下面四条 NULL 换成与 g_plan_add 同构的 plan 即可（目录约定见文件头注释）。 */
static npu_binop_plan_t g_plan_mul = {
    .name = "MUL",
    .asset_dir = "/data/local/tmp/npuops/assets/mul",
    .old_iova = { 0, 0, 0 },   /* 从资产目录 iova.txt 读（与命令流同一次抓取） */
    .bytes = 64,
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 2,
    .cmdbuf_dir = { 1, 2, 1 },   /* 32 B(dir1) / 16 B(dir2) / 368 B(dir1)，顺序有语义 */
    .patch_idx = 2,
    .expect_patches = 3,
    .cmdbuf_file = { "cmd32.bin", "cmd16.bin", "cmd368.bin" },
};
static npu_binop_plan_t g_plan_sub = {
    .name = "SUB",
    .asset_dir = "/data/local/tmp/npuops/assets/sub",
    .old_iova = { 0, 0, 0 },   /* 从资产目录 iova.txt 读（与命令流同一次抓取） */
    .bytes = 64,
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 2,
    .cmdbuf_dir = { 1, 2, 1 },   /* 32 B(dir1) / 16 B(dir2) / 368 B(dir1)，顺序有语义 */
    .patch_idx = 2,
    .expect_patches = 3,
    .cmdbuf_file = { "cmd32.bin", "cmd16.bin", "cmd368.bin" },
};
static npu_binop_plan_t g_plan_maximum = {
    .name = "MAXIMUM",
    .asset_dir = "/data/local/tmp/npuops/assets/maximum",
    .old_iova = { 0, 0, 0 },   /* 从资产目录 iova.txt 读（与命令流同一次抓取） */
    .bytes = 64,
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 2,
    .cmdbuf_dir = { 1, 2, 1 },   /* 32 B(dir1) / 16 B(dir2) / 368 B(dir1)，顺序有语义 */
    .patch_idx = 2,
    .expect_patches = 3,
    .cmdbuf_file = { "cmd32.bin", "cmd16.bin", "cmd368.bin" },
};
static npu_binop_plan_t g_plan_minimum = {
    .name = "MINIMUM",
    .asset_dir = "/data/local/tmp/npuops/assets/minimum",
    .old_iova = { 0, 0, 0 },   /* 从资产目录 iova.txt 读（与命令流同一次抓取） */
    .bytes = 64,
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 2,
    .cmdbuf_dir = { 1, 2, 1 },   /* 32 B(dir1) / 16 B(dir2) / 368 B(dir1)，顺序有语义 */
    .patch_idx = 2,
    .expect_patches = 3,
    .cmdbuf_file = { "cmd32.bin", "cmd16.bin", "cmd368.bin" },
};
static npu_binop_plan_t g_plan_relu = {
    .name = "RELU",
    .asset_dir = "/data/local/tmp/npuops/assets/relu",
    .old_iova = { 0, 0, 0 },
    .bytes = 64,
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 1,
    .cmdbuf_dir = { 1, 2, 1 },
    .patch_idx = 2,
    .expect_patches = 2,
    .cmdbuf_file = { "cmd0.bin", "cmd1.bin", "cmd2.bin" },
};
static npu_binop_plan_t g_plan_logistic = {
    .name = "LOGISTIC",
    .asset_dir = "/data/local/tmp/npuops/assets/logistic",
    .old_iova = { 0, 0, 0 },
    .bytes = 64,
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 1,
    .cmdbuf_dir = { 1, 2, 1 },
    .patch_idx = 2,
    .expect_patches = 2,
    .cmdbuf_file = { "cmd0.bin", "cmd1.bin", "cmd2.bin" },
};
static npu_binop_plan_t g_plan_tanh = {
    .name = "TANH",
    .asset_dir = "/data/local/tmp/npuops/assets/tanh",
    .old_iova = { 0, 0, 0 },
    .bytes = 64,
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 1,
    .cmdbuf_dir = { 1, 2, 1 },
    .patch_idx = 2,
    .expect_patches = 2,
    .cmdbuf_file = { "cmd0.bin", "cmd1.bin", "cmd2.bin" },
};
static npu_binop_plan_t g_plan_fcdyn = {
    .name = "FULLY_CONNECTED(dynW)",
    .asset_dir = "/data/local/tmp/npuops/assets/fcdyn",
    .old_iova = { 0, 0, 0 },
    .bytes = 64,
    .bytes_b = 4096,
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 2,
    .cmdbuf_dir = { 1, 2, 1 },
    .patch_idx = 2,
    .expect_patches = 3,
    .cmdbuf_file = { "cmd0.bin", "cmd1.bin", "cmd2.bin" },
};
static npu_binop_plan_t g_plan_bmm_qk = {
    .name = "BATCH_MATMUL(QK^T T=8 D=64)",
    .asset_dir = "/data/local/tmp/npuops/assets/bmm_qk",
    .old_iova = { 0, 0, 0 },
    .bytes = 512,          /* [1,8,64] int8 */
    .bytes_b = 512,
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 2,
    .cmdbuf_dir = { 1, 2, 1 },
    .patch_idx = 2,
    .expect_patches = 3,
    .cmdbuf_file = { "cmd0.bin", "cmd1.bin", "cmd2.bin" },
};
static npu_binop_plan_t g_plan_softmax16 = {
    .name = "SOFTMAX [1,16,16]",
    .asset_dir = "/data/local/tmp/npuops/assets/softmax16",
    .old_iova = { 0, 0, 0 },
    .bytes = 256,
    .extra_file = "scratch.bin",
    .bytes_b = 4096,
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 2,
    .cmdbuf_dir = { 1, 2, 1 },
    .patch_idx = 2,
    .expect_patches = 3,
    .cmdbuf_file = { "cmd0.bin", "cmd1.bin", "cmd2.bin" },
};
static npu_binop_plan_t g_plan_bmm_qk16 = {
    .name = "BATCH_MATMUL(QK^T T=16)",
    .asset_dir = "/data/local/tmp/npuops/assets/bmm_qk16",
    .old_iova = { 0, 0, 0 },
    .bytes = 1024,
    .bytes_b = 1024,
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 2,
    .cmdbuf_dir = { 1, 2, 1 },
    .patch_idx = 2,
    .expect_patches = 3,
    .cmdbuf_file = { "cmd0.bin", "cmd1.bin", "cmd2.bin" },
};
static npu_binop_plan_t g_plan_bmm_pv16 = {
    .name = "BATCH_MATMUL(PV T=16)",
    .asset_dir = "/data/local/tmp/npuops/assets/bmm_pv16",
    .old_iova = { 0, 0, 0 },
    .bytes = 1024,
    .bytes_b = 1024,
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 2,
    .cmdbuf_dir = { 1, 2, 1 },
    .patch_idx = 2,
    .expect_patches = 3,
    .cmdbuf_file = { "cmd0.bin", "cmd1.bin", "cmd2.bin" },
};
static npu_binop_plan_t g_plan_fc_const = {
    .name = "FULLY_CONNECTED(const W)",
    .asset_dir = "/data/local/tmp/npuops/assets/fc_const",
    .old_iova = { 0, 0, 0 },
    .bytes = 64,
    .extra_file = "scratch.bin",
    .bytes_b = 8192,          /* 常量权重缓冲 */
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 2,            /* A + 常量权重缓冲 */
    .cmdbuf_dir = { 1, 2, 1 },
    .patch_idx = 2,
    .expect_patches = 3,
    .cmdbuf_file = { "cmd0.bin", "cmd1.bin", "cmd2.bin" },
};
static npu_binop_plan_t g_plan_fc512 = {
    .name = "FULLY_CONNECTED(512x512 dynW)",
    .asset_dir = "/data/local/tmp/npuops/assets/fc_dyn512",
    .old_iova = { 0, 0, 0 },
    .bytes = 512,
    .bytes_b = 262144,       /* 512x512 int8 权重走另一段地址（0xcbbc0000） */
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 2,
    .cmdbuf_dir = { 1, 2, 1 },
    .patch_idx = 2,
    .expect_patches = 3,
    .cmdbuf_file = { "cmd0.bin", "cmd1.bin", "cmd2.bin" },
};
static npu_binop_plan_t g_plan_fc1024 = {
    .name = "FULLY_CONNECTED(1024x1024 dynW)",
    .asset_dir = "/data/local/tmp/npuops/assets/fc_dyn1024",
    .old_iova = { 0, 0, 0 },
    .bytes = 1024,
    .bytes_b = 1048576,
    .alloc_size = NPU_BINOP_ALLOC,
    .n_inputs = 2,
    .cmdbuf_dir = { 1, 2, 1 },
    .patch_idx = 2,
    .expect_patches = 3,
    .cmdbuf_file = { "cmd0.bin", "cmd1.bin", "cmd2.bin" },
};
static const npu_binop_plan_t *const g_plans[NPU_OP__COUNT] = {
    [NPU_OP_ADD] = &g_plan_add,
    [NPU_OP_MUL] = &g_plan_mul,
    [NPU_OP_SUB] = &g_plan_sub,
    [NPU_OP_MAXIMUM] = &g_plan_maximum,
    [NPU_OP_MINIMUM] = &g_plan_minimum,
    [NPU_OP_RELU] = &g_plan_relu,
    [NPU_OP_LOGISTIC] = &g_plan_logistic,
    [NPU_OP_TANH] = &g_plan_tanh,
    [NPU_OP_FC_DYNW] = &g_plan_fcdyn,
    [NPU_OP_BMM_QK] = &g_plan_bmm_qk,
    [NPU_OP_SOFTMAX16] = &g_plan_softmax16,
    [NPU_OP_BMM_QK16] = &g_plan_bmm_qk16,
    [NPU_OP_BMM_PV16] = &g_plan_bmm_pv16,
    [NPU_OP_FC_CONST] = &g_plan_fc_const,
    [NPU_OP_FC512] = &g_plan_fc512,
    [NPU_OP_FC1024] = &g_plan_fc1024,
};

static const char *const g_op_names[NPU_OP__COUNT] = {
    "ADD", "MUL", "SUB", "MAXIMUM", "MINIMUM",
};

const npu_binop_plan_t *npu_binop_plan(npu_op_t op) {
  if ((int)op < 0 || op >= NPU_OP__COUNT) return NULL;
  return g_plans[op];
}

uint32_t npu_binop_elems(npu_op_t op) {
  const npu_binop_plan_t *p = npu_binop_plan(op);
  return p ? p->bytes : 0;
}

/* ------------------------------------------------------------ ioctl 基础 */
static int apu_open_handshake(void) {
  static const uint32_t devs[] = {1, 2, 4, 6, 7}, mems[] = {1, 2, 4, 5};
  hs_args_t hs;

  g_fd = open("/dev/apusys", O_RDWR | O_SYNC);
  if (g_fd < 0) { perror("open /dev/apusys"); return -1; }

  /* E-106B-3 实验①: 完整握手序列 BASIC x2 -> DEV(1,2,4,6,7) -> MEM(1,2,4,5) */
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

/* 驱动不支持 MEM op=1 (FREE)，只 UNMAP（E-106B-3 的结论，避免内核刷警告）。 */
static void apu_release(buf_t *b) {
  mem_args_t m;
  if (!b->handle) return;
  memset(&m, 0, sizeof(m)); put32(&m, 0, 3); put64(&m, 16, b->handle);
  ioctl(g_fd, APU_MEM, &m);
  if (b->host) munmap(b->host, b->size);
  memset(b, 0, sizeof(*b));
}

/* DMA_BUF_IOCTL_SYNC(flags)；1<<30 = START, 1<<31 = END，低位 1=READ / 2=WRITE */
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

/* 把命令流里出现的旧 IOVA 换成新的：命令流里存的是 iova>>4（E-106B-3），
   所以按 4 字节对齐扫描 32 位字比较。 */
static int patch_addrs(unsigned char *cmd, unsigned size,
                       const uint64_t oldv[3], const uint64_t newv[3], unsigned *hits) {
  *hits = 0;
  for (unsigned o = 0; o + 4 <= size; o += 4) {
    uint32_t w;
    memcpy(&w, cmd + o, 4);
    for (int k = 0; k < 3; k++) {
      if (oldv[k] == 0) continue;   /* 一元算子的 B 槽没有地址；否则 0 会匹配满屏零字 */
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

/* ------------------------------------------------------------ 通用提交器 */
int npu_bin_binop(const npu_binop_plan_t *plan, const int8_t *a, const int8_t *b, int8_t *out) {
  char p[512];
  buf_t cb[3] = {{0}};
  buf_t a1 = {0}, b1 = {0}, c = {0}, einfo = {0};
  uint32_t cb_size[3] = {0, 0, 0};
  static unsigned char cbs[3 * 24];
  static unsigned char subcmd[256];
  static unsigned char adj[64];
  cmd_args_t cmd;
  uint32_t subcmd_size = 0;
  unsigned hits = 0;
  int rc = NPU_BINOP_OK;

  if (!plan || !a || !b || !out) return NPU_BINOP_EINVAL;
  if (plan->bytes == 0 || plan->bytes > plan->alloc_size) return NPU_BINOP_EINVAL;

  /* old IOVA 与命令流是同一次抓取的配对数据：默认从资产目录 iova.txt 读，
     这样重新抓取资产不需要改代码；plan 里显式给了非零值则用 plan 的。 */
  uint64_t old_iova[3] = { plan->old_iova[0], plan->old_iova[1], plan->old_iova[2] };
  if (old_iova[0] == 0 || old_iova[2] == 0 || (plan->n_inputs == 2 && old_iova[1] == 0)) {
    snprintf(p, sizeof(p), "%s/iova.txt", plan->asset_dir);
    FILE *f = fopen(p, "r");
    if (!f) { fprintf(stderr, "%s: missing (needs 3 hex IOVAs)\n", p); return NPU_BINOP_EASSET; }
    unsigned long long x = 0, y = 0, z = 0;
    if (fscanf(f, "%llx %llx %llx", &x, &y, &z) != 3) { fclose(f); return NPU_BINOP_EASSET; }
    fclose(f);
    old_iova[0] = x; old_iova[1] = y; old_iova[2] = z;
  }

  /* --- 资产大小先看清楚，再分配（这样 cmdbuf 大小不写死） --- */
  for (int k = 0; k < 3; k++) {
    snprintf(p, sizeof(p), "%s/cmd%d.bin", plan->asset_dir, k);
    if (file_size(p, &cb_size[k])) {
      snprintf(p, sizeof(p), "%s/%s", plan->asset_dir, plan->cmdbuf_file[k]);
    }
    if (file_size(p, &cb_size[k])) { fprintf(stderr, "%s: missing asset\n", p); return NPU_BINOP_EASSET; }
  }
  snprintf(p, sizeof(p), "%s/subcmd.bin", plan->asset_dir);
  if (file_size(p, &subcmd_size) || subcmd_size > sizeof(subcmd)) return NPU_BINOP_EASSET;

  if (apu_open_handshake()) { rc = NPU_BINOP_EIO; goto done; }

  /* --- 1. 三个 cmdbuf：分配 + 映射 + 读入 ncc 生成的内容 --- */
  for (int k = 0; k < 3; k++) {
    if (apu_alloc_map(&cb[k], cb_size[k], NPU_BINOP_CMDBUF_ALIGN, 1)) { rc = NPU_BINOP_EIO; goto done; }
  }
  for (int k = 0; k < 3; k++) {
    snprintf(p, sizeof(p), "%s/%s", plan->asset_dir, plan->cmdbuf_file[k]);
    if (read_file(p, cb[k].host, cb_size[k])) { rc = NPU_BINOP_EASSET; goto done; }
  }

  /* --- 2. 数据 buffer：A/B/C（+ exec_infos，必须给 MEM handle，见 E-106B 卡点 1） --- */
  if (apu_alloc_map(&a1, plan->alloc_size, NPU_BINOP_CMDBUF_ALIGN, 1)) { rc = NPU_BINOP_EIO; goto done; }
  if (plan->n_inputs != 2 && plan->n_inputs != 1) return NPU_BINOP_EINVAL;
  {
    uint32_t bsize = plan->bytes_b > plan->alloc_size ? plan->bytes_b : plan->alloc_size;
    if (plan->n_inputs == 2 &&
        apu_alloc_map(&b1, bsize, NPU_BINOP_CMDBUF_ALIGN, 1)) { rc = NPU_BINOP_EIO; goto done; }
  }
  if (apu_alloc_map(&c,  plan->alloc_size, NPU_BINOP_CMDBUF_ALIGN, 1)) { rc = NPU_BINOP_EIO; goto done; }
  if (apu_alloc_map(&einfo, 76, 0, 1)) { rc = NPU_BINOP_EIO; goto done; }
  memset(einfo.host, 0, 76);

  memcpy(a1.host, a, plan->bytes);
  if (plan->n_inputs == 2) {
    if (plan->extra_file && plan->extra_file[0]) {
      snprintf(p, sizeof(p), "%s/%s", plan->asset_dir, plan->extra_file);
      FILE *fx = fopen(p, "rb");
      uint32_t want = plan->bytes_b ? plan->bytes_b : plan->bytes;
      if (!fx || fread(b1.host, 1, want, fx) != want) {
        if (fx) fclose(fx);
        fprintf(stderr, "%s: cannot load extra buffer\n", p);
        rc = NPU_BINOP_EASSET; goto done;
      }
      fclose(fx);
    } else {
      memcpy(b1.host, b, plan->bytes_b ? plan->bytes_b : plan->bytes);
    }
  }

  /* 写回主机 -> 设备前同步 A/B（START|WRITE + END|WRITE） */
  /* 命令缓冲同样要 flush：它们也是 dma-buf，之前漏了同步，导致偶发读到陈旧命令流 */
  for (int k2 = 0; k2 < 3; k2++) {
    if (!cb[k2].handle) continue;
    dma_buf_sync(cb[k2].handle, (1u << 30) | 2u);
    dma_buf_sync(cb[k2].handle, (1u << 31) | 2u);
  }
  dma_buf_sync(a1.handle, (1u << 30) | 2u);
  dma_buf_sync(a1.handle, (1u << 31) | 2u);
  if (plan->n_inputs == 2) {
    dma_buf_sync(b1.handle, (1u << 30) | 2u);
    dma_buf_sync(b1.handle, (1u << 31) | 2u);
  }

  /* --- 3. 地址重定位：把抓取时的 A/B/C IOVA 换成这次的 --- */
  {
    uint64_t newv[3] = { a1.iova, plan->n_inputs == 2 ? b1.iova : 0, c.iova };
    int k = plan->patch_idx;
    if (k < 0 || k > 2) { rc = NPU_BINOP_EINVAL; goto done; }
    patch_addrs((unsigned char *)cb[k].host, cb_size[k], old_iova, newv, &hits);
    /* 调试：把重定位后的命令流落盘，便于和厂商抓取做字节级 diff（NPU_DUMP_DIR=/tmp/x） */
    {
      const char *dd = getenv("NPU_DUMP_DIR");
      if (dd && dd[0]) {
        char dp[600];
        snprintf(dp, sizeof(dp), "%s/%s_cmd%d.bin", dd, plan->name, k);
        /* 文件名里不要空格/括号 */
        for (char *q2 = dp; *q2; q2++) {
          if (*q2 == ' ' || *q2 == '(' || *q2 == ')' || *q2 == '[' || *q2 == ']' || *q2 == '^' || *q2 == ',') *q2 = '_';
        }
        FILE *fd2 = fopen(dp, "wb");
        if (fd2) { fwrite(cb[k].host, 1, cb_size[k], fd2); fclose(fd2); }
      }
    }
    if (hits != (unsigned)plan->expect_patches) {
      fprintf(stderr, "npu_binop(%s): patched %u/%d addresses (command stream mismatch?)\n",
              plan->name, hits, plan->expect_patches);
      rc = NPU_BINOP_EPATCH;
      goto done;
    }
  }

  /* --- 4. cmdbuf 描述符：顺序有语义 dir1 / dir2 / dir1 --- */
  memset(cbs, 0, sizeof(cbs));
  for (int k = 0; k < 3; k++) {
    uint64_t h = cb[k].handle;
    uint32_t sz = cb_size[k], al = NPU_BINOP_CMDBUF_ALIGN, dir = (uint32_t)plan->cmdbuf_dir[k];
    memcpy(cbs + k * 24 + 0, &h, 8);
    memcpy(cbs + k * 24 + 8, &sz, 4);
    memcpy(cbs + k * 24 + 12, &al, 4);
    memcpy(cbs + k * 24 + 16, &dir, 4);
  }

  /* --- 5. subcmd_info：从抓取模板读入，只覆盖 num_cmdbufs 与数组指针 --- */
  memset(subcmd, 0, sizeof(subcmd));
  snprintf(p, sizeof(p), "%s/subcmd.bin", plan->asset_dir);
  if (read_file(p, subcmd, subcmd_size)) { rc = NPU_BINOP_EASSET; goto done; }
  put32(subcmd, subcmd_size - 16, 3);
  put64(subcmd, subcmd_size - 8, (uint64_t)(uintptr_t)cbs);

  memset(adj, 0, sizeof(adj));  /* adj_matrix 全零（E-106B-2 已验证） */

  /* --- 6. mdw_cmd_in --- */
  memset(&cmd, 0, sizeof(cmd));
  snprintf(p, sizeof(p), "%s/cmdin.bin", plan->asset_dir);
  if (read_file(p, &cmd, sizeof(cmd))) { rc = NPU_BINOP_EASSET; goto done; }
  put64(&cmd, 0x10, 0);
  put64(&cmd, 0x18, 0);
  put32(&cmd, 0x48, 1);
  put64(&cmd, 0x50, (uint64_t)(uintptr_t)subcmd);
  put64(&cmd, 0x58, (uint64_t)(uintptr_t)adj);
  put64(&cmd, 0x60, 0xffffffffffffffffULL);
  put64(&cmd, 0x68, einfo.handle);   /* exec_infos 必须是 MEM handle（E-106B 卡点 1） */

  if (ioctl(g_fd, APU_CMD, &cmd) < 0) { perror("CMD RUN"); rc = NPU_BINOP_ECALL; goto done; }

  usleep(200000);                    /* rc=0 只代表提交成功（E-106B 卡点 3） */
  dma_buf_sync(c.handle, (1u << 30) | 1u);
  dma_buf_sync(c.handle, (1u << 31) | 1u);
  memcpy(out, c.host, plan->bytes);
  rc = NPU_BINOP_OK;

done:
  apu_release(&c); apu_release(&b1); apu_release(&a1); apu_release(&einfo);
  apu_release(&cb[2]); apu_release(&cb[1]); apu_release(&cb[0]);
  if (g_fd >= 0) { close(g_fd); g_fd = -1; }
  return rc;
}

int npu_binop(npu_op_t op, const int8_t *a, const int8_t *b, int8_t *out) {
  const npu_binop_plan_t *p = npu_binop_plan(op);
  if (!p) {
    fprintf(stderr,
            "npu_binop(%s): 还没有抓取该算子的命令流资产（返回 ENOASSET）。\n"
            "  抓取方式见 probe/npu_binop.c 头注释（iolog2 + run_shim，每算子一次）。\n",
            ((int)op >= 0 && op < NPU_OP__COUNT) ? g_op_names[op] : "?");
    return NPU_BINOP_ENOASSET;
  }
  return npu_bin_binop(p, a, b, out);
}

/* -------------------------------------------------------------- demo/main */
#ifdef NPU_BINOP_DEMO
/* 自检：ADD 用与 probe/npuadd_demo.c 相同的输入 A=[10,20,...,80]x8、B=[1..8]x8，
   期望 C=[11,22,...,88]x8；再对四个尚未抓取资产的算子演示接口（应全部返回 ENOASSET）。 */
int main(void) {
  int8_t A[64], B[64], C[64], want[64];
  int bad = 0;

  for (int i = 0; i < 64; i++) {
    A[i] = (int8_t)(10 * (i % 8 + 1));
    B[i] = (int8_t)(i % 8 + 1);
    want[i] = A[i] + B[i];
  }

  printf("== npu_binop(ADD) via plan '%s' ==\n", g_plan_add.name);
  int rc = npu_binop(NPU_OP_ADD, A, B, C);
  printf("npu_binop(ADD) rc=%d\n", rc);
  if (rc == NPU_BINOP_OK) {
    printf("C[0..7] =");
    for (int i = 0; i < 8; i++) printf(" %d", C[i]);
    printf("\n");
    for (int i = 0; i < 64; i++) if (C[i] != want[i]) bad++;
    printf("golden check: %s (%d/64 mismatched)\n", bad ? "FAIL" : "PASS", bad);
  }

  for (int op = NPU_OP_MUL; op <= NPU_OP_MINIMUM; op++) {
    int8_t o[64];
    int r = npu_binop((npu_op_t)op, A, B, o);
    printf("npu_binop(%s) rc=%d %s\n", g_op_names[op], r,
           r == NPU_BINOP_ENOASSET ? "(assets not captured yet)" : "");
  }
  return (rc == NPU_BINOP_OK && bad == 0) ? 0 : 1;
}
#endif

int npu_bin_unop(const npu_binop_plan_t *plan, const int8_t *a, int8_t *out) {
  return npu_bin_binop(plan, a, a, out);   /* b 不会被使用：plan->n_inputs == 1 */
}
