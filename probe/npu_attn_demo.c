/* 自建提交器拼"注意力核心"：FC(投影) → FC(投影) → BATCH_MATMUL(QKᵀ)
 *
 *   ./npu_attn_demo <x.bin> <Wq.bin> <Wk.bin> <qk_expected.bin>
 *
 * 用法说明：
 *   * 第 1 段（单算子校验）：把 <x.bin> 当作已经算好的 a[1,8,64]、另一个文件当 b，
 *     跑 BMM 并与 golden 逐字节比对（验证 BMM 资产本身）。
 *   * 第 2 段（串联）：x[64] → FC(Wq) → q[64]（8 个 token 共用同一份 x 仅作演示不行，
 *     所以这里用 8 个 token 的 x 矩阵逐行做 GEMV），再 FC(Wk) → k，最后 BMM(q,k) → s[8,8]。
 *     参考值用**我们实际产生的 int8 q/k** 在 int64 下计算——这样只验证"串联链路"，
 *     不把 FC 的 requant ±1 混进来。
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "npu_binop.h"

#define T 8
#define D 64

static int load(const char *p, int8_t *buf, int n) {
  FILE *f = fopen(p, "rb");
  if (!f) { perror(p); return -1; }
  size_t got = fread(buf, 1, (size_t)n, f);
  fclose(f);
  if (got != (size_t)n) { fprintf(stderr, "%s: short read %zu/%d\n", p, got, n); return -2; }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 7) {
    fprintf(stderr, "usage: %s <a[8x64]> <b[8x64]> <qk_expected[8x8]> <x[8x64]> <Wq[64x64]> <Wk[64x64]>\n", argv[0]);
    return 2;
  }
  static int8_t x[T * D], Wq[D * D], Wk[D * D], exp_s[T * T], ga[T * D], gb[T * D];
  if (load(argv[1], ga, T * D) || load(argv[2], gb, T * D) || load(argv[3], exp_s, T * T) ||
      load(argv[4], x, T * D) || load(argv[5], Wq, D * D) || load(argv[6], Wk, D * D)) return 3;

  /* ---- 1. BMM 单算子：直接把 golden 的 a/b 当输入喂进去 ---- */
  static int8_t a[T * D], b[T * D], s_dev[T * T];
  memcpy(a, ga, sizeof(a));
  memcpy(b, gb, sizeof(b));
  memset(s_dev, 0x5A, sizeof(s_dev));
  int rc1 = npu_binop(NPU_OP_BMM_QK, a, b, s_dev);
  int bad1 = 0;
  for (int i = 0; i < T * T; i++) if (s_dev[i] != exp_s[i]) bad1++;
  printf("stage1 BMM(golden a,b)  rc=%-3d mismatched=%d/%d\n", rc1, bad1, T * T);

  /* ---- 2. 串联：q = x·Wqᵀ, k = x·Wkᵀ（逐 token GEMV），s = q·kᵀ ---- */
  static int8_t q[T * D], k[T * D];
  int rc2 = 0;
  for (int t = 0; t < T && rc2 == 0; t++) {
    rc2 |= npu_binop(NPU_OP_FC_DYNW, x + t * D, Wq, q + t * D);
  }
  for (int t = 0; t < T && rc2 == 0; t++) {
    rc2 |= npu_binop(NPU_OP_FC_DYNW, x + t * D, Wk, k + t * D);
  }
  /* 参考：用我们实际产生的 q/k 做 int64 精确乘加，再夹到 int8 */
  static int8_t s_ref[T * T];
  for (int i = 0; i < T; i++)
    for (int j = 0; j < T; j++) {
      long long acc = 0;
      for (int d = 0; d < D; d++) acc += (int)q[i * D + d] * (int)k[j * D + d];
      if (acc > 127) acc = 127;
      if (acc < -128) acc = -128;
      s_ref[i * T + j] = (int8_t)acc;
    }
  memset(s_dev, 0x5A, sizeof(s_dev));
  int rc3 = npu_binop(NPU_OP_BMM_QK, q, k, s_dev);
  int bad3 = 0, first = -1;
  for (int i = 0; i < T * T; i++)
    if (s_dev[i] != s_ref[i]) { bad3++; if (first < 0) first = i; }
  printf("stage2 FC->FC->BMM      rc=%-3d mismatched=%d/%d", rc3, bad3, T * T);
  if (first >= 0) printf("  first@%d dev=%d ref=%d", first, s_dev[first], s_ref[first]);
  printf("\n");
  printf("RESULT: %s\n", (bad1 == 0 && bad3 == 0) ? "PASS" : "FAIL");
  return (bad1 == 0 && bad3 == 0) ? 0 : 1;
}
