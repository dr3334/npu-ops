/* 自建提交器跑 FULLY_CONNECTED（动态权重）：A[1,64] int8 · W[64,64] int8 -> C[1,64]
 *
 *   ./npu_fc_demo <input.bin> <weights.bin> <expected.bin>
 *
 * 期望值来自 golden（独立生成），这里只做逐字节比较。
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "npu_binop.h"

static int load(const char *p, int8_t *buf, int n) {
  FILE *f = fopen(p, "rb");
  if (!f) { perror(p); return -1; }
  size_t got = fread(buf, 1, (size_t)n, f);
  fclose(f);
  if (got != (size_t)n) { fprintf(stderr, "%s: short read %zu/%d\n", p, got, n); return -2; }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s <input> <weights> <expected>\n", argv[0]); return 2; }
  static int8_t a[64], w[4096], exp[64], c[64];
  if (load(argv[1], a, 64) || load(argv[2], w, 4096) || load(argv[3], exp, 64)) return 3;
  memset(c, 0x5A, sizeof(c));
  int rc = npu_binop(NPU_OP_FC_DYNW, a, w, c);
  int bad = 0, first = -1;
  for (int i = 0; i < 64; i++) if (c[i] != exp[i]) { bad++; if (first < 0) first = i; }
  printf("FC(dynW) rc=%-4d mismatched=%d/64", rc, bad);
  if (first >= 0) printf("  first@%d dev=%d exp=%d", first, c[first], exp[first]);
  printf("\n");
  return bad ? 1 : 0;
}
