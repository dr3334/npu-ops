/* 一元算子对拍（自建提交器）：读设备上的 golden，跑一遍 npu_binop，逐字节比较。
 *
 *   ./npu_unop_demo <op> <input.bin> <expected.bin>
 *   op = RELU | LOGISTIC | TANH
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
  return got == (size_t)n ? 0 : -2;
}

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s <op> <input> <expected>\n", argv[0]); return 2; }
  npu_op_t op;
  if (!strcmp(argv[1], "RELU")) op = NPU_OP_RELU;
  else if (!strcmp(argv[1], "LOGISTIC")) op = NPU_OP_LOGISTIC;
  else if (!strcmp(argv[1], "TANH")) op = NPU_OP_TANH;
  else { fprintf(stderr, "unknown op %s\n", argv[1]); return 2; }

  int8_t a[64], exp[64], c[64];
  if (load(argv[2], a, 64) || load(argv[3], exp, 64)) return 3;
  memset(c, 0x5A, sizeof(c));
  int rc = npu_binop(op, a, a, c);
  int bad = 0, first = -1;
  for (int i = 0; i < 64; i++) if (c[i] != exp[i]) { bad++; if (first < 0) first = i; }
  printf("%-9s rc=%-4d mismatched=%d/64", argv[1], rc, bad);
  if (first >= 0) printf("  first@%d dev=%d exp=%d", first, c[first], exp[first]);
  printf("\n");
  return bad ? 1 : 0;
}
