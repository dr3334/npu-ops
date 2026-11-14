/* 常量权重 FC：A[1,64] -> C[1,64]（权重在 8192 B 常量缓冲里，由资产带上） */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "npu_binop.h"
int main(int argc, char **argv) {
  static int8_t a[64], exp[64], c[64];
  FILE *f;
  if (argc < 3) { fprintf(stderr, "usage: %s in expected\n", argv[0]); return 2; }
  if (!(f = fopen(argv[1], "rb"))) { perror("in"); return 3; }
  if (fread(a, 1, 64, f) != 64) return 4; fclose(f);
  if (!(f = fopen(argv[2], "rb"))) { perror("exp"); return 3; }
  if (fread(exp, 1, 64, f) != 64) return 4; fclose(f);
  memset(c, 0x5A, sizeof(c));
  int rc = npu_binop(NPU_OP_FC_CONST, a, a, c);
  int bad = 0, first = -1;
  for (int i = 0; i < 64; i++) if (c[i] != exp[i]) { bad++; if (first < 0) first = i; }
  printf("FC(constW) rc=%d mismatched=%d/64", rc, bad);
  if (first >= 0) printf("  first@%d dev=%d exp=%d", first, c[first], exp[first]);
  printf("\n");
  return bad ? 1 : 0;
}
