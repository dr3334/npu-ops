/* 1 MB 权重的动态 FC：A[1024] x W[1024x1024] -> C[1024] */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "npu_binop.h"
static int load(const char *p, int8_t *b, int n) {
  FILE *f = fopen(p, "rb"); if (!f) { perror(p); return -1; }
  size_t got = fread(b, 1, (size_t)n, f); fclose(f);
  return got == (size_t)n ? 0 : -2;
}
int main(int argc, char **argv) {
  static int8_t a[1024], w[1048576], exp[1024], c[1024];
  if (argc < 4) { fprintf(stderr, "usage: %s in weights expected\n", argv[0]); return 2; }
  if (load(argv[1], a, 1024) || load(argv[2], w, 1048576) || load(argv[3], exp, 1024)) return 3;
  memset(c, 0x5A, sizeof(c));
  int rc = npu_binop(NPU_OP_FC1024, a, w, c);
  int bad = 0, first = -1;
  for (int i = 0; i < 1024; i++) if (c[i] != exp[i]) { bad++; if (first < 0) first = i; }
  printf("FC1024 rc=%d mismatched=%d/1024", rc, bad);
  if (first >= 0) printf("  first@%d dev=%d exp=%d", first, c[first], exp[first]);
  printf("\n");
  return 0;
}
