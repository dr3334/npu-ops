/* 单位置校验：softmax 资产 vs 抓取时厂商 runtime 的同一输入输出 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "npu_binop.h"
int main(int argc, char **argv) {
  static int8_t in[256], out[256], vendor[256];
  FILE *f;
  if (argc < 3) { fprintf(stderr, "usage: %s in vendor_out\n", argv[0]); return 2; }
  if (!(f = fopen(argv[1], "rb"))) { perror("in"); return 3; }
  if (fread(in, 1, 256, f) != 256) return 4; fclose(f);
  if (!(f = fopen(argv[2], "rb"))) { perror("vendor"); return 3; }
  if (fread(vendor, 1, 256, f) != 256) return 4; fclose(f);
  memset(out, 0x5A, sizeof(out));
  int rc = npu_binop(NPU_OP_SOFTMAX16, in, in, out);
  int bad = 0, first = -1;
  for (int i = 0; i < 256; i++) if (out[i] != vendor[i]) { bad++; if (first < 0) first = i; }
  printf("softmax rc=%d mismatched=%d/256", rc, bad);
  if (first >= 0) printf("  first@%d dev=%d vendor=%d", first, out[first], vendor[first]);
  printf("\n  dev[0..7]=%d %d %d %d %d %d %d %d\n", out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
  return 0;
}
