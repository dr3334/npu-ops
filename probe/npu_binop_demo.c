/* 自建提交器覆盖逐元素全家：ADD / MUL / SUB / MAXIMUM / MINIMUM
 * 期望值在本地用 int64 计算并 clip，和 NPU 输出逐字节比较。 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "npu_binop.h"

static int run_one(npu_op_t op, const char *name, const int8_t *a, const int8_t *b) {
  int8_t c[64], exp[64];
  for (int i = 0; i < 64; i++) {
    long long v;
    switch (op) {
      case NPU_OP_ADD: v = (long long)a[i] + b[i]; break;
      case NPU_OP_MUL: v = (long long)a[i] * b[i]; break;
      case NPU_OP_SUB: v = (long long)a[i] - b[i]; break;
      case NPU_OP_MAXIMUM: v = a[i] > b[i] ? a[i] : b[i]; break;
      default: v = a[i] < b[i] ? a[i] : b[i]; break;
    }
    if (v > 127) v = 127;
    if (v < -128) v = -128;
    exp[i] = (int8_t)v;
  }
  memset(c, 0x5A, sizeof(c));
  int rc = npu_binop(op, a, b, c);
  int bad = 0;
  for (int i = 0; i < 64; i++) if (c[i] != exp[i]) bad++;
  printf("%-8s rc=%-4d mismatched=%d/64  C[0..7]=%d %d %d %d %d %d %d %d\n",
         name, rc, bad, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
  return bad ? 1 : 0;
}

int main(void) {
  int8_t a[64], b[64];
  for (int i = 0; i < 64; i++) { a[i] = (int8_t)((i % 8) + 1); b[i] = (int8_t)(10 * ((i % 8) + 1)); }
  int bad = 0;
  bad += run_one(NPU_OP_ADD, "ADD", a, b);
  bad += run_one(NPU_OP_MUL, "MUL", a, b);
  bad += run_one(NPU_OP_SUB, "SUB", a, b);
  bad += run_one(NPU_OP_MAXIMUM, "MAXIMUM", a, b);
  bad += run_one(NPU_OP_MINIMUM, "MINIMUM", a, b);
  printf(bad ? "RESULT: FAIL\n" : "RESULT: ALL PASS\n");
  return bad ? 1 : 0;
}
