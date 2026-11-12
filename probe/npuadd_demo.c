#include <stdio.h>
#include <stdint.h>
#include "npuadd.h"
int main(void) {
  int8_t a[64], b[64], c[64];
  for (int i = 0; i < 64; i++) { a[i] = (int8_t)(10 * ((i % 8) + 1)); b[i] = (int8_t)((i % 8) + 1); }
  int rc = npu_add64(a, b, c);
  printf("npu_add64 rc=%d\n", rc);
  if (rc == 0) { printf("C[0..7] ="); for (int i = 0; i < 8; i++) printf(" %d", c[i]); printf("\n"); }
  return rc ? 1 : 0;
}
