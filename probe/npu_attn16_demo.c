/* 自建提交器：完整 attention 链路（int8, T=16, D=64）
 *
 *   ./npu_attn16_demo <x[16x64]> <Wq[64x64]> <Wk[64x64]> <Wv[64x64]>
 *
 * 链路（全部由我们自己的 ioctl 提交，无 libneuron/neuronrt）：
 *   q,k,v = x·Wᵀ   （逐 token，16×3 = 48 次 FULLY_CONNECTED 动态权重）
 *   s     = q·kᵀ   （BATCH_MATMUL T=16）
 *   p     = softmax(s)（SOFTMAX [1,16,16] int8）
 *   o     = p·v    （BATCH_MATMUL PV T=16）
 *
 * 数值参考：用**设备实际产生的 int8 q/k/v** 在 float64/int64 下重算，
 * 这样只验证链路本身，不把 FC 的 requant ±1 code 混进来（该偏差已单独记录）。
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "npu_binop.h"

#define T 16
#define D 64

static int load(const char *p, int8_t *buf, int n) {
  FILE *f = fopen(p, "rb");
  if (!f) { perror(p); return -1; }
  size_t got = fread(buf, 1, (size_t)n, f);
  fclose(f);
  if (got != (size_t)n) { fprintf(stderr, "%s: short read %zu/%d\n", p, got, n); return -2; }
  return 0;
}

static int8_t x[T * D], Wq[D * D], Wk[D * D], Wv[D * D];
static int8_t q[T * D], k[T * D], v[T * D];
static int8_t s[T * T], p[T * T], o[T * D];

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <x[16x64]> <Wq> <Wk> <Wv>\n", argv[0]);
    return 2;
  }
  if (load(argv[1], x, T * D) || load(argv[2], Wq, D * D) ||
      load(argv[3], Wk, D * D) || load(argv[4], Wv, D * D)) return 3;

  int rc = 0;
  for (int t = 0; t < T && rc == 0; t++) rc |= npu_binop(NPU_OP_FC_DYNW, x + t * D, Wq, q + t * D);
  for (int t = 0; t < T && rc == 0; t++) rc |= npu_binop(NPU_OP_FC_DYNW, x + t * D, Wk, k + t * D);
  for (int t = 0; t < T && rc == 0; t++) rc |= npu_binop(NPU_OP_FC_DYNW, x + t * D, Wv, v + t * D);
  printf("stage q/k/v (48 submissions) rc=%d\n", rc);

  rc |= npu_binop(NPU_OP_BMM_QK16, q, k, s);
  printf("stage s = q*k^T              rc=%d\n", rc);

  rc |= npu_binop(NPU_OP_SOFTMAX16, s, s, p);
  printf("stage p = softmax(s)         rc=%d\n", rc);

  /* softmax 的输出域是 (scale 1/256, zp -128)，PV 模型的 A 域是 (1/127, zp 0)：
     中间要做一次 requant（TFLite 图里由 ncc 插入，手搓链路就得自己做）。 */
  static int8_t p_pv[T * T];
  for (int i = 0; i < T * T; i++) {
    double real = ((double)p[i] + 128.0) / 256.0;       /* 概率 */
    int qv = (int)lround(real * 127.0);
    if (qv > 127) qv = 127;
    if (qv < -128) qv = -128;
    p_pv[i] = (int8_t)qv;
  }
  rc |= npu_binop(NPU_OP_BMM_PV16, p_pv, v, o);
  printf("stage o = p*v                rc=%d\n", rc);

  /* ---- 参考：用设备实际产出的 int8 q/k/v 重算 ---- */
  /* 1) scores: int64 精确乘加，按 QK 模型的 int8 输出夹取（与 bmm 的 golden 同一口径） */
  static int8_t s_ref[T * T];
  for (int i = 0; i < T; i++)
    for (int j = 0; j < T; j++) {
      long long acc = 0;
      for (int d = 0; d < D; d++) acc += (int)q[i * D + d] * (int)k[j * D + d];
      if (acc > 127) acc = 127;
      if (acc < -128) acc = -128;
      s_ref[i * T + j] = (int8_t)acc;
    }
  int bad_s = 0;
  for (int i = 0; i < T * T; i++) if (s[i] != s_ref[i]) bad_s++;

  /* 2) softmax: 在 float 下按模型量化参数（in 1/16, out 1/256, zp -128）重算 */
  static int8_t p_ref[T * T];
  for (int i = 0; i < T; i++) {
    double mx = -1e9;
    for (int j = 0; j < T; j++) { double sv = s_ref[i * T + j] / 16.0; if (sv > mx) mx = sv; }
    double sum = 0;
    for (int j = 0; j < T; j++) sum += exp(s_ref[i * T + j] / 16.0 - mx);
    for (int j = 0; j < T; j++) {
      double pv = exp(s_ref[i * T + j] / 16.0 - mx) / sum;
      int qv = (int)lround(pv * 256.0) - 128;
      if (qv > 127) qv = 127;
      if (qv < -128) qv = -128;
      p_ref[i * T + j] = (int8_t)qv;
    }
  }
  int bad_p = 0;
  for (int i = 0; i < T * T; i++) if (p[i] != p_ref[i]) bad_p++;
  printf("  p[0..7] = %d %d %d %d %d %d %d %d\n", p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);
  printf("  p_ref[0..7] = %d %d %d %d %d %d %d %d\n", p_ref[0],p_ref[1],p_ref[2],p_ref[3],p_ref[4],p_ref[5],p_ref[6],p_ref[7]);
  printf("  s_ref[0..7] = %d %d %d %d %d %d %d %d   s[0..7]= %d %d %d %d %d %d %d %d\n",
         s_ref[0],s_ref[1],s_ref[2],s_ref[3],s_ref[4],s_ref[5],s_ref[6],s_ref[7],
         s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7]);

  /* 3) o = p·v，按 PV 模型的量化域：a=1/127, b=1/127, out=6.2000124e-05 */
  static int8_t o_ref[T * D];
  const double SA = 0.007874015748031496, SB = 0.007874015748031496, SO = 6.200012400024799e-05;
  for (int i = 0; i < T; i++)
    for (int d = 0; d < D; d++) {
      double acc = 0;
      for (int j = 0; j < T; j++) acc += ((double)p_pv[i * T + j] * SA) * ((double)v[j * D + d] * SB);
      long long r = (long long)llround(acc / SO);
      if (r > 127) r = 127;
      if (r < -128) r = -128;
      o_ref[i * D + d] = (int8_t)r;
    }
  int bad_o = 0, first = -1;
  for (int i = 0; i < T * D; i++) if (o[i] != o_ref[i]) { bad_o++; if (first < 0) first = i; }

  printf("mismatch: s=%d/%d  p=%d/%d  o=%d/%d", bad_s, T * T, bad_p, T * T, bad_o, T * D);
  if (first >= 0) printf("  first_o@%d dev=%d ref=%d", first, o[first], o_ref[first]);
  printf("\n");
  return rc ? 1 : 0;
}
