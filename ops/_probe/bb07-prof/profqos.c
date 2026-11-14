/* profqos.c — pull the runtime's own per-sub-command profile out of libneuron_runtime.
 *
 * WHY: `neuronrt -c N` only reports one total. The public C API exposes
 *      NeuronRuntime_getProfiledQoSData(), whose ProfiledQoSData holds one QoSData per
 *      SUB-COMMAND (execTime in usec, bandwidth in MB/s, boostValue). That is the per-layer
 *      number this ticket is after -- it just has no CLI in front of it.
 *
 * BUILD: see ops/_probe/bb07-prof/build-profqos.sh (aarch64 for the board, x86-64 for a
 *        smoke test against the host runtime).
 *
 * RUN:   profqos <dla> -i <in.bin> [-i <in2.bin>] -o <out.bin> [-o <out2.bin>]
 *                  [-c loops] [-X '<options string>'] [-n] [-q]
 *          -n  null device (host smoke test; no hardware)
 *          -q  quiet: only the SUM/MEASURED summary line
 *
 * EXIT: 0 ok, 2 usage, 3 file IO, 4 runtime API error, 5 no profiled data, 6 SIGFPE in lib.
 */
#define _GNU_SOURCE
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "neuron/api/RuntimeAPI.h"

#define MAXIO 8

static int g_quiet = 0;

/* getProfiledQoSData divides the transferred bytes by the measured exec time. On a device
 * that reports 0 usec (the null device, and any sub-command the firmware timed as 0) that is
 * an integer divide by zero and the *library* traps. Catch it instead of dying silently. */
static void on_sigfpe(int sig) {
  (void)sig;
  fprintf(stderr, "SIGFPE inside libneuron_runtime::getProfiledQoSData -- no usable exec "
                  "time (null device, or a sub-command the firmware timed as 0 usec).\n");
  printf("NO_PROFILED_DATA\n");
  fflush(stdout);
  _exit(6);
}

static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static void *read_file(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); return NULL; }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n < 0) { fclose(f); return NULL; }
  void *buf = malloc((size_t)n);
  if (!buf) { fclose(f); return NULL; }
  if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return NULL; }
  fclose(f);
  *size = (size_t)n;
  return buf;
}

static int write_file(const char *path, const void *buf, size_t size) {
  FILE *f = fopen(path, "wb");
  if (!f) { perror(path); return -1; }
  size_t w = fwrite(buf, 1, size, f);
  fclose(f);
  return w == size ? 0 : -1;
}

static void usage(void) {
  fprintf(stderr, "usage: profqos <dla> -i <in.bin> [-i ...] -o <out.bin> [-o ...] "
                  "[-c loops] [-X options] [-n] [-q]\n");
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IOLBF, 0);
  signal(SIGFPE, on_sigfpe);

  const char *dla = NULL, *optstr = NULL;
  const char *inp[MAXIO], *outp[MAXIO];
  int nin = 0, nout = 0, loops = 1, null_device = 0;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-i") && i + 1 < argc)       { if (nin  < MAXIO) inp[nin++]   = argv[++i]; }
    else if (!strcmp(argv[i], "-o") && i + 1 < argc)  { if (nout < MAXIO) outp[nout++] = argv[++i]; }
    else if (!strcmp(argv[i], "-c") && i + 1 < argc)  { loops = atoi(argv[++i]); }
    else if (!strcmp(argv[i], "-X") && i + 1 < argc)  { optstr = argv[++i]; }
    else if (!strcmp(argv[i], "-n"))                  { null_device = 1; }
    else if (!strcmp(argv[i], "-q"))                  { g_quiet = 1; }
    else if (argv[i][0] != '-' && !dla)                { dla = argv[i]; }
    else { usage(); return 2; }
  }
  if (!dla || nin == 0 || nout == 0) { usage(); return 2; }
  if (loops < 1) loops = 1;

  EnvOptions env;
  memset(&env, 0, sizeof(env));
  env.deviceKind = null_device ? kEnvOptNullDevice : kEnvOptHardware;
  /* CPUThreadNum=0 makes the runtime divide by zero (nbThreads must not be 0). */
  env.CPUThreadNum = 1;

  void *rt = NULL;
  int rc;
  if (optstr) {
    rc = NeuronRuntime_create_with_options(optstr, &env, &rt);
    if (!g_quiet) printf("[rt] create_with_options(%s) rc=%d\n", optstr, rc);
  } else {
    rc = NeuronRuntime_create(&env, &rt);
    if (!g_quiet) printf("[rt] create rc=%d\n", rc);
  }
  if (rc != NEURONRUNTIME_NO_ERROR || !rt) { fprintf(stderr, "create failed rc=%d\n", rc); return 4; }

  rc = NeuronRuntime_loadNetworkFromFile(rt, dla);
  if (!g_quiet) printf("[rt] loadNetworkFromFile rc=%d\n", rc);
  if (rc != NEURONRUNTIME_NO_ERROR) { NeuronRuntime_release(rt); return 4; }

  size_t nin_rt = 0, nout_rt = 0;
  NeuronRuntime_getInputNumber(rt, &nin_rt);
  NeuronRuntime_getOutputNumber(rt, &nout_rt);
  if (!g_quiet) printf("[rt] inputs=%zu outputs=%zu\n", nin_rt, nout_rt);
  if ((int)nin_rt > nin || (int)nout_rt > nout) {
    fprintf(stderr, "need %zu inputs and %zu outputs (got %d/%d)\n", nin_rt, nout_rt, nin, nout);
    NeuronRuntime_release(rt);
    return 2;
  }

  for (size_t h = 0; h < nin_rt; h++) {
    size_t want = 0;
    NeuronRuntime_getInputSize(rt, (uint64_t)h, &want);
    size_t have = 0;
    void *buf = read_file(inp[h], &have);
    if (!buf) { NeuronRuntime_release(rt); return 3; }
    if (have != want && !g_quiet)
      fprintf(stderr, "[warn] input %zu: file %zu B, runtime wants %zu B\n", h, have, want);
    BufferAttribute attr; attr.ionFd = NON_ION_FD;
    rc = NeuronRuntime_setInput(rt, (uint64_t)h, buf, have < want ? have : want, attr);
    if (rc != NEURONRUNTIME_NO_ERROR) { fprintf(stderr, "setInput %zu rc=%d\n", h, rc); return 4; }
  }

  void *obuf[MAXIO];
  for (size_t h = 0; h < nout_rt; h++) {
    size_t want = 0;
    NeuronRuntime_getOutputSize(rt, (uint64_t)h, &want);
    obuf[h] = calloc(1, want ? want : 1);
    BufferAttribute attr; attr.ionFd = NON_ION_FD;
    rc = NeuronRuntime_setOutput(rt, (uint64_t)h, obuf[h], want, attr);
    if (rc != NEURONRUNTIME_NO_ERROR) { fprintf(stderr, "setOutput %zu rc=%d\n", h, rc); return 4; }
  }

  double t0 = now_us();
  for (int i = 0; i < loops; i++) {
    rc = NeuronRuntime_inference(rt);
    if (rc != NEURONRUNTIME_NO_ERROR) { fprintf(stderr, "inference %d rc=%d\n", i, rc); return 4; }
  }
  double measured = now_us() - t0;
  if (!g_quiet) printf("[rt] %d inference(s) in %.0f us\n", loops, measured);

  ProfiledQoSData *pq = NULL;
  uint8_t boost = 0;
  rc = NeuronRuntime_getProfiledQoSData(rt, &pq, &boost);
  if (!g_quiet) printf("[rt] getProfiledQoSData rc=%d execBoostValue=%u\n", rc, (unsigned)boost);

  double sum_us = 0.0;
  int n_cmd = 0;
  if (rc == NEURONRUNTIME_NO_ERROR && pq && pq->qosData && pq->numSubCmd) {
    for (uint32_t g = 0; g < pq->numSubgraph; g++) {
      uint32_t k = pq->numSubCmd[g];
      if (!g_quiet) printf("[subgraph %u] subcmds=%u\n", g, k);
      for (uint32_t c = 0; c < k; c++) {
        const QoSData *d = &pq->qosData[g][c];
        sum_us += (double)d->execTime;
        n_cmd++;
        if (!g_quiet)
          printf("  cmd %3u  exec_us=%-10llu bandwidthMBps=%-8u boost=%-3u suggested_ms=%u\n",
                 c, (unsigned long long)d->execTime, d->bandwidth, (unsigned)d->boostValue,
                 d->suggestedTime);
      }
    }
  }

  printf("SUM_EXEC_US=%.0f CMDS=%d MEASURED_US=%.0f LOOPS=%d PER_INF_US=%.0f\n",
         sum_us, n_cmd, measured, loops, measured / loops);
  if (n_cmd == 0) printf("NO_PROFILED_DATA\n");

  for (size_t h = 0; h < nout_rt; h++) {
    size_t want = 0;
    NeuronRuntime_getOutputSize(rt, (uint64_t)h, &want);
    if (write_file(outp[h], obuf[h], want) != 0) return 3;
  }
  NeuronRuntime_release(rt);
  return n_cmd == 0 ? 5 : 0;
}
