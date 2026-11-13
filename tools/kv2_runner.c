/*
 * kv2_runner.c -- same job as kv1_runner.c, but through RuntimeV2, the only Neuron API that
 * takes a buffer descriptor *per inference request* (IOBuffer{buffer, length, fd, offset}).
 *
 * Why: RuntimeAPI v1 latches a graph input at its first inference (docs/OP-kv1-resident-kv.md
 * §4), so an in-place KV update never reaches the device.  V2 re-supplies the descriptors on
 * every run, which is the remaining candidate for "KV stays in one dma-buf, host writes the
 * new row".
 *
 *   aarch64-linux-android31-clang -O2 -o kv2_runner tools/kv2_runner.c -I"$SDK/host/include" -ldl
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <dlfcn.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* RuntimeV2.h refers to `IOBuffer` without the `struct` tag, i.e. it expects C++ */
typedef struct IOBuffer IOBuffer;
#include "neuron/api/RuntimeV2.h"

struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
#define DMA_HEAP_IOC_MAGIC 'H'
#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)

#define MAXIO 16

static void die(const char *m) { fprintf(stderr, "kv2run: %s\n", m); exit(1); }

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double cpu_ms(void) {
    FILE *f = fopen("/proc/self/stat", "r");
    if (!f) return -1;
    char buf[2048];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    /* fields 14 and 15 (utime, stime in clock ticks) start after the 13th field; the comm
       field may contain spaces, so scan backwards from the last ')' */
    char *r = strrchr(buf, ')');
    if (!r) return -1;
    /* fields after comm: state(3) ... utime(14) stime(15); utime is the 12th token here */
    long ut = 0, st = 0;
    char *tok = strtok(r + 2, " ");
    int idx = 1;
    while (tok && idx < 14) {
        if (idx == 12) ut = atol(tok);
        if (idx == 13) { st = atol(tok); break; }
        tok = strtok(NULL, " ");
        idx++;
    }
    if (!tok) return -1;
    return (double)(ut + st) * 1000.0 / (double)sysconf(_SC_CLK_TCK);
}

static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

typedef struct { int fd; void *ptr; size_t len; } Buf;

static int dmabuf_alloc(const char *heap, size_t len, Buf *o) {
    char p[256];
    snprintf(p, sizeof p, "/dev/dma_heap/%s", heap);
    int h = open(p, O_RDWR | O_CLOEXEC);
    if (h < 0) return -errno;
    struct dma_heap_allocation_data d;
    memset(&d, 0, sizeof d);
    d.len = len;
    d.fd_flags = O_RDWR | O_CLOEXEC;
    if (ioctl(h, DMA_HEAP_IOCTL_ALLOC, &d) != 0) { int e = -errno; close(h); return e; }
    close(h);
    void *q = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, d.fd, 0);
    if (q == MAP_FAILED) { int e = -errno; close(d.fd); return e; }
    o->fd = (int)d.fd; o->ptr = q; o->len = len;
    return 0;
}

static Buf anon_alloc(size_t len) {
    Buf b;
    b.fd = -1; b.len = len;
    b.ptr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (b.ptr == MAP_FAILED) die("mmap failed");
    return b;
}

static void read_into(const char *path, void *dst, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open input");
    size_t n = fread(dst, 1, cap, f);
    fclose(f);
    if (n != cap) fprintf(stderr, "kv2run: warn %s is %zu B, tensor %zu B\n", path, n, cap);
}

int main(int argc, char **argv) {
    const char *dla = NULL, *lib = "libneuron_runtime.so", *heap = "mtk_mm-uncached";
    const char *ins[MAXIO], *outs[MAXIO], *stepdir = NULL, *outdir = NULL;
    int nins = 0, nouts = 0, steps = 0, m0 = 0, kv_idx[4] = {6, 7}, nkv = 2, nb = 1, count = 1;
    for (int i = 1; i < argc; i++) {
        const char *k = argv[i], *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!strcmp(k, "-a")) { dla = v; i++; }
        else if (!strcmp(k, "--lib")) { lib = v; i++; }
        else if (!strcmp(k, "-i")) { ins[nins++] = v; i++; }
        else if (!strcmp(k, "-o")) { outs[nouts++] = v; i++; }
        else if (!strcmp(k, "--kv-heap")) { heap = v; i++; }
        else if (!strcmp(k, "--steps")) { steps = atoi(v); i++; }
        else if (!strcmp(k, "--m0")) { m0 = atoi(v); i++; }
        else if (!strcmp(k, "--stepdir")) { stepdir = v; i++; }
        else if (!strcmp(k, "--outdir")) { outdir = v; i++; }
        else if (!strcmp(k, "--threads")) { nb = atoi(v); i++; }
        else if (!strcmp(k, "-c")) { count = atoi(v); i++; }
        else die("unknown argument");
    }
    if (!dla || nins == 0) die("need -a and -i");

    void *L = dlopen(lib, RTLD_NOW);
    if (!L) die("dlopen failed");
    int (*p_create)(const char *, size_t, void **, size_t) =
        (int (*)(const char *, size_t, void **, size_t))dlsym(L, "NeuronRuntimeV2_create");
    int (*p_run)(void *, SyncInferenceRequest) =
        (int (*)(void *, SyncInferenceRequest))dlsym(L, "NeuronRuntimeV2_run");
    void (*p_rel)(void *) = (void (*)(void *))dlsym(L, "NeuronRuntimeV2_release");
    int (*p_nin)(void *, size_t *) = (int (*)(void *, size_t *))dlsym(L, "NeuronRuntimeV2_getInputNumber");
    int (*p_nout)(void *, size_t *) = (int (*)(void *, size_t *))dlsym(L, "NeuronRuntimeV2_getOutputNumber");
    int (*p_isz)(void *, uint64_t, size_t *) = (int (*)(void *, uint64_t, size_t *))dlsym(L, "NeuronRuntimeV2_getInputSize");
    int (*p_osz)(void *, uint64_t, size_t *) = (int (*)(void *, uint64_t, size_t *))dlsym(L, "NeuronRuntimeV2_getOutputSize");
    if (!p_create || !p_run || !p_nin || !p_nout || !p_isz || !p_osz) die("V2 symbols missing");

    void *rt = NULL;
    int rc = p_create(dla, (size_t)nb, &rt, 8);
    printf("[v2] create(%s, threads=%d) rc=%d rt=%p\n", dla, nb, rc, rt);
    if (rc != 0 || !rt) die("V2 create failed");

    size_t nin = 0, nout = 0;
    p_nin(rt, &nin);
    p_nout(rt, &nout);
    printf("[v2] inputs=%zu outputs=%zu\n", nin, nout);
    if ((int)nin != nins || (int)nout != nouts) die("input/output count mismatch");

    Buf ib[MAXIO], ob[MAXIO];
    struct IOBuffer ri[MAXIO], ro[MAXIO];
    size_t isz[MAXIO], osz[MAXIO];
    for (size_t i = 0; i < nin; i++) {
        isz[i] = 0;
        p_isz(rt, i, &isz[i]);
        int kv = 0;
        for (int t = 0; t < nkv; t++) if (kv_idx[t] == (int)i) kv = 1;
        if (kv && strcmp(heap, "host") != 0) {
            int e = dmabuf_alloc(heap, isz[i], &ib[i]);
            if (e) die("dma-buf alloc failed");
            printf("[v2 in %zu] dma-buf fd=%d len=%zu\n", i, ib[i].fd, isz[i]);
        } else {
            ib[i] = anon_alloc(isz[i]);
            printf("[v2 in %zu] anon len=%zu\n", i, isz[i]);
        }
        read_into(ins[i], ib[i].ptr, isz[i]);
    }
    for (size_t i = 0; i < nout; i++) {
        osz[i] = 0;
        p_osz(rt, i, &osz[i]);
        ob[i] = anon_alloc(osz[i]);
    }

    int total = steps > 0 ? steps : count;
    double cpu0 = 0;
    double *tm = calloc(total, sizeof(double));
    int kv_out[4], kv_out_n = 0;
    for (size_t j = 0; j < nout; j++)
        if (osz[j] % (48 * 128) == 0 && kv_out_n < 4) kv_out[kv_out_n++] = (int)j;
    if (kv_out_n != nkv) die("cannot identify KV outputs");

    size_t slab = ib[kv_idx[0]].len / 48;
    for (int s = 0; s < total; s++) {
        if (steps > 0 && s > 0 && stepdir) {
            static const char *nm[6] = {"x", "mask", "cos14", "sin14", "cos2", "sin2"};
            for (int i = 0; i < 6 && i < nins; i++) {
                char p[512];
                snprintf(p, sizeof p, "%s/step%02d/%s.bin", stepdir, s, nm[i]);
                if (access(p, R_OK) == 0) read_into(p, ib[i].ptr, isz[i]);
            }
        }
        for (size_t i = 0; i < nin; i++) {
            memset(&ri[i], 0, sizeof ri[i]);
            ri[i].buffer = ib[i].ptr;
            ri[i].length = isz[i];
            ri[i].fd = ib[i].fd;
            ri[i].offset = 0;
        }
        for (size_t i = 0; i < nout; i++) {
            memset(&ro[i], 0, sizeof ro[i]);
            ro[i].buffer = ob[i].ptr;
            ro[i].length = osz[i];
            ro[i].fd = -1;
            ro[i].offset = 0;
        }
        if (s == 0) { cpu0 = cpu_ms(); }
        SyncInferenceRequest req;
        req.inputs = ri;
        req.outputs = ro;
        double t0 = now_ms();
        rc = p_run(rt, req);
        tm[s] = now_ms() - t0;
        if (rc != 0) { fprintf(stderr, "kv2run: run rc=%d at step %d\n", rc, s); return 1; }
        if (outdir && steps > 0) {
            char p[512];
            static const char *onm[3] = {"hidden", "k_out", "v_out"};
            snprintf(p, sizeof p, "%s/step%02d", outdir, s);
            mkdir(p, 0777);
            for (size_t j = 0; j < nout && j < 3; j++) {
                snprintf(p, sizeof p, "%s/step%02d/%s.bin", outdir, s, onm[j]);
                FILE *f = fopen(p, "wb");
                if (!f) die("cannot write output");
                fwrite(ob[j].ptr, 1, osz[j], f);
                fclose(f);
            }
        }
        if (steps > 0 && s + 1 < total) {
            int m = m0 + s;
            for (int t = 0; t < nkv; t++) {
                const uint8_t *src = (const uint8_t *)ob[kv_out[t]].ptr;
                uint8_t *dst = (uint8_t *)ib[kv_idx[t]].ptr;
                size_t oslab = osz[kv_out[t]] / 48;
                for (int q = 0; q < 48; q++)
                    memcpy(dst + (size_t)q * slab + (size_t)m * 128, src + (size_t)q * oslab, 128);
            }
        }
    }
    double *srt = malloc(sizeof(double) * total);
    memcpy(srt, tm, sizeof(double) * total);
    qsort(srt, total, sizeof(double), cmpd);
    double sum = 0;
    for (int i = 0; i < total; i++) sum += tm[i];
    printf("[time] n=%d mean=%.3f median=%.3f p10=%.3f p90=%.3f min=%.3f max=%.3f\n",
           total, sum / total, srt[total / 2], srt[total / 10], srt[(total * 9) / 10],
           srt[0], srt[total - 1]);
    {
        double cpu1 = cpu_ms();
        printf("[cpu] wall_total=%.0f ms cpu_total=%.1f ms cpu_per_invoke=%.3f ms\n",
               sum, cpu1 - cpu0, (cpu1 - cpu0) / total);
    }
    if (p_rel) p_rel(rt);
    return 0;
}
