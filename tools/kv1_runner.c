/*
 * kv1_runner.c -- minimal Neuron runtime runner whose KV cache lives in a device dma-buf.
 *
 * Ticket: "KV cache on device, allocated once, reused" (branch impl/kv-resident).
 *
 * Why this exists: the TFLite graph is functional, so the KV cache can only be a graph
 * input/output pair.  The stock CLI (neuronrt) re-reads the whole [48,C,64] cache from a
 * file for every Run, which is where the 25.2 MB/token host traffic comes from.  This
 * runner binds the same buffers once and keeps them, so per token the host writes only the
 * new KV row (48 slabs x 128 B = 6 KB per tensor) into the buffer the NPU already reads.
 *
 * Everything is loaded with dlopen() so there is no link-time dependency and the exact
 * device library on LD_LIBRARY_PATH is the one used.
 *
 * Build (WSL, NDK r26d):
 *   aarch64-linux-android31-clang -O2 -o kv1_runner tools/kv1_runner.c \
 *       -I"$SDK/host/include" -ldl
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <dlfcn.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdbool.h>

#include "neuron/api/RuntimeAPI.h"

/* ---- dma-buf heap (no dependency on linux/dma-heap.h being in the NDK sysroot) ---- */
struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
#define DMA_HEAP_IOC_MAGIC 'H'
#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)

#ifndef DMA_BUF_IOCTL_SYNC
struct dma_buf_sync {
    uint64_t flags;
};
#define DMA_BUF_SYNC_READ (1 << 0)
#define DMA_BUF_SYNC_WRITE (2 << 0)
#define DMA_BUF_SYNC_RW (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
#define DMA_BUF_SYNC_START (0 << 2)
#define DMA_BUF_SYNC_END (1 << 2)
#define DMA_BUF_BASE 'b'
#define DMA_BUF_IOCTL_SYNC _IOW(DMA_BUF_BASE, 0, struct dma_buf_sync)
#endif

#define MAXIO 16

/* ---------------------------------------------------------------- runtime symbols */
static void *g_lib;
static int (*p_create)(const EnvOptions *, void **);
static int (*p_create_opt)(const char *, const EnvOptions *, void **);
static int (*p_load)(void *, const char *);
static int (*p_setin)(void *, uint64_t, const void *, size_t, BufferAttribute);
static int (*p_setoffin)(void *, uint64_t, const void *, size_t, BufferAttribute, size_t);
static int (*p_setout)(void *, uint64_t, void *, size_t, BufferAttribute);
static int (*p_nins)(void *, size_t *);
static int (*p_nouts)(void *, size_t *);
static int (*p_insize)(void *, uint64_t, size_t *);
static int (*p_outsize)(void *, uint64_t, size_t *);
static int (*p_inpad)(void *, uint64_t, size_t *);
static int (*p_outpad)(void *, uint64_t, size_t *);
static int (*p_inrank)(void *, uint64_t, uint32_t *);
static int (*p_outrank)(void *, uint64_t, uint32_t *);
static int (*p_infer)(void *);
static void (*p_release)(void *);
static int (*p_version)(NeuronVersion *);

static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "kvrun: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

static void *need_sym(const char *n) {
    void *s = dlsym(g_lib, n);
    if (!s) die("missing symbol %s in libneuron_runtime", n);
    return s;
}
static void *opt_sym(const char *n) { return dlsym(g_lib, n); }

static void load_runtime(const char *lib) {
    g_lib = dlopen(lib, RTLD_NOW);
    if (!g_lib) die("dlopen(%s) failed: %s", lib, dlerror());
    p_create      = need_sym("NeuronRuntime_create");
    p_create_opt  = opt_sym("NeuronRuntime_create_with_options");
    p_load        = need_sym("NeuronRuntime_loadNetworkFromFile");
    p_setin       = need_sym("NeuronRuntime_setInput");
    p_setoffin    = opt_sym("NeuronRuntime_setOffsetedInput");
    p_setout      = need_sym("NeuronRuntime_setOutput");
    p_nins        = need_sym("NeuronRuntime_getInputNumber");
    p_nouts       = need_sym("NeuronRuntime_getOutputNumber");
    p_insize      = need_sym("NeuronRuntime_getInputSize");
    p_outsize     = need_sym("NeuronRuntime_getOutputSize");
    p_inpad       = opt_sym("NeuronRuntime_getInputPaddedSize");
    p_outpad      = opt_sym("NeuronRuntime_getOutputPaddedSize");
    p_inrank      = opt_sym("NeuronRuntime_getInputRank");
    p_outrank     = opt_sym("NeuronRuntime_getOutputRank");
    p_infer       = need_sym("NeuronRuntime_inference");
    p_release     = need_sym("NeuronRuntime_release");
    p_version     = opt_sym("NeuronRuntime_getVersion");
}

/* ------------------------------------------------------------------------ helpers */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

typedef struct {
    int fd;         /* -1 = not a dma-buf */
    void *ptr;
    size_t len;
    const char *tag;
} Buf;

static int g_sync = 0;   /* DMA_BUF_IOCTL_SYNC around host writes when the heap is cached */

static void dmabuf_sync(Buf *b, int start, int rw) {
    if (!g_sync || b->fd < 0) return;
    struct dma_buf_sync s;
    s.flags = (uint64_t)(rw | (start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END));
    if (ioctl(b->fd, DMA_BUF_IOCTL_SYNC, &s) != 0)
        fprintf(stderr, "kvrun: DMA_BUF_IOCTL_SYNC(%s) failed: %s\n", b->tag, strerror(errno));
}

static int dmabuf_alloc(const char *heap, size_t len, Buf *out) {
    char path[256];
    snprintf(path, sizeof path, "/dev/dma_heap/%s", heap);
    int hfd = open(path, O_RDWR | O_CLOEXEC);
    if (hfd < 0) return -errno;
    struct dma_heap_allocation_data d;
    memset(&d, 0, sizeof d);
    d.len = len;
    d.fd_flags = O_RDWR | O_CLOEXEC;
    if (ioctl(hfd, DMA_HEAP_IOCTL_ALLOC, &d) != 0) {
        int e = -errno;
        close(hfd);
        return e;
    }
    close(hfd);
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, d.fd, 0);
    if (p == MAP_FAILED) {
        int e = -errno;
        close(d.fd);
        return e;
    }
    out->fd = (int)d.fd;
    out->ptr = p;
    out->len = len;
    return 0;
}

static Buf alloc_plain(const char *tag, size_t len) {
    Buf b;
    b.fd = -1;
    b.len = len;
    b.ptr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (b.ptr == MAP_FAILED) die("mmap(%s, %zu) failed: %s", tag, len, strerror(errno));
    b.tag = tag;
    return b;
}

static size_t read_file(const char *path, void *dst, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) die("open %s failed: %s", path, strerror(errno));
    size_t n = fread(dst, 1, cap, f);
    size_t extra = 0;
    if (n == cap) { char c; if (fread(&c, 1, 1, f) == 1) extra = 1; }
    fclose(f);
    if (extra) die("file %s is larger than the %zu byte buffer", path, cap);
    return n;
}

/* ---------------------------------------------------------------- arg parsing */
typedef struct {
    const char *dla;
    const char *lib;
    const char *ins[MAXIO];  int nins;
    const char *outs[MAXIO]; int nouts;
    int kv_idx[4]; int nkv;
    const char *kv_heap;
    int count;
    int steps;
    const char *stepdir;
    const char *outdir;
    int m0;
    int probe_only;
    int noconv;
    int rebind;
    int poison;
    const char *dumpcache;
    int pingpong;        /* bind a second dma-buf per step: forces a real re-transfer */   /* optional: dump the KV input buffers here when done */
    int verbose;
    const char *json;
} Args;

static int is_kv(const Args *a, int idx) {
    for (int i = 0; i < a->nkv; i++) if (a->kv_idx[i] == idx) return 1;
    return 0;
}


static void show_io(const char *when) {
    FILE *f = fopen("/proc/self/io", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "rchar:", 6) || !strncmp(line, "wchar:", 6) ||
            !strncmp(line, "read_bytes:", 11) || !strncmp(line, "syscr:", 6))
            printf("[io %s] %s", when, line);
    }
    fclose(f);
}

static void show_dmabuf_maps(void) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return;
    char line[512];
    int n = 0;
    while (fgets(line, sizeof line, f)) {
        if (strstr(line, "dmabuf")) {
            printf("[maps] %s", line);
            n++;
            if (n > 12) break;
        }
    }
    fclose(f);
}

static void *g_rt;

int main(int argc, char **argv) {
    Args A;
    memset(&A, 0, sizeof A);
    A.lib = "libneuron_runtime.so";
    A.kv_heap = "mtk_mm-uncached";
    A.kv_idx[0] = 6; A.kv_idx[1] = 7; A.nkv = 2;
    A.count = 1;
    A.m0 = 0;
    A.poison = -1;

    for (int i = 1; i < argc; i++) {
        const char *k = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if      (!strcmp(k, "-a"))        { A.dla = v; i++; }
        else if (!strcmp(k, "--lib"))     { A.lib = v; i++; }
        else if (!strcmp(k, "-i"))        { if (A.nins >= MAXIO) die("too many inputs"); A.ins[A.nins++] = v; i++; }
        else if (!strcmp(k, "-o"))        { if (A.nouts >= MAXIO) die("too many outputs"); A.outs[A.nouts++] = v; i++; }
        else if (!strcmp(k, "-c"))        { A.count = atoi(v); i++; }
        else if (!strcmp(k, "--kv-idx"))  {
            A.nkv = 0;
            char *s = strdup(v);
            for (char *t = strtok(s, ","); t && A.nkv < 4; t = strtok(NULL, ","))
                A.kv_idx[A.nkv++] = atoi(t);
            i++;
        }
        else if (!strcmp(k, "--kv-heap")) { A.kv_heap = v; i++; }
        else if (!strcmp(k, "--steps"))   { A.steps = atoi(v); i++; }
        else if (!strcmp(k, "--stepdir")) { A.stepdir = v; i++; }
        else if (!strcmp(k, "--outdir"))  { A.outdir = v; i++; }
        else if (!strcmp(k, "--m0"))      { A.m0 = atoi(v); i++; }
        else if (!strcmp(k, "--sync"))    { g_sync = atoi(v); i++; }
        else if (!strcmp(k, "--probe"))   { A.probe_only = 1; }
        else if (!strcmp(k, "--noconv"))  { A.noconv = 1; }
        else if (!strcmp(k, "--rebind"))  { A.rebind = 1; }
        else if (!strcmp(k, "--poison"))  { A.poison = atoi(v); i++; }
        else if (!strcmp(k, "--dumpcache")) { A.dumpcache = v; i++; }
        else if (!strcmp(k, "--pingpong"))  { A.pingpong = 1; }
        else if (!strcmp(k, "--verbose")) { A.verbose = 1; }
        else if (!strcmp(k, "--json"))    { A.json = v; i++; }
        else if (!strcmp(k, "--help")) {
            printf("usage: kvrun -a <dla> -i <in>... -o <out>... [-c N] [--kv-idx 6,7] "
                   "[--kv-heap NAME|host] [--steps S --stepdir DIR --outdir DIR] [--m0 M] "
                   "[--sync 0|1] [--probe] [--json FILE]\n");
            return 0;
        }
        else die("unknown argument %s", k);
    }
    if (!A.dla) die("need -a <dla>");

    load_runtime(A.lib);

    NeuronVersion ver;
    if (p_version && p_version(&ver) == 0)
        printf("[lib] %s version %u.%u.%u\n", A.lib, ver.major, ver.minor, ver.patch);

    EnvOptions eo;
    memset(&eo, 0, sizeof eo);
    eo.deviceKind = kEnvOptHardware;
    eo.MDLACoreOption = Auto;
    eo.CPUThreadNum = 1;
    eo.suppressInputConversion = A.noconv ? true : false;
    eo.suppressOutputConversion = A.noconv ? true : false;
    int rc = -1;
    if (p_create_opt) rc = p_create_opt(NULL, &eo, &g_rt);
    if (rc != 0 || !g_rt) rc = p_create(&eo, &g_rt);
    if (rc != 0 || !g_rt) die("NeuronRuntime_create failed rc=%d", rc);

    rc = p_load(g_rt, A.dla);
    if (rc != 0) die("loadNetworkFromFile(%s) rc=%d", A.dla, rc);

    size_t nin = 0, nout = 0;
    p_nins(g_rt, &nin);
    p_nouts(g_rt, &nout);
    printf("[graph] %s inputs=%zu outputs=%zu\n", A.dla, nin, nout);

    size_t in_size[MAXIO], in_pad[MAXIO], out_size[MAXIO], out_pad[MAXIO];
    uint32_t rk = 0;
    for (size_t i = 0; i < nin; i++) {
        in_size[i] = in_pad[i] = 0;
        p_insize(g_rt, i, &in_size[i]);
        if (p_inpad) p_inpad(g_rt, i, &in_pad[i]);
        rk = 0;
        if (p_inrank) p_inrank(g_rt, i, &rk);
        printf("[in %2zu] size=%zu padded=%zu rank=%u%s\n", i, in_size[i], in_pad[i], rk,
               is_kv(&A, (int)i) ? "   <-- KV" : "");
    }
    for (size_t i = 0; i < nout; i++) {
        out_size[i] = out_pad[i] = 0;
        p_outsize(g_rt, i, &out_size[i]);
        if (p_outpad) p_outpad(g_rt, i, &out_pad[i]);
        rk = 0;
        if (p_outrank) p_outrank(g_rt, i, &rk);
        printf("[out %2zu] size=%zu padded=%zu rank=%u\n", i, out_size[i], out_pad[i], rk);
    }
    if ((int)nin != A.nins) die("graph has %zu inputs but %d -i were given", nin, A.nins);
    if ((int)nout != A.nouts) die("graph has %zu outputs but %d -o were given", nout, A.nouts);

    /* ---- one persistent buffer per input ---- */
    Buf inb[MAXIO];
    char tagbuf[MAXIO][32];
    for (size_t i = 0; i < nin; i++) {
        snprintf(tagbuf[i], sizeof tagbuf[i], "in%zu", i);
        int kv = is_kv(&A, (int)i);
        if (kv && strcmp(A.kv_heap, "host") != 0) {
            int e = dmabuf_alloc(A.kv_heap, in_size[i], &inb[i]);
            if (e != 0) die("dma_heap alloc(%s, %zu) failed: %s", A.kv_heap, in_size[i], strerror(-e));
            inb[i].tag = tagbuf[i];
            printf("[buf in%zu] dma-buf heap=%s len=%zu fd=%d\n", i, A.kv_heap, inb[i].len, inb[i].fd);
        } else {
            inb[i] = alloc_plain(tagbuf[i], in_size[i]);
            printf("[buf in%zu] %s len=%zu fd=-1\n", i, kv ? "host-anon (KV)" : "host-anon", inb[i].len);
        }
        size_t n = read_file(A.ins[i], inb[i].ptr, inb[i].len);
        if (n != inb[i].len)
            fprintf(stderr, "kvrun: warn: %s is %zu B, tensor is %zu B (rest left zero)\n",
                    A.ins[i], n, inb[i].len);
        dmabuf_sync(&inb[i], 1, DMA_BUF_SYNC_WRITE);
        dmabuf_sync(&inb[i], 0, DMA_BUF_SYNC_WRITE);
    }

    Buf outb[MAXIO];
    for (size_t i = 0; i < nout; i++) {
        char *t = malloc(32);
        snprintf(t, 32, "out%zu", i);
        outb[i] = alloc_plain(t, out_size[i]);
        if (A.verbose) printf("[buf out%zu] host-anon len=%zu\n", i, outb[i].len);
    }

    /* ---- bind once ---- */
    for (size_t i = 0; i < nin; i++) {
        BufferAttribute attr;
        attr.ionFd = inb[i].fd;
        double t0 = now_ms();
        rc = p_setin(g_rt, i, inb[i].ptr, inb[i].len, attr);
        double t1 = now_ms();
        printf("[setInput %2zu] rc=%d ionFd=%d %.3f ms\n", i, rc, attr.ionFd, t1 - t0);
        if (rc != 0) die("setInput(%zu) rc=%d", i, rc);
    }
    for (size_t i = 0; i < nout; i++) {
        BufferAttribute attr;
        attr.ionFd = -1;
        rc = p_setout(g_rt, i, outb[i].ptr, outb[i].len, attr);
        if (rc != 0) die("setOutput(%zu) rc=%d", i, rc);
    }
    Buf kv2[MAXIO];
    if (A.pingpong) {
        for (int t = 0; t < A.nkv; t++) {
            int i2 = A.kv_idx[t];
            int e2 = dmabuf_alloc(A.kv_heap, inb[i2].len, &kv2[t]);
            if (e2 != 0) die("second dma-buf alloc failed: %s", strerror(-e2));
            memcpy(kv2[t].ptr, inb[i2].ptr, inb[i2].len);
            printf("[pingpong] alt buffer for in%d fd=%d\n", i2, kv2[t].fd);
        }
    }
    if (A.probe_only) {
        printf("[probe] binding accepted; not invoking\n");
        p_release(g_rt);
        return 0;
    }

    if (A.verbose) show_dmabuf_maps();
    show_io("before");

    rc = p_infer(g_rt);
    if (rc != 0) die("inference(warmup) rc=%d", rc);
    printf("[warmup] ok\n");

    /* ---- KV geometry ------------------------------------------------
     * cache input is [48, C, 64] int16 (48 = layers*KVhead); the graph outputs only the new
     * rows [48, T, 64] int16.  The KV outputs are the outputs whose byte size is a multiple
     * of 48*64*2; they come in (k, v) order, matching the KV inputs.  Row m of the cache
     * lives at p*slab + m*128 bytes, slab = C*64*2.                                          */
    int kv_out_idx[4]; int kv_out_n = 0;
    for (size_t j = 0; j < nout; j++) {
        if (out_size[j] > 0 && out_size[j] % (48u * 128u) == 0 && kv_out_n < 4) {
            kv_out_idx[kv_out_n] = (int)j;
            printf("[kv-out] output %zu size=%zu (T=%zu) -> KV slot %d\n",
                   j, out_size[j], out_size[j] / (48u * 128u), kv_out_n);
            kv_out_n++;
        }
    }
    if (kv_out_n != A.nkv)
        die("expected %d KV outputs, detected %d", A.nkv, kv_out_n);

    size_t kv_len = inb[A.kv_idx[0]].len;
    size_t slab = kv_len / 48;
    size_t row_bytes = 128;
    if (slab % row_bytes != 0) die("slab=%zu is not a multiple of %zu", slab, row_bytes);
    printf("[kv] cache=%zu B  slab=%zu B (%zu rows)  row=%zu B  m0=%d\n",
           kv_len, slab, slab / row_bytes, row_bytes, A.m0);

    int total = A.steps > 0 ? A.steps : A.count;
    double *times = calloc(total, sizeof(double));
    FILE *js = A.json ? fopen(A.json, "w") : NULL;

    for (int s = 0; s < total; s++) {
        if (A.steps > 0 && s > 0 && A.stepdir) {
            static const char *nm[6] = {"x", "mask", "cos14", "sin14", "cos2", "sin2"};
            for (int i = 0; i < 6 && i < A.nins; i++) {
                char p[512];
                snprintf(p, sizeof p, "%s/step%02d/%s.bin", A.stepdir, s, nm[i]);
                if (access(p, R_OK) != 0) continue;
                size_t n = read_file(p, inb[i].ptr, inb[i].len);
                if (n != inb[i].len)
                    fprintf(stderr, "kvrun: warn step %d %s: %zu B vs %zu\n", s, p, n, inb[i].len);
                dmabuf_sync(&inb[i], 1, DMA_BUF_SYNC_WRITE);
                dmabuf_sync(&inb[i], 0, DMA_BUF_SYNC_WRITE);
            }
        }
        double t0 = now_ms();
        rc = p_infer(g_rt);
        double dt = now_ms() - t0;
        if (rc != 0) die("inference(step %d) rc=%d", s, rc);
        times[s] = dt;
        if (A.verbose) printf("[run %4d] %.3f ms\n", s, dt);
        if (js) fprintf(js, "{\"step\":%d,\"ms\":%.4f}\n", s, dt);

        if (A.outdir && A.steps > 0) {
            char p[512];
            static const char *onm[3] = {"hidden", "k_out", "v_out"};
            snprintf(p, sizeof p, "%s/step%02d", A.outdir, s);
            mkdir(p, 0777);
            for (size_t j = 0; j < nout && j < 3; j++) {
                snprintf(p, sizeof p, "%s/step%02d/%s.bin", A.outdir, s, onm[j]);
                FILE *f = fopen(p, "wb");
                if (!f) die("write %s failed", p);
                fwrite(outb[j].ptr, 1, outb[j].len, f);
                fclose(f);
            }
        }

        /* absorb: write the new KV row into slot m of the SAME device buffer */
        if (A.steps > 0 && s + 1 < total) {
            int m = A.m0 + s;
            for (int t = 0; t < A.nkv; t++) {
                int oj = kv_out_idx[t];
                Buf *dst = &inb[A.kv_idx[t]];
                const uint8_t *src = (const uint8_t *)outb[oj].ptr;
                dmabuf_sync(dst, 1, DMA_BUF_SYNC_WRITE);
                /* the KV *output* is [48, T, 64]: slab stride is T*128, row 0 is the token */
                size_t oslab = out_size[oj] / 48;
                if (A.poison >= 0) {
                    int16_t v16 = (int16_t)A.poison;
                    for (int q = 0; q < 48; q++) {
                        int16_t *d16 = (int16_t *)((uint8_t *)dst->ptr + (size_t)q * slab +
                                                   (size_t)m * row_bytes);
                        for (int r = 0; r < 64; r++) d16[r] = v16;
                    }
                } else {
                    for (int q = 0; q < 48; q++)
                        memcpy((uint8_t *)dst->ptr + (size_t)q * slab + (size_t)m * row_bytes,
                               src + (size_t)q * oslab, row_bytes);
                }
                dmabuf_sync(dst, 0, DMA_BUF_SYNC_WRITE);
            }
            if (A.pingpong) {
                for (int t = 0; t < A.nkv; t++) {
                    int i2 = A.kv_idx[t];
                    memcpy(kv2[t].ptr, inb[i2].ptr, inb[i2].len);
                    BufferAttribute at2;
                    at2.ionFd = kv2[t].fd;
                    rc = p_setin(g_rt, (uint64_t)i2, kv2[t].ptr, kv2[t].len, at2);
                    if (rc != 0) die("pingpong setInput(%d) rc=%d", i2, rc);
                }
            }
            if (A.rebind) {
                for (int t = 0; t < A.nkv; t++) {
                    BufferAttribute at;
                    at.ionFd = inb[A.kv_idx[t]].fd;
                    rc = p_setin(g_rt, (uint64_t)A.kv_idx[t], inb[A.kv_idx[t]].ptr,
                                 inb[A.kv_idx[t]].len, at);
                    if (rc != 0) die("re-setInput(%d) rc=%d", A.kv_idx[t], rc);
                }
            }
        }
    }

    if (!A.outdir) {
        for (size_t j = 0; j < nout; j++) {
            FILE *f = fopen(A.outs[j], "wb");
            if (!f) die("write %s failed", A.outs[j]);
            fwrite(outb[j].ptr, 1, outb[j].len, f);
            fclose(f);
        }
    }

    double sum = 0;
    for (int s = 0; s < total; s++) sum += times[s];
    double *sorted = malloc(sizeof(double) * total);
    memcpy(sorted, times, sizeof(double) * total);
    qsort(sorted, total, sizeof(double), cmp_double);
    printf("[time] n=%d mean=%.3f ms median=%.3f ms p10=%.3f p90=%.3f min=%.3f max=%.3f\n",
           total, sum / total, sorted[total / 2], sorted[total / 10],
           sorted[(total * 9) / 10], sorted[0], sorted[total - 1]);

    if (A.dumpcache) {
        char q[600];
        for (int t = 0; t < A.nkv; t++) {
            snprintf(q, sizeof q, "%s.%d.bin", A.dumpcache, t);
            FILE *f = fopen(q, "wb");
            if (!f) die("write %s failed", q);
            fwrite(inb[A.kv_idx[t]].ptr, 1, inb[A.kv_idx[t]].len, f);
            fclose(f);
            printf("[dumpcache] %s %zu B\n", q, inb[A.kv_idx[t]].len);
        }
    }
    show_io("after");
    if (js) fclose(js);
    p_release(g_rt);
    return 0;
}
