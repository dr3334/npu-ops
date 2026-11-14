// LD_PRELOAD recorder v2: apusys ioctl PRE/POST + pointer dereference + mmap tracking.
//
// Purpose: find the MDLA command buffer (cmdbuf) that libneuron builds before CMD_RUN.
//   1. log every apusys ('A'=0x41) ioctl with caller input (PRE) and driver output (POST)
//   2. track mmap/mmap64 regions (addr, len, fd) to know which host buffers exist
//   3. on CMD (nr=34) RUN, dereference every pointer-looking word of the in-struct with
//      process_vm_readv (safe: EFAULT instead of SIGSEGV) and dump the payload
//
// Build (aarch64 Android):
//   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang \
//     -shared -fPIC -O2 -o libiolog2.so iolog2.c -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>

#define MAXMAP 96
#define MAXPTR 16
#define MAXDUMP 256

static int (*real_ioctl)(int, int, ...);
static void *(*real_mmap)(void *, size_t, int, int, int, off_t);
static int logfd = -1;
static int inited = 0;

struct maprec { uintptr_t addr; size_t len; int fd; };
static struct maprec maps[MAXMAP];
static int nmaps = 0;

static void dump_buffers(void);
static void dump_maps(void);

/* E-105B: single-field mutation of a *valid* submission. The spec file holds one
   line "0xOFFSET 0xVALUE"; the word is rewritten just before the ioctl is issued,
   so we can tell whether a field is used, ignored, or fatal without guessing the
   struct layout. (env vars are scrubbed by neuronrt, hence a file.) */
static int g_patch_off = -1;
static uint32_t g_patch_val = 0;

static unsigned char g_zero[256];/* E-106B: target for the adj_matrix experiment */
static int g_use_zero_adj = 0;
static void load_patch(void) {
  FILE *f = fopen("/data/local/tmp/npuops/probe/patch.txt", "r");
  if (!f) return;
  unsigned off = 0, val = 0;
  char tag[32] = {0};
  if (fscanf(f, "%31s", tag) == 1 && !strcmp(tag, "adjzero")) { g_use_zero_adj = 1; fclose(f); return; }
  rewind(f);
  if (fscanf(f, "%x %x", &off, &val) == 2) {
    g_patch_off = (int)off;
    g_patch_val = (uint32_t)val;
  }
  fclose(f);
}

// Capture the final state at process exit: by then the runtime has copied the NPU
// output back into the output dma-buf, so this is where the result becomes visible.
static void dump_final(void) {
  if (logfd < 0) return;
  setenv("BUFSUFFIX", "_final", 1);
  dprintf(logfd, "EXIT-DUMP begin\n");
  dump_maps();
  dump_buffers();
  dprintf(logfd, "EXIT-DUMP end\n");
  close(logfd);
  logfd = -1;
}

static void open_log(void) {
  char auto_path[128];
  const char *p = getenv("IOLOG");
  if (!p) {
    /* neuronrt scrubs the environment (observed: IOLOG/IODIR both NULL inside the
       process), so fall back to a per-PID path instead of a shared default. */
    snprintf(auto_path, sizeof(auto_path), "/data/local/tmp/npuops/probe/io2.%d.log",
             (int)getpid());
    p = auto_path;
  }
  logfd = open(p, O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (logfd >= 0)
    dprintf(logfd, "OPEN path=%s IOLOG=%s IODIR=%s pid=%d\n", p,
            getenv("IOLOG") ? getenv("IOLOG") : "(null)",
            getenv("IODIR") ? getenv("IODIR") : "(null)", getpid());
}

static void hexdump(const char *tag, const void *buf, size_t n, int indent) {
  char line[768];
  const unsigned char *p = (const unsigned char *)buf;
  size_t i, off = 0;
  for (i = 0; i < n && off < sizeof(line) - 4; i++) {
    int k = snprintf(line + off, sizeof(line) - off, "%02x", p[i]);
    if (k <= 0) break;
    off += (size_t)k;
    if ((i & 15) == 15 && off < sizeof(line) - 2) line[off++] = ' ';
  }
  line[off] = 0;
  dprintf(logfd, "%*s%s[%zu]=%s\n", indent, "", tag, n, line);
}

// Read another address inside our own process without risking SIGSEGV.
static ssize_t safe_read(void *dst, uintptr_t src, size_t n) {
  struct iovec local = {dst, n};
  struct iovec remote = {(void *)src, n};
  return process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
}

static int ptr_like(uint64_t v) {
  return v >= 0x10000ULL && v < 0x800000000000ULL && (v & 7) == 0;
}

static void dump_pointers(const unsigned char *b, unsigned size, const char *why) {
  int dumped = 0;
  for (unsigned off = 0; off + 8 <= size && dumped < MAXPTR; off += 8) {
    uint64_t v;
    memcpy(&v, b + off, 8);
    // Android heap pointers are often tagged in the top byte (HWASAN/MTE): try raw and stripped
    uint64_t cands[2] = {v, v & 0x00ffffffffffffffULL};
    for (int c = 0; c < 2; c++) {
      if (!ptr_like(cands[c])) continue;
      unsigned char buf[MAXDUMP];
      ssize_t got = safe_read(buf, (uintptr_t)cands[c], MAXDUMP);
      if (got >= 16) {
        dprintf(logfd, "DEREF %s off=%u ptr=0x%llx got=%zd\n", why, off,
                (unsigned long long)cands[c], got);
        hexdump("mem", buf, (size_t)got, 2);
        dumped++;
        break;
      }
    }
  }
}

/* E-106B: chase mdw_cmd_in -> subcmd_infos[] -> cmdbufs[] so we can rebuild the
   submission ourselves. Offsets come from mdw_ioctl.h and were confirmed by the
   mutation experiments (E-105B/C): subcmd_infos@0x50, num_cmdbufs@+64, cmdbufs@+72. */
static void dump_subcmds(const unsigned char *cmd, unsigned size) {
  if (size < 0x58) return;
  /* mdw_cmd_in@0x48 = num_subcmds，@0x50 = subcmd_infos[]。**一块 40 op 的图会带 2 个子命令**，
     只抓 subcmd[0] 会漏掉输出地址（整块重放卡在这里，见 reports/npu-lib-whole-block-probe.md）。 */
  uint32_t nsub = 0;
  memcpy(&nsub, cmd + 0x48, 4);
  uint64_t p = 0;
  memcpy(&p, cmd + 0x50, 8);
  if (nsub == 0 || nsub > 8) nsub = 1;
  dprintf(logfd, "SUBSCMDS n=%u ptr=0x%llx\n", nsub, (unsigned long long)p);
  for (uint32_t i = 0; i < nsub; i++) {
    unsigned char sc[128];
    char tag[24];
    ssize_t got = safe_read(sc, (uintptr_t)(p + (uint64_t)i * 80), 80);
    if (got < 80) { dprintf(logfd, "SUBSCMD%d unreadable got=%zd\n", i, got); continue; }
    snprintf(tag, sizeof(tag), "subcmd%u", i);
    hexdump(tag, sc, 80, 2);
    uint32_t n = 0;
    uint64_t cbs = 0;
    memcpy(&n, sc + 64, 4);
    memcpy(&cbs, sc + 72, 8);
    dprintf(logfd, "SUBSCMD%d num_cmdbufs=%u cmdbufs=0x%llx\n", i, n, (unsigned long long)cbs);
    if (n == 0 || n > 8) continue;
    unsigned char cb[8 * 24];
    ssize_t g2 = safe_read(cb, (uintptr_t)cbs, (size_t)n * 24);
    if (g2 <= 0) { dprintf(logfd, "SUBSCMD%d cmdbufs unreadable\n", i); continue; }
    snprintf(tag, sizeof(tag), "cmdbufs%u", i);
    hexdump(tag, cb, (size_t)g2, 2);
  }
}

#include <signal.h>
#include <setjmp.h>

static sigjmp_buf g_jmp;
static volatile sig_atomic_t g_in_guard = 0;

static void guard_handler(int sig) {
  if (g_in_guard) siglongjmp(g_jmp, 1);
}

// dma-buf mappings are neither page-backed (process_vm_readv -> EFAULT) nor readable
// through /proc/self/mem (EIO), but they are mapped in our address space: a guarded
// memcpy is the only way to snapshot them without risking SIGBUS/SIGSEGV.
static ssize_t guarded_copy(void *dst, uintptr_t src, size_t n) {
  struct sigaction sa, old_segv, old_bus;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = guard_handler;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGSEGV, &sa, &old_segv);
  sigaction(SIGBUS, &sa, &old_bus);
  g_in_guard = 1;
  ssize_t done = -1;
  if (sigsetjmp(g_jmp, 1) == 0) {
    memcpy(dst, (const void *)src, n);
    done = (ssize_t)n;
  }
  g_in_guard = 0;
  sigaction(SIGSEGV, &old_segv, NULL);
  sigaction(SIGBUS, &old_bus, NULL);
  return done;
}

struct allocrec { uint64_t handle; uint32_t size; };
#define MAXALLOC 64
static struct allocrec g_allocs[MAXALLOC];
static int g_nallocs = 0;
static uint32_t g_pending_alloc_size = 0;   /* MEM ALLOC 请求里的 size（POST 时被响应覆盖） */

static void dump_buffers(void) {
  const char *dir = getenv("IODIR");
  if (!dir) dir = "/data/local/tmp/npuops/probe/buf";
  char path[256];
  for (int i = 0; i < nmaps; i++) {
    if (maps[i].len < 8 || maps[i].len > (64u << 20)) continue;  /* 64 MiB：真维度的常量区是 28 MB，4 MiB 上限会跳过它 */
    snprintf(path, sizeof(path), "%s/map%02d_fd%d_len%zx%s.bin", dir, i, maps[i].fd,
             maps[i].len, getenv("BUFSUFFIX") ? getenv("BUFSUFFIX") : "");
    int f = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (f < 0) continue;
    unsigned char buf[4096];
    size_t done = 0;
    while (done < maps[i].len) {
      size_t chunk = maps[i].len - done;
      if (chunk > sizeof(buf)) chunk = sizeof(buf);
      ssize_t got = guarded_copy(buf, maps[i].addr + done, chunk);
      if (got <= 0) break;
      if (write(f, buf, (size_t)got) != got) break;
      done += (size_t)got;
    }
    close(f);
    dprintf(logfd, "BUFDUMP %s wrote=0x%zx/0x%zx\n", path, done, maps[i].len);
  }
  /* ALLOC 过但进程没 mmap 的缓冲（整块的 28 MB 常量区就是这样：厂商 runtime 把 .dla 的
     内容直接写进这块 dma-buf，不经过 mmap，所以以前抓不到）。这里自己 mmap 一次再 dump。 */
  dprintf(logfd, "ALLOCS n=%d (dumping unmapped allocations)\n", g_nallocs);
  for (int a = 0; a < g_nallocs; a++) {
    if (g_allocs[a].size < 8 || g_allocs[a].size > (64u << 20)) continue;
    int seen = 0;
    for (int i = 0; i < nmaps; i++)
      if (maps[i].fd == (int)g_allocs[a].handle && maps[i].len == g_allocs[a].size) { seen = 1; break; }
    if (seen) continue;
    void *mp = real_mmap(NULL, g_allocs[a].size, PROT_READ | PROT_WRITE, MAP_SHARED,
                         (int)g_allocs[a].handle, 0);
    if (mp == MAP_FAILED) {
      dprintf(logfd, "BUFDUMP-ALLOC skip fd=%d size=%u (mmap failed errno=%d)\n",
              (int)g_allocs[a].handle, g_allocs[a].size, errno);
      continue;
    }
    snprintf(path, sizeof(path), "%s/map%02d_fd%d_len%zx%s.bin", dir, 100 + a,
             (int)g_allocs[a].handle, (size_t)g_allocs[a].size,
             getenv("BUFSUFFIX") ? getenv("BUFSUFFIX") : "");
    int f = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (f >= 0) {
      size_t done = 0;
      while (done < g_allocs[a].size) {
        size_t chunk = g_allocs[a].size - done;
        if (chunk > 4096) chunk = 4096;
        ssize_t w = write(f, (const char *)mp + done, chunk);
        if (w <= 0) break;
        done += (size_t)w;
      }
      close(f);
      dprintf(logfd, "BUFDUMP-ALLOC %s wrote=0x%zx/0x%zx\n", path, done,
              (size_t)g_allocs[a].size);
    }
    munmap(mp, g_allocs[a].size);
  }
}

static void dump_maps(void) {
  dprintf(logfd, "MAPS n=%d\n", nmaps);
  for (int i = 0; i < nmaps; i++) {
    dprintf(logfd, "  map[%d] addr=0x%llx len=0x%zx fd=%d\n", i,
            (unsigned long long)maps[i].addr, maps[i].len, maps[i].fd);
    if (maps[i].len >= 64 && maps[i].len <= 65536) {
      unsigned char buf[64];
      ssize_t got = safe_read(buf, maps[i].addr, sizeof(buf));
      if (got == (ssize_t)sizeof(buf)) hexdump("head", buf, sizeof(buf), 4);
    }
  }
}

static unsigned req_size(unsigned long req) { return (unsigned)((req >> 16) & 0x3fffu); }

int ioctl(int fd, int req_, ...) {
  va_list ap;
  void *arg;
  unsigned long req = (unsigned long)req_;
  va_start(ap, req_);
  arg = va_arg(ap, void *);
  va_end(ap);

  if (!inited) {
    inited = 1;
    real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    real_mmap = dlsym(RTLD_NEXT, "mmap");
    open_log();
    atexit(dump_final);

  }

  unsigned type = (req >> 8) & 0xffu, nr = req & 0xffu, size = req_size(req);
  int apusys = (logfd >= 0) && (type == 0x41u);

  if (apusys) {
    dprintf(logfd, "PRE  fd=%d nr=%u size=%u\n", fd, nr, size);
    if (arg && size >= 8 && size <= 256) hexdump("in ", arg, size, 2);
    if (nr == 33 && arg && size >= 24 && *(const uint32_t *)arg == 0)
      memcpy(&g_pending_alloc_size, (const char *)arg + 20, 4);
    if (nr == 34 && arg && size >= 8 && ((unsigned char *)arg)[0] == 0 /* RUN */) {
      static int patch_loaded = 0;
      if (!patch_loaded) { patch_loaded = 1; load_patch(); }
      if (g_patch_off >= 0 && (unsigned)g_patch_off + 4 <= size) {
        uint32_t old = 0;
        memcpy(&old, (char *)arg + g_patch_off, 4);
        memcpy((char *)arg + g_patch_off, &g_patch_val, 4);
        dprintf(logfd, "PATCH off=0x%x old=0x%08x new=0x%08x\n", g_patch_off, old,
                g_patch_val);
      }
      if (g_use_zero_adj && size >= 0x60) {
        uint64_t old = 0; memcpy(&old, (char *)arg + 0x58, 8);
        uint64_t nv = (uint64_t)(uintptr_t)g_zero;
        memcpy((char *)arg + 0x58, &nv, 8);
        dprintf(logfd, "PATCH_ADJ old=0x%llx new=0x%llx\n",
                (unsigned long long)old, (unsigned long long)nv);
      }
      dump_pointers((const unsigned char *)arg, size, "cmd_run");
      dump_subcmds((const unsigned char *)arg, size);
      dump_maps();
      dump_buffers();
    }
    // MEM UNMAP is issued after the inference completed: snapshot before teardown.
    if (nr == 33 && arg && size >= 4 && *(const uint32_t *)arg == 3 /* UNMAP */ &&
        !getenv("BUFSUFFIX")) {
      setenv("BUFSUFFIX", "_post", 1);
      dprintf(logfd, "POST-RUN-DUMP begin\n");
      dump_buffers();
      dprintf(logfd, "POST-RUN-DUMP end\n");
    }
  }

  errno = 0;
  int ret = real_ioctl(fd, req_, arg);
  int err = errno;

  if (apusys) {
    dprintf(logfd, "POST fd=%d nr=%u size=%u ret=%d errno=%d\n", fd, nr, size, ret, err);
    if (arg && size >= 8 && size <= 256) hexdump("out", arg, size, 2);
    /* 记下每次 ALLOC 的 (handle,size)，供 dump_buffers 补抓"没被 mmap 过"的缓冲 */
    if (nr == 33 && arg && size >= 24 && ret == 0 && g_nallocs < MAXALLOC) {
      uint64_t h = 0; memcpy(&h, arg, 8);
      if (h && g_pending_alloc_size) {
        g_allocs[g_nallocs].handle = h;
        g_allocs[g_nallocs].size = g_pending_alloc_size;
        g_nallocs++;
      }
      g_pending_alloc_size = 0;
    }
  }  errno = err;
  return ret;
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
  if (!inited) {
    inited = 1;
    real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    real_mmap = dlsym(RTLD_NEXT, "mmap");
    open_log();
    atexit(dump_final);

  }
  void *p = real_mmap(addr, len, prot, flags, fd, off);
  if (logfd >= 0 && p != MAP_FAILED && fd >= 0 && nmaps < MAXMAP && len >= 8 && len <= (1 << 20)) {
    maps[nmaps].addr = (uintptr_t)p;
    maps[nmaps].len = len;
    maps[nmaps].fd = fd;
    nmaps++;
    dprintf(logfd, "MMAP addr=0x%llx len=0x%zx fd=%d prot=0x%x flags=0x%x\n",
            (unsigned long long)(uintptr_t)p, len, fd, prot, flags);
  }
  return p;
}
