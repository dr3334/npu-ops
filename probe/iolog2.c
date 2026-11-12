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

static void dump_buffers(void) {
  const char *dir = getenv("IODIR");
  if (!dir) dir = "/data/local/tmp/npuops/probe/buf";
  char path[256];
  for (int i = 0; i < nmaps; i++) {
    if (maps[i].len < 64 || maps[i].len > 65536) continue;
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
    if (nr == 34 && arg && size >= 8 && ((unsigned char *)arg)[0] == 0 /* RUN */) {
      dump_pointers((const unsigned char *)arg, size, "cmd_run");
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
  if (logfd >= 0 && p != MAP_FAILED && fd >= 0 && nmaps < MAXMAP && len >= 64 && len <= (1 << 20)) {
    maps[nmaps].addr = (uintptr_t)p;
    maps[nmaps].len = len;
    maps[nmaps].fd = fd;
    nmaps++;
    dprintf(logfd, "MMAP addr=0x%llx len=0x%zx fd=%d prot=0x%x flags=0x%x\n",
            (unsigned long long)(uintptr_t)p, len, fd, prot, flags);
  }
  return p;
}
