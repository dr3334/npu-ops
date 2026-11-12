// LD_PRELOAD ioctl recorder for aarch64 Android (no root daemon needed).
// Logs each apusys ('A' = 0x41) ioctl twice: PRE (caller's input buffer) and POST
// (driver's output buffer). The kernel zeroes the in-struct in some paths, so the
// PRE snapshot is the only place where the caller's `op` field is visible.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static int (*real_ioctl)(int, int, ...);
static int logfd = -1;
static int inited = 0;

static void open_log(void) {
  const char *p = getenv("IOLOG");
  if (!p) p = "/data/local/tmp/npuops/probe/io.log";
  logfd = open(p, O_WRONLY | O_CREAT | O_APPEND, 0644);
}

static void hexdump(const char *tag, const void *buf, size_t n) {
  char line[640];
  const unsigned char *p = (const unsigned char *)buf;
  size_t i, off = 0;
  for (i = 0; i < n && off < sizeof(line) - 4; i++) {
    int k = snprintf(line + off, sizeof(line) - off, "%02x", p[i]);
    if (k <= 0) break;
    off += (size_t)k;
    if ((i & 15) == 15 && off < sizeof(line) - 2) line[off++] = ' ';
  }
  line[off] = 0;
  dprintf(logfd, "  %s[%zu]=%s\n", tag, n, line);
}

static unsigned req_size(unsigned long req) { return (unsigned)((req >> 16) & 0x3fffu); }

static void log_pre(int fd, unsigned long req, void *arg) {
  unsigned size = req_size(req);
  dprintf(logfd, "PRE  fd=%d type=0x%02x nr=%u size=%u\n",
          fd, (unsigned)((req >> 8) & 0xffu), (unsigned)(req & 0xffu), size);
  if (arg && size >= 8 && size <= 256) hexdump("in ", arg, size);
}

static void log_post(int fd, unsigned long req, void *arg, int ret, int err) {
  unsigned size = req_size(req);
  dprintf(logfd, "POST fd=%d type=0x%02x nr=%u size=%u ret=%d errno=%d\n",
          fd, (unsigned)((req >> 8) & 0xffu), (unsigned)(req & 0xffu), size, ret, err);
  if (arg && size >= 8 && size <= 256) hexdump("out", arg, size);
}

int ioctl(int fd, int req_, ...) {
  va_list ap;
  void *arg;
  unsigned long req = (unsigned long)req_;
  va_start(ap, req_);
  arg = va_arg(ap, void *);
  va_end(ap);

  if (!inited) {
    inited = 1;
    if (real_ioctl == NULL) real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    open_log();
  }

  int apusys = (logfd >= 0) && (((req >> 8) & 0xffu) == 0x41u);
  if (apusys) log_pre(fd, req, arg);

  errno = 0;
  int ret = real_ioctl(fd, req_, arg);
  int err = errno;

  if (apusys) log_post(fd, req, arg, ret, err);
  errno = err;
  return ret;
}
