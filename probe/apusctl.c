// apusctl — our own /dev/apusys client (E-106 step A: memory pipeline).
//
// Today it performs what libneuron does before a submit, without libneuron:
//   HANDSHAKE (BASIC) -> MEM ALLOC -> MEM MAP -> MEM UNMAP -> MEM FREE
//
//   apusctl probe [size]   alloc+map one buffer, print handle/IOVA
//   apusctl buffers        alloc+map the four buffers of the FC pipeline
//
// Not yet: CMD RUN. mdw_cmd_in / mdw_subcmd_info field semantics are not confirmed by
// experiment, so per AGENTS.md R10 we stop before submitting instead of guessing.
//
// ABI: the argument buffer is a 48-byte union; the driver writes the OUT union starting
// at offset 0 and zeroes what it consumed, so OUT must be read from offset 0.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define APUSYS_MAGIC 0x41

typedef struct { uint32_t op, pad; uint64_t flags; uint64_t w0, w1, w2; } hs_args_t;         /* 40 B */
typedef struct { uint32_t op, pad; uint64_t flags; uint64_t w0, w1, w2, w3; } mem_args_t;  /* 48 B */

#define APU_HANDSHAKE _IOWR(APUSYS_MAGIC, 32, hs_args_t)
#define APU_MEM _IOWR(APUSYS_MAGIC, 33, mem_args_t)

_Static_assert(sizeof(hs_args_t) == 40, "hs args must be 40 bytes");
_Static_assert(sizeof(mem_args_t) == 48, "mem args must be 48 bytes");

static int g_fd = -1;

/* byte-level field access so offsets never depend on compiler packing */
static void put32(void *buf, unsigned off, uint32_t v) { memcpy((char *)buf + off, &v, 4); }
static void put64(void *buf, unsigned off, uint64_t v) { memcpy((char *)buf + off, &v, 8); }
static uint32_t get32(const void *buf, unsigned off) { uint32_t v; memcpy(&v, (const char *)buf + off, 4); return v; }
static uint64_t get64(const void *buf, unsigned off) { uint64_t v; memcpy(&v, (const char *)buf + off, 8); return v; }

static int apu_open(void) {
  g_fd = open("/dev/apusys", O_RDWR | O_SYNC);
  if (g_fd < 0) { perror("open /dev/apusys"); return -1; }
  return 0;
}

static int apu_handshake(void) {
  hs_args_t hs;
  memset(&hs, 0, sizeof(hs));
  put32(&hs, 0, 0);                        /* MDW_HS_IOCTL_OP_BASIC */
  if (ioctl(g_fd, APU_HANDSHAKE, &hs) < 0) { perror("handshake"); return -1; }
  printf("handshake: version=0x%llx dev_bitmask=0x%llx mem_bitmask=0x%llx meta_size=%llu\n",
         (unsigned long long)get64(&hs, 0), (unsigned long long)get64(&hs, 8),
         (unsigned long long)get64(&hs, 16), (unsigned long long)get64(&hs, 32));
  return 0;
}

static uint64_t apu_alloc(uint32_t size, uint32_t align) {
  mem_args_t m;
  memset(&m, 0, sizeof(m));
  put32(&m, 0, 0);                         /* ALLOC */
  put32(&m, 16, 0);                        /* type = MAIN */
  put32(&m, 20, size);
  put32(&m, 24, align);
  put64(&m, 32, 0);                        /* alloc flags */
  if (ioctl(g_fd, APU_MEM, &m) < 0) { perror("mem alloc"); return 0; }
  return get64(&m, 0);                     /* out.alloc.handle @0 */
}

static int apu_map(uint64_t handle, uint32_t size, uint64_t flags, uint64_t *va_out) {
  mem_args_t m;
  memset(&m, 0, sizeof(m));
  put32(&m, 0, 2);                         /* MAP */
  put64(&m, 16, handle);                   /* in.map.handle */
  put32(&m, 24, size);                     /* in.map.size   */
  put64(&m, 28, flags);                    /* in.map.flags  */
  if (ioctl(g_fd, APU_MEM, &m) < 0) { perror("mem map"); return -1; }
  *va_out = get64(&m, 8);                  /* out.map{type@0, device_va@8} */
  return 0;
}

static int apu_mem_op(unsigned op, uint64_t handle) {
  mem_args_t m;
  memset(&m, 0, sizeof(m));
  put32(&m, 0, op);                        /* 1=FREE, 3=UNMAP */
  put64(&m, 16, handle);
  return ioctl(g_fd, APU_MEM, &m);
}

static int cmd_probe(uint32_t size) {
  if (apu_open() || apu_handshake()) return 2;
  uint64_t h = apu_alloc(size, 4096);
  if (!h) return 3;
  uint64_t va = 0;
  if (apu_map(h, size, 2, &va)) return 4;
  printf("alloc: size=0x%x handle=0x%llx iova=0x%llx\n", size,
         (unsigned long long)h, (unsigned long long)va);
  apu_mem_op(3, h);
  apu_mem_op(1, h);
  close(g_fd);
  printf("probe: alloc/map/unmap/free all accepted by the driver\n");
  return 0;
}

static int cmd_buffers(void) {
  if (apu_open() || apu_handshake()) return 2;
  struct { const char *name; uint32_t size; } want[] = {
      {"cmd", 400}, {"param", 8192}, {"input", 4096}, {"output", 4096},
  };
  enum { N = 4 };
  uint64_t h[N], va[N];
  for (int i = 0; i < N; i++) {                 /* 1) allocate all first */
    h[i] = apu_alloc(want[i].size, 4096);
    if (!h[i]) return 3;
  }
  for (int i = 0; i < N; i++) {                 /* 2) map all, hold simultaneously */
    if (apu_map(h[i], want[i].size, 2, &va[i])) return 4;
    printf("buffer %-6s size=%-5u handle=0x%llx iova=0x%llx\n", want[i].name,
           want[i].size, (unsigned long long)h[i], (unsigned long long)va[i]);
  }
  for (int i = 0; i < N; i++) { apu_mem_op(3, h[i]); apu_mem_op(1, h[i]); }
  close(g_fd);
  return 0;
}

int main(int argc, char **argv) {
  const char *sub = (argc > 1) ? argv[1] : "probe";
  if (!strcmp(sub, "probe")) {
    uint32_t size = (argc > 2) ? (uint32_t)strtoul(argv[2], NULL, 0) : 4096;
    return cmd_probe(size);
  }
  if (!strcmp(sub, "buffers")) return cmd_buffers();
  fprintf(stderr, "usage: apusctl {probe [size]|buffers}\n");
  return 1;
}
