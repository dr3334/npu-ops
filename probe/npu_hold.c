// npu_hold — allocate NPU memory through /dev/apusys and hold it, to perturb the
// driver's IOVA allocator for E-102 (same model, different device addresses).
//
//   npu_hold <MiB> <seconds>
//
// This is our own first apusys ioctl caller (precursor of probe/apusctl.c). It only
// allocates memory; it never submits a command, so it does not run work on the NPU.
//
// ABI (see probe/ioctl-catalog.md, decoded from mdw_ioctl.h):
//   HANDSHAKE = _IOWR('A', 32, union mdw_hs_args)   40 bytes: op@0, flags@8, union@16
//   MEM       = _IOWR('A', 33, union mdw_mem_args)  48 bytes: op@0, flags@8,
//               alloc{type@16,size@20,align@24,flags@32}
//   MEM op: 0=ALLOC, 1=FREE
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define APUSYS_MAGIC 0x41

typedef struct {
  uint32_t op, pad;
  uint64_t flags;
  uint64_t u0, u1, u2;       /* union mdw_hs_out (40 bytes total) */
} hs_args_t;

typedef struct {
  uint32_t op, pad;
  uint64_t flags;
  uint32_t type;             /* 0 = MDW_MEM_TYPE_MAIN */
  uint32_t size;
  uint32_t align;
  uint32_t pad1;
  uint64_t aflags;
  uint64_t r0;               /* out: handle lands at offset 0 of the out union */
} mem_args_t;

#define APU_HANDSHAKE _IOWR(APUSYS_MAGIC, 32, hs_args_t)
#define APU_MEM _IOWR(APUSYS_MAGIC, 33, mem_args_t)

int main(int argc, char **argv) {
  long mib = (argc > 1) ? atol(argv[1]) : 64;
  long secs = (argc > 2) ? atol(argv[2]) : 20;
  size_t bytes = (size_t)mib * 1024u * 1024u;

  int fd = open("/dev/apusys", O_RDWR | O_SYNC);
  if (fd < 0) { perror("open /dev/apusys"); return 2; }

  hs_args_t hs;
  memset(&hs, 0, sizeof(hs));
  hs.op = 0;                                     /* MDW_HS_IOCTL_OP_BASIC */
  if (ioctl(fd, APU_HANDSHAKE, &hs) < 0) { perror("handshake"); return 3; }
  /* The driver writes the OUT union starting at offset 0, overwriting op/pad:
     out.basic = {version@0, dev_bitmask@8, mem_bitmask@16, flags@24, meta_size@32}. */
  uint64_t *w = (uint64_t *)&hs;
  printf("handshake ok version=0x%llx dev_bitmask=0x%llx mem_bitmask=0x%llx meta_size=%llu\n",
         (unsigned long long)w[0], (unsigned long long)w[1],
         (unsigned long long)w[2], (unsigned long long)w[4]);

  mem_args_t ma;
  memset(&ma, 0, sizeof(ma));
  ma.op = 0;                                     /* ALLOC */
  ma.type = 0;                                   /* MAIN */
  ma.size = (uint32_t)bytes;
  ma.align = 4096;
  if (ioctl(fd, APU_MEM, &ma) < 0) { perror("mem alloc"); return 4; }
  uint64_t handle = *(uint64_t *)&ma;            /* out.alloc.handle is at offset 0 */
  printf("allocated %ld MiB handle=0x%llx (holding %ld s)\n",
         mib, (unsigned long long)handle, secs);
  fflush(stdout);
  sleep((unsigned)secs);

  mem_args_t fr;
  memset(&fr, 0, sizeof(fr));
  fr.op = 1;                                     /* FREE */
  *(uint64_t *)&fr = handle;                     /* in.free.handle is at offset 0 too */
  ioctl(fd, APU_MEM, &fr);
  close(fd);
  printf("released\n");
  return 0;
}
