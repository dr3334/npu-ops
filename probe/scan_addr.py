import os
import struct
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "probe/assets/fc_const"
targets = {}
# iova.txt: A scratch C
vals = open(os.path.join(d, "iova.txt")).read().split()
targets["A"] = int(vals[0], 16)
targets["scratch"] = int(vals[1], 16)
targets["C"] = int(vals[2], 16)

for name, iova in targets.items():
    if not iova:
        continue
    w32 = iova >> 4
    for f in ["cmd0.bin", "cmd1.bin", "cmd2.bin", "cmdin.bin", "subcmd.bin"]:
        p = os.path.join(d, f)
        if not os.path.exists(p):
            continue
        b = open(p, "rb").read()
        h32 = [o for o in range(0, len(b) - 3, 4) if struct.unpack_from("<I", b, o)[0] == w32]
        h64 = [o for o in range(0, len(b) - 7, 4) if struct.unpack_from("<Q", b, o)[0] == iova]
        if h32 or h64:
            print(f"{name:8s} 0x{iova:x} {f:12s} u32@{[hex(x) for x in h32]} u64@{[hex(x) for x in h64]}")
