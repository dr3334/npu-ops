"""Decode the MEM ioctl sequence from an iolog2 log: ALLOC (op=0) and MAP (op=2).

    /usr/bin/python3 probe/mem_map_flow.py <cap.log>

Prints, per call: op, size, align, handle/addr in, handle/iova out — so we can see which
buffers land in the 0x25xxxxxx device range and which get something else.
"""
import re
import struct
import sys


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def main(path):
    lines = open(path, errors="replace").read().splitlines()
    cur = None
    n = 0
    for ln in lines:
        m = re.match(r"PRE\s+fd=(-?\d+) nr=(\d+) size=(\d+)", ln)
        if m:
            cur = dict(nr=int(m.group(2)), pre=b"", post=None)
            continue
        if ln.startswith("  in ") and cur is not None:
            cur["pre"] = bytes.fromhex(ln.split("=", 1)[1].replace(" ", ""))
            continue
        if ln.startswith("  out") and cur is not None:
            cur["post"] = bytes.fromhex(ln.split("=", 1)[1].replace(" ", ""))
            pre, post = cur["pre"], cur["post"]
            if cur["nr"] == 33 and len(pre) >= 44 and len(post) >= 16:
                op = u32(pre, 0)
                if op == 0:      # ALLOC
                    size, align, typ = u32(pre, 20), u32(pre, 24), u32(pre, 40)
                    handle = u64(post, 0)
                    print(f"  [{n:2d}] ALLOC  size={size:8d} align=0x{align:<5x} type={typ:<3d} -> handle={handle}")
                elif op == 2:    # MAP
                    handle = u64(pre, 16)
                    size = u32(pre, 24)
                    flags = u64(pre, 28)
                    iova = u64(post, 8)
                    dom = "DEV(0x25xxxxxx)" if 0x20000000 <= iova < 0x30000000 else "OTHER"
                    print(f"  [{n:2d}] MAP    handle={handle:3d} size={size:8d} flags={flags:<3d} -> iova=0x{iova:016x}  {dom}")
                elif op == 1:
                    print(f"  [{n:2d}] FREE   handle={u64(pre, 16)}")
                elif op == 4:
                    print(f"  [{n:2d}] op=4   handle={u64(pre, 16)}")
                n += 1
            cur = None


if __name__ == "__main__":
    main(sys.argv[1])
