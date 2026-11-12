#!/usr/bin/env python3
"""Parse an iolog2 log into the ADD submission's memory map.

Log order per call: PRE / "  in [...]" / POST / "  out[...]"  (process on the out line)

    /usr/bin/python3 probe/parse_mem.py probe/ioAdd3.log
"""
import re
import struct
import sys


def u32(b, o): return struct.unpack_from("<I", b, o)[0]
def u64(b, o): return struct.unpack_from("<Q", b, o)[0]


def main(path):
    lines = open(path, "r", errors="replace").read().splitlines()
    cur = None
    for ln in lines:
        m = re.match(r"PRE\s+fd=(-?\d+) nr=(\d+) size=(\d+)", ln)
        if m:
            cur = dict(nr=int(m.group(2)), size=int(m.group(3)), pre=b"", post=b"")
            continue
        if ln.startswith("  in ") and cur is not None:
            cur["pre"] = bytes.fromhex(ln.split("=", 1)[1].replace(" ", ""))
            continue
        if ln.startswith("  out") and cur is not None:
            cur["post"] = bytes.fromhex(ln.split("=", 1)[1].replace(" ", ""))
            b, p = cur["pre"], cur["post"]
            if cur["nr"] == 33 and len(b) >= 24 and len(p) >= 16:
                op = u32(b, 0)
                if op == 0:
                    print(f"ALLOC handle={u64(p,0)} size={u32(b,20)} align=0x{u32(b,24):x}")
                elif op == 2:
                    print(f"MAP   handle={u64(b,16)} size=0x{u32(b,24):x} iova=0x{u64(p,8):x}")
            elif cur["nr"] == 34 and len(b) >= 4 and u32(b, 0) == 0:
                print(f"CMDIN {b.hex()}")
            cur = None
            continue
        m = re.match(r"  subcmd\[(\d+)\]=(.*)$", ln)
        if m:
            print(f"SUBSCMD {bytes.fromhex(m.group(2).replace(' ', '')).hex()}")
            continue
        m = re.match(r"  cmdbufs\[(\d+)\]=(.*)$", ln)
        if m:
            raw = bytes.fromhex(m.group(2).replace(" ", ""))
            print(f"CMDBUF n={len(raw)//24}")
            for k in range(len(raw) // 24):
                e = raw[k * 24:(k + 1) * 24]
                print(f"  [{k}] handle={u64(e,0)} size=0x{u32(e,8):x} "
                      f"align=0x{u32(e,12):x} dir={u32(e,16)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "probe/ioAdd3.log")
