#!/usr/bin/env python3
"""Extract MEM MAP results (handle -> device VA) from an iolog2 log.

    /usr/bin/python3 probe/iova.py probe/ioE102A.log [probe/ioE102B.log ...]

Prints one line per MAP: handle, size, device_va. Use it to prove whether two runs
got different IOVAs (E-102 relocation evidence).
"""
import re
import struct
import sys


def decode(path):
    lines = open(path, "r", errors="replace").read().splitlines()
    out = []
    pending = None
    for ln in lines:
        m = re.match(r"PRE\s+fd=(\d+) nr=(\d+) size=(\d+)", ln)
        if m:
            pending = dict(fd=int(m.group(1)), nr=int(m.group(2)), size=int(m.group(3)),
                           buf=None, outbuf=None)
            continue
        h = re.match(r"\s+in \[(\d+)\]=(.*)$", ln)
        if h and pending is not None:
            pending["buf"] = bytes.fromhex(h.group(2).replace(" ", ""))
            continue
        h = re.match(r"\s+out\[(\d+)\]=(.*)$", ln)
        if h and pending is not None:
            pending["outbuf"] = bytes.fromhex(h.group(2).replace(" ", ""))
            if pending["nr"] == 33 and pending["buf"] and pending["buf"][:4] == b"\x02\x00\x00\x00":
                inb = pending["buf"]
                handle = struct.unpack_from("<Q", inb, 16)[0]
                size = struct.unpack_from("<I", inb, 24)[0]
                va = struct.unpack_from("<Q", pending["outbuf"], 8)[0] if pending["outbuf"] else 0
                out.append((handle, size, va))
            pending = None
    return out


def main(paths):
    for p in paths:
        maps = decode(p)
        print(f"== {p} : {len(maps)} MAP calls")
        for handle, size, va in maps:
            print(f"   handle={handle:<3d} size=0x{size:<6x} device_va=0x{va:x}")
    if len(paths) >= 2:
        a, b = decode(paths[0]), decode(paths[1])
        same = [x[2] for x in a] == [x[2] for x in b]
        print(f"\nIOVA sequences identical across runs: {same}")
        if not same:
            print("  run A:", [hex(x[2]) for x in a])
            print("  run B:", [hex(x[2]) for x in b])


if __name__ == "__main__":
    main(sys.argv[1:])
