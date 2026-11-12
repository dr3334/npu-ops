#!/usr/bin/env python3
"""Decode the LD_PRELOAD ioctl log (probe/iolog.c), separating caller inputs (PRE)
from driver outputs (POST). The kernel zeroes in-structs on some paths, so only PRE
shows the caller's `op` and arguments.

ABI: MTK apusys mdw_ioctl.h (handshake nr=32 / mem nr=33 / cmd nr=34 / util nr=35,
magic 'A'=0x41). Field offsets are validated against the observed payloads; anything
that does not fit is printed raw instead of being guessed.
"""
import re
import struct
import sys
from collections import Counter

NR_NAME = {0: "UNKNOWN(nr=0)", 32: "HANDSHAKE", 33: "MEM", 34: "CMD", 35: "UTIL"}
MEM_OP = {0: "ALLOC", 1: "FREE", 2: "MAP", 3: "UNMAP", 4: "FLUSH", 5: "INVALIDATE"}
MEM_TYPE = {0: "MAIN", 1: "VLM", 2: "LOCAL", 3: "SYSTEM", 4: "SYSTEM_ISP", 5: "SYSTEM_APU"}
CMD_OP = {0: "RUN", 1: "RUN_STALE", 2: "DEL", 3: "ENQ"}


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def decode_mem(b):
    op = u32(b, 0)
    name = MEM_OP.get(op, f"UNKNOWN({op})")
    if op == 0:
        t, size, align, fl = u32(b, 16), u32(b, 20), u32(b, 24), u64(b, 32)
        return (f"MEM ALLOC   type={t}({MEM_TYPE.get(t,'?')}) size={size} (0x{size:x}) "
                f"align=0x{align:x} flags=0x{fl:x}")
    if op == 1:
        return f"MEM FREE    handle=0x{u64(b, 16):x}"
    if op == 2:
        return (f"MEM MAP     handle=0x{u64(b, 16):x} size=0x{u32(b, 24):x} "
                f"flags=0x{u64(b, 28):x}")
    if op == 3:
        return f"MEM UNMAP   handle=0x{u64(b, 16):x}"
    if op in (4, 5):
        return (f"MEM {name:11s} handle=0x{u64(b, 16):x} off=0x{u32(b, 24):x} "
                f"size=0x{u32(b, 28):x}")
    return f"MEM raw op={op} {b[16:40].hex()}"


def decode_cmd(b):
    op = u32(b, 0)
    words = [hex(u64(b, i * 8)) for i in range(min(8, len(b) // 8))]
    return (f"CMD {CMD_OP.get(op, 'op=' + str(op)):9s} words[0:8]={words} "
            f"(offsets beyond op not yet matched to the 6.1 header)")


def parse(path):
    lines = open(path, "r", errors="replace").read().splitlines()
    entries, cur = [], None
    for ln in lines:
        m = re.match(r"PRE\s+fd=(\d+) type=0x([0-9a-f]+) nr=(\d+) size=(\d+)", ln)
        if m:
            cur = dict(fd=int(m.group(1)), type=int(m.group(2), 16), nr=int(m.group(3)),
                       size=int(m.group(4)), pre=b"", post=b"", ret=None)
            entries.append(cur)
            continue
        mp = re.match(r"POST fd=(\d+) type=0x([0-9a-f]+) nr=(\d+) size=(\d+) ret=(-?\d+) ",
                      ln)
        if mp and cur is not None:
            cur["ret"] = int(mp.group(5))
            continue
        mh = re.match(r"\s+(in |out)\[(\d+)\]=(.*)$", ln)
        if mh and cur is not None:
            data = bytes.fromhex(mh.group(3).replace(" ", ""))
            cur["pre" if mh.group(1).strip() == "in" else "post"] = data
    return entries


def main(path):
    es = parse(path)
    print(f"# ioctl catalog — {path}: {len(es)} apusys ioctl calls\n")

    print("## 1. by request")
    print("| type | nr | size | count | name |")
    print("|---|---|---|---|---|")
    for (t, nr, sz), c in sorted(Counter((e["type"], e["nr"], e["size"]) for e in es).items()):
        print(f"| 0x{t:02x} | {nr} | 0x{sz:x} | {c} | {NR_NAME.get(nr,'?')} |")

    print("\n## 2. MEM calls (PRE = caller input)")
    for e in es:
        if e["nr"] != 33:
            continue
        extra = ""
        if len(e["post"]) >= 8:
            extra = f"  -> out first8=0x{u64(e['post'], 0):x}"
        print(f"ret={e['ret']:3d}  {decode_mem(e['pre'])}{extra}")

    print("\n## 3. CMD calls")
    for e in es:
        if e["nr"] == 34:
            print(f"ret={e['ret']}  {decode_cmd(e['pre'])}")

    print("\n## 4. HANDSHAKE sequence")
    for e in es:
        if e["nr"] == 32:
            op, sub = u32(e["pre"], 0), u32(e["pre"], 16)
            print(f"in: op={op} sub={sub}  -> out: {e['post'][:40].hex()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "probe/io.log")
