#!/usr/bin/env python3
"""Turn one captured run (iolog2 log + buffer dump dir) into a submitter asset dir.

    /usr/bin/python3 probe/mkasset.py <log> <bufdir> <a.bin> <b.bin> <outdir>

Produces: cmd368.bin cmd32.bin cmd16.bin cmdin.bin subcmd.bin iova.txt
iova.txt = "<oldA> <oldB> <oldC>" (hex), the IOVAs used by *this* capture — the
command stream still contains them, so the submitter needs them to relocate.
"""
import os
import re
import shutil
import struct
import sys

SIZES = {368: "cmd368.bin", 32: "cmd32.bin", 16: "cmd16.bin"}


def u32(b, o): return struct.unpack_from("<I", b, o)[0]
def u64(b, o): return struct.unpack_from("<Q", b, o)[0]


def parse_log(path):
    """Return (cmd_in_bytes, subcmd_bytes, [(handle,size,iova)...])."""
    lines = open(path, "r", errors="replace").read().splitlines()
    cmd_in = subcmd = None
    maps = []
    cur = None
    for ln in lines:
        m = re.match(r"PRE\s+fd=(-?\d+) nr=(\d+) size=(\d+)", ln)
        if m:
            cur = dict(nr=int(m.group(2)), pre=b"")
            continue
        if ln.startswith("  in ") and cur is not None:
            cur["pre"] = bytes.fromhex(ln.split("=", 1)[1].replace(" ", ""))
            continue
        if ln.startswith("  out") and cur is not None:
            post = bytes.fromhex(ln.split("=", 1)[1].replace(" ", ""))
            if cur["nr"] == 33 and len(cur["pre"]) >= 24 and len(post) >= 16:
                op = u32(cur["pre"], 0)
                if op == 0:
                    maps.append([u64(post, 0), u32(cur["pre"], 20), 0])   # handle,size,iova
                elif op == 2:
                    h = u64(cur["pre"], 16)
                    for e in maps:
                        if e[0] == h:
                            e[2] = u64(post, 8)
            elif cur["nr"] == 34 and len(cur["pre"]) >= 4 and u32(cur["pre"], 0) == 0:
                cmd_in = cur["pre"]
            cur = None
            continue
        m = re.match(r"  subcmd\[(\d+)\]=(.*)$", ln)
        if m and subcmd is None:
            subcmd = bytes.fromhex(m.group(2).replace(" ", ""))
    return cmd_in, subcmd, maps


def find_buffers(bufdir, a, b):
    """Locate the three cmdbufs (by size) and A/B/C (by content) in the dump dir."""
    by_size = {}
    data_4096 = []
    for f in sorted(os.listdir(bufdir)):
        if not f.endswith(".bin") or "_post" in f or "_final" in f:
            continue
        p = os.path.join(bufdir, f)
        n = os.path.getsize(p)
        if n in SIZES:
            by_size[n] = p
        elif n == 4096:
            data_4096.append(p)
    A = B = None
    other = []
    for p in data_4096:
        head = open(p, "rb").read(64)
        if head == a:
            A = p
        elif head == b:
            B = p
        else:
            other.append(p)
    C = other[0] if other else None
    return by_size, A, B, C


def main(log, bufdir, abin, bbin, outdir):
    a = open(abin, "rb").read(64)
    b = open(bbin, "rb").read(64)
    cmd_in, subcmd, maps = parse_log(log)
    if not cmd_in or not subcmd:
        sys.exit("log: cmdin/subcmd not found")
    by_size, A, B, C = find_buffers(bufdir, a, b)
    missing = [SIZES[s] for s in SIZES if s not in by_size]
    if missing or not (A and B and C):
        sys.exit(f"missing buffers: {missing} data={{A:{bool(A)} B:{bool(B)} C:{bool(C)}}}")

    # A/B/C 的文件名 -> fd -> IOVA
    def iova_of(path):
        m = re.search(r"_fd(\d+)_", os.path.basename(path))
        fd = int(m.group(1))
        for h, sz, va in maps:
            if h == fd and sz == 4096:
                return va
        return 0

    iova = [iova_of(A), iova_of(B), iova_of(C)]
    if 0 in iova:
        sys.exit(f"iova not resolved: {[hex(x) for x in iova]}")

    os.makedirs(outdir, exist_ok=True)
    for s, name in SIZES.items():
        shutil.copy(by_size[s], os.path.join(outdir, name))
    open(os.path.join(outdir, "cmdin.bin"), "wb").write(cmd_in)
    open(os.path.join(outdir, "subcmd.bin"), "wb").write(subcmd)
    with open(os.path.join(outdir, "iova.txt"), "w") as f:
        f.write("0x%x 0x%x 0x%x\n" % tuple(iova))
    print(f"{outdir}: cmd368/cmd32/cmd16 + cmdin(152) + subcmd(80) + iova="
          f"{[hex(x) for x in iova]}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
