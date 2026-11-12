#!/usr/bin/env python3
"""Turn one captured run (iolog2 log + buffer dump dir) into a submitter asset dir.

    /usr/bin/python3 probe/mkasset.py <log> <bufdir> <a.bin> (<b.bin>|-) <outdir>

Produces: cmd0.bin cmd1.bin cmd2.bin cmdin.bin subcmd.bin iova.txt [cmdsizes.txt]
  cmdN.bin   — the three command buffers, in the order the subcmd descriptor lists them
               (that order is semantic, see E-106B-3). Sizes are NOT hardcoded: they are
               taken from the mdw_subcmd_cmdbuf array printed by the shim, so unary ops
               (which use a 336-byte command stream instead of 368) work unchanged.
  iova.txt   — "<oldA> <oldB> <oldC>" (hex), 0 for the unused slot of a unary op.
"""
import os
import re
import shutil
import struct
import sys


def u32(b, o): return struct.unpack_from("<I", b, o)[0]
def u64(b, o): return struct.unpack_from("<Q", b, o)[0]


def parse_log(path):
    """Return cmd_in, subcmd, maps[(handle,size,iova)], cmdbufs[(handle,size,align,dir)]."""
    lines = open(path, "r", errors="replace").read().splitlines()
    cmd_in = subcmd = None
    cmdbufs = []
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
                    maps.append([u64(post, 0), u32(cur["pre"], 20), 0])
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
            continue
        m = re.match(r"  cmdbufs\[(\d+)\]=(.*)$", ln)
        if m:
            raw = bytes.fromhex(m.group(2).replace(" ", ""))
            for k in range(len(raw) // 24):
                e = raw[k * 24:(k + 1) * 24]
                cmdbufs.append((u64(e, 0), u32(e, 8), u32(e, 12), u32(e, 16)))
    return cmd_in, subcmd, maps, cmdbufs


def find_by_fd(bufdir, handle, size):
    """Buffer dumps are named map<idx>_fd<fd>_len<size>.bin and fd == MEM handle."""
    for f in sorted(os.listdir(bufdir)):
        if "_post" in f or "_final" in f:
            continue
        m = re.match(r"map\d+_fd(\d+)_len([0-9a-f]+)\.bin$", f)
        if not m:
            continue
        if int(m.group(1)) == handle and int(m.group(2), 16) == size:
            return os.path.join(bufdir, f)
    return None


def find_data(bufdir, a, b, unary):
    A = B = C = None
    other = []
    for f in sorted(os.listdir(bufdir)):
        if "_post" in f or "_final" in f:
            continue
        p = os.path.join(bufdir, f)
        if os.path.getsize(p) != 4096:
            continue
        head = open(p, "rb").read(64)
        if head == a:
            A = p
        elif not unary and head == b:
            B = p
        else:
            other.append(p)
    if other:
        C = other[0]
    return A, B, C


def main(log, bufdir, abin, bbin, outdir):
    unary = (bbin == "-")
    a = open(abin, "rb").read(64)
    b = b"" if unary else open(bbin, "rb").read(64)
    cmd_in, subcmd, maps, cmdbufs = parse_log(log)
    if not cmd_in or not subcmd:
        sys.exit("log: cmdin/subcmd not found")
    if len(cmdbufs) != 3:
        sys.exit(f"expected 3 cmdbuf descriptors, got {len(cmdbufs)}")

    A, B, C = find_data(bufdir, a, b, unary)
    if not (A and C) or (not unary and not B):
        sys.exit(f"data buffers: A={bool(A)} B={bool(B)} C={bool(C)}")

    def iova_of(path):
        m = re.search(r"_fd(\d+)_", os.path.basename(path))
        fd = int(m.group(1))
        for h, sz, va in maps:
            if h == fd and sz == 4096:
                return va
        return 0

    iova = [iova_of(A), iova_of(B) if B else 0, iova_of(C)]
    if iova[0] == 0 or iova[2] == 0 or (not unary and iova[1] == 0):
        sys.exit(f"iova not resolved: {[hex(x) for x in iova]}")

    os.makedirs(outdir, exist_ok=True)
    sizes = []
    for k, (handle, size, align, direction) in enumerate(cmdbufs):
        src = find_by_fd(bufdir, handle, size)
        if src is None:
            sys.exit(f"cmdbuf[{k}] handle={handle} size={size}: no dump file found")
        shutil.copy(src, os.path.join(outdir, f"cmd{k}.bin"))
        sizes.append(size)
    open(os.path.join(outdir, "cmdin.bin"), "wb").write(cmd_in)
    open(os.path.join(outdir, "subcmd.bin"), "wb").write(subcmd)
    with open(os.path.join(outdir, "iova.txt"), "w") as f:
        f.write("0x%x 0x%x 0x%x\n" % tuple(iova))
    with open(os.path.join(outdir, "cmdsizes.txt"), "w") as f:
        f.write(" ".join(str(s) for s in sizes) + "\n")
    print(f"{outdir}: cmdbufs={sizes} dirs={[c[3] for c in cmdbufs]} "
          f"iova={[hex(x) for x in iova]}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
