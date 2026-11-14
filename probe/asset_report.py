"""Summarise every submitter asset: what buffers it has, where they live, and which
command-stream words are relocated addresses.

    /usr/bin/python3 probe/asset_report.py [assets_dir]
"""
import os
import struct
import sys


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def main(root="probe/assets"):
    rows = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "iova.txt")):
            continue
        vals = open(os.path.join(d, "iova.txt")).read().split()
        iovas = [int(v, 16) for v in vals]
        sizes = []
        if os.path.exists(os.path.join(d, "cmdsizes.txt")):
            sizes = [int(x) for x in open(os.path.join(d, "cmdsizes.txt")).read().split()]
        extra = ""
        sp = os.path.join(d, "scratch.bin")
        if os.path.exists(sp):
            nz = sum(1 for x in open(sp, "rb").read() if x)
            extra = f" scratch={os.path.getsize(sp)}B(nonzero {nz})"
        cmd = os.path.join(d, "cmd2.bin")
        hits = []
        if os.path.exists(cmd):
            b = open(cmd, "rb").read()
            words = [u32(b, o) for o in range(0, len(b) - 3, 4)]
            for idx, va in enumerate(iovas):
                if not va:
                    continue
                w = va >> 4
                hits += [(idx, o * 4) for o, x in enumerate(words) if x == w]
        dom = ["0x%x" % v for v in iovas if v]
        rows.append((name, sizes, dom, extra, sorted(hits)))
    w = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'asset':{w}s}  cmdbufs            iovas                                extras")
    for name, sizes, iov, extra, hits in rows:
        print(f"{name:{w}s}  {str(sizes):18s} {' '.join(iov):36s}{extra}")
        if hits:
            print(f"{'':{w}s}    address words at offsets: " +
                  ", ".join(f"buf{i}@0x{o:03x}" for i, o in hits))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "probe/assets")
