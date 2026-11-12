#!/usr/bin/env python3
"""Compare MDLA command buffers captured for different FC shapes.

Usage: /usr/bin/python3 probe/diff_cmd.py probe/buf/map10_64x64.bin probe/buf/map10_64x128.bin probe/buf/map10_128x128.bin
Labels default to the file stem.
"""
import struct
import sys


def main(paths):
    data = []
    for p in paths:
        with open(p, "rb") as f:
            data.append((p.split("/")[-1], f.read()))
    n = min(len(b) for _, b in data)
    print(f"# {len(data)} command buffers, length {n} bytes\n")
    print(f"{'word':<6}{'off':<8}" + "".join(f"{name:>14}" for name, _ in data) + "  class")
    changed = 0
    for i in range(0, n, 4):
        words = [struct.unpack_from("<I", b, i)[0] for _, b in data]
        if len(set(words)) == 1:
            continue
        changed += 1
        cls = "?"
        if len(words) == 3:
            base, mid, hi = words
            if base != mid and mid == hi:
                cls = "K-dependent (N same)"
            elif base == mid and base != hi:
                cls = "N-dependent (K same)"
            elif base != mid and base != hi and mid == hi:
                cls = "K/N coupled"
            else:
                cls = "both/other"
        print(f"w{i // 4:<5d}0x{i:<7x}" + "".join(f"{w:#014x}" for w in words) + f"  {cls}")
    print(f"\nchanged words: {changed}/{n // 4}")


if __name__ == "__main__":
    main(sys.argv[1:])
