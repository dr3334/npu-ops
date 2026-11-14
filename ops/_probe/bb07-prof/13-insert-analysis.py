#!/usr/bin/env python3
"""13-insert-analysis.py — where exactly do -P1/-P2/-P3 insert, and what is in it?

Longest-common-prefix / longest-common-suffix decomposition turns each product into
"prefix + inserted blob + suffix", so the blob can be dumped on its own.
"""
import re
B = "ops/_probe/bb07-prof/build2/"
def load(n):
    return open(B + n + ".dla", "rb").read()

def decompose(a, b):
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b)) - p and a[len(a)-1-s] == b[len(b)-1-s]:
        s += 1
    return p, b[p:len(b)-s], a[p:len(a)-s], s

base = load("q000")
for name in ("q001", "q002", "q003", "q012", "q013"):
    d = load(name)
    p, ins, rem, s = decompose(base, d)
    print(f"== {name}: {len(d)} B; common prefix {p} B (-0x{p:x}), common suffix {s} B; inserted {len(ins)} B, removed {len(rem)} B")
    if ins:
        print("   inserted hexdump (first 256 B):")
        for off in range(0, min(len(ins), 256), 16):
            chunk = ins[off:off+16]
            hx = " ".join(f"{b:02x}" for b in chunk)
            tx = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            print(f"     {off:04x}  {hx:<47}  {tx}")
        txt = [m.group().decode() for m in re.finditer(rb"[ -~]{5,}", ins)]
        print("   ascii in inserted:", txt[:10])
    print()
