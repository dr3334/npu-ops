"""Search a captured constant/code blob for references to a buffer IOVA, in several encodings.

    /usr/bin/python3 probe/scan_blob.py <blob.bin> <iova-hex> [<iova-hex> ...]

Encodings checked: u32 = iova, u32 = iova>>4, u64 = iova, u32 = iova+0x1000*k.
"""
import struct
import sys


def main(path, *iovas):
    b = open(path, "rb").read()
    print(f"blob {path}: {len(b)} bytes")
    for h in iovas:
        va = int(h, 16)
        pats = {
            "u64=va": struct.pack("<Q", va),
            "u32=va": struct.pack("<I", va & 0xFFFFFFFF),
            "u32=va>>4": struct.pack("<I", (va >> 4) & 0xFFFFFFFF),
            "u32=va>>12": struct.pack("<I", (va >> 12) & 0xFFFFFFFF),
        }
        for name, pat in pats.items():
            hits = []
            start = 0
            while True:
                i = b.find(pat, start)
                if i < 0:
                    break
                hits.append(i)
                start = i + 1
            if hits:
                print(f"  {h} {name:10s} hits at {[hex(x) for x in hits[:8]]}"
                      f"{' ...' if len(hits) > 8 else ''}  (total {len(hits)})")


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
