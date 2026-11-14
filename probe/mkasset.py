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


def _cb_desc(raw):
    """24 字节的 mdw_subcmd_cmdbuf 描述符 -> [(handle,size,align,dir)]。"""
    out = []
    for k in range(len(raw) // 24):
        e = raw[k * 24:(k + 1) * 24]
        out.append((u64(e, 0), u32(e, 8), u32(e, 12), u32(e, 16)))
    return out


def parse_log(path):
    """Return (cmd_in, groups, maps)。

    groups: [{'subcmd': bytes|None, 'cmdbufs': [(handle,size,align,dir)]}, ...]
      * 新格式（shim 会枚举 num_subcmds）：每个子命令一组，标签是 subcmd<i>/cmdbufs<i>；
      * 老格式（只有一个 subcmd[80]/cmdbufs[72]）：就是一组。
    一块 40 op 的图会带 2 个子命令 —— 只抓第一个会漏掉输出地址。
    """
    lines = open(path, "r", errors="replace").read().splitlines()
    cmd_in = None
    legacy_sub = None
    legacy_cbs = []
    subs = {}
    cbs = {}
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
        m = re.match(r"  subcmd(\d+)\[(\d+)\]=(.*)$", ln)
        if m:
            subs[int(m.group(1))] = bytes.fromhex(m.group(3).replace(" ", ""))
            continue
        m = re.match(r"  cmdbufs(\d+)\[(\d+)\]=(.*)$", ln)
        if m:
            cbs[int(m.group(1))] = _cb_desc(bytes.fromhex(m.group(3).replace(" ", "")))
            continue
        m = re.match(r"  subcmd\[(\d+)\]=(.*)$", ln)
        if m and legacy_sub is None:
            legacy_sub = bytes.fromhex(m.group(2).replace(" ", ""))
            continue
        m = re.match(r"  cmdbufs\[(\d+)\]=(.*)$", ln)
        if m:
            legacy_cbs.extend(_cb_desc(bytes.fromhex(m.group(2).replace(" ", ""))))
    if subs or cbs:
        n = max(list(subs) + list(cbs)) + 1
        groups = [{"subcmd": subs.get(i), "cmdbufs": cbs.get(i, [])} for i in range(n)]
        groups = [g for g in groups if g["subcmd"] or g["cmdbufs"]]
    else:
        groups = [{"subcmd": legacy_sub, "cmdbufs": legacy_cbs}]
    return cmd_in, groups, maps


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


def iova_of_path(maps, path):
    """(fd, 文件大小) 一起匹配 MAP 表 -> 抓取那次的 IOVA；匹配不上返回 0。

    fd 会重复（.dla 代码缓冲和输入缓冲可能同 fd），所以不能只用 fd。
    """
    m = re.search(r"_fd(\d+)_len([0-9a-f]+)", os.path.basename(path))
    if not m:
        return 0
    fd = int(m.group(1))
    fsz = int(m.group(2), 16)
    for h, sz, va in maps:
        if h == fd and sz == fsz:
            return va
    return 0


def find_output_by_vendor(bufdir, vendor, nbytes, maps=None):
    """有的算子会多分配几个 4 KiB 缓冲（scratch/输出）；用抓取时的厂商输出反查真正的输出缓冲。

    注意：fd 会被复用，同一 fd 的几个 mmap 在 dump 里可能拿到**同一份内容**，于是 .dla 代码缓冲
    的后半段也可能"等于"输出。判据因此加一条：候选必须能在 MAP 表里解析出 IOVA —— 解析不出来的
    是假候选（代码缓冲不是我们 ALLOC/MAP 出来的数据缓冲）。
    """
    if not vendor or not os.path.exists(vendor):
        return None
    want = open(vendor, "rb").read(nbytes)
    for f in sorted(os.listdir(bufdir)):
        if "_post" not in f:
            continue
        p = os.path.join(bufdir, f)
        try:
            if open(p, "rb").read(nbytes) == want:
                base = f.replace("_post.bin", ".bin")
                cand = os.path.join(bufdir, base)
                # 只接受 4 KiB 的数据缓冲：抓取目录里还有 .dla 代码缓冲（几千字节），
                # 它的前 256 字节可能与输出偶然相同，选错会把地址写进代码区。
                if not os.path.exists(cand):
                    continue
                if maps is not None and not iova_of_path(maps, cand):
                    continue
                return cand
        except OSError:
            pass
    return None


def find_by_size(bufdir, size):
    for f in sorted(os.listdir(bufdir)):
        if "_post" in f or "_final" in f:
            continue
        p = os.path.join(bufdir, f)
        if os.path.getsize(p) == size:
            return p
    return None


def find_data(bufdir, a, b, unary):
    A = B = C = None
    other = []
    code_size = None
    for f in sorted(os.listdir(bufdir)):
        if "_post" in f or "_final" in f:
            continue
        p = os.path.join(bufdir, f)
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
    b_full = b"" if unary else open(bbin, "rb").read()
    b = b_full[:64]
    cmd_in, groups, maps = parse_log(log)
    if not cmd_in or not groups:
        sys.exit("log: cmdin/subcmd not found")
    all_cbs = [d for g in groups for d in g["cmdbufs"]]

    A, B, C = find_data(bufdir, a, b, unary)
    if not A:
        A = find_by_size(bufdir, len(a))
    if not B and not unary:
        B = find_by_size(bufdir, len(b_full))
    if not C:
        C = find_by_size(bufdir, 512)
    vendor = os.path.join(outdir, "vendor_out.bin")
    by_vendor = find_output_by_vendor(bufdir, vendor,
                                      os.path.getsize(vendor) if os.path.exists(vendor) else 0,
                                      maps)
    if by_vendor is not None:
        C = by_vendor
    if not (A and C) or (not unary and not B):
        sys.exit(f"data buffers: A={bool(A)} B={bool(B)} C={bool(C)}")

    def iova_of(path):
        """(fd, size) both, because a single fd can be reused by several maps."""
        m = re.search(r"_fd(\d+)_len([0-9a-f]+)", os.path.basename(path))
        fd = int(m.group(1))
        fsz = int(m.group(2), 16)
        for h, sz, va in maps:
            if h == fd and sz == fsz:
                return va
        return 0

    scratch = None
    if not B:
        # any buffer that some cmdbuf references but nobody claimed (LUT / const region)
        used_words = set()
        for handle, size, align, direction in all_cbs:
            src = find_by_fd(bufdir, handle, size)
            if not src:
                continue
            d = open(src, "rb").read()
            used_words |= {u32(d, o) for o in range(0, len(d) - 3, 4)}
        used = {os.path.basename(A), os.path.basename(C)}
        best = None
        for pth in sorted(os.listdir(bufdir)):
            if not (pth.startswith("map") and pth.endswith(".bin")) or "_post" in pth or "_final" in pth:
                continue
            if pth in used:
                continue
            va = iova_of(os.path.join(bufdir, pth))
            if va and (va >> 4) in used_words:
                best = os.path.join(bufdir, pth)
                break
        scratch = best
    iova_b = iova_of(B) if B else (iova_of(scratch) if scratch else 0)
    if iova_b == 0 and not unary:
        for h, sz, va in maps:
            if sz == len(b_full) and va:
                iova_b = va
                break
    iova = [iova_of(A), iova_b, iova_of(C)]
    if iova[0] == 0 or iova[2] == 0 or (not unary and iova[1] == 0):
        sys.exit(f"iova not resolved: {[hex(x) for x in iova]}")

    os.makedirs(outdir, exist_ok=True)
    # 多子命令：每个子命令一套 cmdbuf + 自己的 subcmd 模板；单子命令：保持老文件名
    if len(groups) > 1:
        lines = []
        for gi, g in enumerate(groups):
            if g["subcmd"]:
                open(os.path.join(outdir, f"subcmd{gi}.bin"), "wb").write(g["subcmd"])
            for k, (handle, size, align, direction) in enumerate(g["cmdbufs"]):
                src = find_by_fd(bufdir, handle, size)
                if src is None:
                    sys.exit(f"cmd{gi}_{k} handle={handle} size={size}: no dump file")
                shutil.copy(src, os.path.join(outdir, f"cmd{gi}_{k}.bin"))
            lines.append("%d %d %s" % (gi, len(g["cmdbufs"]),
                                         " ".join("%d:%d" % (d[1], d[3])
                                                  for d in g["cmdbufs"])))
        open(os.path.join(outdir, "subcmds.txt"), "w").write("\n".join(lines) + "\n")
    else:
        g = groups[0]
        if g["subcmd"]:
            open(os.path.join(outdir, "subcmd.bin"), "wb").write(g["subcmd"])
        for k, (handle, size, align, direction) in enumerate(g["cmdbufs"]):
            src = find_by_fd(bufdir, handle, size)
            if src is None:
                sys.exit(f"cmdbuf[{k}] handle={handle} size={size}: no dump file found")
            shutil.copy(src, os.path.join(outdir, f"cmd{k}.bin"))
    if scratch:
        shutil.copy(scratch, os.path.join(outdir, "scratch.bin"))
    open(os.path.join(outdir, "cmdin.bin"), "wb").write(cmd_in)
    open(os.path.join(outdir, "iova.txt"), "w").write("0x%x 0x%x 0x%x\n" % tuple(iova))
    open(os.path.join(outdir, "cmdsizes.txt"), "w").write(
        " ".join(str(d[1]) for d in all_cbs) + "\n")
    print(f"{outdir}: subcmds={len(groups)} cmdbufs={[d[1] for d in all_cbs]} "
          f"iova={[hex(x) for x in iova]}")

if __name__ == "__main__":
    main(*sys.argv[1:6])
