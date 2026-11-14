#!/usr/bin/env python3
"""bbmap —— 把 ncc 的 dump 文本（--show-memory-summary / --show-exec-plan）对到 .dla 里的字节区间。

    /usr/bin/python3 tools/bbmap.py --dla ops/_probe/bb06-dump/v0.dla \
        --tflite ops/_probe/bb06-dump/_build/model.tflite \
        --log ops/_probe/bb06-dump/v0.compile.log

做的事只有两件，都不猜：
  1. 从 .tflite 里读出**常量 buffer**（跳过 buffer 0 = IO 占位），在 .dla 里定位它们：
     先试精确子串；找不到就滑窗比**字节多重集**（编译器会重排常量块，见 docs/OP-dla-structure.md §3）。
  2. 从编译日志里解析 DRAM Usage 表，把 Input/Output/Temp/Static/Code/Total 与实测区间摆在一起，
     逐项判「对得上 / 对不上」，对不上的写原因。

对齐口径（重要）：dump 只印**一位小数的 KiB**（"4.2K"），所以对 Static 的比较必须在
同一个分辨率上做 —— 拿 "4.2K" 反算成 4301 B 再去和 4352 B 比，是**假的不一致**。
本工具把实测字节按 "%.1f" % (n/1024.0) 折成同样的字符串再比（C/Python 的 %.1f 都是
round-half-to-even，4.25 -> "4.2"，与实测一致）。

只读，不写产物；退出码 0 = 至少定位到一个常量块。
"""
import argparse
import json
import re
import sys


def fmt_k(n):
    """按 ncc 的显示口径把字节数折成 "4.2K"（一位小数 KiB）。"""
    return "%.1fK" % (n / 1024.0)


def constant_buffers(buf):
    """-> [(name, bytes)]：模型字节里真正带数据的 buffer（buffer 0 是 IO 占位，跳过）。"""
    import tflite
    m = tflite.Model.GetRootAsModel(buf, 0)
    names = {}
    for i in range(m.SubgraphsLength()):
        sg = m.Subgraphs(i)
        for t in range(sg.TensorsLength()):
            tt = sg.Tensors(t)
            names[tt.Buffer()] = tt.Name().decode() if tt.Name() else "t%d" % t
    out = []
    for b in range(m.BuffersLength()):
        bb = m.Buffers(b)
        if bb.DataLength() == 0:
            continue
        out.append((names.get(b, "buffer%d" % b), bytes(bb.Data(i) for i in range(bb.DataLength()))))
    return out


def tflite_constant_buffers(path):
    """同 constant_buffers，但接受文件路径。"""
    return constant_buffers(open(path, "rb").read())


def find_exact(hay, needle):
    i = hay.find(needle)
    return [] if i < 0 else [i]


def find_multiset(hay, needle):
    """滑窗找**字节多重集相等**的最左区间（长度等于 len(needle)）。O(n*256) 的计数窗。"""
    n, w = len(hay), len(needle)
    if w == 0 or w > n:
        return []
    want = [0] * 256
    for c in needle:
        want[c] += 1
    have = [0] * 256
    for c in hay[:w]:
        have[c] += 1
    hits = []
    for start in range(0, n - w + 1):
        if have == want:
            hits.append(start)
        if start + w < n:
            have[hay[start]] -= 1
            have[hay[start + w]] += 1
    return hits[:8]


def parse_mem_summary(log):
    """从编译日志里抠 DRAM Usage 表 -> {列名: 原文}，另带 L1。"""
    txt = open(log, errors="replace").read()
    m = re.search(r"DRAM Usage:\s*\n(.*?)\n\s*\n", txt, re.S)
    if not m:
        return None
    rows = [r for r in m.group(1).splitlines() if r.strip()]
    if len(rows) < 2:
        return None
    head = [h for h in rows[0].split()]
    vals = [v for v in rows[-1].split() if re.match(r"^[0-9.]+[KM]?$", v)]
    keys = ["Input", "Output", "Temp", "Static", "Code", "Total"]
    if len(vals) < len(keys):
        return None
    out = {k: v for k, v in zip(keys, vals[-6:])}
    l1 = re.search(r"L1:\s*(\S+)", txt)
    out["L1"] = l1.group(1) if l1 else "-"
    out["_header"] = head
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="dump 文本 <-> .dla 字节区间的对应表")
    ap.add_argument("--dla", required=True)
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--log", default="", help="对应的 ncc 编译日志（含 --show-memory-summary）")
    ap.add_argument("--json", default="", help="把结果另存为 JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    dla = open(args.dla, "rb").read()
    consts = tflite_constant_buffers(args.tflite)
    regions = []
    for name, data in consts:
        hits = find_exact(dla, data)
        how = "exact"
        if not hits:
            hits = find_multiset(dla, data)
            how = "multiset"
        for h in hits[:3]:
            regions.append({"tensor": name, "bytes": len(data), "offset": h,
                            "end": h + len(data) - 1, "how": how})
    regions.sort(key=lambda r: r["offset"])
    csum = sum(r["bytes"] for r in regions)
    res = {"dla": args.dla, "dla_bytes": len(dla), "constants": regions,
           "constants_total": csum, "verdicts": []}

    if not args.quiet:
        print("=" * 80)
        print("dla   : %s  (%d B)" % (args.dla, len(dla)))
        print("tflite: %s  (常量 buffer %d 个)" % (args.tflite, len(consts)))
        print("-" * 80)
        print("%-10s %8s %10s %10s %9s" % ("tensor", "bytes", "offset", "end", "how"))
        for r in regions:
            print("%-10s %8d %10d %10d %9s" % (r["tensor"], r["bytes"], r["offset"], r["end"], r["how"]))
        print("-" * 80)
        print("常量合计 %d B；产物 %d B；常量占 %.1f%%" % (csum, len(dla), 100.0 * csum / len(dla)))

    if args.log:
        ms = parse_mem_summary(args.log)
        res["memory_summary"] = ms
        if ms:
            def verdict(item, ok, detail):
                res["verdicts"].append({"item": item, "ok": bool(ok), "detail": detail})
                if not args.quiet:
                    print("[%s] %-22s %s" % ("对得上" if ok else "对不上", item, detail))

            if not args.quiet:
                print("-" * 80)
                print("dump: Input=%s Output=%s Temp=%s Static=%s Code=%s Total=%s L1=%s"
                      % (ms["Input"], ms["Output"], ms["Temp"], ms["Static"], ms["Code"], ms["Total"], ms["L1"]))
            verdict("Static == 常量块合计",
                    ms["Static"] == fmt_k(csum),
                    "实测常量 %d B -> %s；dump 印 %s" % (csum, fmt_k(csum), ms["Static"]))
            total_meas = "%.1fK" % ((int(ms["Input"]) + int(ms["Output"]) + int(ms["Temp"]) + csum
                                     + int(ms["Code"].rstrip("K"))) / 1024.0)
            verdict("Total == 五列之和", ms["Total"] == total_meas,
                    "Input+Output+Temp+Static+Code = %s；dump 印 %s" % (total_meas, ms["Total"]))
            code = int(ms["Code"])
            same = [r for r in regions if r["bytes"] == code]
            verdict("Code == 某个区段", bool(same),
                    "Code=%d B，产物里没有等长的常量区段可对应" % code if not same
                    else "Code=%d B 与区段 %s 等长" % (code, same))

    if args.json:
        open(args.json, "w").write(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    return 0 if regions else 1


if __name__ == "__main__":
    sys.exit(main())
