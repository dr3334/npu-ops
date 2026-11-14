"""可复用的测量与统计（记录 → 汇总 → 配对比较 → 出报告）。

为什么要有这一层：仓库的纪律是「同一份 .dla 的绝对 ms 会漂 ±15%，单次数字不能比」。
所以测量必须：

  1. **带 provenance 落盘**（资产/iter/wait/口径/主机/时间），否则过两天没人知道那是什么；
  2. **报中位数与 MAD**（离散度），不报「最好那一次」；
  3. 跨时间/跨设备比较只用**配对交替**的中位数比值（A/B/A/B…），不用两次绝对数相除。

用法：

    from npu_lib import stats

    xs = stats.measure(lambda: g.run({...}), rounds=5, warmup=1)      # 毫秒
    stats.record("block_1l_fp16/ours", xs, kind="host_wall", note="单次提交含 200ms 等待")

    r = stats.paired(fn_ours, fn_vendor, rounds=7)                    # 交替测量
    print(r["ratio"], r["median_a"], r["median_b"])

    python -m npu_lib.stats --render                                  # -> reports/timings.md

解析器（把已有的日志口径统一进来）：

    stats.parse_neuronrt(text)   # "Total inference time = 12.3 ms (0.61 ms/inf)" -> [0.61]
    stats.parse_runner(text)     # "TIMING iter=10 wait_ms=5 min=.. med=.." -> dict
"""
import json
import os
import statistics
import time

from . import env

DEFAULT_PATH = env.REPO_ROOT / "reports" / "timings.json"


# ------------------------------------------------------------------ 汇总
def summarize(xs):
    """一组毫秒样本 -> 汇总（中位数为主，MAD 表示离散度）。"""
    v = sorted(float(x) for x in xs)
    if not v:
        return {"n": 0}
    med = statistics.median(v)
    mad = statistics.median([abs(x - med) for x in v]) if len(v) > 1 else 0.0

    def pct(q):
        if len(v) == 1:
            return v[0]
        i = (len(v) - 1) * q
        lo, hi = int(i), min(int(i) + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (i - lo)

    return {
        "n": len(v),
        "min": v[0],
        "p10": pct(0.10),
        "median": med,
        "mean": sum(v) / len(v),
        "p90": pct(0.90),
        "max": v[-1],
        "mad": mad,
        "stdev": statistics.pstdev(v) if len(v) > 1 else 0.0,
        "jitter_pct": (100.0 * mad / med) if med else 0.0,
    }


# ------------------------------------------------------------------ 落盘
class Store:
    """reports/timings.json：只追加，不覆盖历史。"""

    def __init__(self, path=None):
        self.path = path or DEFAULT_PATH
        self.records = []
        if os.path.exists(self.path):
            try:
                self.records = json.load(open(self.path))
            except (ValueError, OSError):
                self.records = []

    def add(self, name, values, **meta):
        rec = {
            "name": name,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": os.environ.get("COMPUTERNAME") or os.uname().nodename,
            "values_ms": [round(float(x), 4) for x in values],
            "summary": summarize(values),
            "meta": meta,
        }
        self.records.append(rec)
        self.save()
        return rec

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.records, f, indent=1)

    def latest(self):
        """每个 name 只留最后一条（报告用）。"""
        out = {}
        for r in self.records:
            out[r["name"]] = r
        return out

    def render_md(self):
        rows = ["| 测量 | n | 中位数 ms | MAD | p10 | p90 | 口径 | 备注 |",
                "|---|---|---|---|---|---|---|---|"]
        for name, r in sorted(self.latest().items()):
            s, m = r["summary"], r.get("meta", {})
            if not s.get("n"):
                continue
            rows.append("| `%s` | %d | **%.3f** | %.3f | %.3f | %.3f | %s | %s |" % (
                name, s["n"], s["median"], s["mad"], s["p10"], s["p90"],
                m.get("kind", ""), (m.get("note", "") or "").replace("|", "/")))
        return "\n".join(rows) + "\n"


_STORE = None


def store(path=None):
    global _STORE
    if _STORE is None or path is not None:
        _STORE = Store(path)
    return _STORE


def record(name, values, **meta):
    return store().add(name, values, **meta)


# ------------------------------------------------------------------ 测量
def measure(fn, rounds=5, warmup=1):
    """跑 fn 若干轮，返回每轮毫秒。热跑（warmup）不计入样本。"""
    for _ in range(max(0, warmup)):
        fn()
    xs = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1e3)
    return xs


def paired(fn_a, fn_b, rounds=7, warmup=2):
    """配对交替测量（A,B,A,B,…）——跨时间比较只认这个。

    返回 {median_a, median_b, ratio, n, a_ms, b_ms}；ratio = median_b / median_a。
    """
    for _ in range(max(0, warmup)):
        fn_a()
        fn_b()
    a, b = [], []
    for _ in range(rounds):
        t0 = time.perf_counter(); fn_a(); a.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter(); fn_b(); b.append((time.perf_counter() - t0) * 1e3)
    sa, sb = summarize(a), summarize(b)
    ratio = (sb["median"] / sa["median"]) if sa.get("median") else None
    return {"n": rounds, "median_a": sa.get("median"), "median_b": sb.get("median"),
            "ratio": ratio, "summary_a": sa, "summary_b": sb, "a_ms": a, "b_ms": b}


# ------------------------------------------------------------------ 解析器
def parse_neuronrt(text):
    """厂商 runtime 自己的计时行 -> [ms/inf, ...]（多行时全收）。"""
    import re as _re
    out = []
    for m in _re.finditer(r"Total inference time = ([0-9.]+) ms \(([0-9.]+) ms/inf\)", text or ""):
        out.append(float(m.group(2)))
    return out


def parse_runner(text):
    """npu_run 的 TIMING 行 -> dict（设备侧一次提交的 min/med/max + 输出是否一致）。"""
    import re as _re
    m = _re.search(
        r"TIMING iter=(\d+) wait_ms=(\d+) min=([0-9.]+) med=([0-9.]+) max=([0-9.]+) "
        r"mean=([0-9.]+) all_equal=(\d)", text or "")
    if not m:
        return {}
    return {"iter": int(m.group(1)), "wait_ms": int(m.group(2)), "min": float(m.group(3)),
            "median": float(m.group(4)), "max": float(m.group(5)),
            "mean": float(m.group(6)), "all_equal": bool(int(m.group(7)))}


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="timings.json 的报告/查询")
    ap.add_argument("--render", action="store_true", help="渲染 reports/timings.md")
    ap.add_argument("--list", action="store_true", help="列出所有测量名")
    ap.add_argument("--path", default=None)
    args = ap.parse_args(argv)
    st = store(args.path)
    if args.list:
        for name, r in sorted(st.latest().items()):
            s = r["summary"]
            print("%-44s n=%-3d median=%.3f ms" % (name, s.get("n", 0), s.get("median", 0.0)))
        return 0
    if args.render:
        out = st.path.parent / "timings.md"
        header = "# 板上计时汇总（自动生成，勿手改）\n\n" \
                 "来源：`reports/timings.json`；生成：`PYTHONPATH=tools /usr/bin/python3 -m npu_lib.stats --render`\n\n" \
                 "口径说明：**中位数**是主指标，MAD 是离散度；跨时间/跨设备比较只认配对交替的比值，\n" \
                 "不要拿两次绝对数相除（同一份 `.dla` 会漂 ±15%）。\n\n"
        out.write_text(header + st.render_md())
        print("[ok]", out)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
