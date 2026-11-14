#!/usr/bin/env python3
"""93-static-budget.py — analytic (NOT measured) cost prior for the 61-layer decoder graph.

Why: ticket 08 asks which layers/classes are expensive and whether the cost is arithmetic or
data movement. The board measurement answers it; this analytic budget is the *prior* the
measurement has to agree with, and it is the only answer available if the board is missing.
Every number is derived from model.json shapes only (no device involved) -- it is a model, not
a measurement, and the report labels it that way.

MACs = M*K*N for FC (M = T = 8), weight bytes = K*N*2 (fp16), activation bytes = M*(K+N)*2.
"""
import json, sys

meta = json.load(open(sys.argv[1] if len(sys.argv) > 1 else
                      "ops/_probe/bb08-prof/build/dec/model.json"))
s = meta["shape"]
T, D, Hq, Hkv, HD, I = s["T"], s["D"], s["Hq"], s["Hkv"], s["HD"], s["I"]

def fc(name, K, N, M=T):
    macs = M * K * N
    w = K * N * 2
    act = M * (K + N) * 2
    return dict(name=name, macs=macs, wbytes=w, actbytes=act)

def bmm(name, B, M, K, N):
    return dict(name=name, macs=B * M * K * N, wbytes=0, actbytes=B * (M * K + K * N + M * N) * 2)

rows = [
    fc("q_proj", D, Hq * HD),
    fc("k_proj", D, Hkv * HD),
    fc("v_proj", D, Hkv * HD),
    bmm("qk^T", Hq, T, HD, T),
    bmm("pv", Hq, T, T, HD),
    fc("o_proj", Hq * HD, D),
    fc("gate_proj", D, I),
    fc("up_proj", D, I),
    fc("down_proj", I, D),
]
tot_macs = sum(r["macs"] for r in rows)
tot_w = sum(r["wbytes"] for r in rows)
print("shapes: T=%d D=%d Hq=%d Hkv=%d HD=%d I=%d" % (T, D, Hq, Hkv, HD, I))
print("  %-10s %12s %7s %10s %7s %10s" % ("part", "MACs", "share", "w_bytes", "share", "act_bytes"))
for r in rows:
    print("  %-10s %12d %6.1f%% %10d %6.1f%% %10d" % (
        r["name"], r["macs"], 100.0 * r["macs"] / tot_macs, r["wbytes"],
        100.0 * r["wbytes"] / tot_w if tot_w else 0, r["actbytes"]))
print("  %-10s %12d %6.1f%% %10d %6.1f%%" % ("TOTAL", tot_macs, 100.0, tot_w, 100.0))
grp = {
    "MLP (gate/up/down)": sum(r["macs"] for r in rows if r["name"] in ("gate_proj", "up_proj", "down_proj")),
    "attn FC (q/o)": sum(r["macs"] for r in rows if r["name"] in ("q_proj", "o_proj")),
    "attn FC (k/v)": sum(r["macs"] for r in rows if r["name"] in ("k_proj", "v_proj")),
    "BMM (scores/pv)": sum(r["macs"] for r in rows if r["name"] in ("qk^T", "pv")),
}
print("\n== analytic grouping ==")
for k, v in sorted(grp.items(), key=lambda kv: -kv[1]):
    print("  %-20s %12d MACs  %6.1f%%" % (k, v, 100.0 * v / tot_macs))
print("\nweights dominate the bytes: %.1f MiB of fp16 weights for one layer." % (tot_w / 2**20))
print("arithmetic intensity of the whole layer: %.2f MAC/byte (weights only: %.2f)"
      % (tot_macs / (tot_w + sum(r["actbytes"] for r in rows)), tot_macs / tot_w))
