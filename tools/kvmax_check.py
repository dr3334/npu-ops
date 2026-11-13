#!/usr/bin/env python3
"""Ticket kv-maxlen: build a NON-EMPTY-cache numeric check for a decode graph of any C.

device-vs-quantised-golden at width C (the implementation gate ticket 06 used), without
needing a prefill board run: the cache is seeded with the *quantised golden* prefill K/V
codes (same frozen scale table), 16 real rows out of C+8, so attention really runs over
C+8 slots instead of the trivial valid=0 path build_q06.py's own golden/ uses.

  prepare : write <dir>/live/{x,mask,cos*,sin*,k_in,v_in}.bin + expected h/k/v
  compare : compare <dir>/live/out_*.bin (pulled from the device) with the expectation

Nothing here modifies build_q06.py / build_w4a8.py; they are imported read-only.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_q06 as bq                     # noqa: E402
import build_w4a8 as bw                    # noqa: E402

D, KV, HD = bq.D, bq.KV, bq.HD


def _init():
    bw.set_tier(8, 16)
    bw.SEAM = "native"
    bw.set_source("fixture")
    bw.set_input("same")


def prepare(a):
    _init()
    P = {k: np.asarray(v, np.float64) for k, v in bw.fixture().items()}
    tab = bq.load_tab(a.tab)
    t, C, n = a.t, a.cache, a.layers
    xr = np.asarray(bw.input_hidden(t + 1), np.float64)[1:1 + t]
    gp = bq.golden_prefill(P, 16, n, tab)          # quantised golden prefill K/V codes
    valid = int(gp["k"].shape[1])
    k_in = np.zeros((n * KV, C, HD), np.int16)
    v_in = np.zeros((n * KV, C, HD), np.int16)
    k_in[:, :valid] = gp["k"]
    v_in[:, :valid] = gp["v"]
    cfg = dict(mode="decode", t=t, cache=C, layers=n)
    g = bq.golden_step(P, cfg, tab, xr, 1, valid, k_in, v_in)
    feed = bq.decode_feed(t, C, 1, valid, xr)
    os.makedirs(os.path.join(a.dir, "live"), exist_ok=True)
    bq.write_feed(os.path.join(a.dir, "live"),
                  dict(feed, x=bq.q_x(xr, tab["in.x"]),
                       k_in=k_in, v_in=v_in), bq.DECODE_IO)
    np.save(os.path.join(a.dir, "exp_h.npy"), g["h"])
    np.save(os.path.join(a.dir, "exp_k.npy"), g["k"])
    np.save(os.path.join(a.dir, "exp_v.npy"), g["v"])
    json.dump(dict(C=C, t=t, layers=n, valid=valid, s_h=float(g["s_h"]),
                   S_K=float(bq.S_K), S_V=float(bq.S_V)),
              open(os.path.join(a.dir, "exp_meta.json"), "w"), indent=2)
    print("prepare C=%d t=%d n=%d valid=%d s_h=%.6e  k_in=%d B  sat=%s"
          % (C, t, n, valid, g["s_h"], k_in.nbytes, list(g["sat"].items())[:4]))
    return 0


def compare(a):
    m = json.load(open(os.path.join(a.dir, "exp_meta.json")))
    C, t, n, valid, s_h = m["C"], m["t"], m["layers"], m["valid"], m["s_h"]
    bq.S_K, bq.S_V = m["S_K"], m["S_V"]
    e = os.path.join(a.dir, "live")
    eh = np.load(os.path.join(a.dir, "exp_h.npy"))
    ek = np.load(os.path.join(a.dir, "exp_k.npy"))
    ev = np.load(os.path.join(a.dir, "exp_v.npy"))
    dh = np.fromfile(os.path.join(e, "out_hidden.bin"), np.int16).reshape(t, D)[:1]
    dk = np.fromfile(os.path.join(e, "out_k.bin"), np.int16).reshape(n * KV, t, HD)[:, :1]
    dv = np.fromfile(os.path.join(e, "out_v.bin"), np.int16).reshape(n * KV, t, HD)[:, :1]
    rec = dict(C=C, valid=valid, cache_bytes=int(2 * n * KV * C * HD * 2),
               hidden=bq.code_stats(dh, eh[:1], s_h),
               k=bq.code_stats(dk, ek[:, :1], bq.S_K),
               v=bq.code_stats(dv, ev[:, :1], bq.S_V))
    rec["abs_dcos"] = abs(rec["hidden"]["cos"] - 1.0)
    rec["pass_impl"] = bool(rec["abs_dcos"] <= 0.003 and rec["hidden"]["codes_max"] <= 1200)
    print(json.dumps(rec, indent=2))
    with open(os.path.join(a.dir, "compare.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print("C=%d  |dcos| %.3e (gate 3e-3)  codes max %d (gate 1200)  rel_l2 %.4e  -> %s"
          % (C, rec["abs_dcos"], rec["hidden"]["codes_max"],
             rec["hidden"]["rel_l2"], "PASS" if rec["pass_impl"] else "FAIL"))
    return 0 if rec["pass_impl"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tab", default="ops/q06_tab.json")
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--cache", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=24)
    a = ap.parse_args()
    if a.prepare:
        return prepare(a)
    if a.compare:
        return compare(a)
    ap.error("give --prepare or --compare")


if __name__ == "__main__":
    sys.exit(main())
