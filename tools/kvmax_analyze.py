#!/usr/bin/env python3
"""Ticket kv-maxlen: turn ops/kvmax/timing.jsonl into PAIRED ratios (NOTES 11.5).

Absolute ms from one .dla drifts +-15% inside a session, so nothing here is reported as a
performance conclusion: per cache length we take per_invoke = (T(-c N) - T(-c 1)) / (N-1),
then form the ratio B/A *within the same round* (each round is one contiguous block of the
locked session, order alternating ascending/descending) and report the MEDIAN ratio.
"""
import json
import os
import statistics as st
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "ops/kvmax/timing.jsonl"
recs = [json.loads(l) for l in open(P) if l.strip()]
ok = [r for r in recs if r.get("per_invoke_ms") not in (None, "null") and r.get("rc1") == "0"
      and r.get("rcN") == "0"]
cs = sorted({r["C"] for r in ok})
print("%8s %10s %s" % ("C", "per_ms", "per-round per-invoke ms (magnitude only)"))
cur = {}
for c in cs:
    v = [r["per_invoke_ms"] for r in ok if r["C"] == c]
    cur[c] = v
    print("%8d %10.1f %s" % (c, st.median(v), ["%.1f" % x for x in v]))

print("\npaired ratios (same round, B/A) -- the only quotable numbers:")
byround = {}
for r in ok:
    byround.setdefault(r["round"], {})[r["C"]] = r["per_invoke_ms"]
for i in range(1, len(cs)):
    a, b = cs[i - 1], cs[i]
    ratios = [d[b] / d[a] for d in byround.values() if a in d and b in d and d[a]]
    if ratios:
        print("  C %6d -> %6d : median x%.3f   (rounds: %s)"
              % (a, b, st.median(ratios), ["%.3f" % x for x in ratios]))

if cs:
    lo = cs[0]
    print("\nKV-width share, using C=%d as the tower floor:" % lo)
    for c in cs:
        print("  C %6d : per_invoke %.1f ms = floor %.1f + width %.1f ms"
              % (c, st.median(cur[c]), st.median(cur[lo]),
                 st.median(cur[c]) - st.median(cur[lo])))
