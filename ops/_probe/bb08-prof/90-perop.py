#!/usr/bin/env python3
"""90-perop.py — per-command table out of a level-3 dump, reconciled against a wall clock.

WHY this reading (inherited from 07's perop_table.py, not re-derived):
  execution_result.json : inference -> codes[one per MDLA core] -> flows[one per command]
                          each flow: pmu.endCycle + 14 PMU events (event semantics UNKNOWN).
  endCycle is monotonic per core, so delta(endCycle) is that command's slot on the core's
  completion timeline and the deltas sum to that core's last endCycle by construction.
  dla_info.json maps the same flow id -> opIndices[] + bufferAccesses[] (bytes touched).
  The exec plan printed by ncc-tflite ("FullyConnectedLayer<6>{FULLY_CONNECTED}") uses the same
  index space as opIndices, so --plan turns every command into <layer class>{<op>} + a name.

RECONCILIATION, explicitly:
  A = slowest core's last endCycle / clock -> the device-compute total (us).
      The two cores run concurrently, so the DEVICE total is the slowest core, not the sum.
  B = the same-run wall clock (neuronrt "Total inference time"/N, or profqos MEASURED_US/loops)
  deviation = (B - A)/B -> how much of the wall clock is NOT device compute.
  The clock is only known to 0.67-1.0 GHz (07 6.3), so A carries that scale and the deviation
  is reported at both ends. Shares do not depend on the clock at all (the map is linear).

usage:
  90-perop.py --exec E.json --info I.json --plan P.compile.log --label fc-P3 \
              --exec-us 264 --clock-mhz 1000 [--wall-us 780] [--out out.json]
"""
import argparse, json, re

MOVEMENT = {"TRANSPOSE", "CONCATENATION", "SPLIT", "RESHAPE"}
WEIGHTED = {"FULLY_CONNECTED", "BATCH_MATMUL", "CONV_2D", "DEPTHWISE_CONV_2D"}

ap = argparse.ArgumentParser()
ap.add_argument("--exec", required=True)
ap.add_argument("--info", required=True)
ap.add_argument("--plan", default=None)
ap.add_argument("--label", default="?")
ap.add_argument("--exec-us", type=float, default=None, help="runtime's own exec_us (same product)")
ap.add_argument("--clock-mhz", type=float, default=1000.0)
ap.add_argument("--wall-us", type=float, default=None, help="same-run wall clock, us/inf")
ap.add_argument("--out", default=None)
ap.add_argument("--top", type=int, default=15)
A = ap.parse_args()

er = json.load(open(A.exec))
di = json.load(open(A.info))
sub = di["graphs"][0]["subgraphs"][0]
codes_info = sub["codes"]
inf = er["inferences"][0]

names = {}
if A.plan:
    for ln in open(A.plan, errors="replace"):
        m = re.search(r"([A-Za-z0-9_]+Layer)<(\d+)>\{([A-Z_0-9]+)\}", ln)
        if m:
            names[int(m.group(2))] = (m.group(1), m.group(3))

def classify(ops):
    cls = {names.get(i, ("?", "?"))[1] for i in ops}
    if not cls:
        return "?"
    if cls & WEIGHTED:
        return "weighted-matmul"
    if cls <= MOVEMENT:
        return "movement"
    return "elementwise"

rows = []
for ci, code in enumerate(inf["codes"]):
    acc = {f["id"]: f for f in (codes_info[ci]["flows"] if ci < len(codes_info) else [])}
    prev = 0
    for f in code["flows"]:
        endc = f["pmu"]["endCycle"]
        d = endc - prev
        prev = endc
        a = acc.get(f["id"], {})
        ops = a.get("opIndices", [])
        nbytes = sum(ba["size"] for ba in a.get("bufferAccesses", []))
        opclass = [names.get(i, ("?", "op%d" % i))[1] for i in ops]
        layclass = [names.get(i, ("?", "?"))[0] for i in ops]
        rows.append(dict(core=ci, id=f["id"], cycles=d, bytes=nbytes, ops=ops,
                         opclass="+".join(opclass) or "?", layer="+".join(layclass) or "?",
                         kind=classify(ops)))

core_tot = {}
for ci, code in enumerate(inf["codes"]):
    core_tot[ci] = code["flows"][-1]["pmu"]["endCycle"]
slow = max(core_tot.values())
sum_all = sum(core_tot.values())

clk = A.clock_mhz
print("== %s ==" % A.label)
print("target=%s cores=%d buffers=%d (L1=%d, DRAM=%d)" % (
    inf["target"], len(inf["codes"]), len(sub["buffers"]),
    sum(1 for b in sub["buffers"] if b["memoryType"] == "L1"),
    sum(1 for b in sub["buffers"] if b["memoryType"] == "DRAM")))
print("commands: " + ", ".join("core%d=%d" % (c, len(inf["codes"][c]["flows"])) for c in sorted(core_tot)))
print("clock assumed %.0f MHz  (us = cycles / %.0f)" % (clk, clk))
for c in sorted(core_tot):
    print("core %d: %d cycles = %.1f us" % (c, core_tot[c], core_tot[c] / clk))
print("device compute (slowest core) = %.1f us ; sum over cores = %.1f us" % (slow / clk, sum_all / clk))
if A.exec_us:
    print("runtime exec_us (same product) = %.0f us  -> derived clock = %.1f MHz" % (
        A.exec_us, slow / A.exec_us))

for c in sorted(core_tot):
    tot = core_tot[c]
    sub_rows = [r for r in rows if r["core"] == c]
    print()
    print("== core %d: %d commands, %d cycles = %.1f us ==" % (c, len(sub_rows), tot, tot / clk))
    print("  %4s %-6s %-26s %8s %6s %9s %8s %7s" % ("cmd", "ops", "layer<op>", "cycles", "share", "bytes", "B/us", "us"))
    for r in sub_rows:
        us = r["cycles"] / clk
        print("  %4d %-6s %-26s %8d %5.1f%% %9d %8.1f %7.2f" % (
            r["id"], str(r["ops"])[:6], (r["layer"] + "<" + r["opclass"] + ">")[:26], r["cycles"],
            100.0 * r["cycles"] / tot if tot else 0, r["bytes"], (r["bytes"] / us if us else 0.0), us))

agg = {}
for r in rows:
    a = agg.setdefault(r["opclass"], dict(cycles=0, bytes=0, n=0))
    a["cycles"] += r["cycles"]; a["bytes"] += r["bytes"]; a["n"] += 1
print()
# both cores run concurrently, so summing their cycles is "total device work"; shares are
# taken against that sum (a per-slowest-core denominator would exceed 100% once both cores are
# included). Per-core shares are what the earlier per-core tables show.
print("== by op class (both cores summed = total core-cycles; share of that total) ==")
print("  %-40s %5s %9s %6s %10s %8s" % ("ops", "n", "cycles", "share", "bytes", "B/us"))
for k in sorted(agg, key=lambda k: -agg[k]["cycles"]):
    a = agg[k]
    us = a["cycles"] / clk
    print("  %-40s %5d %9d %5.1f%% %10d %8.1f" % (k[:40], a["n"], a["cycles"], 100.0 * a["cycles"] / sum_all,
                                                  a["bytes"], a["bytes"] / us if us else 0))
print()
print("== by kind ==")
kinds = {}
for r in rows:
    kinds[r["kind"]] = kinds.get(r["kind"], 0) + r["cycles"]
for k in sorted(kinds, key=lambda k: -kinds[k]):
    print("  %-16s %9d cycles  %5.1f%% of total core-cycles" % (k, kinds[k], 100.0 * kinds[k] / sum_all))

print()
print("== top %d commands by cycles ==" % A.top)
for r in sorted(rows, key=lambda r: -r["cycles"])[:A.top]:
    us = r["cycles"] / clk
    print("  core%d cmd%-4d %-32s %8d cyc %6.2f us %9d B %7.1f B/us" % (
        r["core"], r["id"], (r["layer"] + "<" + r["opclass"] + ">")[:32], r["cycles"], us, r["bytes"],
        r["bytes"] / us if us else 0))

print()
print("== reconciliation ==")
if A.wall_us:
    lo = slow / 1000.0
    hi = slow / 670.0
    print("  wall clock (same run)           = %.1f us" % A.wall_us)
    print("  compute, slowest core @1.00GHz  = %.1f us" % lo)
    print("  compute, slowest core @0.67GHz  = %.1f us" % hi)
    print("  deviation (wall-compute)/wall   = %+.1f%% (@1.00GHz) .. %+.1f%% (@0.67GHz)" % (
        100.0 * (A.wall_us - lo) / A.wall_us, 100.0 * (A.wall_us - hi) / A.wall_us))
    print("  compute share of wall           = %.1f%% .. %.1f%%" % (
        100.0 * lo / A.wall_us, 100.0 * hi / A.wall_us))

if A.out:
    json.dump(dict(label=A.label, clock_mhz=clk, core_cycles=core_tot, slowest=slow,
                   sum_cores=sum_all, exec_us=A.exec_us, wall_us=A.wall_us,
                   by_class=agg, by_kind=kinds, rows=rows), open(A.out, "w"), indent=1)
    print("[json] %s" % A.out)
