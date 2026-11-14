#!/usr/bin/env python3
"""41-perop-table.py — per-command (per-layer) breakdown out of the runtime's level-3 dump.

READING (and why):
  * execution_result.json: inference -> codes[one per MDLA core] -> flows[one per command]
    each flow has pmu.endCycle and 14 undocumented PMU events.
  * endCycle is monotonic inside a core, so delta(endCycle) is that command's slot on the
    core's completion timeline; the deltas sum to the core's last endCycle by construction.
  * the alternative reading, event 6914 - event 6912, sums to ~30x the core's endCycle, so it
    is NOT a duration (see the note printed at the end).  We keep endCycle.
  * cycles -> us needs a clock. We do not measure it; we solve it from the runtime's own
    exec_us for the same DLA (NeuronRuntime_getProfiledQoSData). The *check* is that the two
    cores then imply the same clock -- if they did not, the reading would be wrong.

usage: 41-perop-table.py <exec_us>   (exec_us from profqos on the same .dla)
"""
import json, sys

EV = "ops/_probe/bb07-prof/board/"
exec_us = float(sys.argv[1]) if len(sys.argv) > 1 else 264.0

er = json.load(open(EV + "execution_result.json"))
di = json.load(open(EV + "dla_info.json"))
sub = di["graphs"][0]["subgraphs"][0]
code_info = sub["codes"]
inf = er["inferences"][0]

print(f"target={inf['target']} cores={len(inf['codes'])} "
      f"buffers={len(sub['buffers'])} "
      f"(L1={sum(1 for b in sub['buffers'] if b['memoryType']=='L1')}, "
      f"DRAM={sum(1 for b in sub['buffers'] if b['memoryType']=='DRAM')})")

implied = []
for ci, code in enumerate(inf["codes"]):
    cinfo = code_info[ci] if ci < len(code_info) else {"flows": []}
    acc = {f["id"]: f for f in cinfo.get("flows", [])}
    flows = code["flows"]
    core_end = flows[-1]["pmu"]["endCycle"]
    implied.append(core_end / (exec_us * 1e-6))
    print()
    print(f"== core {ci}: {len(flows)} commands, last endCycle {core_end}")
    print("   cmd   ops            cyc      share    bytes   B/us    us")
    prev = 0
    rows = []
    for f in flows:
        endc = f["pmu"]["endCycle"]
        d = endc - prev
        prev = endc
        a = acc.get(f["id"], {})
        nbytes = sum(ba["size"] for ba in a.get("bufferAccesses", []))
        rows.append((f["id"], a.get("opIndices", []), d, nbytes))
    freq = core_end / (exec_us * 1e-6)
    for (fid, ops, d, nbytes) in rows:
        us = d / freq * 1e6
        print(f"  {fid:5d}  {str(ops)[:14]:14s} {d:7d}  {100.0*d/core_end:5.1f}%  "
              f"{nbytes:7d}  {(nbytes/us if us else 0):6.1f}  {us:6.2f}")
    print(f"  TOTAL {'':14s} {core_end:7d}  100.0%          {'':6s}  {exec_us:6.2f}")

print()
print(f"implied cores clock (MHz): {[round(x/1e6, 1) for x in implied]}  "
      f"<- must agree for the endCycle reading to hold")
print(f"exec_us anchor: {exec_us} (NeuronRuntime_getProfiledQoSData)")
