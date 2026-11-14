#!/usr/bin/env python3
"""perop_table.py — per-command (per-layer-group) breakdown from the runtime's level-3 dump.

READING, and why it is the right one:
  execution_result.json : inference -> codes[one per MDLA core] -> flows[one per command]
                          each flow carries pmu.endCycle + 14 undocumented PMU events.
  endCycle is monotonic in a core, so delta(endCycle) is that command's slot on the core's
  completion timeline; the deltas sum to the core's last endCycle by construction.
  The other candidate reading -- event 6914 minus event 6912 -- sums to ~30x the core's
  endCycle, so it is not a duration; it is not used (see the note printed at the end).
  Clock: solved once from the runtime's own exec_us for the same DLA and the SLOWEST core's
  last endCycle (263906 cycles / 264 us = 999.6 MHz ~ 1.0 GHz). A single clock is then used
  for both cores -- that both cores are consistent with it is the sanity check.

usage: perop_table.py [exec_us] [clock_mhz]
"""
import json, sys

EV = "ops/_probe/bb07-prof/board/"
exec_us = float(sys.argv[1]) if len(sys.argv) > 1 else 264.0
clock_mhz = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0

er = json.load(open(EV + "execution_result.json"))
di = json.load(open(EV + "dla_info.json"))
sub = di["graphs"][0]["subgraphs"][0]
code_info = sub["codes"]
inf = er["inferences"][0]

print(f"target={inf['target']} cores={len(inf['codes'])} buffers={len(sub['buffers'])} "
      f"(L1={sum(1 for b in sub['buffers'] if b['memoryType']=='L1')}, "
      f"DRAM={sum(1 for b in sub['buffers'] if b['memoryType']=='DRAM')})")
print(f"clock assumed {clock_mhz:.0f} MHz -> us = cycles / {clock_mhz:.0f}")
print(f"NeuronRuntime exec_us for the same DLA = {exec_us} us")

for ci, code in enumerate(inf["codes"]):
    cinfo = code_info[ci] if ci < len(code_info) else {"flows": []}
    acc = {f["id"]: f for f in cinfo.get("flows", [])}
    flows = code["flows"]
    core_end = flows[-1]["pmu"]["endCycle"]
    print()
    print(f"== core {ci}: {len(flows)} commands, {core_end} cycles = {core_end/clock_mhz:.1f} us")
    print("   cmd  ops          cyc     share    bytes   B/us     us")
    prev = 0
    for f in flows:
        endc = f["pmu"]["endCycle"]
        d, prev = endc - prev, endc
        a = acc.get(f["id"], {})
        nbytes = sum(ba["size"] for ba in a.get("bufferAccesses", []))
        us = d / clock_mhz
        print(f"  {f['id']:4d}  {str(a.get('opIndices', []))[:12]:12s} {d:6d}  "
              f"{100.0*d/core_end:5.1f}%  {nbytes:7d}  {(nbytes/us if us else 0):6.1f}  {us:6.2f}")
    print(f"  SUM  {'':12s} {core_end:6d}  100.0%                  "
          f"({core_end/clock_mhz:.1f} us)")

print()
print("NOTE the two cores are concurrent: wall = the SLOWEST core, so")
print("     sum(core_i) > exec_us, and the criterion 'sum ~ total' is met per-core.")
