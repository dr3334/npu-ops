#!/usr/bin/env python3
"""40-perop-report.py — turn the runtime's level-3 JSON dumps into a per-op table.

Inputs (pulled off the board):
  board/execution_result.json  : per inference -> codes(=MDLA cores) -> flows(=commands)
                                 each flow: {id, pmu{endCycle, events[]}}
                                 event 6914 == endCycle, 6912 == that flow's start counter
  board/dla_info.json          : same flow ids -> bufferAccesses[{buffer_id,size}], opIndices[]
                                 plus buffers[{id,memoryType,size,type}]

Output: per core, per flow: op indices, active cycles, bytes touched, share of that core's
active cycles.  Cycles->us needs a clock; we do NOT assume one -- we report cycles and, for
convenience only, a calibrated column using the runtime's own exec_us (see CALIB_US below).
"""
import json, sys, collections

EV = "ops/_probe/bb07-prof/board/"
CALIB_US = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0   # NeuronRuntime exec_us for the run

er = json.load(open(EV + "execution_result.json"))
di = json.load(open(EV + "dla_info.json"))

sub = di["graphs"][0]["subgraphs"][0]
buffers = {b["id"]: b for b in sub["buffers"]}
code_info = sub["codes"]

inf = er["inferences"][0]
print(f"inference: target={inf['target']} graph={inf['graphIndex']} subgraph={inf['subgraphIndex']} "
      f"codes={len(inf['codes'])} (one per MDLA core)")
print(f"buffers: {len(buffers)}; "
      f"L1={sum(1 for b in buffers.values() if b['memoryType']=='L1')} "
      f"DRAM={sum(1 for b in buffers.values() if b['memoryType']=='DRAM')}")

total_active = 0
per_core_total = []
for ci, code in enumerate(inf["codes"]):
    cinfo = code_info[ci] if ci < len(code_info) else {"flows": []}
    acc = {f["id"]: f for f in cinfo.get("flows", [])}
    rows = []
    core_active = 0
    for f in code["flows"]:
        ev = {e["event"]: e["value"] for e in f["pmu"]["events"]}
        active = ev.get(6914, f["pmu"]["endCycle"]) - ev.get(6912, 0)
        a = acc.get(f["id"], {})
        ops = a.get("opIndices", [])
        nbytes = sum(ba["size"] for ba in a.get("bufferAccesses", []))
        rows.append((f["id"], ops, active, nbytes, f["pmu"]["endCycle"]))
        core_active += active
    per_core_total.append(core_active)
    total_active += core_active
    print()
    print(f"== core {ci}: {len(rows)} flows, active cycles {core_active}, last endCycle {code['flows'][-1]['pmu']['endCycle']}")
    print("  flow   ops                 active_cyc   share     bytes   bytes/cyc")
    for (fid, ops, active, nbytes, endc) in rows:
        share = 100.0 * active / core_active if core_active else 0.0
        bpc = (nbytes / active) if active else 0.0
        print(f"  {fid:5d}  {str(ops)[:18]:18s} {active:10d}  {share:5.1f}%  {nbytes:8d}  {bpc:9.3f}")

print()
print(f"TOTAL active cycles: {total_active}  (per core: {per_core_total})")
if CALIB_US > 0:
    freq = total_active / (CALIB_US * 1e-6)
    print(f"calibration: runtime exec_us={CALIB_US} -> implied clock {freq/1e6:.1f} MHz "
          f"(this makes sum==exec_us BY CONSTRUCTION; it is a unit conversion, not a check)")
    print()
    print("  per-op us (= active_cyc / freq) and MB/s (= bytes / us):")
    for ci, code in enumerate(inf["codes"]):
        cinfo = code_info[ci] if ci < len(code_info) else {"flows": []}
        acc = {f["id"]: f for f in cinfo.get("flows", [])}
        for f in code["flows"]:
            ev = {e["event"]: e["value"] for e in f["pmu"]["events"]}
            active = ev.get(6914, f["pmu"]["endCycle"]) - ev.get(6912, 0)
            a = acc.get(f["id"], {})
            nbytes = sum(ba["size"] for ba in a.get("bufferAccesses", []))
            us = active / freq * 1e6
            mbs = (nbytes / us) if us else 0.0
            print(f"    core{ci} flow {f['id']:3d} ops={str(a.get('opIndices',[]))[:14]:14s} "
                  f"{us:8.2f} us  {nbytes:7d} B  {mbs:9.1f} B/us")
