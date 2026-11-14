#!/usr/bin/env bash
# 42-finalize.sh — per-op table with the cross-core clock argument + a couple of loose ends.
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
cd "$R" || exit 2
EV=ops/_probe/bb07-prof/board
B=ops/_probe/bb07-prof/build3
{
  echo "# per-command profile of ops/_probe/bb07-prof/build3/full_p3.dla (decoder layer, T=8, fp16)"
  echo "# source: MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3 on the board -> execution_result.json"
  echo "# clock: the two cores' last endCycle are 189009 (core0) and 263906 (core1);"
  echo "#        the runtime's exec_us for the same DLA is 264 us, and 263906 / 264us = 999.6 MHz,"
  echo "#        i.e. a round 1.0 GHz -- so us = cycles / 1000.  Core0 finishes 75 us earlier and idles."
  echo "#        Sum over the SLOWEST core therefore equals exec_us by construction; that is the"
  echo "#        criterion's 'sum ~ total', at the granularity of one command per layer-group."
  echo
  /usr/bin/python3 ops/_probe/bb07-prof/41-perop-table.py 264.0 | sed 's/^/  /'
} > "$EV/perop-table.txt" 2>&1
echo "wrote $EV/perop-table.txt ($(wc -l < "$EV/perop-table.txt") lines)"

echo
echo "== loose end: are attn_half / full actually different graphs? =="
cmp -s "$B/attn_half.tflite" "$B/full.tflite" && echo "  attn_half.tflite == full.tflite (byte identical)" || echo "  differ"
ls -l "$B/attn_half.tflite" "$B/full.tflite" | awk '{print "  " $5, $9}'
echo "== exec plan layer count in the -P3 compile log =="
grep -cE "Layer<" "$B/full_p3.compile.log" 2>/dev/null || true
grep -oE "\{[A-Z_]+\}" "$B/full_p3.compile.log" 2>/dev/null | sort | uniq -c | sort -rn | head -12
