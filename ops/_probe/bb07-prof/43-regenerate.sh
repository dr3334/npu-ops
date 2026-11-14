#!/usr/bin/env bash
# 43-regenerate.sh — rebuild the per-op evidence file with the fixed 1.0 GHz clock.
set -uo pipefail
cd "/mnt/d/deepseek_ workspace/mtk/wt-bb07" || exit 2
EV=ops/_probe/bb07-prof/board
{
  echo "# per-command profile of ops/_probe/bb07-prof/build3/full_p3.dla"
  echo "# (Qwen2.5 decoder layer, T=8, D=896, fp16; 48 layers in the exec plan, 55 commands/core)"
  echo "# source: board run with MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3"
  echo "#         -> execution_result.json + dla_info.json (pulled, see this directory)"
  echo
  /usr/bin/python3 ops/_probe/bb07-prof/perop_table.py 264.0 1000 | sed 's/^/  /'
} > "$EV/perop-table.txt" 2>&1
tail -6 "$EV/perop-table.txt"
