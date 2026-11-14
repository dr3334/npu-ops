#!/usr/bin/env bash
# 30-host-levels.sh — the same level matrix on the HOST runtime (-m null), which DOES print the
# level-gating text and the LEVEL 1/2 summary tables (the board prints neither). This is the
# only place the "please use -P{} option" wording can be captured verbatim for OUR products.
#
#   bash ops/_probe/bb08-prof/30-host-levels.sh            # from WSL
#
# Evidence: host/<model>-P<lvl>-ask<ask>.txt (console) + host/gating-matrix.txt
set -uo pipefail
R="${R:-/mnt/d/deepseek_ workspace/mtk/wt-bb08}"
S="$R/../neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk/host"
B="$R/ops/_probe/bb08-prof/build"
EV="$R/ops/_probe/bb08-prof/host"
cd "$R" || exit 2
mkdir -p "$EV"
export LD_LIBRARY_PATH="$S/lib"
NRT="$S/bin/neuronrt"
"$NRT" --version 2>&1 | head -3

: > "$EV/gating-matrix.txt"
for m in fc dec; do
  IN="$B/golden/input.bin"; [ "$m" = dec ] && IN="$B/dec/golden/input.bin"
  for l in 0 1 2 3; do
    for ask in 1 2 3; do
      OUT="$EV/${m}-P${l}-ask${ask}.txt"
      timeout 300 env MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=$ask \
        "$NRT" -m null -a "$B/${m}_p${l}.dla" -i "$IN" -o /tmp/h_${m}_${l}_${ask}.bin -c 3 \
        > "$OUT" 2>&1
      echo "=== $m -P$l ask=$ask rc=$? ===" >> "$EV/gating-matrix.txt"
      grep -iE "profiling level|only supports|please use" "$OUT" >> "$EV/gating-matrix.txt"
      echo >> "$EV/gating-matrix.txt"
    done
  done
done
echo "== gating matrix =="
cat "$EV/gating-matrix.txt"
