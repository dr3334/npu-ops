#!/usr/bin/env bash
# 25-xruntime-and-levels.sh
#  (a) does --Xruntime alone set the profiling level? (env OFF)
#  (b) what level does a plain (no -P) DLA declare?
#  (c) does enabling profiling inflate the measured time? (observer effect)
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
export LD_LIBRARY_PATH="$S/host/lib"
RT="$S/host/bin/neuronrt"
B="$R/ops/_probe/bb07-prof/build2"
IN="$R/ops/_probe/bb07-prof/build/golden/input.bin"
T=/mnt/d/bb07-tmp/lvl
rm -rf "$T"; mkdir -p "$T"; cd "$T" || exit 3

echo "########## (a) --Xruntime with env OFF ##########"
x() {
  local opt="$1" log="$T/x.log"
  "$RT" -m null -a "$B/q003.dla" -i "$IN" -o "$T/o.bin" -c 2 --Xruntime "$opt" > "$log" 2>&1
  printf "  opt=[%-24s] rc=%-2s lines=%-3s prof-hits=%s\n" "$opt" "$?" "$(wc -l < "$log")" "$(grep -ic profil "$log")"
}
for o in "" "profiling-level:3" "profilingLevel=3" "profil:3" "verbose" "nbThreads:2"; do x "$o"; done

echo
echo "########## (b) level a plain DLA declares ##########"
for dl in q000 q001 q002 q003; do
  for lv in 1 3; do
    log="$T/b-$dl-$lv.log"
    MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=$lv "$RT" -m null -a "$B/$dl.dla" -i "$IN" -o "$T/o.bin" -c 2 > "$log" 2>&1
    printf "  DLA=%-5s ask=%-2s -> %s\n" "$dl" "$lv" "$(grep -iE "only supports|profiling level" "$log" | tr "\n" " | ")"
  done
done

echo
echo "########## (c) observer effect: level 1/2/3, -c 20 ##########"
for lv in 1 2 3; do
  for rep in 1 2; do
    log="$T/c-$lv-$rep.log"
    MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=$lv "$RT" -m null -a "$B/q003.dla" -i "$IN" -o "$T/o.bin" -c 20 > "$log" 2>&1
    printf "  level=%s rep=%s : %s\n" "$lv" "$rep" "$(grep -E "Total inference time" "$log")"
  done
done
echo
echo "########## profiling OFF ##########"
for rep in 1 2; do
  "$RT" -m null -a "$B/q003.dla" -i "$IN" -o "$T/o.bin" -c 20 > "$T/off-$rep.log" 2>&1
  printf "  off rep=%s : %s\n" "$rep" "$(grep -E "Total inference time" "$T/off-$rep.log")"
done
