#!/usr/bin/env bash
# 21-Xruntime-probe.sh — option-string vocabulary, probed with the profiler ON (so the
# runtime is chatty). Host runtime, -m null.
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
export LD_LIBRARY_PATH="$S/host/lib"
RT="$S/host/bin/neuronrt"
B="$R/ops/_probe/bb07-prof/build2"
IN="$R/ops/_probe/bb07-prof/build/golden/input.bin"
T=/mnt/d/bb07-tmp/xrt
rm -rf "$T"; mkdir -p "$T"; cd "$T" || exit 3
export MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3

p() { # tag, option-string
  local tag="$1" opt="$2"
  "$RT" -m null -a "$B/q003.dla" -i "$IN" -o "$T/o_$tag.bin" -c 2 --Xruntime "$opt" > "$T/$tag.log" 2>&1
  local rc=$?
  local hit
  hit=$(grep -inE "options|invalid|format|unable|warn|error" "$T/$tag.log" | grep -viE "^1:INFO: The profiling|Runtime Initial|Detail of|Status|^ *-{10}" | head -6)
  printf '=== %-16s rc=%-2s opt=[%s]\n' "$tag" "$rc" "$opt"
  [ -n "$hit" ] && echo "$hit" | sed 's/^/    /'
}
p empty        ''
p threads_colon 'nbThreads:2'
p threads_eq    'nbThreads=2'
p threads_json  '{"nbThreads":2}'
p par_single    'parallelismOption:single'
p par_json      '{"parallelismOption":"single","nbThreads":2}'
p proflevel     'profiling-level:3'
p proflevel_eq  'profilingLevel=3'
p garbage       'garbage'
p dump          'dump-profiling-result'
