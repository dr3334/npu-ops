#!/usr/bin/env bash
# Ticket 06: prefill + decode over ONE shared-weight bin, with a host-owned KV cache.
#
#   ./tools/e2e_q06.sh [--steps N]        # run inside ./tools/devlock.sh run --
#
# One lock session runs the whole sequence: prefill once, then N decode Runs.  Between two
# Runs the host writes the previous Run's KV row into its cache and pushes the whole
# [48, C, 64] buffer back -- that round trip IS the per-token copy cost the ticket asks to
# measure, so every phase is timed (device clock for the Run, host wall clock for the copy)
# and written to seq/timing.jsonl.
#
# NOTE neuronrt 7.3.15 prints NOTHING for these graphs (rc=0, outputs written), so the
# success test is the exit code, not a grep for "Inference" (ticket 05 graphs did print).
set -uo pipefail

STEPS=8
if [ $# -ge 2 ] && [ "$1" = "--steps" ]; then STEPS="$2"; fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
_win="$(cd "$ROOT" && pwd -W 2>/dev/null || echo /d/x)"
_drv="$(printf '%s' "$_win" | cut -c1 | tr 'A-Z' 'a-z')"
_rest="$(printf '%s' "$_win" | cut -c3-)"
WSLROOT="/mnt/$_drv$_rest"
DEV=/data/local/tmp/npuops
SUB=$DEV/q06
ADB=/d/adb/platform-tools/adb
WSL=wsl.exe
BIN=ops/q06_shared.bin
PRE=ops/q06_prefill_t16_24l
DEC=ops/q06_decode_t8_c2048_24l
SEQ=ops/q06_seq
WD='/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen'
REPEAT=20
export MSYS_NO_PATHCONV=1

winpath() { local p="$ROOT/$1"; ( cd "$(dirname "$p")" && printf '%s/%s' "$(pwd -W)" "$(basename "$p")" ); }
wslrun() { "$WSL" -e bash -lc "cd '$WSLROOT' && export QWEN25_WEIGHTS_DIR='$WD' && $*"; }
now() { date +%s%N; }
cd "$ROOT" || exit 2

PIN='-i q06/live/x.bin -i q06/live/mask.bin -i q06/live/cos14.bin -i q06/live/sin14.bin -i q06/live/cos2.bin -i q06/live/sin2.bin'
DIN="$PIN -i q06/live/k_in.bin -i q06/live/v_in.bin"
OUT='-o q06/live/out_hidden.bin -o q06/live/out_k.bin -o q06/live/out_v.bin'

# run_graph <dla> <inputs> <count>  -> prints "rc=<n> ms=<device ms>"
run_graph() {
  local dla=$1; local ins=$2; local cnt=$3
  local cmd="cd $DEV && a=\$(date +%s%N); LD_LIBRARY_PATH=\$PWD/lib ./bin/neuronrt -m hw -a q06/$dla $ins -i q06/shared.bin $OUT -c $cnt >/dev/null 2>&1; rc=\$?; b=\$(date +%s%N); echo rc=\$rc ms=\$(( (b-a)/1000000 ))"
  "$ADB" shell "$cmd" | tr -d '\r'
}

"$ADB" shell "mkdir -p $SUB/live" >/dev/null
"$ADB" push "$(winpath "$BIN")" "$SUB/shared.bin" >/dev/null || exit 5
"$ADB" push "$(winpath "$PRE/model_shared.dla")" "$SUB/prefill.dla" >/dev/null || exit 5
"$ADB" push "$(winpath "$DEC/model_shared.dla")" "$SUB/decode.dla" >/dev/null || exit 5
BSHA="$("$ADB" shell "sha256sum $SUB/shared.bin" | tr -d '\r' | cut -d' ' -f1)"
echo "[bin] $BIN sha256=$BSHA"

# ---- prefill once: its KV output seeds the cache of all 24 layers
for f in x mask cos14 sin14 cos2 sin2; do
  "$ADB" push "$(winpath "$PRE/golden/$f.bin")" "$SUB/live/$f.bin" >/dev/null || exit 5
done
T0=$(now)
R="$(run_graph prefill.dla "$PIN" 1)"
T1=$(now)
echo "[prefill] $R host_wall=$(( (T1-T0)/1000000 ))ms"
case "$R" in *rc=0*) : ;; *) echo PREFILL_RUN_FAIL; exit 7 ;; esac
for f in hidden k v; do
  "$ADB" pull "$SUB/live/out_$f.bin" "$(winpath "$PRE/out_$f.bin")" >/dev/null || exit 6
done

# ---- seed both caches from the prefill device output, write step 0 inputs
wslrun "/usr/bin/python3 tools/build_q06.py --seq-init --steps $STEPS --tab ops/q06_tab.json --op-prefill '$PRE' --op-decode '$DEC' --seqdir '$SEQ'" || exit 8

: > "$SEQ/timing.jsonl"
s=0
while [ "$s" -lt "$STEPS" ]; do
  T0=$(now)
  for f in x mask cos14 sin14 cos2 sin2 k_in v_in; do
    "$ADB" push "$(winpath "$SEQ/live/$f.bin")" "$SUB/live/$f.bin" >/dev/null || exit 5
  done
  T1=$(now)
  CNT=1
  if [ "$s" = "0" ]; then CNT=$REPEAT; fi
  R="$(run_graph decode.dla "$DIN" $CNT)"
  T2=$(now)
  case "$R" in *rc=0*) : ;; *) echo "DECODE_RUN_FAIL step $s ($R)"; exit 7 ;; esac
  for f in hidden k v; do
    "$ADB" pull "$SUB/live/out_$f.bin" "$(winpath "$SEQ/live/out_$f.bin")" >/dev/null || exit 6
  done
  T3=$(now)
  wslrun "/usr/bin/python3 tools/build_q06.py --seq-absorb $s --tab ops/q06_tab.json --seqdir '$SEQ'" || exit 8
  T4=$(now)
  echo "[step $s] $R push=$(( (T1-T0)/1000000 ))ms pull=$(( (T3-T2)/1000000 ))ms absorb=$(( (T4-T3)/1000000 ))ms"
  printf '{"step":%d,"device":"%s","host_push_ms":%d,"host_pull_ms":%d,"host_absorb_ms":%d,"n":%d}\n' "$s" "$R" $(( (T1-T0)/1000000 )) $(( (T3-T2)/1000000 )) $(( (T4-T3)/1000000 )) "$CNT" >> "$SEQ/timing.jsonl"
  s=$((s + 1))
done

# ---- noise floor: the LAST step's inputs, three independent Runs
: > "$SEQ/noise.txt"
i=1
while [ "$i" -le 3 ]; do
  run_graph decode.dla "$DIN" 1 >/dev/null
  "$ADB" shell "sha256sum /data/local/tmp/npuops/q06/live/out_hidden.bin" | tr -d '\r' | cut -d' ' -f1 >> "$SEQ/noise.txt"
  i=$((i + 1))
done
echo '[noise] out_hidden sha256, three Runs of the same inputs:'
cat "$SEQ/noise.txt"

wslrun "/usr/bin/python3 tools/build_q06.py --seq-report --seqdir '$SEQ'"
printf '{"bin_sha256":"%s","pre_dla":"%s","dec_dla":"%s","steps":%d,"repeat":%d}\n' "$BSHA" "$(sha256sum "$PRE/model_shared.dla" | cut -d' ' -f1)" "$(sha256sum "$DEC/model_shared.dla" | cut -d' ' -f1)" "$STEPS" "$REPEAT" > "$SEQ/e2e.json"
echo done
