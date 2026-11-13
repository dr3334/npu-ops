#!/usr/bin/env bash
# tools/chain_two_layer.sh -- ticket 10: run the whole two-layer chain on the board, in ONE lock.
#
#   export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
#   ./tools/devlock.sh run -- ./tools/chain_two_layer.sh
#
# WHAT IT DOES (all board steps inside the single devlock session):
#   healthcheck --quick
#   l0 : e2e x2 (noise floor) -> timed reload round           (x[8,896]  -> h1[8,896])
#        cp ops/two_layer_l0_t8_fp16/out_device.bin -> ops/two_layer_l1_t8_fp16/golden/input.bin
#   l1 : e2e x2 -> timed reload round                          (board h1 -> h2[8,896])
#        cp ops/two_layer_l1_t8_fp16/out_device.bin -> ops/two_layer_logits_t8_fp16/golden/input.bin
#   logits : e2e x2 -> timed reload round                      (board h2 -> logits[8,151936])
#   healthcheck --quick
#
# The handoff is a plain file copy of the DEVICE output over the next graph's input: no CPU
# correction in between (NOTES 10.4 item 5).  e2e.sh only rebuilds model.tflite / golden when
# they are missing, so overwriting golden/input.bin is exactly the hook that makes the second
# graph consume what the first graph actually produced.
#
# Evidence kept per layer (renamed so a later e2e cannot overwrite it):
#   out_r1.bin / out_r2.bin / out_r3.bin   three device runs (r1 vs r2 = noise floor)
#   compare.txt                            canonical tools/compare.py verdict (last e2e run)
#   compare_r1.txt / compare_r2.txt        per-run copies of the same verdict
#   transfer_<layer>.json                  timed .dla push / input push / run / output pull
#   run.log                                full neuronrt output (inference time, load cost)
#   diagnostics_<layer>.txt                build_two_layer.py --diag: propagation vs own error
#
# No public tool is modified: e2e.sh / compare.py / devlock.sh / healthcheck.sh are only called.
set -uo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2
_win="$(cd "$ROOT" && pwd -W 2>/dev/null || echo /d/x)"
_drv="$(printf '%s' "${_win:0:1}" | tr 'A-Z' 'a-z')"
WSLROOT="${WSLROOT:-/mnt/$_drv${_win:2}}"
DEV=/data/local/tmp/npuops
ADB="${ADB:-/d/adb/platform-tools/adb}"
WSL="${WSL:-wsl.exe}"
: "${QWEN25_WEIGHTS_DIR:=/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen}"
export QWEN25_WEIGHTS_DIR

L0=ops/two_layer_l0_t8_fp16
L1=ops/two_layer_l1_t8_fp16
LG=ops/two_layer_logits_t8_fp16

now_ms() { date +%s%3N; }
winpath() { local p="$ROOT/$1"; ( cd "$(dirname "$p")" && printf '%s/%s' "$(pwd -W)" "$(basename "$p")" ); }
wslrun()  { "$WSL" -e bash -lc "cd '$WSLROOT' && export QWEN25_WEIGHTS_DIR='$QWEN25_WEIGHTS_DIR'; $*"; }

# ---- one timed reload round: push .dla + input, run, pull ------------------------
# This is the "reload the whole static weight block again" cost that a 24-layer chain
# would pay per layer (each layer is its own .dla).
timed_round() {
  local op="$1" tag="$2" name
  name="$(basename "$op")"
  local t push_ms pushin_ms run_ms pull_ms dla_b in_b out_b inf_ms load_ms
  "$ADB" shell "mkdir -p $DEV/dla" >/dev/null
  t=$(now_ms); "$ADB" push "$(winpath "$op/model.dla")" "$DEV/dla/$name.dla" >/dev/null; push_ms=$(( $(now_ms) - t ))
  t=$(now_ms); "$ADB" push "$(winpath "$op/golden/input.bin")" "$DEV/dla/$name.in.bin" >/dev/null; pushin_ms=$(( $(now_ms) - t ))
  t=$(now_ms)
  "$ADB" shell "cd $DEV && LD_LIBRARY_PATH=\$PWD/lib ./bin/neuronrt -m hw -a dla/$name.dla -i dla/$name.in.bin -o dla/$name.out.bin -c 3 2>&1" \
    | tr -d '\r' | tee "$op/run.log"
  run_ms=$(( $(now_ms) - t ))
  t=$(now_ms); "$ADB" pull "$DEV/dla/$name.out.bin" "$(winpath "$op")/out_r3.bin" >/dev/null; pull_ms=$(( $(now_ms) - t ))
  dla_b=$(stat -c %s "$op/model.dla")
  in_b=$(stat -c %s "$op/golden/input.bin")
  out_b=$(stat -c %s "$op/out_r3.bin" 2>/dev/null || echo 0)
  inf_ms=$(grep -oE 'Total inference time = [0-9.]+' "$op/run.log" | grep -oE '[0-9.]+' | head -1)
  inf_ms="${inf_ms:-0}"
  load_ms=$(awk -v r="$run_ms" -v i="$inf_ms" 'BEGIN{ d=r-i; if (d<0) d=0; printf "%.0f", d }')
  cat > "$op/transfer_$tag.json" <<EOF
{"op":"$name","dla_bytes":$dla_b,"input_bytes":$in_b,"output_bytes":$out_b,
 "push_dla_ms":$push_ms,"push_input_ms":$pushin_ms,"run_ms":$run_ms,"pull_out_ms":$pull_ms,
 "inference_total_ms":$inf_ms,"load_and_startup_ms_est":$load_ms,
 "push_dla_MBps":$(awk -v b=$dla_b -v ms=$push_ms 'BEGIN{ if(ms>0) printf "%.2f", b/1e6/(ms/1000); else print 0 }'),
 "pull_out_MBps":$(awk -v b=$out_b -v ms=$pull_ms 'BEGIN{ if(ms>0) printf "%.2f", b/1e6/(ms/1000); else print 0 }'),
 "ts":"$(date -Iseconds)"}
EOF
  echo "[$tag] push_dla=${push_ms}ms push_in=${pushin_ms}ms run=${run_ms}ms (inference ${inf_ms}ms -> load+startup ~${load_ms}ms) pull=${pull_ms}ms size=${dla_b}B"
}

# ---- one layer: two e2e runs (noise floor) + one timed reload round + diagnostics --
run_layer() {
  local op="$1" tag="$2" graph="$3" rc1 rc2
  echo "===== $tag : e2e run 1 ====="
  ./tools/e2e.sh "$op" > "/tmp/e2e_${tag}_r1.log" 2>&1; rc1=$?
  [ -f "$op/out_device.bin" ] && cp "$op/out_device.bin" "$op/out_r1.bin"
  [ -f "$op/compare.txt" ] && cp "$op/compare.txt" "$op/compare_r1.txt"
  echo "===== $tag : e2e run 2 (same .dla, same input -> noise floor) ====="
  ./tools/e2e.sh "$op" > "/tmp/e2e_${tag}_r2.log" 2>&1; rc2=$?
  [ -f "$op/out_device.bin" ] && cp "$op/out_device.bin" "$op/out_r2.bin"
  [ -f "$op/compare.txt" ] && cp "$op/compare.txt" "$op/compare_r2.txt"
  echo "e2e rc: run1=$rc1 run2=$rc2"
  sed -n '1,12p' "$op/compare.txt"
  if [ "$rc1" = "0" ] && [ "$rc2" = "0" ]; then echo "$tag e2e: PASS"; else echo "$tag e2e: FAIL"; fi
  echo "===== $tag : timed reload round ====="
  timed_round "$op" "$tag"
  echo "===== $tag : diagnostics (propagation vs own error) ====="
  wslrun "/usr/bin/python3 tools/build_two_layer.py --graph $graph --diag" | tee "$op/diagnostics_$tag.txt"
  LAST_RC=$(( rc1 != 0 || rc2 != 0 ))
}

handoff() {  # $1 = from op dir, $2 = to op dir   (device output becomes the next input)
  local from="$1" to="$2"
  cp "$from/out_device.bin" "$to/golden/input.bin"
  if cmp -s "$from/out_device.bin" "$to/golden/input.bin"; then
    echo "[handoff] $from/out_device.bin -> $to/golden/input.bin  (byte-identical, $(stat -c %s "$to/golden/input.bin") B, sha256 $(sha256sum "$to/golden/input.bin" | cut -c1-16))"
  else
    echo "[handoff] FAILED: copy is not byte-identical"; return 1
  fi
}

FAILED=0
LAST_RC=0
T0=$(now_ms)
echo "===== device capacity before ====="
"$ADB" shell 'df -h /data | tail -1; grep -E "MemTotal|MemAvailable" /proc/meminfo' | tr -d '\r'
echo "===== healthcheck --quick (before) ====="
./tools/healthcheck.sh --quick; HC1=$?
echo "healthcheck_before_rc=$HC1"

run_layer "$L0" l0 l0; FAILED=$(( FAILED + LAST_RC ))
handoff "$L0" "$L1"
run_layer "$L1" l1 l1; FAILED=$(( FAILED + LAST_RC ))
handoff "$L1" "$LG"
run_layer "$LG" logits logits; FAILED=$(( FAILED + LAST_RC ))

echo "===== device capacity after ====="
"$ADB" shell 'df -h /data | tail -1; grep -E "MemAvailable" /proc/meminfo' | tr -d '\r'
echo "===== healthcheck --quick (after) ====="
./tools/healthcheck.sh --quick; HC2=$?
echo "healthcheck_after_rc=$HC2"
echo "session_s=$(( ( $(now_ms) - T0 ) / 1000 ))  failed_layers=$FAILED  healthcheck=$HC1/$HC2"
if [ "$FAILED" = "0" ] && [ "$HC1" = "0" ] && [ "$HC2" = "0" ]; then exit 0; else exit 1; fi
