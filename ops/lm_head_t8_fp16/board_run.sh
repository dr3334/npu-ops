#!/usr/bin/env bash
# Board session for ticket 08 (LM head). MUST be run under the shared board lock:
#   export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
#   ./tools/devlock.sh run -- ./ops/lm_head_t8_fp16/board_run.sh
#
# What it produces (all inside ops/lm_head_t8_fp16/):
#   compare.txt          from the canonical tools/e2e.sh (run 1 and run 2)
#   out_r1.bin / out_r2.bin   two e2e runs of the same .dla + input  -> noise floor
#   out_r3.bin           third run from the timed block (bonus determinism point)
#   run.log              full neuronrt output (inference time, load time)
#   transfer.json        measured push / run / pull wall-clock times
set -uo pipefail
export MSYS_NO_PATHCONV=1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OP="ops/lm_head_t8_fp16"
NAME=lm_head_t8_fp16
DEV=/data/local/tmp/npuops
ADB="/d/adb/platform-tools/adb"
cd "$ROOT" || exit 2
winpath() { local p="$ROOT/$1"; ( cd "$(dirname "$p")" && printf '%s/%s' "$(pwd -W)" "$(basename "$p")" ); }
now_ms() { date +%s%3N; }
T0=$(now_ms)

echo "===== device capacity before ====="
"$ADB" shell 'df -h /data | tail -1; grep -E "MemTotal|MemAvailable" /proc/meminfo' | tr -d '\r'

echo "===== healthcheck --quick (before) ====="
./tools/healthcheck.sh --quick
echo "healthcheck_before_rc=$?"

echo "===== e2e run 1 ====="
./tools/e2e.sh "$OP"
RC1=$?
[ -f "$OP/out_device.bin" ] && cp "$OP/out_device.bin" "$OP/out_r1.bin"
echo "e2e_run1_rc=$RC1"

echo "===== e2e run 2 (same .dla, same input -> noise floor) ====="
./tools/e2e.sh "$OP"
RC2=$?
[ -f "$OP/out_device.bin" ] && cp "$OP/out_device.bin" "$OP/out_r2.bin"
echo "e2e_run2_rc=$RC2"

echo "===== timed push / run / pull ====="
"$ADB" shell "mkdir -p $DEV/dla" >/dev/null
t=$(now_ms)
"$ADB" push "$(winpath "$OP/model.dla")" "$DEV/dla/$NAME.dla" >/dev/null
PUSH_MS=$(( $(now_ms) - t ))
t=$(now_ms)
"$ADB" push "$(winpath "$OP/golden/input.bin")" "$DEV/dla/$NAME.in.bin" >/dev/null
PUSHIN_MS=$(( $(now_ms) - t ))
t=$(now_ms)
"$ADB" shell "cd $DEV && LD_LIBRARY_PATH=\$PWD/lib ./bin/neuronrt -m hw -a dla/$NAME.dla -i dla/$NAME.in.bin -o dla/$NAME.out.bin -c 3 2>&1" | tr -d '\r' | tee "$OP/run.log"
RUN_MS=$(( $(now_ms) - t ))
t=$(now_ms)
"$ADB" pull "$DEV/dla/$NAME.out.bin" "$(winpath "$OP/out_r3.bin")" >/dev/null
PULL_MS=$(( $(now_ms) - t ))
DLA_B=$(stat -c %s "$OP/model.dla")
OUT_B=$(stat -c %s "$OP/out_r3.bin" 2>/dev/null || echo 0)
IN_B=$(stat -c %s "$OP/golden/input.bin")
PUSH_MBPS=$(awk -v b=$DLA_B -v ms=$PUSH_MS 'BEGIN{ if (ms>0) printf "%.2f", b/1e6/(ms/1000); else print 0 }')
PULL_MBPS=$(awk -v b=$OUT_B -v ms=$PULL_MS 'BEGIN{ if (ms>0) printf "%.2f", b/1e6/(ms/1000); else print 0 }')
cat > "$OP/transfer.json" <<EOF
{"op":"$NAME","dla_bytes":$DLA_B,"input_bytes":$IN_B,"output_bytes":$OUT_B,
 "push_dla_ms":$PUSH_MS,"push_input_ms":$PUSHIN_MS,"run_ms":$RUN_MS,"pull_out_ms":$PULL_MS,
 "push_dla_MBps":$PUSH_MBPS,"pull_out_MBps":$PULL_MBPS,"ts":"$(date -Iseconds)"}
EOF
cat "$OP/transfer.json"

echo "===== device capacity after ====="
"$ADB" shell 'df -h /data | tail -1; grep -E "MemAvailable" /proc/meminfo' | tr -d '\r'

echo "===== healthcheck --quick (after) ====="
./tools/healthcheck.sh --quick
echo "healthcheck_after_rc=$?"
echo "total_session_s=$(( ( $(now_ms) - T0 ) / 1000 ))"
