#!/usr/bin/env bash
# KV cache resident on the device (branch impl/kv-resident).
#
#   ./tools/e2e_kv1.sh [--steps N]        # run inside ./tools/devlock.sh run --
#
# The SAME decode sequence is executed twice, seeded from the same prefill output:
#   BASE   neuronrt CLI, KV cache owned by the host  (ticket 06 path, unchanged)
#   KVRES  tools/kv1_runner, KV cache in one persistent dma-buf, bound once with
#          NeuronRuntime_setInput(..., BufferAttribute{ionFd}) and updated in place.
# Acceptance: byte-identical outputs at every phase, and KVRES writes only the new KV
# row (48 slabs x 128 B = 6 KB per tensor) instead of re-feeding the whole 24 MiB cache.
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
KV=$DEV/kv1
ADB=${ADB:-/d/adb/platform-tools/adb}
WSL=wsl.exe
BIN=ops/q06_shared.bin
PRE=ops/q06_prefill_t16_24l
DEC=ops/q06_decode_t8_c2048_24l
SEQ=ops/kv1_seq_base
SEQKV=ops/kv1_seq_kv
PLAN=ops/kv1_plan
BASE=ops/kv1_base_out
KVOUT=ops/kv1_kv_out
RSEQ=ops/kv1_run
WD='/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen'
export MSYS_NO_PATHCONV=1

winpath() { local p="$ROOT/$1"; ( cd "$(dirname "$p")" && printf '%s/%s' "$(pwd -W)" "$(basename "$p")" ); }
wslrun() { "$WSL" -e bash -lc "cd '$WSLROOT' && export QWEN25_WEIGHTS_DIR='$WD' && $*"; }
now() { date +%s%N; }
snap() { local d=$1; mkdir -p "$PLAN/step$d"; for f in x mask cos14 sin14 cos2 sin2; do cp "$SEQ/live/$f.bin" "$PLAN/step$d/$f.bin"; done; }
cd "$ROOT" || exit 2

PIN='-i q06/live/x.bin -i q06/live/mask.bin -i q06/live/cos14.bin -i q06/live/sin14.bin -i q06/live/cos2.bin -i q06/live/sin2.bin'
DIN="$PIN -i q06/live/k_in.bin -i q06/live/v_in.bin"
OUT='-o q06/live/out_hidden.bin -o q06/live/out_k.bin -o q06/live/out_v.bin'

cli_run() {
  "$ADB" shell "cd $DEV && a=\$(date +%s%N); LD_LIBRARY_PATH=\$PWD/lib ./bin/neuronrt -m hw -a q06/$1 $2 -i q06/shared.bin $OUT -c $3 >/dev/null 2>&1; rc=\$?; b=\$(date +%s%N); echo rc=\$rc ms=\$(( (b-a)/1000000 ))" | tr -d '\r'
}

rm -rf "$SEQ" "$SEQKV" "$PLAN" "$BASE" "$KVOUT" "$RSEQ"
mkdir -p "$PLAN" "$BASE" "$KVOUT" "$RSEQ"

"$ADB" shell "mkdir -p $SUB/live $KV/plan $KV/out8" >/dev/null
"$ADB" push "$(winpath "$BIN")" "$SUB/shared.bin" >/dev/null || exit 5
"$ADB" push "$(winpath "$PRE/model_shared.dla")" "$SUB/prefill.dla" >/dev/null || exit 5
"$ADB" push "$(winpath "$DEC/model_shared.dla")" "$SUB/decode.dla" >/dev/null || exit 5
"$ADB" push "$(winpath out/kv1/kv1_runner)" "$DEV/bin/kv1_runner" >/dev/null || exit 5
"$ADB" shell "chmod 755 $DEV/bin/kv1_runner" >/dev/null
echo "[bin] $BIN sha256=$("$ADB" shell "sha256sum $SUB/shared.bin" | tr -d '\r' | cut -d' ' -f1)"

echo '--- phase 0: prefill once (CLI) -> seeds the cache of all 24 layers ---'
for f in x mask cos14 sin14 cos2 sin2; do
  "$ADB" push "$(winpath "$PRE/golden/$f.bin")" "$SUB/live/$f.bin" >/dev/null || exit 5
done
R="$(cli_run prefill.dla "$PIN" 1)"
case "$R" in *rc=0*) : ;; *) echo "PREFILL_RUN_FAIL $R"; exit 7 ;; esac
echo "[prefill] $R"
for f in hidden k v; do "$ADB" pull "$SUB/live/out_$f.bin" "$(winpath "$PRE/out_$f.bin")" >/dev/null || exit 6; done

echo '--- phase 1: seed both caches from that prefill output, write phase-0 inputs ---'
wslrun "/usr/bin/python3 tools/build_q06.py --seq-init --steps $STEPS --tab ops/q06_tab.json --op-prefill '$PRE' --op-decode '$DEC' --seqdir '$SEQ'" || exit 8
cp "$SEQ/live/k_in.bin" ops/kv1_seed_k.bin
cp "$SEQ/live/v_in.bin" ops/kv1_seed_v.bin
snap 00
cp -r "$SEQ" "$SEQKV"

echo "--- phase 2: BASE = neuronrt CLI, host-owned cache, $STEPS phases ---"
: > "$RSEQ/cli.txt"
s=0
while [ "$s" -lt "$STEPS" ]; do
  nn=$(printf '%02d' "$s")
  T0=$(now)
  for f in x mask cos14 sin14 cos2 sin2 k_in v_in; do
    "$ADB" push "$(winpath "$SEQ/live/$f.bin")" "$SUB/live/$f.bin" >/dev/null || exit 5
  done
  T1=$(now)
  CNT=1
  if [ "$s" = 0 ]; then CNT=20; fi
  R="$(cli_run decode.dla "$DIN" $CNT)"
  case "$R" in *rc=0*) : ;; *) echo "DECODE_RUN_FAIL step $s ($R)"; exit 7 ;; esac
  for f in hidden k v; do "$ADB" pull "$SUB/live/out_$f.bin" "$SEQ/live/out_$f.bin" >/dev/null || exit 6; done
  mkdir -p "$BASE/step$nn"
  for f in hidden k v; do cp "$SEQ/live/out_$f.bin" "$BASE/step$nn/out_$f.bin"; done
  T2=$(now)
  wslrun "/usr/bin/python3 tools/build_q06.py --seq-absorb $s --tab ops/q06_tab.json --seqdir '$SEQ'" || exit 8
  printf '[base step %s] %s  host_push=%dms run+pull=%dms\n' "$s" "$R" $(( (T1-T0)/1000000 )) $(( (T2-T1)/1000000 )) | tee -a "$RSEQ/cli.txt"
  s=$((s+1))
  if [ "$s" -lt "$STEPS" ]; then snap "$(printf '%02d' "$s")"; fi
done

echo '--- phase 3: KVRES = one process, KV in a dma-buf, $STEPS invokes ---'
"$ADB" push "$(winpath ops/kv1_seed_k.bin)" "$KV/seed_k.bin" >/dev/null || exit 5
"$ADB" push "$(winpath ops/kv1_seed_v.bin)" "$KV/seed_v.bin" >/dev/null || exit 5
s=0
while [ "$s" -lt "$STEPS" ]; do
  nn=$(printf '%02d' "$s")
  "$ADB" shell "mkdir -p $KV/plan/step$nn" >/dev/null
  for f in x mask cos14 sin14 cos2 sin2; do
    "$ADB" push "$(winpath "$PLAN/step$nn/$f.bin")" "$KV/plan/step$nn/$f.bin" >/dev/null || exit 5
  done
  s=$((s+1))
done
KVIN='-i kv1/plan/step00/x.bin -i kv1/plan/step00/mask.bin -i kv1/plan/step00/cos14.bin -i kv1/plan/step00/sin14.bin -i kv1/plan/step00/cos2.bin -i kv1/plan/step00/sin2.bin -i kv1/seed_k.bin -i kv1/seed_v.bin -i q06/shared.bin'
"$ADB" shell "cd $DEV && ./bin/kv1_runner -a q06/decode.dla $KVIN -o kv1/out8/o.bin -o kv1/out8/ok.bin -o kv1/out8/ov.bin --steps $STEPS --m0 16 --stepdir kv1/plan --outdir kv1/out8 --json kv1/out8/run.json --verbose 2>&1 | grep -E '^\[kv|^\[time|^\[io|^\[setInput  6|^\[setInput  7' " | tr -d '\r' | tee "$RSEQ/kvres.txt"
s=0
while [ "$s" -lt "$STEPS" ]; do
  nn=$(printf '%02d' "$s")
  mkdir -p "$KVOUT/step$nn"
  for f in hidden k_out v_out; do
    "$ADB" pull "$KV/out8/step$nn/$f.bin" "$KVOUT/step$nn/$f.bin" >/dev/null || exit 6
  done
  s=$((s+1))
done

echo "--- phase 4: byte comparison BASE vs KVRES ($STEPS phases x 3 tensors) ---"
: > "$RSEQ/cmp.txt"
ident=0; differ=0
s=0
while [ "$s" -lt "$STEPS" ]; do
  nn=$(printf '%02d' "$s")
  line="step $nn:"
  for pair in hidden:hidden k:k_out v:v_out; do
    a="${pair%%:*}"; b="${pair##*:}"
    if cmp -s "$BASE/step$nn/out_$a.bin" "$KVOUT/step$nn/$b.bin"; then
      line="$line $a=IDENTICAL"; ident=$((ident+1))
    else
      line="$line $a=DIFFERS($(cmp -l "$BASE/step$nn/out_$a.bin" "$KVOUT/step$nn/$b.bin" 2>/dev/null | wc -l))"
      differ=$((differ+1))
    fi
  done
  echo "$line" | tee -a "$RSEQ/cmp.txt"
  s=$((s+1))
done
echo "[cmp] identical=$ident differ=$differ"

echo '--- phase 5: score KVRES outputs against the same numpy golden as BASE ---'
s=0
while [ "$s" -lt "$STEPS" ]; do
  nn=$(printf '%02d' "$s")
  for f in hidden k v; do
    case "$f" in
      hidden) cp "$KVOUT/step$nn/hidden.bin" "$SEQKV/live/out_hidden.bin" ;;
      k)      cp "$KVOUT/step$nn/k_out.bin"  "$SEQKV/live/out_k.bin" ;;
      v)      cp "$KVOUT/step$nn/v_out.bin"  "$SEQKV/live/out_v.bin" ;;
    esac
  done
  wslrun "/usr/bin/python3 tools/build_q06.py --seq-absorb $s --tab ops/q06_tab.json --seqdir '$SEQKV'" || exit 8
  s=$((s+1))
done
echo '--- report: BASE (neuronrt, host KV) ---'
wslrun "/usr/bin/python3 tools/build_q06.py --seq-report --seqdir '$SEQ'"
echo '--- report: KVRES (kv1_runner, dma-buf KV) ---'
wslrun "/usr/bin/python3 tools/build_q06.py --seq-report --seqdir '$SEQKV'"
echo done
