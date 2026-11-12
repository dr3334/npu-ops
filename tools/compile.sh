#!/usr/bin/env bash
# Compile one single-op TFLite model to .dla with ncc-tflite (NeuroPilot 7.0.8 / 7.3.15).
#
#   ./tools/compile.sh <model.tflite> <out.dla> [extra ncc flags...]
#
# Fixed policy (see reports/S0-env.md):
#   - target MT6897: --arch=mdla5.0 --num-mdla=2 --l1-size-kb=3072
#   - --disallow-bridge: any CPU fallback makes this fail instead of silently mixing
#   - --show-exec-plan / --show-memory-summary recorded into the log
#   - exit code, command line, ncc version, .dla sha256 recorded next to the model
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_SDK="$ROOT/../neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
SDK="${NEURO_SDK:-$DEFAULT_SDK}"
HOST="$SDK/host"
NCC="$HOST/bin/ncc-tflite"
ARCH="${ARCH:-mdla5.0}"
NUM_MDLA="${NUM_MDLA:-2}"
L1_KB="${L1_KB:-3072}"

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <model.tflite> <out.dla> [extra ncc flags...]" >&2
  exit 2
fi

MODEL="$1"; OUT="$2"; shift 2
LOG="${OUT%.dla}.compile.log"

if [ ! -x "$NCC" ]; then
  echo "ncc-tflite not found at $NCC" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUT")"

export LD_LIBRARY_PATH="$HOST/lib:${LD_LIBRARY_PATH:-}"

{
  echo "# cmd: ncc-tflite --arch=$ARCH --opt=3 --disallow-bridge --num-mdla=$NUM_MDLA --l1-size-kb=$L1_KB $* $MODEL -d $OUT"
  echo "# version: $("$NCC" --version 2>&1 | head -1)"
  echo "# date: $(date -Iseconds)"
  echo
} > "$LOG"

"$NCC" \
  --arch="$ARCH" \
  --opt=3 \
  --disallow-bridge \
  --num-mdla="$NUM_MDLA" \
  --l1-size-kb="$L1_KB" \
  --show-exec-plan \
  --show-memory-summary \
  "$@" \
  "$MODEL" \
  -d "$OUT" >> "$LOG" 2>&1
RC=$?

echo "EXIT=$RC" >> "$LOG"
if [ -f "$OUT" ]; then
  SIZE=$(stat -c %s "$OUT")
  SHA=$(sha256sum "$OUT" | cut -d' ' -f1)
  echo "SIZE=$SIZE" >> "$LOG"
  echo "SHA256=$SHA" >> "$LOG"
  printf "%-52s exit=%s  size=%-9s sha256=%s\n" "$(basename "$OUT")" "$RC" "$SIZE" "${SHA:0:16}"
else
  echo "SIZE=0" >> "$LOG"
  printf "%-52s exit=%s  (no .dla produced)\n" "$(basename "$OUT")" "$RC"
fi
exit $RC
