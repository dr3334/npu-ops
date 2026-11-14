#!/usr/bin/env bash
# Pack one .dla into a .dlb at a given compression level (NeuroPilot 7.0.8 / dla-packer).
#
#   ./tools/pack.sh <level 0..9> <in.dla> <out.dlb>
#
# Same policy as compile.sh: one entry point, every run records cmd/version/rc/size/sha256
# into <out>.pack.log next to the artifact. After packing it also runs readdlb on the result
# (the official reader) and appends its output -- readdlb may exit non-zero, so its exit code
# is recorded but never fails this script.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_SDK="$ROOT/../neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
SDK="${NEURO_SDK:-$DEFAULT_SDK}"
HOST="$SDK/host"
PACKER="$HOST/bin/dla-packer"
READDLB="$HOST/bin/readdlb"

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <level> <in.dla> <out.dlb>" >&2
  exit 2
fi

LEVEL="$1"; DLA="$2"; OUT="$3"
LOG="${OUT%.dlb}.pack.log"

if [ ! -x "$PACKER" ]; then
  echo "dla-packer not found at $PACKER" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUT")"
export LD_LIBRARY_PATH="$HOST/lib:${LD_LIBRARY_PATH:-}"

{
  echo "# cmd: dla-packer -l $LEVEL -o $OUT $DLA"
  echo "# version: $("$PACKER" --version 2>&1 | head -1)"
  echo "# date: $(date -Iseconds)"
} > "$LOG"

"$PACKER" -l "$LEVEL" -o "$OUT" "$DLA" >> "$LOG" 2>&1
RC=$?
echo "EXIT=$RC" >> "$LOG"
if [ -f "$OUT" ]; then
  SIZE=$(stat -c %s "$OUT")
  echo "SIZE=$SIZE" >> "$LOG"
  echo "SHA256=$(sha256sum "$OUT" | cut -d' ' -f1)" >> "$LOG"
  echo "=== readdlb $OUT" >> "$LOG"
  "$READDLB" "$OUT" >> "$LOG" 2>&1
  echo "READDLB_EXIT=$?" >> "$LOG"
  printf "%-46s level=%s exit=%s  size=%s\n" "$(basename "$OUT")" "$LEVEL" "$RC" "$SIZE"
else
  echo "SIZE=0" >> "$LOG"
  printf "%-46s level=%s exit=%s  (no .dlb produced)\n" "$(basename "$OUT")" "$LEVEL" "$RC"
fi
exit $RC
