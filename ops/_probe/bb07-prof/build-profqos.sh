#!/usr/bin/env bash
# build-profqos.sh — build the per-sub-command profiler harness three ways.
#
#   bash ops/_probe/bb07-prof/build-profqos.sh host           # x86-64, smoke test vs host runtime
#   bash ops/_probe/bb07-prof/build-profqos.sh device         # aarch64, links the SDK 7.3.15 runtime
#   bash ops/_probe/bb07-prof/build-profqos.sh device-vendor  # aarch64, links the board's own
#                                                             # /vendor/lib64 runtime (SONAME
#                                                             # libneuron_runtime.so -> resolved
#                                                             # by the device linker for free)
#
# Nothing lands inside the repo: build products go to /mnt/d/bb07-tmp.
#
# NOTE on the NDK: only the *Windows* prebuilt ships here. Its extensionless wrappers are bash
# scripts that exec clang.exe, so they work from WSL; the .cmd ones do not (they need cmd.exe).
# Windows clang also needs Windows-resolvable paths, hence realpath --relative-to: relative
# paths with forward slashes work on both sides.
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
SRC="ops/_probe/bb07-prof/profqos.c"
OUTDIR=/mnt/d/bb07-tmp
mkdir -p "$OUTDIR"
cd "$R" || exit 2

rel() { realpath --relative-to="$R" "$1"; }

mode="${1:-host}"
case "$mode" in
  host)
    CC="${CC:-gcc}"
    "$CC" -O2 -Wall -Wextra -std=gnu11 \
      -I "$(rel "$S/host/include")" -o "$(rel "$OUTDIR")/profqos_host" "$SRC" \
      -L "$(rel "$S/host/lib")" -l:libneuron_runtime.so -Wl,-rpath,"$S/host/lib" \
      || { echo BUILD_FAIL; exit 3; }
    echo "[ok] $OUTDIR/profqos_host  $(stat -c %s "$OUTDIR/profqos_host") B"
    ;;
  device|device-vendor)
    NDK="${ANDROID_NDK:-/mnt/d/liteRT/android-ndk-r27c}"
    BIN="$NDK/toolchains/llvm/prebuilt/windows-x86_64/bin"
    CC="$BIN/aarch64-linux-android31-clang"
    [ -f "$CC" ] || CC="$BIN/aarch64-linux-android31-clang.cmd"
    [ -f "$CC" ] || { echo "no aarch64 NDK clang under $NDK" >&2; exit 2; }
    if [ "$mode" = "device-vendor" ]; then
      LIB="vendor-runtime.so"; OUT="$OUTDIR/profqos_vendor"
      [ -f "$OUTDIR/$LIB" ] || { echo "pull it first: adb pull /vendor/lib64/mt6897/libneuron_runtime.so" >&2; exit 2; }
    else
      LIB="libneuron_runtime.so"; OUT="$OUTDIR/profqos"
    fi
    MSYS_NO_PATHCONV=1 "$CC" -O2 -Wall -Wextra -std=gnu11 \
      -I "$(rel "$S/mt6897/include")" -o "$(rel "$OUT")" "$SRC" \
      -L "$(rel "$OUTDIR")" -l:"$LIB" \
      || { echo BUILD_FAIL; exit 3; }
    echo "[ok] $OUT  $(stat -c %s "$OUT") B  $(file -b "$OUT")"
    echo "     NEEDED: $(readelf -d "$OUT" | grep -o 'lib[a-z_]*\.so[0-9.]*' | tr '\n' ' ')"
    ;;
  *) echo "usage: $0 {host|device|device-vendor}" >&2; exit 2 ;;
esac
