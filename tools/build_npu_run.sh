#!/usr/bin/env bash
# Cross-compile probe/npu_run.c to a static aarch64 Android binary with the NDK.
#
#   ./tools/build_npu_run.sh            # -> probe/npu_run
#
# The NDK may be a Windows install (toolchain .../prebuilt/windows-x86_64/bin/*.cmd) or a
# Linux one (WSL). Set ANDROID_NDK to override. Static linkage because the device has no
# libc++/bionic stubs for us and neuronrt's LD_LIBRARY_PATH is not ours to touch.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2

SRC="probe/npu_run.c"
OUT="${OUT:-probe/npu_run}"
API="${API:-31}"

find_cc() {
  local ndk candidates=()
  for ndk in "${ANDROID_NDK:-}" /d/liteRT/android-ndk-r27c /d/liteRT/android-ndk-r26d \
             /home/dr/android-ndk-r26d /opt/android-ndk; do
    [ -n "$ndk" ] || continue
    candidates+=(
      "$ndk/toolchains/llvm/prebuilt/windows-x86_64/bin/aarch64-linux-android${API}-clang.cmd"
      "$ndk/toolchains/llvm/prebuilt/windows-x86_64/bin/aarch64-linux-android${API}-clang"
      "$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${API}-clang"
      "$ndk/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android${API}-clang"
    )
  done
  for c in "${candidates[@]}"; do
    if [ -x "$c" ] || [ -f "$c" ]; then printf '%s' "$c"; return 0; fi
  done
  return 1
}

CC="$(find_cc)" || { echo "no aarch64 NDK clang found (set ANDROID_NDK)" >&2; exit 2; }
echo "[cc] $CC"
# Windows .cmd wrappers must be invoked through cmd.exe; MSYS_NO_PATHCONV keeps -o/-I paths intact.
export MSYS_NO_PATHCONV=1
"$CC" -O2 -static -Wall -Wextra -o "$OUT" "$SRC" || { echo "BUILD_FAIL" >&2; exit 3; }
echo "[ok] $OUT  $(stat -c %s "$OUT") B"
