#!/usr/bin/env bash
# Cross-compile probe/iolog2.c into probe/libiolog2.so (aarch64 Android shared lib).
#
#   ./probe/build_iolog2.sh
#
# The shim is what records the vendor runtime's submission (ioctls + mmap + buffer dumps).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2
SRC=probe/iolog2.c
OUT="${OUT:-probe/libiolog2.so}"
API="${API:-31}"
CC=""
for ndk in "${ANDROID_NDK:-}" /d/liteRT/android-ndk-r27c /d/liteRT/android-ndk-r26d /opt/android-ndk; do
  [ -n "$ndk" ] || continue
  for c in "$ndk/toolchains/llvm/prebuilt/windows-x86_64/bin/aarch64-linux-android${API}-clang.cmd" \
           "$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${API}-clang"; do
    if [ -f "$c" ]; then CC="$c"; break 2; fi
  done
done
[ -n "$CC" ] || { echo "no aarch64 NDK clang found (set ANDROID_NDK)" >&2; exit 2; }
echo "[cc] $CC"
export MSYS_NO_PATHCONV=1
"$CC" -shared -fPIC -O2 -Wall -o "$OUT" "$SRC" -ldl || { echo BUILD_FAIL >&2; exit 3; }
echo "[ok] $OUT $(stat -c %s "$OUT") B"
