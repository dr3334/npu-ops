#!/usr/bin/env bash
# Build the KV-resident runner (tools/kv1_runner.c) for the board.
#
#   ./tools/build_kv1.sh            # -> out/kv1/kv1_runner (aarch64, static-ish, dlopen only)
#
# Compilers: the NDK r26d inside WSL is the one this repo has used for probe/*.c; on Windows
# the same NDK works from Git Bash if ANDROID_NDK points at it.  The binary only needs libdl
# and libc, and it dlopen()s libneuron_runtime.so at runtime so it does not pin a runtime
# version at link time.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${NEURO_SDK:-$ROOT/../neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk}"
OUT="${1:-$ROOT/out/kv1/kv1_runner}"

if [ -n "${ANDROID_NDK:-}" ] && [ -x "${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang" ]; then
  CC="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang"
elif [ -x /home/dr/android-ndk-r26d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang ]; then
  CC=/home/dr/android-ndk-r26d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang
else
  echo "no aarch64 NDK clang found (set ANDROID_NDK)" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT")"
"$CC" -O2 -Wall -Wextra -Wno-unused-parameter -o "$OUT" "$ROOT/tools/kv1_runner.c" \
      -I"$SDK/host/include" -ldl
echo "built $OUT"

# The RuntimeV2 variant lives next to it: same job, buffers re-supplied per inference request.
OUT2="$(dirname "$OUT")/kv2_runner"
"$CC" -O2 -Wall -Wno-unused-parameter -o "$OUT2" "$ROOT/tools/kv2_runner.c" \
      -I"$SDK/host/include" -ldl
echo "built $OUT2"
