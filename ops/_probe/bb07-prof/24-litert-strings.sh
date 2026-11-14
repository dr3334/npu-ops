#!/usr/bin/env bash
# 24-litert-strings.sh — unpack the LiteRT zips outside the repo and look for profiling hooks.
set -uo pipefail
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
T=/mnt/d/bb07-tmp/litert
rm -rf "$T"; mkdir -p "$T"
for z in "$S"/litert_npu_*.zip; do
  d="$T/$(basename "$z" .zip)"
  mkdir -p "$d"; unzip -q "$z" -d "$d"
done
echo "== every .so / binary found =="
find "$T" -type f \( -name "*.so" -o -name "*.bin" -o -perm -u+x \) | sed "s|$T/||" | sort
echo
echo "== any file whose NAME suggests bench/prof/trace =="
find "$T" -type f | grep -iE "bench|prof|trace|perf" | head -20
echo
echo "== strings in libLiteRtDispatch_MediaTek.so: MTKNN / profil =="
D="$S/litert_mediatek_runtime/mediatek_runtime/src/main/jni/arm64-v8a/libLiteRtDispatch_MediaTek.so"
strings -a "$D" | grep -iE "MTKNN|profil|MVPU Per-OP|neuron" | sort -u | head -40
echo
echo "== strings in the JIT compiler plugin =="
C="$S/litert_mediatek_runtime_jit/mediatek_runtime/src/main/jni/arm64-v8a/libLiteRtCompilerPlugin_MediaTek.so"
strings -a "$C" | grep -iE "MTKNN|profil|PER_OP" | sort -u | head -20
echo
echo "== strings.xml (runtime_strings) =="
for f in $(find "$T" -name strings.xml); do echo "--- $f"; cat "$f"; done
