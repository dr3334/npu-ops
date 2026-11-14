#!/usr/bin/env bash
# 24b-litert-strings.sh — profiling hooks inside the MediaTek LiteRT dispatch/plugin libs.
set -uo pipefail
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
D="$S/litert_mediatek_runtime/mediatek_runtime/src/main/jni/arm64-v8a/libLiteRtDispatch_MediaTek.so"
C="$S/litert_mediatek_runtime_jit/mediatek_runtime/src/main/jni/arm64-v8a/libLiteRtCompilerPlugin_MediaTek.so"
echo "== dispatch .so: MTKNN / profil / MVPU =="
strings -a "$D" | grep -iE "MTKNN|profil|MVPU Per-OP|PER_OP" | sort -u | head -30
echo "(count: $(strings -a "$D" | grep -icE 'MTKNN|profil|PER_OP'))"
echo
echo "== dispatch .so: does it link the neuron runtime? =="
strings -a "$D" | grep -iE "libneuron|libmvpu|dlopen|libMdlaCmodel" | sort -u | head -15
echo
echo "== compiler plugin .so: profil =="
strings -a "$C" | grep -iE "MTKNN|profil|PER_OP|profiling-level" | sort -u | head -20
echo
echo "== env-var-shaped strings in dispatch =="
strings -a "$D" | grep -E "^[A-Z][A-Z0-9_]{6,}$" | sort -u | head -40
