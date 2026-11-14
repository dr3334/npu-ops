#!/usr/bin/env bash
# 00-survey.sh — offline survey of the SDK (no board access).
set -uo pipefail
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
echo "== neuron_sdk top =="
ls "$S"
echo "== host =="
ls "$S/host"
echo "== host/bin =="
ls -la "$S/host/bin" 2>&1 | head -30
echo "== mt6897/bin =="
ls -la "$S/mt6897/bin" 2>&1 | head -30
echo "== sdk root =="
ls -la "/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807"
echo "== litert dirs =="
ls -la "$S"/litert_mediatek_runtime* 2>&1 | head -60
