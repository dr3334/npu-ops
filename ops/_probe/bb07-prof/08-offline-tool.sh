#!/usr/bin/env bash
# 08-offline-tool.sh — is there a simulator / cmodel that can run without hardware?
set -uo pipefail
SDK="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807"
S="$SDK/neuron_sdk"
echo "== SDK root / offline_tool =="
ls -R "$SDK/offline_tool" 2>&1 | head -60
echo
echo "== files named *cmodel* / *sim* in SDK =="
find "$SDK" -maxdepth 4 -iname "*cmodel*" -o -maxdepth 4 -iname "*simulat*" 2>/dev/null | head -20
echo
echo "== mtk_sdk_pkg =="
ls -R "$S/mtk_sdk_pkg" 2>&1 | head -40
