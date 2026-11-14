#!/usr/bin/env bash
# 02-api-hunt.sh — find the runtime's own profiling API surface (offline).
set -uo pipefail
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
echo "== host/lib =="
ls -la "$S/host/lib"
echo "== host/include tree =="
find "$S/host/include" -type f | head -60
echo
echo "== NeuronRuntime_* exported, by lib =="
for f in "$S/host/lib"/*.so*; do
  n=$(nm -D --defined-only "$f" 2>/dev/null | grep -c 'NeuronRuntime_')
  [ "$n" = "0" ] && continue
  echo "--- $f ($n) ---"
  nm -D --defined-only "$f" 2>/dev/null | grep 'NeuronRuntime_' | awk '{print $3}' | sort -u
done
echo
echo "== headers mentioning profil =="
grep -ril "profil" "$S/host/include" 2>/dev/null | head -20
