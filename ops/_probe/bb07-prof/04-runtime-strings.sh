#!/usr/bin/env bash
# 04-runtime-strings.sh — what option vocabulary does libneuron_runtime accept? (offline)
set -uo pipefail
S="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk"
R="$S/host/lib/libneuron_runtime.so"
echo "== strings: lines mentioning profil (unique, <=200) =="
strings -a "$R" | grep -i "profil" | sort -u | head -60
echo
echo "== strings: option-ish keys (hints / dump / trace / log) =="
strings -a "$R" | grep -iE "^(dump|trace|log|enable|disable|use)[A-Za-z_.]*$" | sort -u | head -80
echo
echo "== strings: anything with an '=' default or a colon-key shape near 'Option' =="
strings -a "$R" | grep -i "option" | sort -u | head -60
echo
echo "== neuronrt CLI binary: full string table of short opts =="
strings -a "$S/host/bin/neuronrt" | grep -nE "^-[a-zA-Z]$|Xruntime|only supported" | head -30
echo
echo "== neuronrt: strings mentioning runtime option names =="
strings -a "$S/host/bin/neuronrt" | sed -n '250,300p'
