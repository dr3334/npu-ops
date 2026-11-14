#!/usr/bin/env bash
# 47-opnames.sh — name the op indices that the PMU dump reports (read-only use of probe/).
set -uo pipefail
cd "/mnt/d/deepseek_ workspace/mtk/wt-bb07" || exit 2
/usr/bin/python3 probe/dump_tflite.py ops/_probe/bb07-prof/build3/full.tflite 2>&1 | head -60
