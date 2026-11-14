#!/usr/bin/env bash
set -uo pipefail
R="/mnt/d/deepseek_ workspace/mtk/wt-bb07"
cd "$R" || exit 2
echo "== execution_result.json structure =="
/usr/bin/python3 ops/_probe/bb07-prof/_peek.py ops/_probe/bb07-prof/board/execution_result.json 2>&1 | head -40
echo
echo "== dla_info.json structure =="
/usr/bin/python3 ops/_probe/bb07-prof/_peek.py ops/_probe/bb07-prof/board/dla_info.json 2>&1 | head -30
echo
echo "== raw head of execution_result.json =="
head -c 1200 ops/_probe/bb07-prof/board/execution_result.json
echo
