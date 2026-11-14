#!/usr/bin/env bash
# Rebuild + compile the ticket-08 LM-head dimension probes (compile-only, no board).
# Run it from anywhere: the repo root comes from this script's own location, never from a
# hardcoded worktree path (AGENT-BRIEF 1).
set -uo pipefail
export PATH=/usr/bin:/bin:/usr/local/bin:$PATH
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" || exit 2
# ncc / python live in WSL, so pass a /mnt/<drive>/... path even when this was started from
# Git Bash (same derivation as tools/rebuild_qwen25.sh).
_win="$(cd "$ROOT" && pwd -W 2>/dev/null || true)"
if [ -n "${_win:-}" ] && [ "${ROOT#/mnt/}" = "$ROOT" ]; then
  _drv="$(printf %s "${_win:0:1}" | tr "A-Z" "a-z")"
  WSLROOT="/mnt/$_drv${_win:2}"
else
  WSLROOT="$ROOT"
fi
cd "$WSLROOT" || exit 2
/usr/bin/python3 ops/lm_head_t8_fp16/probe/make_probes.py || exit 3
echo "=================== compile ==================="
for d in ops/lm_head_t8_fp16/probe/*/; do
  n=$(basename "$d")
  ./tools/compile.sh "$d/model.tflite" "$d/model.dla" >/dev/null 2>&1
  rc=$?
  reason=$(grep -m1 -E "Dimension should be|unsupported|Bridging|Fail to process|Cannot find" "$d/model.compile.log" | sed 's/^ *//' || true)
  printf '%-16s rc=%d  %s\n' "$n" "$rc" "$reason"
done
