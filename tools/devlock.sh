#!/usr/bin/env bash
# devlock.sh — serialize board access across agents (AGENTS.md R5).
#
#   ./tools/devlock.sh run -- <command...>   # acquire, run, always release
#   ./tools/devlock.sh status                # who holds it
#   ./tools/devlock.sh wait [secs]           # block until free
#
# The lock is a directory (mkdir is atomic on every filesystem we use here, including
# the 9p/drvfs mounts under /mnt/d). Stale locks older than STALE_SECS are reclaimed.
set -uo pipefail

LOCK="${NPU_LOCK:-$HOME/.npuops-board.lock}"
STALE_SECS="${NPU_LOCK_STALE:-1800}"

now() { date +%s; }

holder() { cat "$LOCK/owner" 2>/dev/null; }

locked() { [ -d "$LOCK" ]; }

acquire() {
  local waited=0 limit="${1:-1800}"
  while ! mkdir "$LOCK" 2>/dev/null; do
    if locked; then
      local age=$(( $(now) - $(stat -c %Y "$LOCK" 2>/dev/null || echo $(now)) ))
      if [ "$age" -gt "$STALE_SECS" ]; then
        echo "[devlock] reclaiming stale lock (age ${age}s): $(holder)"
        rm -rf "$LOCK"
        continue
      fi
    fi
    if [ "$waited" -ge "$limit" ]; then
      echo "[devlock] TIMEOUT after ${limit}s; holder: $(holder)" >&2
      return 1
    fi
    sleep 2; waited=$((waited + 2))
  done
  { echo "pid=$$ host=$(hostname) ts=$(date -Iseconds) cmd=$*"; } > "$LOCK/owner"
  trap 'release' EXIT INT TERM
  return 0
}

release() {
  if [ -d "$LOCK" ] && grep -q "pid=$$ " "$LOCK/owner" 2>/dev/null; then
    rm -rf "$LOCK"
  fi
}

case "${1:-}" in
  run)
    shift; [ "${1:-}" = "--" ] && shift
    [ "$#" -ge 1 ] || { echo "usage: devlock.sh run -- <command...>" >&2; exit 2; }
    acquire 1800 || exit 3
    "$@"; rc=$?
    exit $rc
    ;;
  wait)
    acquire "${2:-1800}" && { echo "[devlock] acquired"; release; } || exit 3
    ;;
  status)
    if locked; then echo "LOCKED: $(holder)"; else echo "FREE"; fi
    ;;
  *)
    echo "usage: devlock.sh {run -- <cmd>|wait [secs]|status}" >&2; exit 2 ;;
esac
