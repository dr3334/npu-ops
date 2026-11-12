#!/usr/bin/env bash
# Merge every feat/* worktree branch into the current branch, then run the full suite.
#   ./tools/merge_worktrees.sh [--no-verify]
# Expects: branches feat/elem feat/shapeops feat/actreduce feat/convpool (missing ones are skipped).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
VERIFY=1; [ "${1:-}" = "--no-verify" ] && VERIFY=0
BRANCHES="${BRANCHES:-feat/elem feat/shapeops feat/actreduce feat/convpool}"

git rev-parse --is-inside-work-tree >/dev/null || { echo "not a git repo"; exit 2; }
BASE=$(git rev-parse --abbrev-ref HEAD)
echo "== merging into $BASE"
for b in $BRANCHES; do
  if ! git rev-parse --verify "$b" >/dev/null 2>&1; then echo "  skip $b (missing)"; continue; fi
  if git merge-base --is-ancestor "$b" "$BASE"; then echo "  skip $b (already merged)"; continue; fi
  echo "  merging $b"
  if ! git merge --no-ff -m "merge $b (parallel operator work)" "$b"; then
    echo "  CONFLICT while merging $b — resolve manually, then rerun"; exit 3
  fi
done

if [ "$VERIFY" = "1" ]; then
  echo "== full regression"
  MSYS_NO_PATHCONV=1 ./tools/run_all.sh || { echo "REGRESSION FAILED"; exit 4; }
  echo "== healthcheck"
  MSYS_NO_PATHCONV=1 ./tools/healthcheck.sh || exit 5
fi
echo "== done: $(git log --oneline -1)"
