#!/usr/bin/env bash
# Regenerate every op's model.tflite from its builder (tflite/dla are gitignored artifacts).
#   ./tools/rebuild_ops.sh [--list-only]
# Builders differ in --out convention:
#   file-path builders : build_ops, build_add, build_dynb, build_actreduce   (--out <dir>/model.tflite)
#   dir builders       : build_shapeops                                     (--out <dir>)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
_win="$(cd "$ROOT" && pwd -W 2>/dev/null || echo /d/x)"
_drv="$(printf %s "${_win:0:1}" | tr A-Z a-z)"
WSLROOT="${WSLROOT:-/mnt/$_drv${_win:2}}"
WL() { wsl.exe -e bash -lc "cd '$WSLROOT' && $*"; }
LIST_ONLY=0; [ "${1:-}" = "--list-only" ] && LIST_ONLY=1

FILE_BUILDERS="build_ops build_add build_dynb build_actreduce build_convpool build_rmsnorm build_silu build_bmm"
DIR_BUILDERS="build_shapeops build_rope"

total=0; made=0; failed=0; exists=0
run_one() { # builder, spec, kind
  local b=$1 s=$2 kind=$3
  [ -d "ops/$s" ] || return 0
  total=$((total+1))
  [ "$LIST_ONLY" = "1" ] && { echo "  $b -> $s"; return 0; }
  if [ -f "ops/$s/model.tflite" ]; then exists=$((exists+1)); return 0; fi
  local cmd
  if [ "$kind" = "dir" ]; then cmd="/usr/bin/python3 tools/$b.py $s --out ops/$s";
  else cmd="/usr/bin/python3 tools/$b.py $s --out ops/$s/model.tflite"; fi
  if WL "$cmd" >/dev/null 2>&1; then made=$((made+1)); else echo "  BUILD FAIL $b -> $s"; failed=$((failed+1)); fi
}

for b in $FILE_BUILDERS; do
  [ -f "tools/$b.py" ] || continue
  for s in $(WL "/usr/bin/python3 tools/$b.py --list 2>/dev/null"); do run_one "$b" "$s" file; done
done
for b in $DIR_BUILDERS; do
  [ -f "tools/$b.py" ] || continue
  for s in $(WL "/usr/bin/python3 tools/$b.py --list 2>/dev/null"); do run_one "$b" "$s" dir; done
done
[ "$LIST_ONLY" = "1" ] || echo "---- built=$total regenerated=$made already=$exists failed=$failed"

# --- fc_<m>x<k>x<n>_<dtype>: shape comes from the directory name, not from any builder's SPECS list ---
# 这些 op 是用 build_ops.py 的显式参数建的，rebuild 的 SPECS 循环覆盖不到；删掉 model.tflite 后
# 它们会静默保持「无 tflite + 旧 golden」，一旦 golden 规则变了就会悄悄不一致（2026-11-13 踩过）。
for d in ops/fc_*/; do
  n=$(basename "$d"); [ -f "$d/model.json" ] || continue; [ -f "$d/model.tflite" ] && continue
  case "$n" in
    fc_dynb_*|fc_bmm_*|fc_const_*) continue ;;   # 由各自的 builder 负责
  esac
  shp=${n#fc_}; m=${shp%%x*}; rest=${shp#*x}; k=${rest%%x*}; nn=${rest#*x}
  case "$nn" in *_int8) dtype=int8 ;; *_fp16) dtype=fp16 ;; *_fp32) dtype=fp32 ;; *) continue ;; esac
  nn=${nn%_*}
  case "$m$k$nn" in *[!0-9]*) continue ;; esac
  if WL "/usr/bin/python3 tools/build_ops.py fc --m $m --k $k --n $nn --dtype $dtype --out ops/$n/model.tflite" >/dev/null 2>&1; then
    echo "  rebuilt(fc-name) $n"; else echo "  BUILD FAIL fc-name -> $n"; fi
done
# --- build_binop.py: no spec list; regenerate from each op's model.json (op + shape) ---
for d in ops/*/; do
  n=$(basename "$d"); [ -f "$d/model.json" ] || continue; [ -f "$d/model.tflite" ] && continue
  op=$(grep -m1 '"op"' "$d/model.json" | sed 's/.*: *"//;s/".*//')
  case "$op" in MUL|SUB|MAXIMUM|MINIMUM) : ;; *) continue ;; esac
  m=$(grep -m1 '"M"' "$d/model.json" | tr -dc '0-9'); k=$(grep -m1 '"K"' "$d/model.json" | tr -dc '0-9')
  [ -n "$m" ] && [ -n "$k" ] || continue
  if WL "/usr/bin/python3 tools/build_binop.py --op $op --m $m --k $k --out ops/$n" >/dev/null 2>&1; then
    echo "  rebuilt(binop) $n"
  else echo "  BUILD FAIL binop -> $n"; fi
done