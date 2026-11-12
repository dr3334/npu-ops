#!/usr/bin/env bash
# Run every op under ops/ through e2e.sh and print a one-line verdict each.
#   MSYS_NO_PATHCONV=1 ./tools/run_all.sh
#   HEALTHCHECK=0 ./tools/run_all.sh      # skip the pre/post device gates
#
# E-104: the suite is bracketed by the known-good program (healthcheck), so a run that
# pollutes the board (or starts from a polluted board) cannot be reported as green.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
export MSYS_NO_PATHCONV=1
HC="${HEALTHCHECK:-1}"
if [ "$HC" = "1" ]; then
  printf '%-28s ' "[pre] healthcheck"
  if ./tools/healthcheck.sh --quick >/tmp/hc_pre.log 2>&1; then echo "PASS"; else
    echo "FAIL"; tail -5 /tmp/hc_pre.log; exit 20; fi
fi
pass=0; fail=0; skip=0; xfail=0
printf "%-28s %-8s %-10s %s\n" "op" "result" "contract" "detail"
for d in ops/*/; do
  n=$(basename "$d")
  if [ ! -f "$d/model.json" ]; then printf "%-28s %-8s\n" "$n" "SKIP"; skip=$((skip+1)); continue; fi
  # 预期失败档：提交里记录的编译日志写着 EXIT=1（形状边界/不支持的反例）。
  # 这类目录不跑设备，只复核"它仍然编译失败"，避免把已知反例误报成回归。
  if grep -q "EXIT=1" "$d/model.compile.log" 2>/dev/null; then
    if ./tools/compile.sh "$d/model.tflite" "/tmp/xfail_$n.dla" >/dev/null 2>&1; then
      printf "%-28s %-8s %-10s %s\n" "$n" "REGRESS" "-" "expected compile failure, now compiles"
      fail=$((fail+1))
    else
      printf "%-28s %-8s %-10s %s\n" "$n" "XFAIL" "-" "compile fails as recorded"
      xfail=$((xfail+1))
    fi
    continue
  fi
  out=$(./tools/e2e.sh "$d" 2>&1)
  rc=$?
  contract=$(echo "$out" | grep -oE "int8_tol=[0-9]+" | head -1)
  [ -z "$contract" ] && contract=$(echo "$out" | grep -oE "atol=[0-9.e-]+" | head -1)
  detail=$(echo "$out" | grep -E "^(exact|max_abs)" | head -1 | tr -s ' ')
  if [ $rc -eq 0 ]; then pass=$((pass+1)); res=PASS; else fail=$((fail+1)); res=FAIL; fi
  printf "%-28s %-8s %-10s %s\n" "$n" "$res" "$contract" "$detail"
done
echo "---- PASS=$pass FAIL=$fail XFAIL=$xfail SKIP=$skip"
if [ "$HC" = "1" ]; then
  printf '%-28s ' "[post] healthcheck"
  if ./tools/healthcheck.sh --quick >/tmp/hc_post.log 2>&1; then echo "PASS"; else
    echo "FAIL (the suite may have polluted the board)"; tail -5 /tmp/hc_post.log; exit 21; fi
fi
[ $fail -eq 0 ]
