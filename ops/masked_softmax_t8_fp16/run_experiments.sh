#!/usr/bin/env bash
# Board experiments for ops/masked_softmax_t8_fp16 (ticket 05).
#
#   ./ops/masked_softmax_t8_fp16/run_experiments.sh <case>
#   case = health | mag2 | zero | l1 | noise | primary | restore
#
# Every board touch goes through tools/devlock.sh (6 agents share one device). Only the
# e2e call itself is inside the lock; building/compiling happened outside it beforehand
# (tools/build_masked_softmax.py + tools/compile.sh).
#
# The .dla is built ONCE from a graph whose mask is the **second runtime input**, so all
# the mask variants below reuse the same binary - the only thing that changes is
# golden/input1.bin (and which golden the compare step uses).
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 2
OP=ops/masked_softmax_t8_fp16
G="$OP/golden"
export NPU_LOCK="${NPU_LOCK:-/d/deepseek_ workspace/mtk/.npu-board.lock}"

restore() {
  cp "$G/_p_input.bin"    "$G/input.bin"
  cp "$G/_p_input1.bin"   "$G/input1.bin"
  cp "$G/_p_expected.bin" "$G/expected.bin"
}

save() {   # save <tag>
  cp "$OP/out_device.bin" "$OP/out_$1.bin"
  cp "$OP/compare.txt"    "$OP/compare_$1.txt"
  echo "[saved] out_$1.bin compare_$1.txt"
}

e2e() { ./tools/devlock.sh run -- ./tools/e2e.sh "$OP"; }

case "${1:-}" in
  health)  ./tools/devlock.sh run -- ./tools/healthcheck.sh --quick ;;
  restore) restore; echo "[restored] primary inputs" ;;
  mag2)    cp "$G/input1_mag2.bin" "$G/input1.bin"; cp "$G/expected_mag2.bin" "$G/expected.bin"
           e2e; save mag2; restore ;;
  zero)    cp "$G/input1_zero.bin" "$G/input1.bin"; cp "$G/expected_zero.bin" "$G/expected.bin"
           e2e; save zero; restore ;;
  l1)      cp "$G/input_l1.bin" "$G/input.bin"; cp "$G/input1_l1.bin" "$G/input1.bin"
           cp "$G/expected_l1.bin" "$G/expected.bin"
           e2e; save l1; restore ;;
  # constmask: the production form (mask baked in as a constant, 1 runtime input).
  # Swap the const-mask .tflite/.dla in as the active model and hide input1.bin.
  constmask)
           mv "$G/input1.bin" "$G/_input1.bak"
           mv "$OP/model.tflite" "$OP/_rt.tflite"; mv "$OP/model.dla" "$OP/_rt.dla"
           cp "$OP/model_constmask.tflite" "$OP/model.tflite"
           cp "$OP/model_constmask.dla"    "$OP/model.dla"
           e2e; save constmask
           mv "$OP/_rt.tflite" "$OP/model.tflite"; mv "$OP/_rt.dla" "$OP/model.dla"
           mv "$G/_input1.bak" "$G/input1.bin" ;;
  # expprobe: (1) negative control — the naive "MUL(1/8) + EXP" chain on the SAME input, then
  # (2) the primary run again so compare.txt on disk stays the deliverable. Both e2e calls
  # happen under ONE lock hold (they are two device runs, not two lock cycles).
  expprobe)
           ./tools/devlock.sh run -- bash "$SELF" _expprobe_inner ;;
  _expprobe_inner)
           cp "$OP/model.tflite" "$OP/_rt.tflite"; cp "$OP/model.dla" "$OP/_rt.dla"
           cp "$OP/probe_expchain.tflite" "$OP/model.tflite"
           cp "$OP/probe_expchain.dla"    "$OP/model.dla"
           mv "$G/input1.bin" "$G/_input1.bak"
           cp "$G/expected_exp.bin" "$G/expected.bin"
           ./tools/e2e.sh "$OP"; save expchain
           mv "$OP/_rt.tflite" "$OP/model.tflite"; mv "$OP/_rt.dla" "$OP/model.dla"
           mv "$G/_input1.bak" "$G/input1.bin"
           restore
           ./tools/e2e.sh "$OP"; save primary ;;
  noise)   e2e; save noise ;;
  primary) e2e; save primary ;;
  *) echo "usage: $0 <health|mag2|zero|l1|noise|primary|restore>" >&2; exit 2 ;;
esac