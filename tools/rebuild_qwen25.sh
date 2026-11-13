#!/usr/bin/env bash
# Regenerate the qwen25-attention batch's model.tflite (+ golden/*.bin) from its builders.
#   ./tools/rebuild_qwen25.sh [--force]
#
# 为什么单独一个脚本：这批 builder 的 CLI 各不相同（--which / --variant / --chunks / 目录 vs 文件），
# 塞进 tools/rebuild_ops.sh 的通用 --list 协议会把它们的差异藏起来。这里的每一条都是显式命令。
#
# 只重建缺失的 model.tflite；--force 时连已存在的也重建（会重写 model.json，用于核对可复现性）。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
_win="$(cd "$ROOT" && pwd -W 2>/dev/null || echo /d/x)"
_drv="$(printf %s "${_win:0:1}" | tr 'A-Z' 'a-z')"
WSLROOT="${WSLROOT:-/mnt/$_drv${_win:2}}"
WL() { wsl.exe -e bash -lc "cd '$WSLROOT' && export QWEN25_WEIGHTS_DIR='$WSLROOT/weights/qwen25_frozen' && $*"; }
FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1
built=0; failed=0; skip=0

# build <op-dir> <builder command relative to repo root>
build() {
  local d="$1"; shift
  if [ ! -d "ops/$d" ]; then echo "  no-dir  $d"; skip=$((skip+1)); return 0; fi
  if [ "$FORCE" = "0" ] && [ -f "ops/$d/model.tflite" ]; then skip=$((skip+1)); return 0; fi
  if WL "$*" >/dev/null 2>&1; then echo "  built   $d"; built=$((built+1));
  else echo "  FAILED  $d"; failed=$((failed+1)); fi
}

echo "[qwen25] rebuilding missing artifacts in $WSLROOT"
# --- 票 03：q/k/v 投影 + bias + RoPE（三张图，--out 是目录） ---
for w in q k v; do
  build "qkv_${w}_bias_rope_t8_fp16" "/usr/bin/python3 tools/build_qkv_bias_rope.py --which $w --out ops/qkv_${w}_bias_rope_t8_fp16"
done
# --- 票 04：GQA 打分 + KV 复制（主图 + ops/_gqa_variants 下的对照） ---
build gqa_scores_t8_fp16 "/usr/bin/python3 tools/build_gqa_scores.py --out ops/gqa_scores_t8_fp16"
for v in bmm_only nary14 twox7 unroll14 wrong_reshape rank5; do
  build "_gqa_variants/$v" "/usr/bin/python3 tools/build_gqa_scores.py --variant $v --out ops/_gqa_variants/$v"
done
# --- 票 05：掩码 + 原生 SoftMax（--out 是目录；同时产出 model_constmask.tflite） ---
build masked_softmax_t8_fp16 "/usr/bin/python3 tools/build_masked_softmax.py --out ops/masked_softmax_t8_fp16"
# --- 票 06：PV + O + 残差（主图 + 三个对照，--out 是目录） ---
build pv_o_residual_t8_fp16 "/usr/bin/python3 tools/build_pv_o_residual.py --variant main --out ops/pv_o_residual_t8_fp16"
build _pv_bmm3d_control_t8_fp16 "/usr/bin/python3 tools/build_pv_o_residual.py --variant bmm3d --out ops/_pv_bmm3d_control_t8_fp16"
build _pv_bmm4d_control_t8_fp16 "/usr/bin/python3 tools/build_pv_o_residual.py --variant bmm4d --out ops/_pv_bmm4d_control_t8_fp16"
build _pv_o_badlayout_control_t8_fp16 "/usr/bin/python3 tools/build_pv_o_residual.py --variant main-badlayout --out ops/_pv_o_badlayout_control_t8_fp16"
# --- 票 07：SwiGLU FFN（主图 + 去 silu 对照；--out 是目录） ---
build swiglu_ffn_t8_fp16 "/usr/bin/python3 tools/build_swiglu_ffn.py swiglu_ffn_t8_fp16 --out ops/swiglu_ffn_t8_fp16"
build _swiglu_ffn_t8_fp16_nosilu_ctrl "/usr/bin/python3 tools/build_swiglu_ffn.py swiglu_ffn_t8_fp16_nosilu_ctrl --out ops/_swiglu_ffn_t8_fp16_nosilu_ctrl"
# --- 票 08：LM head（tied，N>65535 必须分块；--out 是**文件**） ---
build lm_head_t8_fp16 "/usr/bin/python3 tools/build_lm_head.py --chunks 4 --out ops/lm_head_t8_fp16/model.tflite"
# --- 票 01：多头整块（--out 是目录；14 头的真实形状要显式给 --hidden/--head_dim） ---
build block_qwen25_multhead_h2_fp16 "/usr/bin/python3 tools/build_block2.py --heads 2 --out ops/block_qwen25_multhead_h2_fp16"
build block_qwen25_multhead_h14_fp16 "/usr/bin/python3 tools/build_block2.py --heads 14 --hidden 896 --head_dim 64 --norm fctile --atol 0.03 --rtol 0.03 --out ops/block_qwen25_multhead_h14_fp16"
build _block_qwen25_multhead_h14_meannorm_control "/usr/bin/python3 tools/build_block2.py --heads 14 --hidden 896 --head_dim 64 --norm mean --out ops/_block_qwen25_multhead_h14_meannorm_control"
# --- 票 09：1 层 decoder 整块（--out 是目录） ---
build decoder_layer_t8_fp16 "/usr/bin/python3 tools/build_decoder_layer.py --out ops/decoder_layer_t8_fp16"
# --- 票 10：两张层图 + tied LM head 的 logits 图（--graph l0|l1|logits，--out 是目录） ---
for g in l0 l1 logits; do
  build "two_layer_${g}_t8_fp16" "/usr/bin/python3 tools/build_two_layer.py --graph $g --out ops/two_layer_${g}_t8_fp16"
done

echo "---- built=$built failed=$failed skipped=$skip"
[ "$failed" = "0" ]
