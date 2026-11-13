#!/usr/bin/env bash
set -u
cd '/d/deepseek_ workspace/mtk/wt-t128' || exit 1
export NPU_LOCK='/d/deepseek_ workspace/mtk/.npu-board.lock'
O=ops/decoder_layer_t128_fp16
./tools/devlock.sh run -- bash -c "./tools/healthcheck.sh --quick && \
  ./tools/e2e.sh $O --repeat 3 && \
  cp $O/out_device.bin $O/out_r1.bin && \
  ./tools/e2e.sh $O --repeat 3 && \
  cp $O/out_device.bin $O/out_r2.bin && \
  ./tools/healthcheck.sh --quick"
echo "BOARD_SCRIPT_EXIT=$?"
