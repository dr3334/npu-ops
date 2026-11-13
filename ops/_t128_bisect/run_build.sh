#!/usr/bin/env bash
set -u
cd '/mnt/d/deepseek_ workspace/mtk/wt-t128' || exit 1
export QWEN25_WEIGHTS_DIR='/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen'
rm -rf /tmp/t8e
/usr/bin/python3 tools/build_decoder_layer.py --out /tmp/t8e >/dev/null || exit 9
echo -n "T8 tflite sha256: "; sha256sum /tmp/t8e/model.tflite | cut -c1-16
diff -q ops/decoder_layer_t8_fp16/model.json /tmp/t8e/model.json && echo "T8 model.json identical"
/usr/bin/python3 tools/build_decoder_layer.py --T 128 || exit 9
./tools/compile.sh ops/decoder_layer_t128_fp16/model.tflite ops/decoder_layer_t128_fp16/model.dla
grep -E "ERROR|Execution Step|EXIT=" ops/decoder_layer_t128_fp16/model.compile.log | head -12
