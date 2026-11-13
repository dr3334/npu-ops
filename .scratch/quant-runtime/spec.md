# Spec — 量化档与运行形态：w4a8 单层 → 24 层共享权重 → prefill/decode 双图

Status: implemented (2026-11-13)，两处待设备恢复后补测
Feature: quant-runtime
Date: 2026-11-13

## Problem Statement

上一轮（`.scratch/qwen25-attention/`）把 fp16 的 1 层 decoder 整块做到 61 层 / 单执行步 / T=8 与 T=128 都上板 PASS，但那是**研究形态**：每层一个 `.dla`、每层一个 `neuronrt` 进程、没有 KV cache、没有 decode、没有量化档。

厂商在同一块板上的部署形态完全不同，而且资产就在设备里：`24layer_128t2048c_0_shared.dla`（3.2 MB，prefill）、`24layer_1t2048c_0_shared.dla`（21.6 MB，decode）、`shared_weights_2048c_0.bin`（250.7 MB，24 层 W4 权重）、`embedding_..._int16.bin`（272.3 MB）。权重外部化、两张图、`SwapModel` 只在 prefill↔decode 边界发生。

三件事因此变成必须先回答的：
1. **量化档**：4-bit 权重在这条栈上已能编、能跑（w4a8 板端 0/64 逐位；w4a16 64/64 ≤1 码），但「选哪档、判据怎么定」没有定论。
2. **共享权重**：厂商把 24 层权重放在独立 bin 里由两张图引用；我们`每层一个 .dla`的做法把权重编进图，24 层就是 24 次装载（≈3.5 s/token）。ncc 的外部/共享权重机制我们**完全不知道**。
3. **执行形态**：prefill 与 decode 是两份图 + KV cache；我们只有 prefill，KV cache 一行没有。

## Solution

按依赖顺序五步，每步都落在一个可判定的证据上：

1. **量测厂商现状**（只读为主 + 一次实跑）：拿 `main` + benchmark config 跑一次，采三个时点的 `meminfo` / `dumpsys meminfo` / `dmabuf_dump`，把 tok/s、PSS、DMA-BUF、加载发生在哪个阶段变成我们自己的数字。
2. **把 int8 判据修到逐位**：golden 改成量化契约（两次舍入 + M0 取整），离线重放已有 dump 验证；此后所有量化档共用同一套 golden 规则。
3. **摸清共享权重机制**：ncc 有没有外部/共享权重的编译开关、厂商 `.dla` 如何引用那个 bin、我们能不能自己造一份。这一步的结论决定第 5 步是否成立。
4. **4W8A 单层**：用我们自己的 builder 把 1 层 decoder 做成 4W8A（int4 权重 + int8 激活），上板对拍；这一档同时在为「混合精度图」准备构件。
5. **24 层单图 + 外部共享权重（prefill）**：先只做 prefill，验证「一个图装 24 层 + 权重在外面」能编能跑；decode/KV 留到下一轮。

## Implementation Decisions

- 精度档：**先 4W8A，目标形态是 attention 4W16A + MLP 4W8A 的混合**（ADR 0005）。4-bit 档不承诺逐位。
- 判据两层：算子级 `|Δ| ≤ 1 码` + `cosine ≥ 0.9999`；整网级看 token 命中率/困惑度。
- 执行形态：抄厂商（两张图 + 外部共享权重 + KV），`每层一个 .dla` 只作诊断（ADR 0006）。
- 量化契约：int8 落码 = `round_half_up(acc·M0/2^31)` → `RoundingDivideByPOT(shift)`；M0 按「1/32 这类整乘子」取，不从 float32 scale 反算（后者会引入 0.03–0.2% 的 ±1）。
- 混合精度图的 dtype 缝一律显式写 `DEQUANTIZE`（ncc 实测：原生层、不 bridge、每缝 +1 层）。
- 所有板端操作持同一把 `devlock`；每次前后 `healthcheck --quick`。

## Testing Decisions

- 每个 op 目录仍然是「一个图 + 一份 `compare.txt`」这条缝；量化档在 `model.json` 里登记 `precision_tier`、`int8_tol`/`atol`/`rtol` 与依据。
- 逐位判据只在「同一档、同一量化参数、同一输入」的比较里使用；跨档比较一律用码级/余弦。
- **判据分两层**（ADR 0005 第六次订正）：实现门 = device == 量化 golden（码级）+ `|Δcos(device, golden)| ≤ 0.003`；质量期望 = fp16/16-bit 权重 ≥0.9999、**int8 权重 ≥0.9998**、int4 ≈0.97（固有上限）。**最终质量门在整网**（24 层 logits cosine / token 命中率），不用单层 cosine 的 1e-4 差当硬门。
- 共享权重的可复现性用「bin 的 sha256 + 图的 sha256 + 一次运行输出」三联记录。

## Out of Scope

- int4 以下（2-bit/1.5-bit）、稀疏化；
- 采样/tokenizer/对话模板；
- 性能调优（本轮只回答「能不能跑成厂商那个形态」）；
- decode 的正确性对拍（KV cache 的数值验证留到下一轮 spec）。

## Further Notes

- 厂商文档：`llm_sdk_document.txt:951-957`（精度菜单与推荐）、`:958,989`（允许 attention 4W16A + MLP 4W8A 混用）、`:1265-1266,1350-1351`（device 4W16A 不是 bit-true）。
- 本地实测矩阵：`skills/mtk-llm-pipeline/references/02-quantization-matrix.md:23-34`（4W16A 11.6–14.5 ms / 4W8A 5.94 ms / 8W8A 8.09 ms / W8A16F 15.08 ms）。
- 探针产物：`.qwen25-impl/probe-int16/RESULTS.md`、`.qwen25-impl/probe-precision/RESULTS.md` 是完整的；int4 那一档的结论文档在仓库内 `reports/Q02-int8-bitexact.md` 与 `docs/OP-shared-weights.md`（`.qwen25-impl/probe-w4a8/` 只有脚本与编译日志，没有 RESULTS.md）。
## 状态（2026-11-13 收尾）

十一个工作项（01、02、03、04、04b、04c、04d、04e、04f、05、06）全部完成并合并到 `feat/qwen25-attention`。

**已定的结论**
- 量化档：**8W16A**（int8 per-channel 权重 + int16 激活 + native int16 掩码 softmax）。真权重单层 0.9998598、24 层 0.9961–0.9968（串联累积）、8 token decode 逐步 `|Δcos| ≤ 8.2e-5`。
- 判据：实现门（device == 量化 golden + `|Δcos| ≤ 0.003`）与质量期望（fp16/16-bit ≥0.9999、int8 ≥0.9998、int4 ≈0.97）分离；最终质量门在整网 logits（ADR 0005 第六次订正）。
- 执行形态：**两张图 + 一份外部共享权重**（prefill T=16 / decode T=8-padded，bin 16,674,816 B 与 T 无关），权重真常驻（dma-buf、CPU 不触页、200–400 次 Run 零重读）。
- 不可行项（都留了最小复现）：4-bit per-channel（权重误差 14–16%，整层 ≤0.9733）、分组 scale（属性级不可表达；K 切段代价 2G−1 层/矩阵且救不了 4-bit）、**T=1 的图**（SOFTMAX command field / Input↔MDLA gap，两个不同的报错）。

**两项已补测完成（2026-11-13）**：
1. decode 每步 `rel_l2` ⇒ 0.00251–0.01280（门 0.05 PASS，kv1 补齐）；
2. KV 常驻 ⇒ **RuntimeV2 每请求 IOBuffer** 可行（v1 的 `setInput` 是绑一次的快照）；host 写 12,288 B/token、端到端 19–30×。
另：24 层 decode 的 KV 上限实测 **C=8192**（撞的是编译期内存规划，不是设备内存）。

**原计划的待测项（已由上面取代）**
1. decode 各步的 `rel_l2`：补跑 `./tools/e2e_q06.sh --steps 8`（第二次补跑时设备掉线：`adb: no devices/emulators found`，kill-server/reconnect 均未恢复）。
2. KV 常驻（RuntimeAPI 形态）：把 per-token 的 25.2 MB 上行拷贝降到 O(256 B/层)——这是与厂商结构性差距的最后一块（厂商 decode 17.2 ms/token，我们 68.5 ms/token，其中 KV 宽度 33.4 ms）。