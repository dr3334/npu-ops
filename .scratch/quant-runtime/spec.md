# Spec — 量化档与运行形态：w4a8 单层 → 24 层共享权重 → prefill/decode 双图

Status: ready-for-agent
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
- 共享权重的可复现性用「bin 的 sha256 + 图的 sha256 + 一次运行输出」三联记录。

## Out of Scope

- int4 以下（2-bit/1.5-bit）、稀疏化；
- 采样/tokenizer/对话模板；
- 性能调优（本轮只回答「能不能跑成厂商那个形态」）；
- decode 的正确性对拍（KV cache 的数值验证留到下一轮 spec）。

## Further Notes

- 厂商文档：`llm_sdk_document.txt:951-957`（精度菜单与推荐）、`:958,989`（允许 attention 4W16A + MLP 4W8A 混用）、`:1265-1266,1350-1351`（device 4W16A 不是 bit-true）。
- 本地实测矩阵：`skills/mtk-llm-pipeline/references/02-quantization-matrix.md:23-34`（4W16A 11.6–14.5 ms / 4W8A 5.94 ms / 8W8A 8.09 ms / W8A16F 15.08 ms）。
- 探针产物（仓库外）：`.qwen25-impl/probe-w4a8/RESULTS.md`、`.qwen25-impl/probe-int16/RESULTS.md`、`.qwen25-impl/probe-precision/RESULTS.md`。
