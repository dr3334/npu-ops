# 量化档目标：先做 4W8A，目标形态采用「attention 4W16A + MLP 4W8A」的混合

**Status**: accepted

厂商 LLM runtime 的出厂档是 `asym4W_sym16A`（w4a16）。但本机实测（`skills/mtk-llm-pipeline/references/02-quantization-matrix.md`）在同样 4-bit 权重下，w4a8 的 decode 是 **5.94 ms**、w4a16 是 **11.6–14.5 ms**（慢约 2 倍）；4-bit 的质量本身也可疑（device token match 23/268，加 hessian 只到 28/268）。厂商文档的推荐列表只有 `sym4W_sym16A` / `asym4W_sym16A` / `dynamic_quant`，4W8A 不在其中，但明确允许 attention 用 4W16A、MLP 用 4W8A 混用。

因此：**先做 4W8A**（本仓库已有板端证据：int4 权重 + int8 激活单执行步、0/64 逐位），**目标形态是厂商允许的混合**（attention 4W16A + MLP 4W8A）。4-bit 档**不承诺逐位**——厂商自己声明 device 4W16A 不是 bit-true，我们的探针也测到 int4 解压后 ≤1 码抖动。

**Considered Options**
- 全局 4W16A：与厂商默认一致，但速度只有 w4a8 的一半，且质量并未因此变好（23/268）。
- 全局 4W8A：最快，但 attention 的分数分布在 4-bit 激活下抖动会被 softmax 放大（T=128 实测 probs 已有 ~12% 抖动）。
- W8A16F（dynamic quant）：质量接近 fp32，但 decode 15.08 ms、内存 2×，不适合作为主档。

**Consequences**
- 判据升级为两层：算子级 `|Δ| ≤ 1 码` + `cosine ≥ 0.9999`；整网级看 token 命中率/困惑度，**不看逐位**。
- golden 必须按「量化契约」实现（两次舍入 + M0 取整规则），否则 ±1 码会被误读成实现错误。
- 混合精度图意味着每条 dtype 缝要显式写 DEQUANTIZE（ncc 已实测：原生层、不 bridge、每缝 +1 层）。
