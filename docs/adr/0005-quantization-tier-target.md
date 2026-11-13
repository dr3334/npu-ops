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
---

## 实测订正（2026-11-13，票 q04 之后）

4W8A 的整层实测把本 ADR 的两条判据逼到了墙上：

- **组件级全部逐位**：4W8A 的 FC 投影、SwiGLU、BMM-QK、BMM-PV 板端与量化 golden **0 码**（per-channel int4 + int32 per-channel bias 可编、单执行步）。int8 RMSNorm 是例外：**861/14336、max 2 码**（eps 那层被折掉，仓库对该链的预算是 4 码——保留 FAIL，未改阈值）。
- **整层不达判据**：61 层 / 单执行步 / Static 7.2 MiB / 噪声 0，但 device vs 量化 golden **均值 4.13 码、max 41**；device vs 独立 fp64 链 **cosine 0.9107**，而量化 golden vs fp64 已经是 **0.9130** ⇒ **设备是忠实的，差距全在量化**。
- **误差归因**：int4 每通道权重 **每矩阵 14.0–15.8%**（主项；MLP 无 bias 藏不住），int8 激活 2.4%（次项）。掉码从 `probs` 开始：int8 分数 1 码 = 37，softmax 在近并列 row 上翻转 17/224。

因此本 ADR 追加三条：

1. **「算子级 cosine ≥ 0.9999」只适用于 8-bit 及更宽的档**。4-bit per-channel 在本栈（TFLite 只有 per-channel，没有 group-wise scale）**达不到** 0.9999，这是量化方案的性质，不是实现错。4-bit 档的算子级判据改为：**device == 量化 golden（码级 0 / ≤1）** + **device 与 golden 的 cosine 差 ≤ 0.003**（本票实测 0.9107 vs 0.9130 = 0.0023，说明设备忠实）；质量由整网/端到端判据回答。
2. **4-bit 的质量缺口要从「量化方案」解决，不是从「精度档」解决**：厂商的 4-bit 是 GPTQ/hessian + 分组 scale（他们的 bin 体积 0.5075 B/param ≈ 4bit + 1.5% scale，反推每 32–64 个权重一个 scale），而 TFLite per-channel 只有每输出通道一个 scale —— 这一条差异就能解释 14–16% 对厂商 4-bit 的差距。**下一步先查「分组 scale 能否在本栈表达」与「8W8A 的中间档是否够用」**，再决定锁哪个档。
3. **落档顺序改为：等 04b/04c 的结果再定**。若 group-wise 不可表达而 8W8A 的质量可接受，则主档改 **8W8A**（速度 8.09 ms、内存 2× 4W），混合精度（attention 与 MLP 分档）作为次选；若两者都不行，则回到 fp16 + 只在能证明不掉质量的位置用 4-bit。

三个必须记住的负结果（都有最小复现，见 `docs/OP-w4a8-layer.md`）：
- **int8 掩码 SOFTMAX 编不过**（无 exec plan；−4096 在 int8 分数域不可表示）→ 该层走 fp16 缝（+2 层）。
- **int8→int8 `QUANTIZE` 能编但算错**（rel_l2 0.85；去掉它就对 9e-5）→ 后续票不要用显式 QUANTIZE 做 requant。
- **T=8 的 P·V 被 16 B 行规则挡住**（EXIT=1）→ 4W8A 整层取 T=16。
