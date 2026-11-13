# 04c: 8W8A 参考档（同一 harness）

**What to build:** 用票 04 的同一套组件与整层 harness，把权重从 int4 换成 int8（8-bit per-channel），量出「质量—速度—内存」的中间点。

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] 组件级：FC / SwiGLU / BMM-QK / BMM-PV 的 8W8A 版，板端 device vs 量化 golden（预期逐位）与 vs fp64 链的 cosine
- [ ] 整层 8W8A（沿用 T=16 的 seam）：层数/执行步/Static 字节、噪声底、码级误差、cosine
- [ ] 记录：每矩阵量化误差（对照 4-bit 的 14–16%）、.dla 体积、板端单层前向时间
- [ ] 结论：8W8A 是否达到「算子级 cosine ≥ 0.9999」或至少落在可接受区间；与 4W8A 的 2× 内存/速度差是否值得