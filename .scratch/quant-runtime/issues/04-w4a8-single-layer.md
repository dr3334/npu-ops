# 04: 4W8A 单层 decoder

**What to build:** 用我们自己的 builder 把 1 层 decoder 整块做成 4W8A（int4 权重 + int8 激活），上板与 dequant 参考对拍；同时为「混合精度图」准备构件。

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] int4 权重按 TFLite 约定打包（2 个/字节、低 nibble 在前），per-channel scale，op version 取 ncc 接受的那几档（1,3,4,5,6,7,9；**canonical v13 会被拒**）
- [ ] 逐算子先过编译：FC / BMM / CONCAT / SPLIT / SOFTMAX（已知 int8 可行）+ RMSNorm（注意 eps 在低比特下会被折掉——按 ADR 0005 把 eps 折进下游 FC bias）
- [ ] 上板对拍：算子级 `|Δ| ≤ 1 码` + `cosine ≥ 0.9999`；逐层记录哪一段开始掉码
- [ ] 记录 16 B 行规则在 int8 激活下的约束（小行宽形状会 bridge）
- [ ] 噪声底与阈值按上一轮的方法登记后再跑