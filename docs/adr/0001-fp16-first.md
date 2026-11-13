# 整条 Qwen2.5 路径先做全 fp16，int8 延后

**Status**: accepted

Qwen2.5-0.5B 的自建算子路径统一用 fp16 权重与激活，int8 留作后续性能优化。int8 累加虽已逐位证明（E-002），但 requant 只保证 ±1 LSB、比例 ≤0.4%、边界行为未复现（E-003）；24 层串联后这点误差会累积，届时无法区分"算子写错"还是"量化偏了"，而这恰是整件事最难判的部分。fp16 在 K≤128 已逐位一致（E-004），`head_dim=64` 正好落在该区间内。

**Considered Options**
- 全 int8：更快、权重小 4 倍，但失去逐位判据，且 requant 边界本身还没复现
- 混合（linear/FFN 用 int8、attention 打分与 softmax 用 fp16）：时间不省、复杂度翻倍，两套 golden 与两套判据要同时维护

**Consequences**
- 0.5B 的 fp16 权重约 1 GB，常量缓冲装不下整模，"权重驻留方式"因此从实现细节升格成必须显式决定的问题
- `docs/ISSUE-const-fc.md` 记的"常量缓冲必须恰好落在抓取时的 IoVA 上"这一契约，会立刻成为主风险
- 先把"能不能跑通"和"能不能跑快"分开；性能档另起一轮，届时 int8 的 requant 边界要先复现再上
