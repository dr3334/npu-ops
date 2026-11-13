# 02: 冻结随机权重夹具

**What to build:** 一套"生成一次、随后冻结"的随机权重，让板端与 CPU 参考跑**完全相同**的数值。目的是把"权重从哪来"这个变量彻底隔离在数值对拍之外——任何失配都只能指向实现，不能指向权重。分布不能用随手拍的均匀分布：fp16 在多层串联下会饱和，尺度必须贴近真实权重的实测统计量。

**Blocked by:** None (can start immediately)

**Status:** done

- [ ] 固定 seed 生成，同一 seed 两次生成**逐字节相同**
- [ ] 分布为正态，尺度对齐真实权重的实测 std（并记录实测值）
- [ ] 覆盖单层全部张量：q/k/v 投影（**含 bias**）、o 投影、gate/up/down、两个 RMSNorm 增益，以及 embed / LM head
- [ ] 每份权重记录 sha256，板端与 CPU 侧能各自校验同一份
- [ ] 生成脚本与 seed 入库，权重产物不入 git
- [ ] 记录"真实权重只在最后一档做一次性检查"这一取舍，以及它随后带来的风险
## Comments

- 2026-11-13 Lead: 合并到 feat/qwen25-attention（merge 655109a）。夹具 25 张量 331.9 MB，weights_digest=898fa319…dad18a4；尺度取真实权重 D:\ai\Qwen2.5-0.5B-Instruct 逐张量实测（hf 不可达但本地有副本），票面预留的 initializer_range 代理分支不适用。

