# 09: 整块 —— 1 层 decoder 拼装

**What to build:** 把 03–07 五个组件拼成一个**自包含**的 decoder 层：`[T,896]` fp16 进、`[T,896]` fp16 出，host 全程不参与任何数学。层内包含两次归一化、位置编码、GQA 注意力、掩码 softmax、前馈网络和两处残差。这是第一个"看起来像真实模型"的产物。

**Blocked by:** 03, 04, 05, 06, 07

**Status:** done

- [ ] 输入输出都只是 `[T,896]` fp16，host 不参与数学
- [ ] 层内两次归一化用真实 epsilon（1e-6）；与现有玩具块的 epsilon=0 之间的口径差**必须**记录
- [ ] 位置编码、GQA、掩码 softmax、前馈网络、两处残差全部在图内
- [ ] 单层 hidden 与 golden 对拍通过，阈值来自噪声底
- [ ] 单执行步、无 bridge；记录层数
- [ ] 记录整块的层数与大张量分布，作为"能否塞进单执行步"的实测依据

## Comments

- 2026-11-13 Lead: 合并到 feat/qwen25-attention（merge aadd651）。**61 层 / 1 个执行步 / 无 bridge**，板端 PASS：max_abs 2.319e-03（9.50 ULP @max|out|=0.4836）、rel_l2 3.690e-03，噪声底 0；阈值 9.0332e-3 跑前登记，设备无关下限 9.25 ULP（实测只多 0.25 ULP）。首跑 FAIL（rel_l2 2.198）已保留为证据；二分 attn_half/ffn_half 定位到共用 RMSNorm 的 MUL(x,x) 接成 MUL(x,1)。副产物：device-free 的 numpy 图执行器（--execcheck），0 板端周期即可发现「编译通过但数学错」。层数结论：**层数不是单步瓶颈，常驻内存才是**（每层 28.5 MiB 静态）。
