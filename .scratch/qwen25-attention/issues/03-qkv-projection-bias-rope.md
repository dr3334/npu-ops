# 03: QKV 投影 + bias + RoPE

**What to build:** 真实形状的 Q/K/V 投影与位置编码。q 是 896×896 且带 bias，k/v 各是 128×896 且带 bias（2 个 KV head × 64，**不是** 896 宽），o 方向另票。这是唯一会把"bias 必须存在"和"RoPE 怎么进图"一起钉死的地方：bias 在配置文件里根本没有对应键，是上游实现硬编码的，漏掉它数值会系统性偏移。

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] fp16 下 q/k/v 三个投影与独立 numpy golden 对上，**bias 参与计算**
- [ ] k/v 输出宽度确认为 128，并记录"如果误接成 896 会怎样"
- [ ] RoPE 用预算常量表（sin/cos 不能作为图内算子），head_dim=64、T=8，输出与 golden 对上
- [ ] 输出为 3-D 形状（`[T,14,64]` / `[T,2,64]`），记录 rank 上限是 4
- [ ] 无 bridge、单执行步
- [ ] 记录本算子的实测噪声底，作为该算子阈值的依据
