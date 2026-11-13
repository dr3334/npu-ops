# 08: LM head + logits

**What to build:** 从 896 维隐藏态算出 151936 维的词表 logits。真实模型里输出投影与词嵌入是**同一个矩阵**（tied），不能物化两次——那会白多出约 260 MB。判据是 logits 向量本身的数值关系，**不是** argmax：随机权重下 top-1 不携带语义，且 fp16 末位噪声就能让近似平局翻转。

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] 复用词嵌入矩阵作为输出投影（tied），不额外物化一份
- [ ] 输出 vocab=151936 的 logits，与 golden 的 atol/rtol 对上
- [ ] **top-1 判据不启用**，并记录它要冻结到真实权重阶段才启用
- [ ] 记录输出张量的尺寸与设备→host 搬运成本
- [ ] 记录本算子的实测噪声底
