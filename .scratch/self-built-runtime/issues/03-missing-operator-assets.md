# 03: 一层 decoder 缺哪些算子资产、工时多少

**Type:** task
**Status:** open
**Blocked by:** None

## Question

一层 decoder（ops/decoder_layer_t8_fp16）用到 13 类原生层：FC / BatchMatMul / SoftMax / Sigmoid / ElementWise(Mul,Add,Sub) / Reshape / Transpose / Concat / Slice / Mean / RSqrt。自建库已实测：五元逐元素 + RELU + FC(动态权重) + BMM + SoftMax。

**要做**：为缺失的 6 类（Slice / Transpose / Reshape / Concat / Mean / RSqrt）各捕获一次资产，按 docs/OWN-SUBMITTER.md §3 的四步法跑通并与 CPU golden 对拍；记录工作量、坑、形状相关字数量（供票 02 用）。

**验收**：每类一个「0 失配」或「1 码且有解释」；Sigmoid 现在 64/64 失配的问题也在这张票里定性。
