# 一个 Qwen2.5 风格 decoder block 的整图试验

日期：2026-11-12 · 板 Titan 1 / MT6897 · ncc 7.3.15 `--disallow-bridge`。
模型：`ops/block_qwen25_1l_fp16`（T=8, H=64, 1 head / head_dim 32, FFN=128, fp16）。
生成器：`tools/build_block.py`（可复现，含 numpy 参考）。

## 1. 结果

**整块 16 个层全部落在同一个 `MDLA_5_0` 执行步，没有 bridge；板端数值通过。**

| 指标 | 值 |
|---|---|
| 编译 | `EXIT=0`，`.dla` 78.8 KB，`Target: MDLA_5_0`（单步） |
| 层清单 | FullyConnected×10（含 2 个**动态权重**）、SoftMax、Transpose、ElementWiseAdd×2、Sigmoid、ElementWiseMul×2 |
| 板端 | 512 个 fp16 输出 |
| 数值 | **max_abs = 9.766e-04**，rmse = 1.986e-04，rel_l2 = 3.730e-04（≈ fp16 量化步长），NaN/Inf = 0 |
| 对照 | 独立的 numpy float32 参考（同一份权重） |

## 2. 图里有什么 / 图外有什么

图内（全在 NPU）：
```
q=FC(x,Wq) k=FC(x,Wk) v=FC(x,Wv) → s=FC(q,k)[动态B] → softmax → a=FC(p,vᵀ)[动态B]
→ o=FC(a,Wo) → r1=x+o → g=FC(r1,Wg) u=FC(r1,Wu) → sigmoid(g) → m=g*sig → y=m*u
→ d=FC(y,Wd) → out=r1+d
```
图外（有意保留，见 §4）：
- **RMSNorm**：需要 rsqrt，是否能在 MDLA 上是 open（rms-agent 在查）；
- **RoPE**：`SIN`/`COS` 被 importer 拒绝，只能用 CPU 预算的 sin/cos 常量，本轮没有拼进图。

## 3. 三个值得记的点

1. **动态权重 FC 在图内可用**：`s=FC(q,k)` 与 `a=FC(p,vᵀ)` 的权重是运行时张量（不是常量），
   编译器照样把它们放进 MDLA 步——这解决了"attention 的两个矩阵乘能不能上 NPU"的最大疑问。
2. **Transpose 也是真实层**（`TransposeLayer<5>`），不是被折叠掉的视图。
3. **整块没有 bridge**，说明这套算子组合对 ncc 是"可整块下沉"的；之前 BMM 单独测出来的 shape 边界
   （16 字节行规则）在这个形状集里刚好都满足。

## 4. 尚未覆盖（必须说清）

- RMSNorm / RoPE 不在图内 → **这还不是"完整 block"**；
- 单 head（真实模型 14 Q heads / 2 KV heads）、T=8（真实 prefill 128/2048）；
- 没有 KV cache、没有残差之外的跨 block 状态；
- **这是厂商编译路径**：`.dla` 由 ncc 生成，不是我们自建提交器；
- 数值是 fp16 路径（atol 2e-2 合同），没有做 int8 版整块。

## 5. 下一步

1. 等 rms/rope/silu/bmm 四个 agent 的结论，把 RMSNorm 与 RoPE 拼进图 → **完整 block**；
2. int8 版整块（需要更多对齐检查：E-003 的 requant 规则已经可用于写 golden）；
3. 多 head / T=128 的 prefill 变体；
4. 把整块作为"基线"，再逐个用自建提交器替换其中的 Linear（openmtk M2/M3 的路径）。
