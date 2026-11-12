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

## 6. v2 尝试（RMSNorm 进图）—— 目前失败，记为失败知识

`tools/build_block2.py` 把 RMSNorm 链（`MUL→MEAN→ADD→RSQRT→MUL→MUL`）放进同一个图，
ncc 在 **importer 阶段**就拒了：

```
ERROR: Fail to process the given model     EXIT=1
```

（没有层清单，说明还没走到 lowering；RMSNorm 单独编译是能过且落 MDLA 的，见 `docs/OP-rmsnorm.md`。）

**待排除的候选**（下一轮按顺序做，一次只改一处）：
1. `gamma` 的 **广播 MUL**：`[T,H] × [H]` 在 TFLite 合法，但 ncc 可能不接受非标量广播 → 先换成 `[1,H]` 常量或把 gamma 乘进相邻 FC；
2. `RSQRT` 在这个图里的 **op version**（单独模型用 version=1 通过，这里可能被折叠成别的形式）；
3. `MEAN` 的 axis 常量张量形状（`[1]` vs 标量）；
4. 图规模（16 → 27 层）触发的 importer 限制。

结论：**"带 Norm 的整块"还没有跑通**；v1（无 Norm）已跑通，四类新算子（RMSNorm/RoPE/SiLU/BMM）
各自单独在 NPU 上都已证明。把 v2 的 importer 错误排掉，才谈得上"完整 block"。

## 7. v2 排错记录（截至本轮）

按"一次只改一处"推进，importer 错误随修改在移动，说明结构在逐步对，但还没收敛：

| 尝试 | 结果 |
|---|---|
| `mean[T,1] + eps[T,1]`（两个 fp16 常量/张量） | `ERROR: Invalid ElementWiseAddLayer<2>{ADD}: input types are incompatible with each other` |
| 去掉 eps ADD（eps=0，直接 `RSQRT(mean)`） | `ERROR: Invalid RSqrtLayer<2>{RSQRT}: output shape is incompatible with inputs` |
| 把 `mean`/`rsqrt` 改成 rank-1 `[T]`（keep_dims=false） | 仍是 RSQRT shape 不兼容，并新增 `ElementWiseMulLayer<3>: input shapes are incompatible` |

**关键反差**：同一套 `MUL→MEAN→(ADD)→RSQRT→MUL→MUL` 在**独立模型**里（`docs/OP-rmsnorm.md`，
形状 `[1,64]`/`[8,64]`）能编译并在 MDLA 上跑出 2 ULP 的结果；拼进整图就报形状/类型不兼容。
⇒ 问题不在算子能力，而在**整图里的形状传播**（最可能是 MEAN 的 `keep_dims` 与后续张量声明不匹配，
或 ncc 要求这组链路的张量保持与图其它部分一致的 rank）。

**下一步（具体动作）**：把 `tools/build_rmsnorm.py` 里独立模型使用的
`tensor()` 声明（shape/keep_dims/axis 常量形状）与 `build_block2.py` 逐字段对比，
先把**只含 RMSNorm + 一个 FC** 的两算子图跑通，再逐步加回 attention/MLP。

**当前可用的整块**：v1（无 Norm、无 RoPE）已通过，`ops/block_qwen25_1l_fp16`。

### 7.1 bisect 结果（RMSNorm + 1 个 FC，两算子图）

把整块缩到最小复现：`MUL(x,x) → MEAN(axis=1,keep) → RSQRT → MUL(x,r)` 再 `FC`，**仍然失败**：

```
层次=1  ERROR: Invalid RSqrtLayer<2>{RSQRT}: output shape is incompatible with inputs
```

两种张量声明都试过：

| mean/rsqrt 声明 | keep_dims | 结果 |
|---|---|---|
| `[T,1]` | true | RSQRT output shape incompatible（层次=1 就停） |
| `[T]` | false | RSQRT 不兼容 + `ElementWiseMulLayer<3>: input shapes are incompatible` |

而**完全相同的链**在 `tools/build_rmsnorm.py` 的独立模型里（`ops/rmsnorm_8x64_fp16`）能编译并跑出 2 ULP。
两者张量声明（mean `[T,1]`、eps `[T,1]`、rsqrt `[T,1]`、axis `[1] int32`）**看起来一致**。

⇒ 结论：不是形状声明本身，**是构造顺序/索引细节**（很可能是 flatbuffer 里 tensor 与 operator 的登记顺序，
或 MEAN 的 axis 张量在 buffer 表中的位置）。下一步应当做的是 **把两份 .tflite 的 tensor 表逐字段 diff**
（`flatc -t --raw-binary schema` 或直接解析），而不是继续在手写图里试错。

> 记录这条的意义：它说明"独立算子能跑"与"同一算子拼进整图能跑"是**两个不同的验证**，
> 后者才是 block 级别真正需要的。
