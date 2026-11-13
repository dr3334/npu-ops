# OP-pv-o —— PV（3-D BMM）+ O 投影 + 残差（Qwen2.5-0.5B 真实形状 · T=8 · fp16）

范围：注意力输出那一半，一个可复现的单算子图：

```
p [14,8,8] × v [14,8,64]  --BATCH_MATMUL(3-D, adj 0/0)-->  a3 [14,8,64]
  --TRANSPOSE perm=[1,0,2]--> [8,14,64] --RESHAPE--> a [8,896]
  --FULLY_CONNECTED(W=o_proj[896,896], bias=0)--> o [8,896]
  --ADD(x, o)--> out [8,896]                     # x 是残差输入
```

- 分支 `impl/06-pv-o`（worktree `wt-t06`）· builder `tools/build_pv_o_residual.py` · 产物 `ops/pv_o_residual_t8_fp16/`
- 权重：票 02 的冻结夹具（`weights/qwen25_frozen`，`weights_digest=898fa319…`），**真实实测尺度**；
  取 `layer0.attn.q_proj.*` / `k_proj.*` / `v_proj.*` / `o_proj.weight` / `input_layernorm.weight` 与 `embed.weight`
- 编译器 `ncc-tflite` 7.3.15 · mdla5.0 / MT6897 · `--disallow-bridge`

---

## 0. 事前登记：假设与预测（写于任何编译/板端动作之前）

### 0.1 三个设计选择（先说清楚，避免结果被误读）

| # | 选择 | 理由 | 代价 / 风险 |
|---|---|---|---|
| D1 | **PV 走 3-D**（`[H,T,T]×[H,T,HD]`），不走 4-D `[1,H,T,T]` | E-112（`docs/OP-gqa.md`）实测 3-D 是 1 个 `BatchMatMulLayer`，4-D 被降成 5 层；层数预算按 3-D 记 | 3-D 的输出布局是 `[H,T,HD]`，必须先 `TRANSPOSE perm=[1,0,2]` 再 `RESHAPE` 成 `[T,896]`（NOTES §10.2 的教训：**直接 reshape 会编译通过但数学错**） |
| D2 | **p、v 打包成一个运行时输入** `[14,8,72]`，图内用两次 `SLICE` 拆成 p `[14,8,8]` / v `[14,8,64]`；第二个运行时输入是残差 x `[8,896]` | `tools/e2e.sh` 的缝只有 `golden/input.bin` + `golden/input1.bin`（NOTES §3、§8 不许改公共工具）。三个语义输入（p / v / x）只能压成两个文件。**不用**"把 x 做成常量"——常量 ADD 有被折进 FC bias 的风险，那样"残差加法"就没有层证据了 | 图上多 2 个 `SLICE` 层；"PV 一层"的判据要按层名数 `BatchMatMulLayer`，不能按总层数 |
| D3 | **p 用"真实权重 + 固定输入走一遍完整 attention"得到**（不是随便造的概率矩阵）：`x0 = embed.weight[0:8]` → RMSNorm(`input_layernorm.weight`, eps=1e-6) → q/k/v 投影（含 bias）→ RoPE(θ=1e6, split-half) → scores/√64 → 因果掩码 → softmax(`axis=-1`, float32)；v 按 GQA 分组（head h 用 kv head `h//7`）复制到 14 头 | 票据给的两条路里，这条路让 p / v / x 的**动态范围**来自真实权重尺度（`q_proj.bias` std≈7.85、`k_proj.bias` std≈29.8），而不是人手挑的数；同时它把"上游算子（票 03/04/05）在 CPU 上的语义"也钉住一份可复现数据 | p 只是**图输入数据**，其推导不进图；若它退化成近 one-hot（见 P3），单次对拍的"累加"判别力会变弱——已准备 alt-input 对照 |

### 0.2 假设 → 预测

| # | 假设 | 若成立，预期观察 |
|---|---|---|
| H1 | MDLA 把 3-D `BATCH_MATMUL` 映成一个原生 matmul 层（E-112 的结论在"头数=14、T=8、fp16、B 侧 64 宽"这组形状上仍成立） | exec plan 里 `BatchMatMulLayer` **恰好 1 个**，`EXIT=0`，`Target: MDLA_5_0`，无 bridge |
| H2 | 4-D 版 PV 被降成 5 层（切分/矩阵乘/切分/矩阵乘/拼接） | 4-D 对照模型的 exec plan 出现 **5 个层**（`Slice`/`BatchMatMul` 各 2 + `Concat` 1），与 3-D 的 1 层形成对照 |
| H3 | 整图（SLICE×2 → BMM → TRANSPOSE → RESHAPE → FC → ADD）全落 MDLA、单执行步 | `Graph[0] Step[0]` 只有一个 step；层清单含 2×Slice + BatchMatMul + Transpose + Reshape + FullyConnected + ElementWiseAdd（编译器可能把 TRANSPOSE/RESHAPE 之一折进相邻层，若层数少于此，按实际层名解释，**不硬凑**） |
| H4 | 板端数值 = "每层边界舍入到 fp16 + 层内 fp32 累加"的量级 | `max_abs` 与 builder 里 `--fp16sim` 算出的**层边界 fp16 下限**同量级（设备只多 1–2 ULP），而不是 `2e-2` 这种松阈值 |
| H5 | 硬件对同一 `.dla`、同一输入是确定性的 | 两次运行的 `max|r1-r2| = 0`（噪声底为 0），阈值退化为 fp16 末位口径 |
| H6 | 输入布局错（少一次 TRANSPOSE / SLICE 偏移错 / FC 权重转置）会大面积失配，而不是"看起来差不多" | 若 H1–H5 全部成立但数值 FAIL，第一嫌疑按 D1/D2 的顺序排（reshape 顺序 → slice 偏移 → FC 权重方向） |

### 0.3 三个数值预测（可在结果里被推翻）

| # | 预测 | 若被推翻意味着 |
|---|---|---|
| P1 | `p` **近 one-hot**：`k_proj.bias` std≈29.8 主导 scores，单头 scores 量级 ~10²，softmax 后 `max(p) > 0.99` | 说明我对 bias 量级的估计错了 → 要复查 q/k/v 的推导（阈值/缩放/RoPE 位置） |
| P2 | `max|out|` ≈ `max|x|` ≈ 0.05 量级（layer-0：x 是 embed 行 std≈0.0152，O 投影输出更小），即 `2e-2` 绝对阈值比输出本身还大 → **不适用 NOTES §4 的默认口径**，阈值必须由 fp16 模拟下限支撑（NOTES §10.1） | 若 `max|out| >> 1`，说明激活尺度与我的推导不符（例如 v 的量级被低估） |
| P3 | 板端 `rel_l2` 在 1e-3 量级或更小（fp16 末位） | 若 `rel_l2 ~ 1e-1`，几乎必然是布局错（§10.2 那类"编译通过但数学错"）而不是精度问题 |

---
