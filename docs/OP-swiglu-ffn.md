# OP-swiglu-ffn —— T=8 SwiGLU FFN 整段（Qwen2.5-0.5B layer0.mlp，fp16）

> 范围：票 07（`feat/qwen25-attention` 的 `impl/07-swiglu` 分支，worktree `wt-t07`）。
> 本文只记录本票的实验；`tools/build_swiglu_ffn.py`、`ops/swiglu_ffn_t8_fp16/`、
> `ops/_swiglu_ffn_t8_fp16_nosilu_ctrl/` 是本分支新增产物。方法约束见 `AGENTS.md`（R1–R10）
> 与 `docs/AGENT-BRIEF.md`。公共索引（`docs/EXPERIMENTS.md`、`PLAN.md`、`docs/CAPABILITY.md`）
> **未改**；编号与索引登记留给 Lead。

---

## 0. TL;DR

| 项 | 结果 |
|---|---|
| 图 | `x[8,896] → FC(gate_proj)[8,4864] / FC(up_proj)[8,4864] → LOGISTIC → MUL → MUL → FC(down_proj) → [8,896]`，fp16 |
| 权重 | 冻结夹具 `layer0.mlp.{gate,up,down}_proj.weight`（sha256 加载即校验），**静态常量**，`weights_digest=898fa319…` |
| 编译 | **EXIT=0**，`Target: MDLA_5_0`，**1 个 Execution Step**，**6 层**，**无 bridge**（`--disallow-bridge`） |
| 融合 | **SiLU 没有被融进前一个全连接**：主图 6 层，去掉 silu 的对照图 4 层 ⇒ silu（`Sigmoid`+`Mul`）**恰好 2 层**，且仍在**同一个执行步**、`Temp=0` |
| 内存 | DRAM：Input 14K / Output 14K / **Temp 0** / **Static 24M** / Code 1.7K；L1 228K |
| 板端 | `compare.txt` **PASS**：max_abs **2.441e-04**（= max\|out\|=0.5264 处 **1 个 fp16 码**）、rmse 7.030e-05、rel_l2 5.476e-04 |
| 噪声底 | **0.000e+00**（同一 `.dla`、同一输入连跑两次，输出**逐字节相同**，硬件确定性） |
| 阈值 | 跑**之前**登记 `atol=2e-3, rtol=1e-2`（`tolerance_revision=repo-fp16-default`）；实测相对误差 4.64e-4，**比"噪声底退化后的最紧口径（3×1 ULP=1.39e-3）还小 3×**，事后未调阈值 |
| 权重驻留实测 | 三个大张量各 **8,716,288 B（8.3125 MiB）**，合计 **26,148,864 B（24.94 MiB）**；`.tflite` 26,171,640 B、`.dla` 26,155,123 B；板端 push 26,155,123 B **0.68 s（36.7 MB/s）**；编译期表现为 `Static 24M`（DRAM），**不进 L1** |
| 单次推理 | 2.155 ms/inf（run1）/ 2.226 ms/inf（run2），`-c 3` 平均，仅记录不作结论 |
| 票面 checklist | **5/5**（§8） |

**一句话**：`896→4864→896` 的 SwiGLU 段在 ncc 7.3.15 / mdla5.0 上是**单执行步、无 bridge、6 层**的一段，
26 MB 权重以 DRAM 静态张量驻留（`Temp=0`，中间激活不落 DRAM）；板端误差正好是输出量级处的 **1 个 fp16 码**，
两次运行逐字节相同（噪声底 0）。

---

## 1. 证据绑定（R1）

| 项 | 值 |
|---|---|
| 仓库/分支 | `npu-ops` worktree `wt-t07`，分支 `impl/07-swiglu`（基于 `feat/qwen25-attention`） |
| 本票 commit | `afcca36`（builder + 登记容差）/ `0abbd27`（编译证据）/ `21f98e2`（板端 PASS + 噪声底） |
| 编译器 | `ncc-tflite` **Version 7.3.15**，`--arch=mdla5.0 --opt=3 --disallow-bridge --num-mdla=2 --l1-size-kb=3072` |
| 设备 | adb serial `0123456789ABCDEF`（mt6897 / Android 16，同 OP-silu §1）；执行器为厂商 `neuronrt -m hw -c 3` |
| 权重 | `weights/qwen25_frozen`（票 02 夹具），`tools/qwen25_weights.get()` 加载即比 sha256；本 worktree 内 `--verify` → `25/25 RESULT: PASS`，`weights_digest=898fa3195d627a4ce055cbdb6fc762a12e74665b431e3ae09b530b4abdad18a4` |
| 模型生成 | `tools/build_swiglu_ffn.py`（新增；只 import `tools/build_ops.py` 的 flatbuffer 脚手架，**未改**公共工具） |
| 板端串行 | 整批（healthcheck → e2e → 保存 out_r1 → e2e → 保存 out_r2 → healthcheck）在**一次** `NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" ./tools/devlock.sh run -- …` 内完成 |
| 日期 | 2026-11-13 10:17–10:18 (+08:00) |

产物指纹（只标识某次产物，**不是**可复现性判据，NOTES §4）：

| 产物 | bytes | sha256[:16] |
|---|---|---|
| `ops/swiglu_ffn_t8_fp16/model.tflite` | 26,171,640 | –（不入 git） |
| `ops/swiglu_ffn_t8_fp16/model.dla` | 26,155,123 | `3c548ea09ad7e947` |
| `ops/_swiglu_ffn_t8_fp16_nosilu_ctrl/model.dla`（对照） | 26,154,484 | `0d66996a703c1425` |

> **.dla 不可字节复现：同一 tflite 连编多次 SIZE/SHA256 会差几个字节，以该目录 model.compile.log 为准；回归只比数值**（上表 `.dla` bytes 即 `SIZE`）。

设备状态门（AGENTS §3.4）：批**前** `healthcheck.sh --quick` → **PASS**，批**后** → **PASS**；
批后 `dmesg | grep -E 'apusys.*(timeout|fail|error)|mdw.*(timeout|fail)'` → **0 行**。

---

## 2. 假设与预测（写于编译与板端之前）

任务书的三个问题：**① SiLU 会不会被 ncc 融进前一个全连接、融合对层数的影响；② 三个大张量的字节数；
③ 单执行步 / 无 bridge**。据此事先登记：

| # | 预测 | 若预测不成立意味着 |
|---|---|---|
| P1 | 层清单为 **6 层**：`FullyConnected(gate)`、`FullyConnected(up)`、`Sigmoid`、`ElementWiseMul`、`ElementWiseMul`、`FullyConnected(down)`；silu **不融合**（TFLite 的 FC `fused_activation` 只有 NONE/RELU/RELU_N1_TO_1/RELU6，本 schema **没有 SILU 算子**，OP-silu §7.1 已实测） | 若层数 < 6 ⇒ 存在我们没见过的 FC+激活折叠，层数预算要重算 |
| P2 | 去掉 silu 两个算子的对照图是 **4 层** ⇒ silu 的代价 = **+2 层**，且**仍在同一执行步** | 若对照图不是 4 层，说明"层数差"不能单独归因到 silu |
| P3 | **1 个执行步、无 bridge**，`Target: MDLA_5_0` | 出 bridge = 没上 NPU = 本票失败 |
| P4 | 26 MB 静态权重放不进 L1（3 MB），必然落 **DRAM Static**；`[8,4864]` fp16 中间张量（77.8 KB/个）放得进 L1 ⇒ **`Temp = 0`** | 若 `Temp > 0`，说明中间激活要落 DRAM，整块预算里要多算一次 DRAM 往返 |
| P5 | 数值上误差应是输出量级处的**几个 fp16 码**；由"逐张量 fp16 舍入"的 numpy 模拟给出**与设备无关的下限**（`golden/expected_alt.bin` vs `expected.bin`） | 若设备误差远大于该下限 ⇒ 某层不是"正确舍入的 fp16"，需要逐层归因 |
| P6 | 同一 `.dla` + 同一输入连跑两次**逐字节相同**（既有 fp16 实验均如此）⇒ 噪声底 0，阈值退化为 fp16 末位量级（NOTES §4） | 若噪声底 > 0，阈值要按 3–5× 噪声底重新登记（**跑之前**才算数） |

**容差登记（跑之前，写进 `model.json`）**：`max|expected| = 0.5264 ≤ 1` ⇒ 用仓库既有 fp16 口径
`atol=2e-3, rtol=1e-2`（OP-actreduce §1.3 / OP-silu 同款），`tolerance_revision="repo-fp16-default"`，
理由写进 `model.json.tolerance_rationale`：该口径已**宽于** 4 ULP（4·2⁻¹¹·0.526 = 1.03e-3），
并且**严于** NOTES §4 的 2e-2 上限；因为 `max|out| ≤ 1`，**不需要** NOTES §10.1 的例外规则。

---

## 3. 做法

### 3.1 图（`tools/build_swiglu_ffn.py`）

```
输入 x [8,896] fp16（model input，不带 data buffer）
  gate = FC(x, W=gate_proj[4864,896], b=0)  -> [8,4864]
  up   = FC(x, W=up_proj  [4864,896], b=0)  -> [8,4864]
  sig  = LOGISTIC(gate)                     -> [8,4864]
  s    = MUL(gate, sig)                     -> [8,4864]      # silu(gate)
  y    = MUL(s, up)                         -> [8,4864]
  out  = FC(y, W=down_proj[896,4864], b=0)  -> [8,896]
```

- **权重直接来自夹具，不转置**：TFLite FC 是 `out[m,n] = Σ_k in[m,k]·W[n,k] + b[n]`，而夹具的
  `gate/up_proj` 已是 `[4864,896]`、`down_proj` 已是 `[896,4864]`，正好是 `[N,K]`。
- **零 bias 是格式要求不是数学要求**：Qwen2.5 的 MLP 三投影**无 bias**，但 TFLite 的
  `FULLY_CONNECTED` 必须有第三个输入（NOTES §5 坑 1：只给两个输入会在**加载期**报
  `Fail to process the given model`）。因此喂三个全零常量（`[4864]`×2、`[896]`，共 11,264 B）。
- silu 只能写成 `LOGISTIC + MUL`：本 schema **没有 SILU 内置算子**（OP-silu §7.1）。

### 3.2 输入（真实尺度，不是"玩具 ±1"）

`x = γ ⊙ z`，其中 `z` 是逐行归一化到 RMS=1 的标准正态，`γ = layer0.post_attention_layernorm.weight`
（**夹具里的真实张量**，mean 0.678 / std 0.151，见 OP-weights §4）。也就是"post-attention RMSNorm 的输出"
这个真实位置上的激活。另有 4 个元素钉成 `+6 / −6 / 0 / +3` 以覆盖比高斯主体更宽的范围。
实测 `x`：rms 0.709、abs_max 6.0。

### 3.3 golden（R3：独立于生成 flatbuffer 的代码路径）

`golden/expected.bin`：从**存下来的** fp16 输入与 fp16 权重出发，float64 里做精确链
（`gate`、`up`、`sigmoid` 用 `exp` 分段写法、`y=silu(gate)·up`、`out=y·Wᵈᵀ`），最后**只做一次**舍入到 fp16。
不经过 ncc / TFLite interpreter / 设备 runtime。

`golden/expected_alt.bin`：**每个张量边界都落一次 fp16** 的模拟（TFLite 参考 kernel 的语义）。
`max|alt − exact|` 就是"任何把中间张量存成 fp16 的实现"的**与设备无关的下限** —— 本票实测
**2.441e-04（1 ULP @ max|out|）**，用于给阈值提供依据（NOTES §10.1 的"独立于设备的下限测量"这条要求）。

对照图 `ops/_swiglu_ffn_t8_fp16_nosilu_ctrl/`：同图**去掉 silu 两个算子**（`gate·up` 直接进 `down_proj`），
**只编译、不上板**，用途只有一个 —— 用层数差把"silu 的代价"从"整段的代价"里分离出来。

---

## 4. 结果

### 4.1 编译与层清单（回答"融合了吗"）

`ops/swiglu_ffn_t8_fp16/model.compile.log`（逐字）：

```
===Graph[0] has 1 Execution Step(s)===
Graph[0] Step[0]
ExecutionStep
 ├ StepId: 0
 ├ Target: MDLA_5_0
 └ Subgraph:
   ├ FullyConnectedLayer<0>{FULLY_CONNECTED}
   ├ SigmoidLayer<2>{LOGISTIC}
   ├ ElementWiseMulLayer<3>{MUL}
   ├ FullyConnectedLayer<1>{FULLY_CONNECTED}
   ├ ElementWiseMulLayer<4>{MUL}
   ├ FullyConnectedLayer<5>{FULLY_CONNECTED}
   └ Output: OpResult (external)

DRAM Usage:
       Target       Input  Output  Temp  Static  Code  Total
 [ 0]: MDLA 5.0     14K    14K     0     24M     1.7K  24M
Total Memory:
  DRAM: 24M
  L1:   228K
EXIT=0
SIZE=26155123
```

**判定**：**P1、P3、P4 成立**。

1. **silu 没有被融合**：`SigmoidLayer` 与两个 `ElementWiseMulLayer` 都是**独立层**，`FullyConnectedLayer<0>`
   的输出接的是 `SigmoidLayer<2>` 而不是"带激活的 FC"。机制上也可预期：TFLite 的 FC `fused_activation`
   只有 NONE/RELU/RELU_N1_TO_1/RELU6，**没有 SILU 档位**，而本 schema 连 SILU 内置算子都没有（OP-silu §7.1）。
2. **层数 6**：3 个 FC + 1 个 Sigmoid + 2 个 MUL。**层序不是图的拓扑序**：
   `FullyConnectedLayer<1>`（up 投影）被排在 `Sigmoid<2>`/`Mul<3>`（gate 分支）**之后** —— 两条分支
   互相独立，调度器按自己的顺序发；`<n>` 是原图节点下标，不是执行序号。
3. **1 个 Execution Step、无 bridge、`Target: MDLA_5_0`**：`--disallow-bridge` 下 `EXIT=0`。
4. **`Temp = 0`**：中间张量 `gate/up/sig/s/y`（各 `[8,4864]` fp16 = **77.8 KB**）全在片内，不落 DRAM；
   DRAM 只有 Input/Output 各 14 KB（`8·896·2`）与 **Static 24M**（三个权重张量）。
5. **权重驻留不是"复制一份"**：`.tflite` 26,171,640 B、`.dla` 26,155,123 B，两者几乎相等
   （dla 比 tflite 小 16,517 B = 去掉 flatbuffer 元数据）⇒ 权重是**原样搬进 `.dla` 的静态区**。

### 4.2 no-silu 对照：silu 的代价 = 恰好 2 层（回答"对层数的影响"）

| 图 | 算子 | 层数 | 执行步 | Temp | Static | Code | L1 |
|---|---|---|---|---|---|---|---|
| 主图 `swiglu_ffn_t8_fp16` | FC,FC,**LOGISTIC**,**MUL**,**MUL**,FC | **6** | 1 | 0 | 24M | 1.7K | 228K |
| 对照 `…_nosilu_ctrl` | FC,FC,**MUL**,FC | **4** | 1 | 0 | 24M | 1.4K | 228K |
| 差 | silu = `Sigmoid` + 一个 `Mul` | **+2** | 0 | 0 | 0 | +0.3K | 0 |

⇒ **P2 成立**：silu 不融合，**固定花 2 层**；但这 2 层**不额外付执行步、不额外付 DRAM**
（都在同一个 MDLA step 里，`Temp` 仍是 0）。**给整块的层数预算**：一层 decoder 的 FFN 段 = 6 层；
若要做 1 层整块，attention 段（QKV 投影 + RoPE + BMM + SoftMax + PV + O + 残差，见票 03–06 的实测）
与本段相加就是总层数，**这里不要再按"4 层"估**。

> 注意：对照图的 `Model` 只用于**编译侧**结论；它没有上板，也没有数值主张。

### 4.3 板端数值与噪声底

`ops/swiglu_ffn_t8_fp16/compare.txt`（两次 e2e 完全一致，第二次为存档）：

```
op        : SWIGLU_FFN {'T': 8, 'H': 896, 'I': 4864}  dtype=fp16
expected  : 7168 vals   device: 7168 vals   logical shape=(8, 896)
max_abs   : 2.441e-04
rmse      : 7.030e-05
rel_l2    : 5.476e-04
nan/inf   : exp 0/0  dev 0/0
range     : exp [-0.4827,0.5264]  dev [-0.4827,0.5264]
tol       : atol=0.002 rtol=0.01
PASS
```

**噪声底（NOTES §4 的做法）**：同一 `.dla`、同一输入连跑两次，`out_r1.bin` / `out_r2.bin` 对比：

| 量 | 值 |
|---|---|
| `max|r1 − r2|` | **0.000e+00** |
| 两次输出逐字节相同 | **True**（7168/7168） |

⇒ **硬件确定性**，噪声底为 0，按 NOTES §4 阈值退化为 **fp16 末位量级**：
`ULP16 @ max|out| = 2.441e-04`（相对 **4.64e-4**）；3–5× 即 **7.3e-4 – 1.2e-3**（相对）。
**实测误差恰好等于 1 ULP**（`max_abs = 2.441e-04 = 1·ULP16`，相对 4.64e-4），
**比"3× 退化下限"还小 3 倍**，与 §3.3 的 fp16 层边界下限（2.441e-04）**相同量级**。

**归因**（`build_swiglu_ffn.py --analyze`）：

| 参考 | max\|d\| | 相对(max) | rel_l2 | 逐位相同 |
|---|---|---|---|---|
| `expected.bin`（float64 精确 + 一次舍入） | 2.441e-04 | 4.64e-04 | 5.476e-04 | 2679/7168 |
| `expected_alt.bin`（逐张量 fp16 舍入） | 2.441e-04 | 4.64e-04 | 4.750e-04 | 3051/7168 |

⇒ 设备既不是"精确链"，也不严格是"逐张量 fp16 链"，但**两者都只差 1 个码**。可用的结论只有：
**这一段在 fp16 下与精确参考的最大偏差 = 输出量级处的 1 个 fp16 码**（相对 4.6e-4）。
"到底哪一层贡献了这 1 个码"本票**没有**逐层 dump 去分离（见 §6 UNKNOWN 1）。

**容差结论**：预登记 `atol=2e-3 / rtol=1e-2` 下 **PASS**，且**没有**事后调阈值（`tolerance_revision`
仍是 `repo-fp16-default`，与 `afcca36` 提交的值逐字相同）。为便于比较，两种口径都给：
绝对 `max_abs = 2.441e-4`（登记 atol 的 **1/8**）；相对 `max_abs/max|exp| = 4.64e-4`。

### 4.4 三个大张量的字节数（"权重驻留方式"的实测数字）

| 张量 | 形状 | bytes | MiB | 占比 |
|---|---|---|---|---|
| `layer0.mlp.gate_proj.weight` | [4864, 896] | **8,716,288** | 8.3125 | 1/3 |
| `layer0.mlp.up_proj.weight` | [4864, 896] | **8,716,288** | 8.3125 | 1/3 |
| `layer0.mlp.down_proj.weight` | [896, 4864] | **8,716,288** | 8.3125 | 1/3 |
| **三个合计** | – | **26,148,864** | **24.9375** | – |
| 三个零 bias（格式需要） | [4864]/[4864]/[896] | 11,264 | 0.011 | 0.04 % |
| `.tflite` 全文件 | – | 26,171,640 | 24.96 | – |
| `.dla` 全文件 | – | 26,155,123 | 24.94 | – |

驻留方式（实测，不是推断）：

- 编译期：`Static 24M`（DRAM），**L1 只用 228K** ⇒ 26 MB 权重**不进片内 SRAM**，每次推理从 DRAM 读；
- 板端搬运：`adb push` 26,155,123 B **0.68 s（36.7 MB/s）**，`.dla` 就是权重的载体（tflite 与 dla 几乎同大小）；
- 每层块的权重预算（给票 09/10）：一层 decoder 的 FFN 段 = **26.15 MB**；
  **24 层 = 627,572,736 B ≈ 598 MiB**（纯线性外推，未测），再加 embed/LM head 272 MB ⇒ 约 **860 MiB**；
  与 NOTES §6 的 `/data` 6.4 GB / 空闲 RAM ~273 MB 对照：**磁盘够、常驻内存不够**，整块/多层方案必须
  考虑"分图加载"或"权重按需换入"，这是票 10 的实测项，本票**只提供数字**。

---

## 5. 适用范围

**适用**：

- 算子：`FC(静态 fp16 权重) ×3 + LOGISTIC + MUL ×2`，[8,896]→[8,4864]→[8,896]，`mdla5.0` / ncc 7.3.15 / `--opt=3`；
- 数值：fp16 输出，误差界 **1 个 fp16 码（相对 ≈4.6e-4）@ |out|≈0.5**；误差随 `|out|` 线性（fp16 相对精度），
  若整块的 `|out|≈8.5`（票 01 实测），对应绝对误差约 **2e-3**，登记阈值时要按 NOTES §10.1 走；
- 层数预算：FFN 段 **6 层 / 1 执行步**（silu 不融合，固定 +2 层，`Temp` 仍为 0）；
- 权重驻留：26 MB/层 静态 DRAM，不占 L1，`.dla` 大小 ≈ `.tflite` 大小。

**不适用**：

- int8（本 spec 全 fp16；且 OP-silu §6 已实测 696 宽 int8 会卡在 16 字节行规则）；
- `T ≠ 8`（只测 T=8）、多 batch、paged/prefill 形态；
- 动态权重（运行时换权重，走 E-005 路径，本票未测）；
- 真实权重（本票全部用冻结随机夹具；"随机权重 PASS"**不能**推出真权重下 logits 一致，见 OP-weights §8.2）；
- 任何语义/精度结论（top-1、困惑度）。

---

## 6. 剩余疑问（UNKNOWN，不圆）

1. **这 1 个码来自哪一层**：设备 vs `exact` 与 vs `alt` 的 `max|d|` 都是 2.441e-4，无法只凭最终输出区分
   "LOGISTIC 的 1 码 LUT 误差"（OP-silu 实测 sigmoid 单算子在 696/704 宽上不是逐位精确）与"某个 MUL 的舍入"。
   要分离需要逐层 dump（把中间张量做成模型输出），本票没做。
2. **FC 的累加精度**：本票只测到"整段 = 1 码"。E-004 说单 FC fp16 是**逐位**的（fp32 累加 + 一次舍入），
   但 4864 宽的 K 上是否仍是 fp32 累加、累加是否有分块求和顺序差异，未单独验证。
3. **`Temp=0` 的实际含义**：`.dla` 里说 Temp=0，但设备上 26 MB 权重每次推理都要从 DRAM 流一遍；
   "权重不占 L1"是否意味着每层都有 DRAM 带宽压力，**没有性能对照**（本票只有 2.16/2.23 ms/inf 的量级数字，
   含进程内提交开销、无控温、无对照实现）。
4. **24 层 × 26 MB = 598 MiB 的可加载性**：纯字节外推，未测；NOTES §6 的空闲 RAM ~273 MB 提示这是真风险。
5. **`.dla` 里的权重是否被重新排版/压缩**：tflite 与 dla 大小差 16.5 KB，只能说明"没有整体压缩"，
   看不出是否做了分块/重排（未逆解析）。

---

## 7. 复现

```bash
cd "/d/deepseek_ workspace/mtk/wt-t07"
# 0) 夹具（本 worktree 内没有 .npz 时生成一次，约 8 s；sha256 与 manifest 一致）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t07' && /usr/bin/python3 tools/gen_qwen25_weights.py \
  && /usr/bin/python3 tools/qwen25_weights.py --verify"

# 1) 生成 model.tflite + golden + model.json（锁外）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t07' && /usr/bin/python3 tools/build_swiglu_ffn.py \
  swiglu_ffn_t8_fp16 --out ops/swiglu_ffn_t8_fp16"

# 2) 编译（锁外，~2.8 s）：exec plan / 层数 / Static / EXIT 全进 model.compile.log
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t07' && \
  ./tools/compile.sh ops/swiglu_ffn_t8_fp16/model.tflite ops/swiglu_ffn_t8_fp16/model.dla"

# 3) 对照（只编译，不上板）：去掉 silu 的图是 4 层
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t07' && \
  /usr/bin/python3 tools/build_swiglu_ffn.py swiglu_ffn_t8_fp16_nosilu_ctrl --out ops/_swiglu_ffn_t8_fp16_nosilu_ctrl \
  && ./tools/compile.sh ops/_swiglu_ffn_t8_fp16_nosilu_ctrl/model.tflite ops/_swiglu_ffn_t8_fp16_nosilu_ctrl/model.dla"

# 4) 板端（持锁）：healthcheck -> e2e -> 存 out_r1 -> e2e -> 存 out_r2 -> healthcheck
export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
./tools/devlock.sh run -- bash -c './tools/healthcheck.sh --quick; \
  ./tools/e2e.sh ops/swiglu_ffn_t8_fp16; cp ops/swiglu_ffn_t8_fp16/out_device.bin ops/swiglu_ffn_t8_fp16/out_r1.bin; \
  ./tools/e2e.sh ops/swiglu_ffn_t8_fp16; cp ops/swiglu_ffn_t8_fp16/out_device.bin ops/swiglu_ffn_t8_fp16/out_r2.bin; \
  ./tools/healthcheck.sh --quick'

# 5) 噪声底 + 两种参考 + 相对口径
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t07' && \
  /usr/bin/python3 tools/build_swiglu_ffn.py --analyze ops/swiglu_ffn_t8_fp16"
```

> `tools/e2e.sh` 在 `model.tflite` 缺失时会回落到 `tools/build_ops.py`（它不认识本 op）。
> 复跑前若删了 `model.tflite`/`model.dla`，先按上面第 1 步重建，或把它加进 `tools/rebuild_ops.sh`
> 的映射表（**公共文件，留给 Lead**，见 §8）。

---

## 8. 与票面 checklist 对照 + 交接

| 票面要求 | 状态 | 证据 |
|---|---|---|
| gate/up 两条 4864 宽投影 + `silu(gate)*up` + down 投影，fp16 与 golden 对上 | ✅ | §4.1 层清单；§4.3 `compare.txt` PASS（max_abs 1 ULP、rel_l2 5.48e-4） |
| 记录激活是否被融合进前一个全连接、对层数的影响 | ✅ | §4.1（`Sigmoid`/`Mul` 是独立层，未融合）+ §4.2 对照图 4 层 ⇒ **silu = +2 层**，仍在同一执行步、`Temp=0` |
| 单执行步、无 bridge | ✅ | §4.1 `===Graph[0] has 1 Execution Step(s)===`、`--disallow-bridge` 下 `EXIT=0`、`Target: MDLA_5_0` |
| 记录三个大张量的字节数，为"权重驻留方式"提供实测数字 | ✅ | §4.4：各 8,716,288 B，合计 26,148,864 B；`Static 24M`(DRAM)、L1 228K、push 0.68 s |
| 记录本算子的实测噪声底 | ✅ | §4.3：`max|r1−r2| = 0.000e+00`（逐字节相同）⇒ 阈值退化到 1 ULP=2.441e-4（相对 4.64e-4） |

**需要 Lead 处理**：

1. `docs/EXPERIMENTS.md` / `docs/CAPABILITY.tsv` / `docs/OP-MATRIX.md` 登记本票（占号留给 Lead；
   建议一行：`SWIGLU_FFN(FC×3+LOGISTIC+MUL×2)` fp16 ✅ 6 层/1 step/Temp 0/1 码）；
2. `tools/rebuild_ops.sh` 的映射表要加 `swiglu_ffn_t8_fp16` → `tools/build_swiglu_ffn.py`
   （否则删了 `model.tflite` 后 `e2e.sh` 会去调 `build_ops.py` 而失败，同 OP-silu §11 第 3 条）；
   `ops/_swiglu_ffn_t8_fp16_nosilu_ctrl` 是**只编译**的对照，不应进 `run_all.sh` 的板端回归；
3. 票 09 的层数预算：FFN 段按 **6 层**算（silu 不融合），并注意 `Temp=0` 依赖"中间张量 ≤ L1 228K"这个现状；
4. 票 10 的权重预算：24 层 × 26.15 MB ≈ **598 MiB** 静态权重（未测可加载性），与 NOTES §6 的空闲 RAM 冲突，
   需要决策"分图/换入"策略。

**本票未改公共文件**：`tools/e2e.sh`、`tools/compare.py`、`tools/compile.sh`、`tools/devlock.sh`、
`tools/healthcheck.sh`、`tools/build_ops.py`（只 import）、`probe/`、`PLAN.md`、`README.md`、
`docs/EXPERIMENTS.md`、`docs/CAPABILITY.md`、`CONTEXT.md`、`AGENTS.md` 全部原样。
