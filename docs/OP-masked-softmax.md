# OP-masked-softmax —— 因果掩码 + 原生 SoftMax（Qwen2.5-0.5B · T=8 · 14Q/2KV · fp16）

> 范围：**带因果掩码的 softmax 单算子**（票 05）。头文件之外的注意力（QKV 投影 / RoPE / GQA 复制 / PV）见
> `docs/OP-gqa.md` 与票 03/04；整块拼装见票 09。本文件不占公共实验号（`docs/EXPERIMENTS.md` 由 Lead 维护），
> 内部编号 **MSM-01…MSM-07**，可直接折进公共索引。
>
> · 分支 `impl/05-masked-softmax`（worktree `wt-t05`）· 基线 `feat/qwen25-attention`
> · 交付 `ops/masked_softmax_t8_fp16/` + `tools/build_masked_softmax.py`（新增）+ 本文
> · 权重夹具：ticket 02 冻结夹具（`weights_digest=898fa319…`，`QWEN25_WEIGHTS_DIR` 指向主检出的那份，已 `--verify` 25/25）
> · 编译器 ncc-tflite 7.3.15 / `--arch=mdla5.0 --opt=3 --disallow-bridge`；设备 mt6897（`0123456789ABCDEF`）/ Android 16
>
> **一句话结论**：`MUL(1/8) → ADD(掩码) → 原生 SoftMax` 在 MDLA_5_0 上**单执行步、无 bridge**，
> 板端 `compare.txt` **PASS**（max_abs 4.883e-04 = 2 个 fp16 code，噪声底实测 **0**）；
> **392/392 个被掩码位置的输出恰好等于 0.0**；掩码 −4096 与 −8192 的两份板端 dump **逐位相同**（有限掩码已饱和，
> 等价于理想 −inf）；非掩码位置与"精确剔除"的 golden 一致。**副产物**：票面/spec 的前提"推理路径里没有 EXP 算子"
> 经配对正对照实测**不成立**（EXP 是原生 `ExpLayer`），真正把方案钉死的是 fp16 的 exp 值域（见 §3.2）。

---

## 0. TL;DR

| 项 | 结果 |
|---|---|
| 图 | `scores[14,8,8] --MUL(1/8 const)--> --ADD(mask)--> SOFTMAX(beta=1)--> probs[14,8,8]`（fp16） |
| 掩码 | 有限负常数 **−4096 = −2¹²**（0 在可见位置），**加在 scale 之后**；−inf / −10000 均不用（§3.4） |
| 两个运行时输入 | `input.bin` = 原始 `q·k`，`input1.bin` = 掩码 → 同一个 `.dla` 能跑"真掩码 / 双倍幅度 / 全 0"三档 |
| 编译 | **EXIT=0**，`ElementWiseMulLayer + ElementWiseAddLayer + SoftMaxLayer`，**1 Execution Step @ MDLA_5_0**，无 bridge；`SIZE=5233`（常量掩码形态 `SIZE=6868`，同样 1 步） |
| 板端主判据 | `max_abs 4.883e-04`、`rmse 8.165e-05`、`rel_l2 3.125e-04`、nan/inf **0/0**、**PASS** |
| 噪声底（同 .dla 同输入两次） | **0.000e+00**（0/896 个 fp16 code 不同）；再加 run1 vs run3 也是 0 |
| 阈值 | `atol=1e-3` / `rtol=5e-3`（板端之前登记在 `model.json`；实测最大偏差是它的 1/2 与 1/7.6） |
| 判据一：被掩码位置 = 0 | **392/392 逐位等于 0.0**（读设备 dump 直接断言，不经过容差） |
| 判据二：非掩码一致 | 与"精确剔除"golden：可见位置 319/504 逐位相同、最大 2 个 code；"掩码幅度翻倍"→ **0/896 个 code 不同** |
| 第二个 regime | layer1（每头 offset 只有 ±45）：同样 PASS、392/392 掩码位为 0、最大 2 code |
| 生产形态 | 掩码烘成常量（1 个运行时输入）：与运行期掩码形态 dump **逐位相同**（0/896） |
| 负面控制 | 手写 `MUL+EXP` 链：**640/896 个输出溢出成 inf**（板端与 golden 都是 640）→ 这条路在真实量级下不可用 |
| ✱ 前提更正 | **EXP 可编译**（`ExpLayer`，1 步，EXIT=0）；LOG / COS 不支持（MDLA+EDPA 都不支持）。"没有 EXP"这句话是**假设，不是事实** |

---

## 1. 这一票要回答什么

票面 6 条（`.scratch/qwen25-attention/issues/05-masked-softmax.md`）：

1. 掩码加在 scale **之后**（顺序影响量级，必须记清楚）；
2. 掩码取受控有限负值（不用 −inf，不用 −10000）；
3. **判据一**：被掩码位置的 softmax 输出为 0；
4. **判据二**：非掩码位置与无掩码版本的输出一致；
5. 整张量与 golden 的偏差在阈值内，阈值由本算子的**噪声底实测**决定；
6. 记录"无 EXP 算子"这一约束，以及它如何把方案收敛到唯一形态。

背景（`spec.md` §Problem 2）：仓库有原生 `SoftMaxLayer`（β 生效），但**带掩码的 softmax 从未上过设备**；
`docs/OP-MATRIX.md` §5.1 把它列为"未覆盖 / 下一步"。所以本票是这个问题第一次有实测答案。

---

## 2. 事前登记：假设与预测（写于板端之前，逐条复核在 §5）

| # | 预测 | 若不成立意味着 |
|---|---|---|
| P1 | 编译 `EXIT=0`、单执行步、三层原生（MUL/ADD/SoftMaxLayer）、无 bridge | 掩码加法把 softmax 挤出 DLA，方案要换形态 |
| P2 | 被掩码位置的设备输出**恰好** 0.0（全部 fp16 位为 0） | 掩码压不下去 → −4096 不够，或设备内部 exp 精度比 fp32 更高 |
| P3 | 掩码 −4096 与 −8192 两份 dump **逐位相同** | 有限掩码还没进入饱和区 → 值选择依赖数据，需重新推导 |
| P4 | 掩码全 0 时退化为普通 softmax；且"掩码 run / 零掩码 run"在可见位置的比值 = 每行的 `Σ_all/Σ_vis`（常数） | 掩码泄漏到了可见位置，或 ADD 不是纯加法 |
| P5 | 噪声底 = 0（同 .dla 同输入两次逐位相同） | 硬件不确定 → 阈值要按 §4 放大 |
| P6 | `|v|max=513.8`（含每头 ±500 的 bias 偏置）不会把原生 softmax 打成 NaN/inf | 设备不做 max 减法且 exp 用 fp16 → 真实块也需要前置归一化 |
| P7 | 票面前提"没有 EXP 算子"**不成立**：EXP 应能编译成原生层（配正对照），LOG/COS 应被后端拒 | 若 EXP 真的不可用，则"唯一形态"的论证只需一条；若可用，论证必须换成"fp16 值域 + 层预算"（§3.2） |

> 观测数据（板端之前）：`layer0` 的 `q·k` 由 `q_proj.bias`(std 7.80) 与 `k_proj.bias`(std 29.78) **主导**，
> `|q·k|max = 4110.7`，scale 后 `|u|max = 513.8`，每头常数偏置落在 **−503.7…+458.0**，头内逐行离散只有 std≈1.19。
> 这正是 `docs/OP-weights.md` §4 警告的"别用 `initializer_range=0.02` 估动态范围"的后果。

---

## 3. 做法

### 3.1 图与形状

```
scores [14,8,8] fp16 ──MUL(scale=1/8, 常量 [14,8,8])──> scaled ──ADD(mask, 第二输入)──> masked ──SOFTMAX(β=1)──> probs [14,8,8]
```

* `[H,T,T] = [14,8,8]`：14 个 query head、T=8、每行 8 个 fp16 = **16 B**，正好满足"最内层行字节数 ≥16 且 %16==0"；
  整张量 1792 B = 112×16。
* 走 **3-D**（`docs/OP-bmm.md`：4-D 会被降成多层，3-D 才是 1 个原生层）。
* `scale = 1/8 = 2^-3`：`1/sqrt(head_dim) = 1/sqrt(64)`。**2 的幂在 fp16 里是精确乘法**（只可能上/下溢），
  所以在可见位置 `(mask=0)` 上，softmax 的输入与"理论上的 `u = s/8`"**逐位相同**——scale 与掩码都不引入舍入。

### 3.2 "无 EXP 算子"这条约束：先测试，再使用（MSM-01）

`docs/OP-rope.md` §"SIN/COS 被拒"里写过"不要试图用 EXP/LOG 拼三角函数"，spec 把它继承成了
"推理路径里没有 EXP 算子"。这条是**假设**，从来没在 EXP 本身上测过。本票把它变成实测，并且**每个探针都配正对照**：

| 探针 | 编译结果 | 落点 |
|---|---|---|
| **EXP** | **EXIT=0，1 Execution Step** | `ExpLayer<0>{EXP}` @ MDLA_5_0 |
| LOG | EXIT=1 | `MDLA: Cannot support Float16 input/output` + `EDPA: unsupported operation`（**只说明 fp16 下不可用**，别的 dtype 未测） |
| COS | EXIT=1 | `>> Unsupport COS (version: 1)` / MDLA+EDPA 都不支持 |
| SOFTMAX（正对照） | EXIT=0，1 步 | `SoftMaxLayer<0>` |
| RSQRT（正对照） | EXIT=0，1 步 | `RSqrtLayer<0>` |
| LOGISTIC（正对照） | EXIT=0，1 步 | `SigmoidLayer<0>` |

日志：`ops/masked_softmax_t8_fp16/probe_{exp,log,cos,softmax,rsqrt,logistic}.compile.log`。

**结论（MSM-01）**：**"推理路径里没有 EXP"是错的**。EXP 是 MDLA 原生算子；真正不被支持的是 LOG 和 COS。
→ 那么"唯一形态"的论证不能建立在"没有 EXP"上，必须换成下面两条**实测**理由：

1. **fp16 的 exp 值域不够**：fp16 的 `exp` 在输入 > ln(65504) = **11.09** 时上溢成 inf。
   真实分数 scale 后 `|u|` 最大 **513.8**（每头偏置 ±500 主导），所以"手写 exp 链"（`MUL → EXP`）
   在真实量级下 **640/896 = 71% 的输出直接变成 inf**（板端实测，§4.8）。要救它必须先在行内做 **max 减法**，
   而仓库已验证的 reduce 只有 **MEAN / SUM**（`MAXIMUM` 是逐元素、不支持广播），行内 max 要靠
   PAD/SLICE + MAXIMUM 树手搭（≥6 层），再拼 EXP、行和、除法——**至少 10 层**，且数值要自己重推。
2. **原生 SoftMax 只有 1 层，且实测能吸收这个量级**：零掩码档喂进去 `|v|max=513.8` 依然有限、PASS（§4.6），
   说明它的内部稳定化（max 减法或 ≥fp32 的 exp）替我们做掉了第 1 条里那 10 层的活。

→ **唯一形态**：`MUL → ADD(mask) → 原生 SoftMax`。掩码只能是**加在分数上的有限负常数**，没有第二种写法。

> **方法学附注（值得记，因为它伪装成了"编译器限制"）**：第一版探针把 `bo.opcode()` 返回的**表偏移**
> 当成了 `bo.operator()` 要的 **opcode 索引**，于是 `SOFTMAX` 的正对照也报
> `ERROR: Fail to process the given model`（加载期）。如果只测 EXP、不配正对照，这个 bug 会被写成
> "EXP 被 importer 拒绝"——**和 E-112 那条假限制是同一类错误**。修好后 EXP 才露出 `ExpLayer`。

### 3.3 掩码为什么必须在 scale **之后**相加（MSM-02）

设原始分数 `s = q·k`（`|s|max = S_raw = 4110.7`），缩放后 `u = s/8`（`|u|max = S = 513.8`）。

| | 掩码加在 **scale 之后**（本票） | 掩码加在 scale 之前（对照） |
|---|---|---|
| softmax 看到的被掩码值 | `−M + u ∈ [−M−S, −M+S]` | `(−M + s)/8`，等效掩码只有 `M/8` |
| 要赢过谁 | 缩放后的行最大（≈ +467） | 未缩放的行最大（≈ +4110） |
| 所需 M（推导见 §3.4） | `M ≥ 2S + 104 = 1131.7` | `M ≥ 2·S_raw + 8·104 = 9053.4`（**8 倍**） |
| 实际取值与余量 | `M = 4096 = 2^12`，**3.62×** | 得取 `2^14 = 16384`，余量只有 **1.81×** |
| 中间量 `|v|max` | 4609.5 → 距 fp16 上溢 **14.2×** | 20494.7 → 只剩 **3.0×** |

一句话：**掩码加在 scale 之后，等于让掩码和"缩放后的分数"比大小，量级差 8 倍**；加在前面白扔 8 倍余量，
还把 fp16 的动态范围预算花掉 3/4。这就是票面说的"顺序影响量级"的定量版本。

### 3.4 掩码值 −4096 的推导（为什么不是 −inf，也不是 −10000）（MSM-02）

**目标**：让被掩码位置的输出在 fp16 下**确定性**地等于 0，且所有中间量保持有限。

记 `v = u + m`（`m ∈ {0, −M}`），`rowmax` 是该行（可见位置）的最大值。

1. 被掩码值的界：`v_mask ≤ −M + S`。
2. 设备内部两种可能的稳定化策略，取更严的一支：
   * 不做 max 减法：指数 `p = v ≤ −M + S`；
   * 做 max 减法：`p = v − rowmax ≤ (−M + S) − (−S) = −M + 2S`（因为 `rowmax ≥ −S`）。
   ⇒ 要求 `−M + 2S ≤ T₀`。
3. `T₀` 取**让 `exp` 恰好下溢为 0** 的阈值：fp16 最小次正规数 `5.96e-8` → `T₀ = ln(2⁻²⁵) = −17.33`；
   fp32 最小次正规数 `1.4e-45` → `T₀ = −103.3`。取更严的 **`T₀ = −104`**（这样即使内部用 fp32 做 exp 也精确为 0，
   不必猜设备精度）。分母 `Σexp(v−rowmax) ≥ 1`，所以 exp 一旦为 0，输出就**恒为 0**。
4. ⇒ **下界** `M ≥ 2S + 104 = 2×513.84 + 104 = 1131.7`。
5. **上界**：不能上溢/不能把掩码位变成正数 → `M > S` 且 `M + S ≤ 65504` → `M ≤ 64990`。
6. **取 `M = 4096 = 2^12`**：比下界高 **3.62×**；实测最坏 `(v_mask − rowmax) = −4089.8`，相对 `T₀ = −104`
   是 **39.3×** 余量（相对 fp16 的 −17.33 是 235×）；`|v|max = 4609.5 ≪ 65504`。取 2 的幂是为了**精确可表示**，
   且便于审计（`−4096` 没有表示误差）。

**为什么不用 −inf**：−inf 把数值推进 IEEE 特殊值域，任何一处 `inf − inf` / `inf × 0` 都会变 NaN，属于未定义行为面；
本方案里所有中间量都是有限值（`|v| ≤ 4609.5`），NaN 的可能性被**构造性**排除。

**为什么不用 −10000**：它在数值上并不"错"（`10000 < 64990`），错在没有推导——它比下界大 8.8 倍，
白占 2.2 倍的 fp16 动态范围预算，且"掩码管用"与"掩码大得离谱"这两件事无法区分。
本票把"挑一个常数"变成"**挑一个区间 + 证明区间内部不敏感**"：区间 `[1132, 64990]`，再用 §4.5 的
−4096 vs −8192 **逐位相同**证明已进入饱和区。于是常数的具体取值不再是一个可以赌的旋钮。

### 3.5 两个运行时输入：为什么把掩码做成 `input1`

`model.tflite` 的子图输入是 `[scores, mask]`（顺序即 e2e 的 `-i input.bin -i input1.bin`）。这样**同一份 `.dla`**
能跑四种掩码（真掩码 / 双倍幅度 / 全 0 / 另一档数据），把"判据一""判据二""幅度不敏感"三个实验压成
**一个编译产物 + 四次板端运行**，而不是再造三张图。生产形态（掩码烘成常量、1 个运行时输入）另编一份
`model_constmask.tflite/.dla` 作为对照证据（§4.7），它是票 09 真正要用的形态。

### 3.6 输入数据从哪来（不许猜动态范围）

`golden/input.bin` 不是随手写的随机数，而是**用 ticket 02 的冻结夹具真算出来的注意力分数**：

* `x = `单位 RMS 随机方向 × 该层 `input_layernorm` 实测增益（layer0 γ mean 0.034 / std 0.094 → 注意力输入 RMS≈0.102，
  与真实模型同量级）；
* `q = x·Wqᵀ + bq`、`k = x·Wkᵀ + bk`（**带 bias**，`docs/OP-weights.md` §4 的 std 7.80 / 29.78），
  GQA 把 2 个 KV head 复制到 14，`s = q·k`（**还没 scale**，因为 scale 在图里）；
* 两档数据：`layer0`（偏置主导，`|u|max=513.8`，最严）与 `layer1`（`|u|max=74.4`，偏置只有 ±45，概率更"散"）；
* RoPE 不在本票内：它是正交旋转，不改变 `|q|`/`|k|`，也就不改变这里要定的**量级**。

实测（板端之前，`python3 tools/build_masked_softmax.py` 会重打印）：

| regime | `|q·k|max` | `|u|max` | 每头常数偏置 | 头内逐行 std | 用到的掩码余量 |
|---|---|---|---|---|
| layer0 | 4110.7 | **513.8** | −503.7 … +458.0 | 1.19 | `(−4089.8)/104 = 39.3×` |
| layer1 | 595.3 | 74.4 | −45.9 … +31.4 | 0.70 | 更宽松 |

### 3.7 golden 独立性（R3）与"参考实现不预设设备策略"

`golden/expected.bin` 由 numpy/float64 从**落盘后的 fp16 输入字节**重算，与写 flatbuffer 的代码**零共享**：

```
u = fp16(input) * 2^-3            # 精确
v = u + fp16(mask)
p = exp(v) / Σexp(v)              # 规范形式，**不做 max 减法**
expected = fp16(p)
```

刻意**不做** max 减法：这样参考不预设设备内部策略（`|v| ≤ 514` 在 float64 里 exp 既不上溢也不下溢）。
交叉校验：max-减法版本与规范形式的最大差 **2.220e-16**（float64 舍入级），说明两种写法在这个量级上等价，
参考本身是可靠的。被掩码位置在 float64 里 `exp(−3582…−4096) = 0`，所以 `expected` 在 392 个掩码位上是**精确 0**，
构建脚本里有一条断言守住这件事。

---

## 4. 结果

### 4.1 编译（MSM-03）

```
===Graph[0] has 1 Execution Step(s)===
Graph[0] Step[0]
ExecutionStep
 ├ StepId: 0
 ├ Target: MDLA_5_0
 └ Subgraph:
   ├ ElementWiseMulLayer<0>{MUL}
   ├ ElementWiseAddLayer<1>{ADD}
   ├ SoftMaxLayer<2>{SOFTMAX}
   └ Output: OpResult (external)
DRAM [0]: MDLA 5.0  Input 3.5K  Output 1.8K  Temp 0  Static 16  Code 1.4K  Total 6.7K
EXIT=0   SIZE=5233   SHA256=cc3d62b9199efa1f…
```

常量掩码形态（`model_constmask.compile.log`）层清单**完全相同**、同样 1 步：`EXIT=0 SIZE=6868`。
（**.dla 不可字节复现：同一 tflite 连编多次 SIZE/SHA256 会差几个字节，以该目录 model.compile.log 为准；回归只比数值**。`docs/OP-elementwise.md` 已实测同 tflite 连编三次哈希不同。）

### 4.2 板端主判据：`ops/masked_softmax_t8_fp16/compare.txt`

```
op        : MUL+ADD+SOFTMAX (masked softmax) {'H': 14, 'T': 8, 'KVH': 2, 'HD': 64, 'D': 896}  dtype=float16
expected  : 896 vals   device: 896 vals   logical shape=(14, 8, 8)
max_abs   : 4.883e-04
rmse      : 8.165e-05
rel_l2    : 3.125e-04
nan/inf   : exp 0/0  dev 0/0
range     : exp [0,1]  dev [0,1]
tol       : atol=0.001 rtol=0.005
PASS
```

### 4.3 判据一：被掩码位置的输出**恰好**为 0（MSM-04）

直接读设备 dump，**不经过容差**（`atol` 允许 1e-3，所以必须单独断言）：

```
J1 masked == 0   : 392/392 masked outputs are exactly 0.0 ; max |masked out| = 0.000e+00
```

layer1 档同样：`392/392`，`max |masked out| = 0.000e+00`。

### 4.4 判据二：非掩码位置与"无掩码版本"一致（MSM-04）

票面把这条写成"非掩码位置与无掩码版本完全一致"。**先把它说清楚**：softmax 在遮掩处是"把被掩码项从分母里去掉"，
所以拿"真掩码 run" 和"全 0 掩码 run"在可见位置做**逐位比较在数学上不可能相等**——分母不同。实测把这个差
定量化了（同一 `.dla`、同一 scores、只换 `input1`）：

```
zero-mask run    : vs plain-softmax golden max_abs = 4.883e-04
                   per-row S_all/S_vis prediction: max rel dev = 1.475e-03  predicted ratio range [1.0000, 1563.4461]
                   literal 'primary == zero at visible positions' would need ratio==1:
                   measured ratio range [1.0000, 1563.8717]
```

—— 两档之间的比值**逐行是常数**，且与独立预测 `Σ_all/Σ_vis` 的偏差只有 **1.475e-03**（= fp16 末位量级）。
这恰好是比"相等"更强的命题：**掩码只从分母里拿掉了被掩码项，别的什么都没动**。

于是判据二按**它真正想问的**（"能不能把掩码错误和 softmax 错误分开"）分三支实测：

| 支 | 测什么 | 结果 |
|---|---|---|
| **2a（主）** | 可见位置 vs **精确剔除**的 golden（= 理想 −inf 掩码，被掩码项**不存在**） | max_abs **4.883e-04**，319/504 逐位相同，最大 **2 个 fp16 code**，mean|dev| 4.06e-05 |
| **2b（更强）** | 掩码幅度翻倍（−4096 → −8192）后两份 dump 是否相同 | **0/896 个 code 不同**（= 有限掩码已饱和，与理想 −inf 不可区分） |
| **2c（对照）** | 全 0 掩码是否退化为普通 softmax | 对普通 softmax golden **PASS**（`compare_zero.txt`，max_abs 4.883e-04） |

```
J2 visible vs    : golden(visible-only softmax) max_abs = 4.883e-04 at (0, 3, 0) (dev=0.738281 exp=0.738770) ; rel = 6.61e-04
                   visible positions: 319/504 bit-exact, max fp16 code distance = 2, mean |dev| = 4.06e-05
J2 mask magnitude: -4096 vs -8192 dump  max_abs = 0.000e+00 ; differing fp16 codes 0/896
```

### 4.5 第二个 regime（layer1）与生产形态（MSM-04/05）

```
layer1 regime    : visible max_abs = 4.883e-04 ; masked exactly 0: 392/392 (max 0.000e+00) ;
                   visible bit-exact 325/504, max code distance 2
const-mask run   : vs runtime-mask dump max_abs = 0.000e+00 (fp16 codes differing 0/896)
                   -> production form (mask baked as constant) is the same op
```

layer1 的每头偏置只有 ±45、行内更"散"（概率从 1 覆盖到 7e-5），结论与 layer0 一致 → 不是单一样本侥幸。
常量掩码（票 09 要用的形态）与运行期掩码**逐位相同**，说明"把掩码做成第二输入"这个测试手法本身不改变数值。

### 4.6 量级上限：`|v|max = 513.8` 没有把设备 softmax 打坏

零掩码档的输入里 `|v|` 最大 513.8（每头偏置 ±500 主导）。若设备不做内部稳定化、且 exp 用 fp16，
`exp(467.5)` 上溢 → NaN/inf。实测：**nan/inf 0/0、PASS**（`compare_zero.txt`）。
所以这个量级下的 softmax 是可用的；设备究竟靠"max 减法"还是"fp32 exp"实现，**从外部不可区分**（见 §9.3）。

### 4.7 负面控制：手写 exp 链在真实量级下不可用（MSM-06）

同一份 scores，只把 `ADD+SOFTMAX` 换成 `EXP`（`probe_expchain.tflite` = `MUL(1/8) + ExpLayer`，编译 1 步 EXIT=0）：

```
exp-chain probe  : NEGATIVE CONTROL, exp(s*1/8) with no max-subtraction ->
                   golden 640/896 inf, device 640/896 inf (fp16 exp saturates above 11.09)
compare_expchain.txt: nan/inf : exp 0/640  dev 0/640 ; range exp [0,inf] dev [0,inf] ; FAIL
```

设备与 numpy 都数出**同样的 640 个 inf** → 不是编译器/驱动问题，就是 fp16 的 exp 值域不够。
这条把 §3.2 的论证从"推理"变成"实测"：**手写 exp 链在真实分数尺度上 71% 溢出**，
原生 SoftMax 是唯一现实形态。

### 4.8 运行元数据

`e2e.json`: `{"op":"masked_softmax_t8_fp16","repeat":3,"compare_exit":0,"ts":"2026-11-13T10:25:52+08:00"}`；
板端 3 次重复约 0.85 ms/次（mag2 档 1028 FPS）——性能不在本 spec 范围，只作记录。
板端实验前后各一次 `healthcheck.sh --quick`：**PASS / PASS**。

---

## 5. 预测复核

| # | 预测 | 结果 |
|---|---|---|
| P1 | 1 步、三层原生、无 bridge | ✅（§4.1） |
| P2 | 被掩码位置恰好 0 | ✅ 392/392（§4.3） |
| P3 | −4096 vs −8192 逐位相同 | ✅ 0/896（§4.4） |
| P4 | 零掩码退化为普通 softmax；比值 = 每行常数 | ✅ 偏差 1.475e-03（§4.4） |
| P5 | 噪声底 = 0 | ✅ 0.000e+00（§6） |
| P6 | `|v|max=513.8` 不打坏设备 softmax | ✅ 无 nan/inf（§4.6） |
| P7 | "没有 EXP"不成立；EXP 原生、LOG/COS 不支持 | ✅ **前提被推翻**（§3.2），设计论证已改写 |

---

## 6. 噪声底与阈值

按 `NOTES.md` §4：**同一个 `.dla`、同一个输入、两次独立 e2e**，各存一份输出再比。

```
noise floor : max|r1-r2| = 0.000e+00 (fp16 codes differing: 0/896)
              run1 vs run3（另一次独立运行）: max = 0.000e+00 ; 0/896 codes
```

**噪声底 = 0（硬件确定性）** → 按 `NOTES.md` §4 的退化规则处理：阈值退化为 fp16 末位量级（约 1e-3 相对）。
登记值（**板端之前**就写在 `model.json`，可在 commit `e535cb1` 里核对，不是事后改的）：

| 项 | 值 | 依据 |
|---|---|---|
| `atol` | **1e-3** | 0.125 处 1 code = 1.22e-4，1e-3 ≈ 8 code；同时必须覆盖"期望值恰为 0"的掩码位 |
| `rtol` | **5e-3** | 0.125 处相对 1 code = 8.5e-4，5e-3 ≈ 5.9 code |
| 实测最大偏差 | 4.883e-04（= 2 code @0.7388，相对 6.61e-04） | 是 atol 的 1/2、rtol 的 1/7.6 |
| 上限约束 | 两者都 **< 2e-2** | spec 的硬要求 |

注：`atol` **不承担**"掩码是否为 0"的判据——那由判据一（逐位等于 0）独立承担，§4.3。

---

## 7. 与票面 checklist 对照

| 票面要求 | 状态 | 证据 |
|---|---|---|
| 掩码在 scale **之后**相加，顺序记清楚 | ✅ | §3.3（8 倍量级差 / 余量 3.62× vs 1.81× 的表）；图中 `MUL → ADD → SOFTMAX` 的顺序见编译层清单 |
| 掩码取受控有限负值，不用 −inf、不用 −10000 | ✅ | §3.4：区间 `[1132, 64990]` 推导 + 取 `2^12`；−inf 的 NaN 面与 −10000 的"无推导"逐条说明；§4.4 的幅度不敏感实验把"常数值"变成"区间内不敏感" |
| 判据一：被掩码位置输出为 0 | ✅ | §4.3，**392/392 逐位 0.0**（layer0 与 layer1 都有） |
| 判据二：非掩码位置与无掩码版本一致 | ✅ | §4.4 的 2a/2b/2c 三支；并**更正**票面的字面表述（dump-vs-dump 相等在数学上不成立，实测比值 1563） |
| 整张量偏差在阈值内，阈值由噪声底实测决定 | ✅ | §6：噪声底 0 → 退化规则；`compare.txt` PASS，max_abs 4.883e-04 |
| 记录"无 EXP"约束及其如何收敛到唯一形态 | ✅ **且被更正** | §3.2：EXP 实测可用（`ExpLayer`），LOG/COS 不可用；唯一形态的论证换成"fp16 exp 值域 + 层预算"，并由 §4.7 的 640/896 inf 实测支撑 |
| （任务书的额外要求）`compare.txt` PASS + 噪声底 + docs 记录 + commit | ✅ | `15c11cf` / `e535cb1` |

---

## 8. 适用范围 / 不适用

**适用**：
* 票 06/09 的 attention 链：T=8、14Q/2KV、head_dim=64、fp16、scale 后 `|u| ≤ 500` 量级（当前夹具的 layer0 最严）；
* 掩码形态：`ADD(0/−M)` + 原生 `SoftMax(β=1)`，`M ∈ [2S+104, 65504−S]`（本数据下 `[1132, 64990]`）；
* 常量掩码（1 输入）与运行期掩码（2 输入）两种形态数值等价（§4.5）。

**不适用**：
* 任何语义/精度结论（top-1、困惑度）；本票只回答"掩码能不能在设备上成立"。
* `T ≠ 8`：T=128 未测（掩码张量会是 `[14,128,128]` = 458 KB/输入）。
* **int8**：`[14,8,8]` 的 int8 行只有 8 B < 16 B，按 `docs/OP-MATRIX.md` §2.1 的 16 字节行规则**预计会 bridge**，
  本票未测也不建议在 int8 下用这个形状（全 fp16 是 spec 的决定）。
* 真实权重档：本票用的是冻结随机夹具（尺度取自真实权重实测），真权重下 `k_proj.bias` 的 abs_max 是 130 而非 77.8，
  `S` 会更大——见 §9.1 的有效区间复核条件。
* 性能/功耗。

---

## 9. 剩余疑问（UNKNOWN，不圆）

1. **掩码有效区间的数据依赖**：`M = 4096` 允许 `S` 长到 `(4096−104)/2 = 1996`（当前 513.8 的 **3.9 倍**）才失效。
   真权重档 `k_proj.bias` abs_max 130（夹具 77.8，比值 1.67×）→ 保守估计 `S` 可能到 ~900，仍在区间内，
   但**未实测**。票 09/真权重档必须重新量一次 `|u|max`；若 `S > 1996`，掩码要按 §3.4 重新推导。
2. **设备内部到底怎么稳定化的**：`|v|max=513.8` 下无 NaN，只能说明"要么做了 max 减法、要么 exp 用 ≥fp32"，
   从外部无法区分（§4.6）。对设计无影响（§3.4 取了更严的一支），但**不能说成"已验证设备做了 max 减法"**。
3. **T=128 未测**：掩码张量 `[14,128,128]`、scores 同形，两个运行时输入是否仍单步、是否撞 L1/DRAM 预算，
   均未知（L=128 的裸 softmax 有先例 `softmax_1x12x128_fp16` PASS，但带掩码未测）。
4. **常量掩码 vs 运行期掩码在整块里的取舍未做**：常量形态 `Static 1.8K / Input 1.8K`，动态形态 `Static 16 / Input 3.5K`；
   层数与数值都一样，选哪个只影响 host 侧接口。票 09 决定。
5. **掩码张量本身的生成位置**：本票的 mask 是 host 送进来的常量数据；整块里它应该在图内由常量 buffer 提供（已编译验证），
   但"每个 head 都要一份完整 [T,T]"意味着 `14×8×8` 的常量（1792 B）——没有尝试用广播/共享降低它，
   因为 ncc 的逐元素算子不支持广播（`docs/OP-elementwise.md` 只验证了同形状）。
6. **`LOG`/`COS` 的失败点与 `docs/OP-rope.md` 的措辞不一致**：`OP-rope` 说"SIN/COS 在 importer 阶段被拒"，
   我这里 COS 是**后端期**被拒（`Cannot find an execution plan`），EXP 则完全可用。措辞差异说明
   "importer 拒 SIN/COS" 这句话也需要重新核实（不在本票范围）。

---

## 10. 复现

```bash
# 0) 环境：WSL 的 /usr/bin/python3（有 numpy + tflite schema），路径含空格必须加引号
export QWEN25_WEIGHTS_DIR='/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen'

# 1) 构建 .tflite + golden（锁外；约 3 s）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t05' && /usr/bin/python3 tools/build_masked_softmax.py --out ops/masked_softmax_t8_fp16"

# 2) 编译（锁外）。常量掩码形态也编一份
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t05' &&   ./tools/compile.sh ops/masked_softmax_t8_fp16/model.tflite ops/masked_softmax_t8_fp16/model.dla &&   ./tools/compile.sh ops/masked_softmax_t8_fp16/model_constmask.tflite ops/masked_softmax_t8_fp16/model_constmask.dla"

# 3) 板端（每一步都持锁；脚本见 ops/masked_softmax_t8_fp16/run_experiments.sh）
export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
./ops/masked_softmax_t8_fp16/run_experiments.sh health      # 实验前
./ops/masked_softmax_t8_fp16/run_experiments.sh mag2        # 掩码 -8192
./ops/masked_softmax_t8_fp16/run_experiments.sh zero        # 掩码全 0
./ops/masked_softmax_t8_fp16/run_experiments.sh l1          # layer1 regime
./ops/masked_softmax_t8_fp16/run_experiments.sh noise       # 噪声底（第一次）
./ops/masked_softmax_t8_fp16/run_experiments.sh primary     # 判据主档（compare.txt）
./ops/masked_softmax_t8_fp16/run_experiments.sh constmask    # 常量掩码（生产形态）
./ops/masked_softmax_t8_fp16/run_experiments.sh expprobe     # 负面控制 + 复跑 primary
./ops/masked_softmax_t8_fp16/run_experiments.sh health      # 实验后

# 4) 判据汇总（噪声底 / 判据一 / 判据二 / 负面控制），不需要板子
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t05' && /usr/bin/python3 tools/build_masked_softmax.py --analyze --out ops/masked_softmax_t8_fp16"

# 5) 一元算子探针（只有编译，不碰板子）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t05' && for op in SOFTMAX RSQRT LOGISTIC EXP LOG COS; do   /usr/bin/python3 tools/build_masked_softmax.py --probe $op --out ops/masked_softmax_t8_fp16;   ./tools/compile.sh ops/masked_softmax_t8_fp16/probe_$(echo $op | tr A-Z a-z).tflite ops/masked_softmax_t8_fp16/probe_$(echo $op | tr A-Z a-z).dla; done"
```

**可复现性实测**：重建 `.tflite` 与 `golden/*.bin` 逐字节相同（`MODELS IDENTICAL AFTER REBUILD`、
`GOLDEN REPRODUCIBLE: identical bytes`）；`.dla` 不可复现（仓库既有结论，本票只把它当产物标识）。

---

## 11. 对上游 / 下游的影响

1. **spec 的前提需要更正**：`spec.md` §Problem 2 与票面写的"推理路径里没有 `EXP` 算子"，经配对正对照实测
   **不成立**（EXP → `ExpLayer`，1 步，EXIT=0；LOG/COS 才是不可用的）。本票的设计结论不变，但**理由变了**：
   是 fp16 的 exp 值域（>11.09 上溢，实测 640/896 溢出）与新算子层预算把它钉死，不是"算子不存在"。
   建议 Lead 在 spec / `docs/OP-MATRIX.md` §5.1 里把这一条从"未覆盖"改成"已覆盖：加法掩码 + 原生 SoftMax"，
   并顺手更正"EXP 不可用"的说法。
2. **给票 09 的接线要求**：
   * 掩码必须是 `MUL(1/8)` **之后**的 `ADD`，形态 `0 / −4096`，常量烘进图（已验证 1 步、逐位等价）；
   * 若沿用运行期掩码，子图输入顺序是 `[scores, mask]`，e2e/neuronrt 的 `-i` 顺序必须一致；
   * 复用前先量一次该层的 `|u|max`，确认 `S ≤ 1996`（§9.1），否则重推 `M`；
   * 整块里这一层是 3 个原生层、1 个执行步，**不要**再插归一化/额外 clamp（会破坏"可见位置逐位相同"）。
3. **给票 06/10 的量级提醒**：layer0 的分数被 `q/k_proj.bias` 主导到 ±500，任何下游对分数做饱和/clamp 的设计
   都会与"bias 主导"这个真实事实冲突。