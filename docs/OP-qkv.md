# OP-qkv —— Q / K / V 投影（含 bias）+ RoPE（T=8, fp16, 真实 Qwen2.5-0.5B 形状）

> 票 **03**（`impl/03-qkv`，worktree `wt-t03`，基线 `feat/qwen25-attention` @ `655109a`）。
> 三个 op 目录（q / k / v 形状不同，而 `e2e.sh`+`compare.py` 只支持单输入单输出）：
> `ops/qkv_q_bias_rope_t8_fp16/`、`ops/qkv_k_bias_rope_t8_fp16/`、`ops/qkv_v_bias_rope_t8_fp16/`。
> 一个 builder：`tools/build_qkv_bias_rope.py --which q|k|v`（+ `--diag` / `--emit-neg-width896`）。
> 权重来源：票 02 的冻结夹具（`weights_digest=898fa31…18a4`，25 张量 sha256 加载即校验）；RoPE 表本票自算。

---

## 0. TL;DR

**结论三句话**：
1. **三个 op 全部编译 + 板端 + 数值 PASS**：`EXIT=0`、单 `ExecutionStep`、`Target: MDLA_5_0`、无 bridge、
   层数 **11**（三个图结构完全相同，只差 heads/宽度/权重张量）；板端 `max_abs` = **1.00 ULP of max|out|**
   （q 1.562e-2 / k 6.25e-2 / v 1.221e-4），`rel_l2` ≈ 3.1–3.2e-4。
2. **bias 真的进了图**：三条独立的 `bias=0` 反事实与真值差 q 25.23 / k 77.82 / v 0.0428，是登记阈值的
   **254× / 257× / 58×**；板端贴着"有 bias"的 golden 到 1 ULP。k/v 宽度钉死在 **128**，并把"误接 896"的两种失败形态
   做成可复现的负对照（都 `EXIT=1`，加载期 `Fail to process the given model`）。
3. **设备与"逐 op fp16 舍入模拟"几乎逐位相同**（q 7167/7168、k 1024/1024、v 1023/1024 个 code 相同）——
   说明 MDLA 的 FC/MUL/SUB/ADD/CONCAT 就是逐 op IEEE fp16 语义；噪声底实测 **0**（同输入两次运行逐位相同）。

| op | 形状 | tflite / dla | 编译 | 层数 | 板端 max_abs（=ULP） | rel_l2 | 判定 |
|---|---|---|---|---|---|---|---|
| `qkv_q_bias_rope_t8_fp16` | `[8,896]→[8,14,64]` fp16 | 1,623,820 / 1,631,384 B | EXIT=0, 1 step, MDLA_5_0, 无 bridge | **11** | 1.562e-2（**1.00** ULP，rel 6.15e-4） | 3.138e-4 | **PASS** |
| `qkv_k_bias_rope_t8_fp16` | `[8,896]→[8,2,64]` fp16 | 233,740 / 239,768 B | 同上 | **11** | 6.250e-2（**1.00** ULP，rel 8.06e-4） | 3.233e-4 | **PASS** |
| `qkv_v_bias_rope_t8_fp16` | `[8,896]→[8,2,64]` fp16 | 233,740 / 239,768 B | 同上 | **11** | 1.221e-4（**1.00** ULP，rel 9.53e-4） | 3.242e-4 | **PASS** |

| 项 | 结果 |
|---|---|
| 噪声底（同一 .dla 同输入两轮） | **0**（7168/7168、1024/1024、1024/1024 个 code 逐位相同）→ 硬件确定性 |
| 阈值（**跑之前**登记） | q `atol=0.0496, rtol=1.953e-3`；k `atol=0.1515, rtol=1.953e-3`；v `atol=4.883e-4, rtol=1.953e-3` = `4·2⁻¹¹·max|out|` / `4·2⁻¹¹` |
| 设备无关下限（numpy 逐 op fp16 模拟） | 三者都是 **1.00 ULP**；与设备实测一致 |
| 反事实（bias / 旋转 / 布局 / 错权重） | 全部 58×–27752× 阈值 |

---

## 1. 这一票要回答什么

票面（`.scratch/qwen25-attention/issues/03-qkv-projection-bias-rope.md`）要的四件事：

1. fp16 下 q/k/v 三个投影与**独立** numpy golden 对上，**bias 参与计算**；
2. k/v 输出宽度确认为 **128**，并记录"如果误接成 896 会怎样"；
3. RoPE 用**预算常量表**（sin/cos 不能作为图内算子）、head_dim=64、T=8，与 golden 对上；
4. 输出为 **3-D**（`[T,14,64]` / `[T,2,64]`），记录 rank 上限是 4；外加：无 bridge、单执行步、实测噪声底。

本票**不做**：attention 打分（票 04）、掩码/SoftMax（05）、PV+O（06）、RMSNorm 本身的板端验证
（`docs/OP-rmsnorm.md` 已有独立证据）、o_proj。

---

## 2. 事前登记：假设与预测（写于任何板端运行之前，2026-11-13）

量级数字来自**只读夹具 + numpy** 的静态分析（设备无关），不是实验结果。它是登记的**预期**。

| # | 预测 | 若预测不成立意味着 |
|---|---|---|
| P1 | 三个图都 `EXIT=0`、**单** `ExecutionStep`、`Target: MDLA_5_0`、无 bridge、层数 **11**（FC 1 + RESHAPE 1 + Slice 2 + Mul 4 + Sub 1 + Add 1 + Concat 1） | 有算子落到 CPU（bridge）或逐元素层被拆成 Slice/op/Concat 块（rank 或行宽触发） |
| P2 | 板端 `max|dev−golden|` ≈ **1 ULP of max|out|**：q ≤ 1.6e-2（max|out|=25.41, ULP=1.56e-2）、k ≤ 6.3e-2（77.56, ULP=6.25e-2）、v ≤ 1.3e-4（0.128, ULP=1.22e-4）；`rel_l2` ≤ 1e-3 | 大于 1 ULP 说明不是"存量 fp16 舍入"，而是数学/布局错 |
| P3 | 板端输出与**逐 op fp16 模拟**（`fp16_sim`）逐位相同的比例 ≥ 90% | 若 dev 与 sim 差很多，MDLA 的 FC 累加/舍入口径与模拟不同（要改口径，不改判据） |
| P4 | **bias 反事实**：`bias_dropped` 与真值差 q 25.22 / k 77.81 / v 0.0428，分别是登记阈值的 **254× / 257× / 58×** ⇒ 板端若匹配真 bias golden，就排除了"bias 被静默丢掉" | 若板端同时接近"有 bias"和"无 bias"（差 250×，不可能），或与两者都差很多 ⇒ 图里 bias 接错 |
| P5 | **布局/旋转反事实**：`no_rotation` q 35.7 / k 108.2 / v 0.253；`head_major_reshape`（`[H,T,64]`，NOTES §10.2 那类）q 35.4 / k 99.2 / v 0.256；k/v 用 q_proj（"k 当成 896 宽"的前 128 维）k 82.7 / v 20.2 —— 全部 ≫ 阈值 | 任何一个接近阈值 ⇒ 该布局/旋转没生效 |
| P6 | **896 负对照**（编译期，无需板子）：(a) 896 宽 FC 接 `[8,2,64]` reshape（元素 7168 vs 1024）在 ncc **逐层校验期**报错；(b) 权重张量声明 `[896,896]` 但只写 128×896 个数，在**加载期**报 `Fail to process the given model`（NOTES §5 坑 2 / E-112 根因） | 两者都"能编过" ⇒ 这两类错误编译器不挡，只能靠 golden/形状检查挡 |
| P7 | **eps 口径不可判别**：eps=1e-6 vs 0 只让 q 输出动 1.56e-2 = **1 ULP**（阈值的 16%）；而"漏掉 gamma"会让 q 动 8.10（阈值的 82 倍）。即 gamma 必须记清、eps 在本算子里测不出来 | 若 eps 造成 >1 ULP 差异，说明 n1 的 rms 尺度接近 eps（本夹具 rms≈1.5e-2，离 1e-6 很远，不会发生） |

**容差登记（跑之前写进 `model.json`，规则见 NOTES §4/§10.1）**：`max|out|` 是 25.41 / 77.56，远大于 1，
§4 的"阈值 < 2e-2"在这两个 op 上等价于 7.9e-4 / 2.6e-4 **相对**误差，比 1 个 fp16 ULP 还紧 ⇒ 走 §10.1 例外：

* 下限由**设备无关**的 `fp16_sim`（numpy 逐 op 舍入模拟）实测：三者都是 **1.00 ULP**；
* 登记 `atol = 4·2⁻¹¹·max|out|`、`rtol = 4·2⁻¹¹`（k=4，阈值 ≈ 实测下限的 4 倍）；
* v 的 `max|out|=0.128 < 1`，§4 **原样适用**（atol/rtol 同样由实测下限推，远小于 2e-2）。

---

## 3. 做法

### 3.1 图（三个 op 只差 heads / 宽度 / 权重张量）

```
input [8,896] fp16
  → FULLY_CONNECTED(W[N,896] const, bias[N] const, keep_num_dims)   [8,N]
  → RESHAPE(NewShape=[8,H,64])                                      [8,H,64]     token-major
  → SPLIT(axis=2, num_splits=2)                                     [8,H,32]×2
  → MUL ×4（cos/sin 常量表，与激活同形状 [8,H,32]，**不用广播**）
  → SUB / ADD                                                       lo=a·cos−b·sin, hi=b·cos+a·sin
  → CONCAT(axis=2)                                                  [8,H,64]  = output
```

* q：`H=14, N=896`；k/v：`H=2, N=128`。
* 输入 `[8,896]`、输出 `[8,14,64]`/`[8,2,64]`：**rank 3**（`rank ≤ 4` 是 ncc 硬限制；
  rank-4 的 `[1,T,H,64]` 在 RoPE 上要 35 层，rank-3 只要 9–11 层，`docs/OP-rope.md` §5.3）。
* `keep_num_dims=True`、`weights_format=DEFAULT([N,K])`、`fused_activation=NONE`。

### 3.2 输入口径（票面要求记录"选了哪个"）

**选 B：图输入 = input_layernorm 之后的 hidden `n1`，RMSNorm 不在图里。**

```
n1 = fp16( embed.weight[TOKEN_IDS] / sqrt(mean(x²) + 1e-6) · layer0.input_layernorm.weight )
```

* `eps = 1e-6`（transformers `rms_norm_eps`），**不是**玩具块的 0；
* `gamma = layer0.input_layernorm.weight` **在 norm 里用掉，不折进 FC 权重**（折了的话 numpy golden
  也必须同步折，多一个静默错的机会）；
* `TOKEN_IDS = [9707, 11, 1879, 828, 11780, 13, 220, 16]` 固定，`golden/input.bin` 逐字节可复现
  （实测：重跑 builder 后 tflite + input/expected 的 sha256 全不变）；
* 为什么不在图里：RMSNorm 有自己的票与板端证据（`docs/OP-rmsnorm.md` 的 FC-tile+CONCAT 变体），
  塞进来只会让失配变成双因；为什么不用裸 embed（不 norm）：那样输入尺度是 0.015 的原始嵌入，
  与"真正喂给 q_proj 的东西"差一个量级。

`n1` 实测：`max|n1| = 1.0508`、std 0.0989 —— 与真实 Qwen2.5-0.5B（`layer0.input_layernorm`
增益 mean 0.032 / std 0.092）量级一致。

### 3.3 RoPE：常量表 + 约定 + 一个陷阱

* **Qwen2 split-half**：`out_lo = a·cos − b·sin`，`out_hi = b·cos + a·sin`（a=x[...,:32], b=x[...,32:]）。
  任务书里曾写错成 `b·sin + a·cos`，那不是旋转（不保 2-norm）——`docs/OP-rope.md` §2 的活教材。
  builder 里保留两道防线：golden 写盘前断言**每对 (a,b) 的 2-norm 不变**（实测
  `rotation_invariant_rel_err` = 5.09e-4 / 4.39e-4 / 2.64e-4，正是 fp16 存表本身的代价），
  且 `--diag` 里 `no_rotation` 是登记反事实（板端与它差 343–360× 阈值）。
* **sin/cos 是常量表**：ncc importer 拒绝 `SIN`/`COS`（`docs/OP-rope.md` §0），所以表必须是数据；
  **代价是 T 必须静态**。本票按 prefill 的固定窗口做 const 表（positions 0..7）。
  decode 每步换表就得走"表当第二输入"的 dyn 路径（+2 个 SliceLayer，`docs/OP-rope.md` §4），
  **本票没做**（见 §9）。
* **表按 head 复制**成 `[8,H,32]`，而不是用 `[8,1,32]` 广播：rank-3 激活 × `[T,1,h]` 表在本仓库
  **没有板端证据**（只有 D=64 的 `[T,64]×[T,1]`，`docs/OP-rmsnorm.md`）。复制代价是 Static ~7 KB（q）。
* **heads 是 token-major**：`[T,896] → [T,14,64]`。`[14,T,64]` 是登记反事实（板端与它差 357× 阈值）——
  NOTES §10.2 的事故类型（`[T,H]` 直接 reshape 成 `[heads,T,HD]`）在这里**不适用**：本算子不做
  head 与 token 的转置，只做最后一维的切分；能不能真对上由板端 `compare.txt` 说话，不由"编译过了"说话。

### 3.4 bias：为什么它是本票的核心

`config.json` 里**没有任何 bias 键**（q/k/v 的 bias 是 transformers 源码里硬编码的），而夹具实测
`q_proj.bias` std≈7.8、`k_proj.bias` std≈29.8（真实权重实测 7.85 / 32.2），**远大于权重本身**（0.067 / 0.081）。
所以 bias 是 graph 的第三个 FC 输入，且"漏掉它"有一个量化上界很紧的判据：
`bias_dropped` 反事实与真值差 25.2 / 77.8，而登记阈值是 0.099 / 0.303 ⇒ 差 **250 倍**。
反过来说：本算子输出里 matmul 项（std 0.195 / 0.238）远小于 bias（7.8 / 29.8），所以这个对拍
**主要约束的是 bias 与布局**，对"权重本身算得对不对"只有 ~1e-2 量级的判别力（见 §9 Q4）。

### 3.5 golden 与 fp16 下限

* `golden/expected.bin`：numpy float64，输入是**已存储的 fp16**（输入/权重/bias/表），**只在最后**舍入一次到 fp16。
  不调用 ncc / TFLite interpreter / 设备，也不与 flatbuffer 构造共享代码（R3）；
* `fp16_sim`（诊断用，不进 golden）：float32 FC + FC 输出按 fp16 存储 + RoPE 逐 op 舍入到 fp16 —— 模拟
  MDLA 的行为（`docs/OP-rope.md` §6 实测"逐 op fp16 模拟与设备逐位相同"）。它与 golden 的差就是
  **设备无关下限**，用来定阈值（`model.json.fp16_floor`）。实测设备与它 99.99% 逐位相同（§5.2）。

---

## 4. 编译器结果

三个 op **结构完全相同的 11 层**，单执行步，无 bridge：

```
===Graph[0] has 1 Execution Step(s)===
Graph[0] Step[0]
ExecutionStep
 ├ StepId: 0
 ├ Target: MDLA_5_0
 └ Subgraph:
   ├ FullyConnectedLayer<0>{FULLY_CONNECTED}
   ├ ReshapeLayer<1>{RESHAPE}
   ├ SliceLayer<2>{SPLIT}            <- a = x[...,:32]
   ├ ElementWiseMulLayer<6>{MUL}     <- b*sin
   ├ ElementWiseMulLayer<3>{MUL}     <- a*cos
   ├ SliceLayer<2>{SPLIT}            <- b = x[...,32:]
   ├ ElementWiseMulLayer<5>{MUL}     <- b*cos
   ├ ElementWiseAddLayer<8>{ADD}     <- hi = b*cos + a*sin
   ├ ElementWiseMulLayer<4>{MUL}     <- a*sin
   ├ ElementWiseSubLayer<7>{SUB}     <- lo = a*cos - b*sin
   └ ConcatLayer<9>{CONCATENATION}   <- [T,H,HD]
```

| op | `model.tflite` | `model.dla` | EXIT | 执行步 | Target | 层数 | DRAM | L1 | bridge | ncc |
|---|---|---|---|---|---|---|---|---|---|---|
| q | 1,623,820 B | 1,631,384 B | **0** | **1** | MDLA_5_0 | **11** | 1.6M | 44K | 无 | 7.3.15 |
| k | 233,740 B | 239,768 B | **0** | **1** | MDLA_5_0 | **11** | 245K | 8.0K | 无 | 7.3.15 |
| v | 233,740 B | 239,768 B | **0** | **1** | MDLA_5_0 | **11** | 245K | 8.0K | 无 | 7.3.15 |

* `--disallow-bridge` 恒开（`tools/compile.sh` 固定），日志里没有任何 `Bridging` 字样（除命令行本身）；
* **P1 成立**，层数与预测逐层一致（ReshapeLayer 确实被物化成 1 层、没被折叠——与 `docs/OP-shapeops.md` 一致）；
* `.dla` 的 sha256 只标识某一次产物，**不作回归判据**（AGENTS §4）；q 的 DRAM 1.6M 就是 896×896 fp16 权重。

## 5. 板端结果 / 噪声底

### 5.1 数值（`compare.txt` 原文见各 op 目录）

| op | max_abs（dev−golden） | = ULP | 相对（max_abs / max_abs_out） | `rmse` | `rel_l2` | 登记 atol / rtol | 判定 |
|---|---|---|---|---|---|---|---|
| q（max_abs_out 25.4062, ULP 1.5625e-2） | **1.5625e-2** | **1.00** | 6.150e-4 | 2.449e-3 | 3.138e-4 | 0.0496 / 1.953e-3 | **PASS** |
| k（77.5625, 6.2500e-2） | **6.2500e-2** | **1.00** | 8.058e-4 | 9.831e-3 | 3.233e-4 | 0.1515 / 1.953e-3 | **PASS** |
| v（0.1281, 1.2207e-4） | **1.2207e-4** | **1.00** | 9.533e-4 | 1.200e-5 | 3.242e-4 | 4.883e-4 / 1.953e-3 | **PASS** |

`nan/inf` 全 0，range 与 golden 一致（q exp/dev `[-25.41,22.27]`、k `[-75.94,77.56]`、
v `[-0.1149,0.1281]`）。

### 5.2 设备 = "逐 op fp16 舍入"（P3 成立，且比预测更强）

| op | dev vs `fp16_sim` max_abs | 逐位相同 code 数 | 说明 |
|---|---|---|---|
| q | 1.907e-6 | **7167 / 7168** | 唯一不同的 1 个 code 在极小量级 |
| k | 0 | **1024 / 1024** | 完全逐位相同 |
| v | 5.961e-8 | **1023 / 1024** | 同上，1 个极小量级 code |

即：**板端 FC/MUL/SUB/ADD/CONCAT 就是逐 op IEEE fp16 语义**；与"精确数学 + 一次最终舍入"的 golden 相差 1 ULP
不是误差，而是"设备在每个层边界各自舍入"的必然结果。这条同时解释了 P2 的"恰好 1.00 ULP"为什么不是巧合。

### 5.3 噪声底（NOTES §4 的测法）

同一 `.dla`、同一输入、连跑两轮（每轮 `-c 3`），各存一份输出（`out_r1.bin` / `out_r2.bin`）：

| op | max_abs(r1−r2) | 相对 | 逐位相同 |
|---|---|---|---|
| q | **0** | 0 | 7168 / 7168 |
| k | **0** | 0 | 1024 / 1024 |
| v | **0** | 0 | 1024 / 1024 |

**噪声底 = 0（硬件确定）**。按 §4 的退化规则：阈值退化为"fp16 末位量级（约 1e-3 相对）"——登记值
`rtol = 4·2⁻¹¹ = 1.953e-3` 正好是这个退化口径（≈1e-3 相对），`atol = 4·2⁻¹¹·max_abs_out` = 4 ULP；
实测误差 1 ULP，只占阈值的 16% / 21% / 25%，**没有为了让 compare 变绿而调阈值**。

**容差登记的时间次序（证据）**：

```
10:19:56  model.tflite + golden/input.bin + golden/expected.bin + model.json（含 atol/rtol）
10:21:02  model.compile.log
10:22:49  第一次板端运行 -> compare.txt
```

容差公式只依赖**夹具 + numpy 逐 op fp16 模拟**（设备无关），不含任何设备数据；本次未修改。

## 6. 反事实与负对照

### 6.1 板端 vs 登记反事实（`--diag`，数字是 max_abs，括号是相对登记阈值的倍数）

| 反事实（若图真是这样） | q | k | v |
|---|---|---|---|
| **bias 被丢掉** | 25.227（**254.2×**） | 77.817（**256.8×**） | 0.0428（**58.0×**） |
| **根本没旋转**（表被忽略） | 35.750（360.2×） | 108.16（357.0×） | 0.2535（343.4×） |
| **head-major reshape** `[H,T,64]` | 35.423（356.9×） | 99.206（327.4×） | 0.2564（347.3×） |
| **k/v 用 q_proj（896 宽）的前 128 维** | n/a（形状不可表达） | 82.695（272.9×） | 20.492（27752×） |

⇒ ① **bias 真进图**：板端与"有 bias"golden 差 1 ULP，与"无 bias"差 254–257 个阈值 ⇒ 不可能漏；
② **旋转真生效**：与"无旋转"差 343–360 个阈值；③ **布局是 token-major**：与 head-major 差 327–357 个阈值。

附加（P7 的落地）：`eps=1e-6 vs 0` 只让 q 输出动 1.56e-2 = **1 ULP**（阈值 16%），而**漏掉 gamma** 会动
8.10（阈值 82 倍）⇒ **gamma 口径必须记清、eps 在本算子里不可判别**。

### 6.2 k/v 宽度 128，误接成 896 会怎样（负对照，编译期即可判死）

负对照图由 `--emit-neg-width896` 生成，编译日志一并入库（`ops/qkv_k_bias_rope_t8_fp16/neg_width896/`）：

| 变体 | 内容 | 结果 |
|---|---|---|
| (a) `a_896wide_fc_to_kv_reshape` | FC 按 896 宽（用 q_proj 权重+bias）→ 再 RESHAPE 成 `[8,2,64]` | `EXIT=1`，**加载/解析期** `ERROR: Fail to process the given model`（元素 7168 vs 1024） |
| (b) `b_declared_896_with_128_data` | 权重张量**声明** `[896,896]` 但缓冲区只写了 128×896 个数（E-112 根因） | `EXIT=1`，同样 `Fail to process the given model` |

**P6 部分修正**：预测 (a) 会在**逐层校验期**报错，实际两个变体都在**更早的加载/解析期**就被拒（NOTES §5 坑 10
的"错误措辞区分阶段"把两者归到第一类）。结论反而更强：**这两类 896 误接在编译期就进不了 NPU**。

数值层面（不需要板子）：若把 896 宽的 q_proj 结果当 k 只用前 128 维，与真 k 差 **82.70**（阈值的 273 倍）；
若只看形状，`[8,896]` 是 7168 个数而 k 需要 1024 个，`compare.py` 会在 `size mismatch` 处直接 FAIL；
权重文件 1,605,632 B vs 229,376 B（7 倍）。

## 7. 票面 checklist 对照

| 票面要求 | 状态 | 证据 |
|---|---|---|
| fp16 下 q/k/v 三个投影与独立 numpy golden 对上，**bias 参与计算** | ✅ | §5.1 三份 `compare.txt` 全 PASS（1.00 ULP）；§6.1 `bias_dropped` 254×/257×/58× 阈值 |
| k/v 输出宽度确认为 **128**，并记录"误接成 896 会怎样" | ✅ | §0 表 `[8,2,64]`；§6.2 两个负对照 `EXIT=1` + 数值 82.70（272.9×） |
| RoPE 用预算常量表（sin/cos 不进图）、head_dim=64、T=8，与 golden 对上 | ✅ | §3.3；§5.1/§5.2；`no_rotation` 反事实 343–360× |
| 输出为 3-D（`[T,14,64]`/`[T,2,64]`），记录 rank 上限 4 | ✅ | §0 表；§3.1 `rank_note`（ncc 硬限制 4；rank-4 同工作量 35 层 vs rank-3 9–11 层） |
| 无 bridge、单执行步 | ✅ | §4：`EXIT=0`、1 `ExecutionStep`、`--disallow-bridge` 恒开且日志无 `Bridging` |
| 记录本算子的实测噪声底，作为阈值依据 | ✅ | §5.3：实测 **0**（逐位相同）→ 阈值退化为 fp16 末位口径（rtol 1.953e-3）；下限另由设备无关模拟独立支撑（1.00 ULP） |
| 额外（本票自己加的）：模型可复现 | ✅ | 重跑 builder 后 `model.tflite`(3) + `golden/*.bin`(9) + `model.json`(3) 字节全同 |

## 8. 适用范围 / 不适用

**适用**：fp16；`T=8`（静态）；hidden `D=896`；`head_dim=64`（h=32）；heads 14（q）/ 2（k,v）；
`theta=1e6`；positions 0..7 的**常量**表；Qwen2 split-half 约定；`mdla5.0` / ncc 7.3.15 /
`--disallow-bridge`；单输入单输出；layer0 权重（layer1 同形状，换权重即可）。

**不适用**：decode（每步换位置 → 必须 dyn 表，本票未建）；`T≠8` 或 batch>1；`partial rotary` /
interleaved / `rope_scaling`；int8；真权重（bf16 源）；与 attention(BMM) 的直接对接；o_proj；任何性能结论。

## 9. 剩余疑问（UNKNOWN）

1. **位置表当运行时输入（dyn）未测**：decode 每步只有一个位置，表必须每次换 → 需要 `golden/input1.bin` 那条路。
   `docs/OP-rope.md` §4 在 `[1,T,64]` 上实测可行（+2 个 SliceLayer、Static→DRAM）；本票的
   `[8,H,64]` 形状下**未验证**，"层数 11→13"只是推断。
2. **表广播（`[8,1,32]`）未测**：本票用复制表避开广播。若广播可用，Static 可省 ~4 KB（q）且层数不变，
   但 `[T,64]×[T,1]` 之外没有证据，**不猜**。
3. **其它位置窗口 / 更长 T**：`p0≠0`、`T=128/1024` 的表大小、层数与数值都没测
   （表线性变大：`T=8` 时 q 的 cos+sin ≈ 14 KB）。
4. **本对拍对"权重乘法算得对不对"的判别力有限**：matmul 项 std 0.195/0.238 远小于 bias 7.8/29.8，所以
   "FC 换成 fp16 累加"这类错误在本实验里只造成 ~1e-2 差异（在阈值内，**测不出**）。本实验钉死的是
   **bias、布局、RoPE 约定**；要单独验证权重乘法灵敏度，需换输入（例如把 `n1` 放大到 ~1 量级）——记录为
   UNKNOWN，不当作已覆盖。
5. **真权重档未测**：夹具 `k_proj.bias` abs_max=77.8（2.6σ），真权重 130（4.04σ）；真权重下 q/k 输出可达
   ~150–200，fp16 末位 ULP 随之变大，阈值口径要按同一公式重算（参数不同）。
6. **与票 04（GQA 打分）的对接未测**：本 op 输出 `[8,H,64]` 直接喂 BMM 是否零搬运/是否需要转置未测；
   `[T,H,HD]→[H,T,HD]` 的 TRANSPOSE 代价另计（NOTES §10.2 那条）。
7. **性能**：只记录（单次 0.86–1.6 ms/inf；同一个 `.dla` 两次运行报 595 vs 713 FPS）——**不作结论**
   （与 `docs/OP-rope.md` §5.4 同一现象：启动/同步开销与板子状态把差异淹没了）。

## 10. 复现

锁外三步（生成 / 编译 / 负对照），板端一步（持锁），诊断一步：

```bash
export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
# 权重夹具：直接用集成分支已生成的那份（或在本 worktree 里 gen_qwen25_weights.py 生成一次，sha256 相同）
export QWEN25_WEIGHTS_DIR="/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen"

# 1) 生成 tflite + golden + model.json（三个 op，锁外；WSL 的 /usr/bin/python3）
for W in q k v; do wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t03' && QWEN25_WEIGHTS_DIR='$QWEN25_WEIGHTS_DIR' /usr/bin/python3 tools/build_qkv_bias_rope.py --which $W"; done

# 2) 编译（锁外；EXIT / 层清单 / SIZE / SHA256 都写进 model.compile.log）
for W in q k v; do wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t03' && ./tools/compile.sh ops/qkv_${W}_bias_rope_t8_fp16/model.tflite ops/qkv_${W}_bias_rope_t8_fp16/model.dla"; done

# 3) 896 负对照（锁外；两个都会 EXIT=1，日志保留）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t03' && QWEN25_WEIGHTS_DIR='$QWEN25_WEIGHTS_DIR' /usr/bin/python3 tools/build_qkv_bias_rope.py --which k --emit-neg-width896 --out ops/qkv_k_bias_rope_t8_fp16"
for N in a_896wide_fc_to_kv_reshape b_declared_896_with_128_data; do wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t03' && ./tools/compile.sh ops/qkv_k_bias_rope_t8_fp16/neg_width896/$N.tflite ops/qkv_k_bias_rope_t8_fp16/neg_width896/$N.dla"; done

# 4) 板端（唯一持锁的一步）：healthcheck + 每个 op 两轮（第二轮=噪声底）+ healthcheck
./tools/devlock.sh run -- ./tools/board_t03.sh

# 5) 诊断（锁外）：设备 vs golden / fp16 模拟 / 反事实
for W in q k v; do wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t03' && QWEN25_WEIGHTS_DIR='$QWEN25_WEIGHTS_DIR' /usr/bin/python3 tools/build_qkv_bias_rope.py --which $W --diag"; done
```

噪声底复算见 §5.3 的两份 `out_r*.bin`（numpy 一行 max|r1−r2|）。

## 11. 交接

* **新增文件**：`tools/build_qkv_bias_rope.py`（builder + `--diag` + `--emit-neg-width896`）、
  `tools/board_t03.sh`（一次持锁跑完三条 op 的板端序列，本票专用）、三个 `ops/qkv_*_bias_rope_t8_fp16/`
  （含 `model.json` / `model.compile.log` / `compare.txt` / `e2e.json` /
  `neg_width896/*.compile.log`）、本文。
* **未改公共文件**（`e2e.sh` / `compare.py` / `compile.sh` / `devlock.sh` / `healthcheck.sh` /
  `build_ops.py` 只 import；`.scratch/` 未动）。
* **给票 04/09 的直接输入**：q `[8,14,64]` 与 k/v `[8,2,64]` 已经是"投影 + bias + RoPE 之后"的张量；
  做 GQA 打分时注意 `[T,H,HD]→[H,T,HD]` 的 TRANSPOSE（NOTES §10.2），别假设 reshape 就够。
* **回归判据**：`compare.txt`（阈值写在各 `model.json`，公式 `4·2⁻¹¹·max_abs_out` / `4·2⁻¹¹`）；
  `model.json.fp16_floor` 可作"设备无关下限"的快速体检；`.dla` 哈希不作判据。
