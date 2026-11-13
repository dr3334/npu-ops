# OP-decoder-layer —— 1 层 Qwen2.5-0.5B decoder 整块（T=8 与 T=128 / fp16 / 14Q-2KV GQA）

> 范围：票 **09**（`impl/09-decoder-block`，worktree `wt-t09`，基线 `feat/qwen25-attention`）；
> **T=128 档**（`impl/t128-decoder`，worktree `wt-t128`，基线 `077db15`）见 §14，那份实现只是同一个 builder 的 `--T 128`。
> 交付：`ops/decoder_layer_t8_fp16/`（+ `ops/_decoder_bisect/{attn_half,ffn_half}/` 二分证据）
> + `tools/build_decoder_layer.py` + 本文。权重：票 02 冻结夹具（`layer0.*`，sha256 加载即校验）。
> 编译器 ncc-tflite **7.3.15** / `--arch=mdla5.0 --opt=3 --disallow-bridge --num-mdla=2 --l1-size-kb=3072`；
> 设备 mt6897 / Android 16（`0123456789ABCDEF`），全部板端操作持 `/d/deepseek_ workspace/mtk/.npu-board.lock`。

---

## 0. TL;DR

| 项 | 结果 |
|---|---|
| 图 | 一个 `.tflite`：`x[8,896] fp16 → RMSNorm → q/k/v(+bias) → RoPE(常量表) → KV 2→14 → 3-D BMM → MUL(1/8) → ADD(掩码) → SoftMax → PV 3-D → TRANSPOSE → RESHAPE → o_proj(无 bias) → +x → RMSNorm → SwiGLU → +r1 → out[8,896] fp16` |
| host 参与数学 | **没有**：1 个运行时输入、1 个输出，掩码/表/权重全是图内常量 |
| 编译 | **EXIT=0**，**61 层**，**1 Execution Step**（`Target: MDLA_5_0`），**无 bridge**；`SIZE=29,913,364` |
| 层数预测 vs 实测 | 预测 **61**，实测 **61**（逐段一致，见 §8） |
| 内存账 | DRAM Input 14K / Output 14K / **Temp 14K** / **Static 28M** / Code 34K / Total 28M；L1 256K |
| 板端主判据 | `compare.txt` **PASS**：`max_abs 2.319e-03`（= max\|out\|=0.4836 处的 **9.50 ULP**，相对 **4.80e-03**）、`rmse 4.801e-04`、`rel_l2 3.690e-03`、nan/inf 0/0 |
| 噪声底 | **0.000e+00**（同 `.dla` 同输入两轮，7168/7168 个 fp16 code 逐位相同 ⇒ 硬件确定性） |
| 阈值（**跑之前**登记，`2bf6575` 已在 git 里） | `atol = rtol = 9.0332e-3 = max(4·2⁻¹¹·max\|out\|, 4·floor)`；**设备无关下限** floor = 2.258e-3 = **9.25 ULP** ⇒ 实测 9.50 ULP 只比下限多 **0.25 ULP** |
| 与设备语义模拟的关系 | dev vs 「逐层 fp16 舍入」模拟 = **1.709e-3（7.00 ULP）**，3388/7168 逐位相同 ⇒ 误差是 fp16 存储的必然，不是数学错 |
| 反事实判别力 | 丢掩码 50.6×、丢 RoPE 47.3×、丢 bias 44.1×、丢残差 39.4× 阈值（**eps=0 只有 0.20×** ⇒ 不可判别，见 §7） |
| 第一次板端 | **FAIL**（`max_abs 2.0475`、`rel_l2 2.198`）→ 二分定位 → 修 `MUL(x,x)` 接线 → PASS。失败证据全保留（§5） |
| 票面 checklist | **6/6**（§9） |

**一句话**：**真实的 61 层 decoder 整块（两次归一化 + RoPE + GQA 复制 + 掩码 softmax + SwiGLU + 两处残差、28.5 MiB 静态权重）在 mdla5.0 上是 1 个执行步、无 bridge，板端与独立 numpy golden 的差 = max|out| 处 9.5 个 fp16 码，且与「逐层 fp16 舍入」的设备语义模拟只差 7 个码 —— 即误差全部由 fp16 存储决定，不是实现错。**

**T=128（§14，同一张图、`--T 128`）**：仍然 **61 层 / 1 个执行步 / 无 bridge**，板端 **PASS**，
`max_abs = 1.733e-2` = 35.50 ULP，噪声底 0，**设备只比设备无关下限高 0.75 ULP**。
**但这一档不等于「真实 prefill 可用」**——24 层、logits、KV cache 都还没在 T=128 上做过。

---

## 1. 这一票要回答什么

票面（`.scratch/qwen25-attention/issues/09-single-decoder-block.md`）：

1. 输入输出都只是 `[T,896]` fp16，**host 不参与数学**；
2. 层内两次归一化用**真实 eps = 1e-6**（玩具块是 0），**口径差必须记录**；
3. RoPE、GQA、掩码 softmax、FFN、两处残差**全部在图内**；
4. 单层 hidden 与 golden 对拍通过，**阈值来自噪声底**；
5. **单执行步、无 bridge；记录层数**；
6. 记录层数与大张量分布，作为「能否塞进单执行步」的实测依据。

本票**不做**：24 层串联（票 10）、T≠8、int8、KV cache、真实权重档、任何性能结论。

---

## 2. 事前登记（写于任何编译/板端动作之前，2026-11-13）

### 2.1 假设 → 预测

| # | 预测 | 若不成立意味着 |
|---|---|---|
| P1 | 整图 **EXIT=0、1 个执行步、无 bridge**，层数 **61**（分段预算见 §8） | 出 bridge = 没上 NPU = 本票失败；>1 个执行步说明 28.5 MiB 静态权重 + 61 层把单步撑破了，要按「切成两半各自编译」二分 |
| P2 | 层数预测 61 与实测逐段一致 | 层数只能从 `model.compile.log` 抄（票 04 的教训：SPLIT 每个输出各发一个 `SliceLayer`） |
| P3 | 板端与独立 golden 的差 ≈ **fp16 逐层舍入下限**（本图 9.25 ULP of max\|out\|），不是 1e-2 量级 | 远大于下限 ⇒ 布局/接线错（NOTES §10.2 那一类），不是精度问题 |
| P4 | 噪声底 = 0（仓库所有 fp16 链都是逐字节相同） | 阈值按 3–5× 噪声底重定，但**必须在跑之前**重登记 |
| P5 | 掩码复用合规：本层 `S = max\|scaled score\| = 578.5 ≤ 1996`，−4096 仍在有效区间中部 | 若 S 超限 ⇒ 按 `M ∈ [2S+104, 65504−S]` 重推 M（票 05 §11.2 的要求） |
| P6 | eps=1e-6 与 eps=0 在本层**不可判别**（差 < 1 ULP） | 若可判别，说明某处 rms 尺度接近 eps（本夹具 rms ≈ 0.1–1，离 1e-6 很远，不会发生） |
| P7 | **「编译通过 + 单执行步」不等于数学正确**（spec C4）：必须靠板端 `compare.txt` 说话 | —— 这条在本次**真的兑现了**（§5） |

### 2.2 容差登记（跑之前写进 `model.json`，规则见 NOTES §4 / §10.1）

`max|out| = 0.4836`，fp16 在该量级 ULP = 2.44e-4：

* **设备无关下限** `floor = 2.258e-3`（= 9.25 ULP）= numpy 逐层模拟「每个张量边界都落一次 fp16、层内 fp32」
  与「精确链 + 一次最终舍入」的差。这条下限**不含任何设备数据**；
* 登记 `atol = rtol = max(4·2⁻¹¹·max|out|, 4·floor) = 9.0332e-3`（4·2⁻¹¹·max|out| = 9.44e-4，被 4·floor 支配）；
* `tolerance_revision` 与理由写在 `model.json`，且**在第一次板端运行之前就提交进 git（`2bf6575`）**；
* 修复 bug 后重建模型时 `model.json` 的容差块**逐字节未变**（`git diff` 只有 `tflite_bytes` 一项变化）——
  即**没有为了让 compare 变绿而调阈值**（NOTES §10.1 的硬要求）。

### 2.3 为什么这个阈值不是「太松」

相对口径 `atol/max|out| = 1.87e-2`，看起来松。它由两件事撑着：

1. **下限是实测的**，不是拍的：本图的 fp16 逐层舍入下限就是 9.25 ULP（§6.2 的分段表显示它主要来自
   `scores` 在 4628 量级上的 fp16 量化——bias 主导的分数是输出的 ~10⁴ 倍，fp16 在那里只有 4.0 的粒度）；
2. **真正的判据是「dev vs 设备语义模拟」**：1.709e-3（7 ULP），3388/7168 逐位相同。如果图里有布局错，
   这一项会像 §5 那样差 10³ 倍，而不会停在个位数 ULP。

---

## 3. 做法

### 3.1 图（`tools/build_decoder_layer.py`，61 层）

`@
x [8,896] fp16 ──MUL(x,x)─MEAN(axis=1)─ADD(eps[T,1])─RSQRT─MUL(x)─MUL(γ1)──> n1            (6 层)
n1 ─FC(Wq[896,896],bq)─RESHAPE[8,14,64]─SPLIT(axis=2)─MUL×4(cos/sin 常量表)─SUB/ADD─CONCAT─> q3   (11 层)
n1 ─FC(Wk[128,896],bk)─RESHAPE[8,2,64]─ 同上 RoPE ─────────────────────────────────────> k3      (11 层)
n1 ─FC(Wv[128,896],bv)─RESHAPE[8,2,64]────────────────────────────────────────────────> v3      (2 层)
k3 ─SPLIT(axis=1,2)─CONCAT([k0]×7+[k1]×7, axis=1)────────────────────────────────────> k14     (3 层)
v3 ─SPLIT(axis=1,2)─CONCAT([v0]×7+[v1]×7, axis=1)────────────────────────────────────> v14     (3 层)
q3/k14/v14 ─TRANSPOSE[1,0,2]×3───────────────────────────────────────────────────────> qt/kt/vt  (3 层)
scores = BMM(qt, kt, adj_y=1) [14,8,8]                                                          (1 层)
scores ─MUL(1/8 常量)─ADD(mask 常量 0/−4096)─SOFTMAX(β=1)────────────────────────────> p         (3 层)
a3 = BMM(p, vt) [14,8,64]                                                                       (1 层)
a3 ─TRANSPOSE[1,0,2]─RESHAPE[8,896]─FC(Wo, 零 bias)─ADD(x)──────────────────────────> r1        (4 层)
r1 ─RMSNorm(γ2)──────────────────────────────────────────────────────────────────────> n2       (6 层)
n2 ─FC(Wg) / FC(Wu)─LOGISTIC─MUL─MUL─FC(Wd)──────────────────────────────────────────> down     (6 层)
down ─ADD(r1)────────────────────────────────────────────────────────────────────────> out      (1 层)
`@

每一段都**照抄一个有板端证据的组件**，没有新造形状：

| 段 | 抄自 | 抄的是什么 |
|---|---|---|
| RMSNorm | `docs/OP-rmsnorm.md` §「订正：D=896 的 MEAN 可用性」 | `MUL→MEAN→ADD(eps)→RSQRT→MUL(x)→MUL(γ)` 6 层，D=896 板端 PASS（`ops/rms896_d896_T8_vA`）；eps 常数写成满形状 `[T,1]` 让 ADD 不需要广播 |
| q/k/v + RoPE | `docs/OP-qkv.md` | `FC(+bias) → RESHAPE[T,H,64] → SPLIT(axis=2) → MUL×4 → SUB/ADD → CONCAT`，11 层；bias 是 FC 第三个输入；k/v 宽 128 |
| KV 复制 | `docs/OP-gqa-scores.md` §4.6 | `SPLIT + 单条 14 输入 CONCAT`（`nary14`）**3 层**且板端逐位 PASS；3-CONCAT 写法是 5 层 —— 本票用 3 层那版 |
| scores / PV | `docs/OP-gqa-scores.md` §4.1、`docs/OP-pv-o.md` | 3-D `BATCH_MATMUL`（`adj_y=1` 打分、`adj 0/0` PV）= 各 1 个原生层；布局用**真正的 TRANSPOSE**（`RESHAPE` 会编译通过但数学错，实测 rel_l2 0.92/1.23） |
| 掩码 softmax | `docs/OP-masked-softmax.md` §11.2 | `MUL(1/8) → ADD(0/−4096 常量) → 原生 SoftMax(β=1)` 3 层、1 步；掩码烘成常量（生产形态，与运行期掩码逐位等价） |
| SwiGLU | `docs/OP-swiglu-ffn.md` | `FC(gate)/FC(up) → LOGISTIC → MUL → MUL → FC(down)` 6 层，silu 不融合 |

两个**没有**做的事，以及理由：

* **不把 γ 折进下游 FC 权重**（NOTES §5.8 允许、可省 2 层）：折了之后 numpy golden 必须同步折，
  多一个静默错的机会，而本票的任务是「拼装」不是「压层数」；
* **不把 1/8 折进 q 投影**（可省 1 层并把 score 量化误差降 8×）：那会改动 `docs/OP-qkv.md` 已验证的
  「q 是 RoPE 之后的张量」这个语义；本票保留组件原样，代价记在 §6.2（floor 由它主导）。

### 3.2 输入数据

图输入 = `embed.weight[TOKEN_IDS]` 的 fp16 行（`TOKEN_IDS = [9707, 11, 1879, 828, 11780, 13, 220, 16]`，
与票 03/06 同一组，`golden/input.bin` 逐字节可复现）；实测 `max|v| = 0.0704`、rms 0.0152
——**这就是「上一层交给这一层的 hidden」**。图的输出就是这一层交给下一层的 hidden，接口只有 `[8,896]` fp16。

图内中间量的实测尺度（决定 tolerance 的量级，也解释 §6.2 的下限）：

| 张量 | max\|v\| | rms | 张量 | max\|v\| | rms |
|---|---|---|---|---|
| x | 0.0704 | 0.0152 | r1 | 0.0821 | 0.0204 |
| n1（RMSNorm#1） | 1.0508 | 0.0989 | n2（RMSNorm#2） | 3.3535 | 0.7031 |
| q3 | 25.41 | 7.80 | mlp_y | 1.1289 | 0.1007 |
| k14 | 77.56 | 30.41 | down | 0.4709 | 0.1280 |
| v14 | 0.1254 | 0.0370 | **out** | **0.4836** | 0.1301 |
| scores | 4628 | 2306 | probs | 1.000 | 0.348 |

### 3.3 三个 numpy 参考（都独立于 flatbuffer 写盘代码，R3）

| 名字 | 是什么 | 用途 |
|---|---|---|
| **golden** | float32 精确链（输入/权重/表都取**落盘的 fp16**），**只在最后**舍入一次到 fp16 → `golden/expected.bin` | 登记的被测参考 |
| golden64 | 同链的 float64 版 | 只为回答「golden 自己够不够准」：与 float32 golden 的 fp16 码相同率 **7078/7168**，最大差 1.22e-4 |
| **sim** | float32 算术 + **每个张量边界舍入到 fp16**（设备语义：`docs/OP-qkv.md` §5.2 实测 7167/7168 逐位相同） | **设备无关下限**，支撑阈值（NOTES §10.1），**不**写进 `golden/` |

flatbuffer 写出代码里**没有任何数学**（只有张量、形状、常量），所以「golden 独立」这条不是形式上的：
两者除了夹具数据与 sin/cos 表/掩码常量之外零共享。

### 3.4 新增：device-free 的 numpy 图执行器（本票的排障工具，`--execcheck`）

票面要求「先算预算、再跑；失败要二分」。本票在**第一次板端失败之后**加了第三件武器：
`--execcheck` 把**已经写好的 `.tflite`** 反解出来（读 flatbuffer 里的算子、形状、常量缓冲），
用 numpy 按 TFLite 语义逐算子执行，再与 golden 比。它不是数值判据（它可以被和错误相同的理解污染），
而是**接线判据**：抓的是「编译通过但数学错」那一类（算子下标错、SPLIT/CONCAT 的输入顺序、
MUL 接错操作数、漏 TRANSPOSE、RESHAPE 顺序、常量数据没写进去……）。本次的 bug 正是被它一眼抓到
（§5.3），代价 0 个板端周期。三个图（full / attn_half / ffn_half）现在都是 **1.00 ULP 的 WIRING OK**。

---

## 4. 编译结果（`model.compile.log` 原文摘要）

`@
===Graph[0] has 1 Execution Step(s)===
Graph[0] Step[0]
 ├ StepId: 0
 ├ Target: MDLA_5_0
 └ Subgraph:  61 层（见 §8 的分段表，逐层与预测一致）
DRAM Usage:
       Target       Input  Output  Temp  Static  Code  Total
 [ 0]: MDLA 5.0     14K    14K     14K   28M     34K   28M
Total Memory:
  DRAM: 28M
  L1:   256K
EXIT=0   SIZE=29913364
`@

* **EXIT=0**、**1 个 Execution Step**、`Target: MDLA_5_0`、**无 bridge**（`--disallow-bridge` 恒开，
  日志里除命令行本身没有任何 `Bridging` 字样）；
* `Static 28M` = 28.50 MiB 权重（§8.2）；`Temp 14K` = 需要一个 `[8,896]` fp16 的 DRAM 暂存
  （14336 B 恰好是这个大小；与「r1 要跨整个 FFN 段存活」一致——**这是推断，未做逐张量验证**）；
* `.tflite` 29,878,152 B / `.dla` 29,913,364 B（权重原样搬进静态区，与 `docs/OP-swiglu-ffn.md` §4.4 一致）。

---

## 5. 第一次板端：FAIL，以及二分定位（证据全保留）

### 5.1 现象

第一次跑（git 里 `2bf6575` 的模型，61 层 / 1 步 / EXIT=0）板端：

| 图 | max_abs | rel_l2 | 判定 |
|---|---|---|---|
| 整块 | **2.0475** | **2.198** | **FAIL**（dev 范围 ±2.27，exp ±0.48） |
| attn_half（`x → r1`，48 层） | 4.571e-02 | 0.577 | FAIL |
| ffn_half（`r1 → out`，13 层） | 4.224e-01 | 0.643 | FAIL |

证据：三个目录的 `compare_firstrun_fail.txt`（**保留，未删**）；噪声底那一对也存了
（`out_firstrun_device.bin`；`.bin` 按仓库约定不入库，可由本节数字复现）。

### 5.2 二分：两个半图都错 ⇒ 错在**共用的那一段**

按票面要求「把图切成两半各自编译」：

* `attn_half` = `x → RMSNorm#1 → q/k/v+RoPE → 复制 → scores → 掩码softmax → PV → o_proj → +x = r1`（48 层）；
* `ffn_half` = `r1 → RMSNorm#2 → SwiGLU → +r1 = out`（13 层）；48 + 13 = **61**，两半没有重复层。

两半**各自都 FAIL 且 rel_l2 同档（0.577 / 0.643）** ⇒ 不可能是「某半段独有的 glue」，
只能是两半共用的东西：**RMSNorm 链**（唯一在两侧都出现的组件）或 FC/残差的通用写法。

### 5.3 定位到根因：`MUL(x,x)` 被接成了 `MUL(x, 1)`

device-free 的 `--execcheck` 在**修好之前**对这个模型直接报 NaN：`RSQRT` 吃到了负的 `mean`。
顺着看到 `rmsnorm_nodes()` 里写的是 `g.mul(x, name+"_sq")`，而 helper 的 `b=None` 分支会造一个
**全 1 常量** —— 于是 `sq = x`，`mean = mean(x)`（可为负），`rsqrt(负数)` 在设备上不产生 NaN
（板端 nan/inf 0/0）却把整块数值毁掉。**这是「编译 EXIT=0 + 单执行步 + 形状全对」的第三个独立反例**
（前两个：NOTES §10.2 的 reshape、`docs/OP-pv-o.md` §3.4 的 badlayout）。

修复：`g.mul(x, name+"_sq", b=x)`（同一张量作两次输入，`docs/OP-rmsnorm.md` §2.2 已验证可行）。
修复后 `--execcheck`：整块 1.00 ULP、attn_half 1.00 ULP、ffn_half 1.00 ULP（§3.4）。

> **方法学教训**：`--execcheck` 这类「设备无关的整图执行」应当**在第一次上板之前**就做。
> 本次是失败之后补的；它花 0 个板端周期就定位了问题，而两次板端往返花了 ~2 分钟设备时间。

---

## 6. 修复后的板端结果

### 6.1 整块（`ops/decoder_layer_t8_fp16/compare.txt`）

`@
op        : DECODER_LAYER_T8_FP16 {'T': 8, 'D': 896, 'Hq': 14, 'Hkv': 2, 'HD': 64, 'I': 4864, ...}
expected  : 7168 vals   device: 7168 vals   logical shape=(8, 896)
max_abs   : 2.319e-03
rmse      : 4.801e-04
rel_l2    : 3.690e-03
nan/inf   : exp 0/0  dev 0/0
range     : exp [-0.4626,0.4836]  dev [-0.4626,0.4839]
tol       : atol=0.0090332 rtol=0.0090332
PASS
`@

| 口径 | 值 |
|---|---|
| 绝对 | `max_abs = 2.319e-03` |
| 相对（max\|out\|） | **4.796e-03**（= 9.50 ULP @ 0.4836） |
| `rel_l2` | **3.690e-03** |
| 逐位相同 | 791 / 7168 |
| 登记阈值 | 9.0332e-3（用掉 **25.7%**） |
| 设备无关下限 | 2.258e-3（= 9.25 ULP）⇒ 实测是下限的 **1.03×** |
| dev vs 设备语义模拟 | **1.709e-3（7.00 ULP）**，3388/7168 逐位相同 |
| dev vs float64 精确链 | 2.353e-3 |
| 噪声底 | **0.000e+00**（两轮 7168/7168 code 逐位相同） |

### 6.2 分段下限（`--analyze`，设备无关）

每段「sim vs golden」，同一段自己量级下的 ULP：

| 段 | max\|v\| | floor（abs） | ULP(该段) | 说明 |
|---|---|---|---|---|
| n1 | 1.051 | 4.88e-04 | 0.50 | RMSNorm 只花半个码 |
| q3 / k14 | 25.4 / 77.6 | 1.56e-02 / 6.25e-02 | 1.00 / 1.00 | 与票 03 的实测一致 |
| scores | 4628 | 4.00 | 1.00 | **本图误差的主要来源**：bias 主导的分数在 4.6e3 量级上 fp16 粒度是 4.0 |
| probs | 1.000 | 1.47e-02 | 15.0 | softmax 把这些差别放大到概率上 |
| attn | 0.1254 | 1.72e-03 | 14.1 | 加权和继承 probs 的差 |
| r1 | 0.0821 | 2.21e-04 | 3.62 | 残差把注意力那部分摊薄 |
| n2 | 3.354 | 8.55e-03 | 4.38 | 归一化重新放大尺度 |
| mlp_y | 1.129 | 4.64e-03 | 4.75 | |
| down / out | 0.471 / 0.484 | 2.23e-03 / 2.26e-03 | 9.12 / **9.25** | FFN 把相对误差带到输出 |

⇒ **误差不随层数发散**（每段都是 0.5–15 ULP 的相对量），最终 9.25 ULP 主要是「scores 在 4.6e3 上的
fp16 量化 → softmax → PV」这条链的固有代价。要更小只有两条路（本票**不做**）：
把 1/8 折进 q 投影（score 量化粒度 /8），或用更高精度存 scores（设备没有这个选项）。

### 6.3 两个半图（二分证据，也各自 PASS）

| 图 | 层数 | 执行步 | max_abs | ULP | rel_l2 | 登记阈值 | 判定 |
|---|---|---|---|---|---|---|---|
| `attn_half`（x→r1） | **48** | 1 | 2.823e-04 | 4.62（max\|out\|=0.0821） | 2.618e-03 | 8.85e-04 | **PASS** |
| `ffn_half`（r1→out） | **13** | 1 | 7.324e-04 | 3.00（max\|out\|=0.4836） | 9.349e-04 | 1.95e-03 | **PASS** |

48 + 13 = **61** 层 = 整块层数（没有重复，也没有遗漏）。两半的 `EXIT=0` / 1 step / DRAM 见各自
`model.compile.log`（`ffn_half` 的 `Temp = 0`、`attn_half` 的 `Temp = 14K`）。

### 6.4 容差时序（可核对）

`@
10:34:47  第一次编译（EXIT=0，61 层，1 步）          -> commit 2bf6575 内含 model.json + compile.log
10:36:52  第一次板端运行（FAIL，max_abs 2.0475）      -> compare_firstrun_fail.txt
10:37:51  修复后重建（model.json 的容差块逐字节未变，只有 tflite_bytes 变）
10:38:17  重新编译
10:38:26  板端 PASS（max_abs 2.319e-03）              -> compare.txt
10:38:27  attn_half PASS / ffn_half PASS（同一次持锁会话内）
`@

---

## 7. 反事实：判据有没有牙齿（`--diag`，纯 numpy）

把「如果图真是这样」的几种可能错误写成 golden，再与设备输出比（阈值 1.34e-2 = atol + rtol·max\|out\|）：

| 反事实 | 与设备输出的 max_abs | 相对阈值 |
|---|---|---|
| 丢因果掩码（ADD 被折掉） | 6.781e-01 | **50.6×** |
| 没做 RoPE（表被忽略） | 6.339e-01 | **47.3×** |
| q/k/v bias 被丢 | 5.914e-01 | **44.1×** |
| 残差 1 被丢 | 5.283e-01 | **39.4×** |
| **eps = 0**（玩具块口径） | 2.677e-03 | **0.20×** |

⇒ ① 掩码、RoPE、bias、残差**都在图上真的生效**（差 40–51 个阈值，不可能看错）；
② **eps 在本层不可判别**：1e-6 vs 0 的差只有 0.20× 阈值（甚至小于 1 ULP），与票 03 的 P7 结论一致。
这就是 §2.1 P6 的落地，也是票面要求「记录 eps 口径差」的定量答案：

* 图里放的是**真实的 1e-6**（`fp16(1e-6) = 9.537e-07`，次正规，只有 ~6% 表示精度；以 `[T,1]` 满形状加入，不走广播）；
* 玩具块用的 0 与本层在数值上**无法区分**（差 < 1 ULP）；两者只在 `mean(x²) → 0` 的极端行上才分道扬镳，
  本夹具的 `mean(x²)` 在 1e-4–1 量级，离 1e-6 还有 2–6 个数量级；
* **所以：不要用「这一层 PASS」当「eps 通路正确」的证据**；eps 通路的证据在
  `docs/OP-rmsnorm.md` §2.4 的 eps=1.0 消融（离线差 0.374，板端仍 PASS）。

---

## 8. 层数与大张量分布（「能不能塞进单执行步」的实测依据）

### 8.1 层数：预测 61 / 实测 61，逐段一致

| 段 | 预测 | 实测（`model.compile.log` 里的层名） |
|---|---|---|
| RMSNorm #1 | 6 | `Mul,Mean,Add,RSqrt,Mul,Mul` |
| q 投影+RoPE | 11 | `FC,Reshape,Slice,Slice,Mul×4,Add,Sub,Concat` |
| k 投影+RoPE | 11 | 同上（H=2） |
| v 投影 | 2 | `FC,Reshape` |
| KV 复制 k / v | 3 + 3 | 各 `Slice,Slice,Concat`（14 输入单条 CONCAT） |
| TRANSPOSE ×3 | 3 | `Transpose`×3 |
| scores BMM | 1 | `BatchMatMul` |
| MUL+ADD+SoftMax | 3 | `Mul,Add,SoftMax` |
| PV BMM | 1 | `BatchMatMul` |
| TRANSPOSE+RESHAPE | 2 | `Transpose,Reshape` |
| o_proj + 残差 | 2 | `FC,Add` |
| RMSNorm #2 | 6 | 同 #1 |
| SwiGLU | 6 | `FC,Sigmoid,Mul,FC,Mul,FC`（层序不是拓扑序） |
| 残差 #2 | 1 | `Add` |
| **合计** | **61** | **61** |

> 任务书给的粗估是 40–50 层；按组件实测累加是 **61**。差的 11–21 层来自两处：q/k 的 RoPE 各 11 层
> （粗估按更早的口径只算 9 层）、两个 RMSNorm 各 6 层（eps 的 ADD 是 1 层）；反向的是 KV 复制按 3 层
> （`nary14`）而不是 5 层。**结论：层数必须从组件实测累加，不能粗估。**

### 8.2 大张量分布

| 张量 | 形状 | 字节 | 驻留 |
|---|---|---|---|
| `q_proj` / `o_proj` | [896,896] | 各 1,605,632（1.53 MiB） | DRAM Static |
| `gate` / `up` / `down` | [4864,896] / [896,4864] | 各 8,716,288（8.31 MiB） | DRAM Static |
| `k_proj` / `v_proj` | [128,896] | 各 229,376（0.22 MiB） | DRAM Static |
| γ1/γ2 + qkv bias + 零 bias | — | 28,928 | DRAM Static |
| RoPE 表（q+k）+ 掩码 | — | 34,560 | DRAM Static |
| **静态合计** | — | **29,882,368 B ≈ 28.50 MiB** | `Static 28M` |
| 中间最大激活 | `gate/up/silu/mlp_y` 各 [8,4864] | 各 77,824（76 KB） | L1（`L1 256K`） |
| 第二梯队 | `attn/q` [8,896] 等 | 各 14,336 | L1 + 1 个 DRAM 暂存（`Temp 14K`） |
| 注意力小张量 | `scores/probs` [14,8,8] | 各 1,792 | L1 |

**「塞进单执行步」的实测结论**：61 层 + 28.5 MiB 静态权重仍然只有 **1 个** Execution Step；
限制单步的不是层数，而是**静态权重体积**（28M 远小于 `/data` 6.4 GB），L1 只用了 256K（3 MB 里的 8%）。
→ **票 10 的 24 层串联在「单图单步」这条路上没有层数障碍**，真正的约束是 NOTES §6 的**常驻内存**
（24 × 28.5 MiB ≈ 684 MiB 静态权重 vs 空闲 RAM ~273 MB），那需要另一个实验（分图加载/换入），不是本票的结论。

### 8.3 掩码复用合规（票 05 §11.2 的硬要求）

* 本层实测 `S = max|scaled score| = 578.5`（未缩放 4628.2，÷8）；
* 票 05 要求 `S ≤ 1996` 才能沿用 `M = 4096`：**满足**（余量 3.45×）；
* 等价校验：把掩码从 −4096 换成 −8192，fp16 概率 **0/896 个 code 不同**（已饱和），
  被掩码位置的 `p` 最大值 **0.000e+00**（392 个位置）——与票 05 的板端结论一致。

---

## 9. 与票面 checklist 对照

| 票面要求 | 状态 | 证据 |
|---|---|---|
| 输入输出都只是 `[T,896]` fp16，host 不参与数学 | ✅ | `model.json` 只有 1 个输入 `x[8,896]`、1 个输出 `out[8,896]`；掩码/表/权重全为图内常量；`tools/e2e.sh` 只推 1 个输入文件 |
| 两次归一化用真实 eps=1e-6，口径差必须记录 | ✅ | `model.json.eps_register`（`graph_eps=1e-6`、`eps_stored_fp16=9.537e-07`、`toy_block_eps=0`）；§7 定量：eps=0 反事实只有 **0.20× 阈值** ⇒ **本层不可判别**，eps 通路的证据在 `docs/OP-rmsnorm.md` §2.4 |
| RoPE、GQA、掩码 softmax、FFN、两处残差全在图内 | ✅ | §4 的 61 层清单逐层可点名；§7 四个反事实（掩码/RoPE/bias/残差）各差 39–51× 阈值 |
| 单层 hidden 与 golden 对拍通过，阈值来自噪声底 | ✅ | `compare.txt` **PASS**（max_abs 2.319e-3 = 9.50 ULP、rel_l2 3.690e-3）；噪声底实测 **0**（两轮逐位相同）⇒ 阈值按 §10.1 走「设备无关 fp16 下限」分支：`atol=rtol=max(4·2⁻¹¹·max|out|, 4·floor)=9.0332e-3`，跑**之前**登记 |
| 单执行步、无 bridge；记录层数 | ✅ | §4：`===Graph[0] has 1 Execution Step(s)===`、`Target: MDLA_5_0`、`--disallow-bridge` 下 EXIT=0、**61 层** |
| 记录层数与大张量分布，作为「能否塞进单执行步」的实测依据 | ✅ | §8：逐段 61 层（预测=实测）、28.50 MiB 静态/DRAM、`Temp 14K`、L1 256K；并给出「层数不是障碍、常驻内存才是」的边界 |
| （任务书额外）第一次失败保留证据 | ✅ | §5：三个目录的 `compare_firstrun_fail.txt`；`2bf6575` 保留了 pre-fix 的 compile.log 与容差登记 |
| （任务书额外）golden 独立 + 记录 eps 口径 + 先预测再跑 | ✅ | §3.3（三种 numpy 参考，与 flatbuffer 写出码零共享）、§2.1（P1–P7 写于编译之前）、§2.2（容差先登记，修复后未变） |

---

## 10. 适用范围 / 不适用

**适用（下游可直接引用）**

1. **61 层整块 = 1 个执行步、无 bridge**，fp16、T=8、D=896、14Q/2KV、HD=64、28.5 MiB 静态权重
   （mdla5.0 / ncc 7.3.15 / `--disallow-bridge`）；
2. **数值口径**：与独立 float32 golden 的差 = max|out| 处 **9.5 个 fp16 码**（相对 4.8e-3），
   其中 **9.25 个码是 fp16 逐层舍入的固有下限**（设备无关模拟），设备只多 0.25 码；
   与「设备语义模拟」逐位相同率 47%；
3. **层数预算（逐段实测）**：RMSNorm 6 / q+RoPE 11 / k+RoPE 11 / v 2 / KV 复制 3+3 /
   TRANSPOSE 3 / 两个 BMM 各 1 / 掩码 softmax 3 / 收尾 4 / RMSNorm 6 / SwiGLU 6 / 残差 1；
4. **KV 复制用 `nary14`（3 层）**而不是 3-CONCAT（5 层）：本图在整块规模上复验了票 04 §4.6；
5. **掩码**：常数烘进图、`0/−4096`、加在 `MUL(1/8)` 之后，3 层；本层 `S=578.5 ≤ 1996`，饱和性 0/896；
6. **`--execcheck`（numpy 反解 `.tflite`）**：任何新拼装图都应在第一次上板前过一遍（本次的 bug 就是它抓的）。

**不适用**

* 24 层串联 / 多层共享一张图（票 10）：本票只证「1 层 + 28.5 MiB 权重 = 1 步」，多层需要重测；
* T≠8、batch>1、decode（sin/cos 是常量表 ⇒ T 必须静态）；
* int8（全 fp16 是 spec 的决定）；真实权重档（`k_proj.bias` abs_max 130 vs 夹具 77.8 ⇒ S 会更大，
  阈值要按同一公式重算）；
* **性能**：2.25 / 2.60 ms/inf 只作记录（小图启动主导，仓库既有结论：不作性能判断）；
* **eps 通路的正确性**：本层的 PASS 不构成 eps 证据（§7）。

---

## 11. 剩余疑问（UNKNOWN，不圆）

1. **`Temp 14K` 是哪个张量**：字节数恰好等于一个 `[8,896]` fp16 激活，与「r1 跨 FFN 段存活」一致，
   但没有逐张量证据（`.dla` 未逆解析，也没做「把 r1 换成常量」的对照）。标注为推断。
2. **设备在 `scores` 层的内部精度**：sim 与设备在整块上只有 3388/7168 逐位相同（半图分别是 5420、4121），
   与票 06 §3.5 的「7166/7168 相同」相比更散。候选解释：61 层里每一层的取整方向差异累积；
   要定死需要逐段把中间量做成图输出（本票只做了 3 个粒度：整块 / 48 层 / 13 层）。
3. **`floor = 9.25 ULP` 能不能压到 1–2 ULP**：只有「把 1/8 折进 q 投影」这条（未做，见 §3.1）。
   如果票 10 的 24 层串联把误差累到不可接受，这是第一个该试的改动；代价是 golden 同步折。
4. **`nary14` 在 int8 下**未测（票 04 §7.5 同一条）；本 spec 全 fp16，不影响。
5. **61 层是否是「单步」的上限**：本票没做「加更多层」的扫描。已知层数不是这里的瓶颈（§8.2 的推理），
   但**没有实测**上限。
6. **板端 `-c 3` 的 3 次重复是否覆盖了所有数据相关分支**：仓库既有做法都是这么测的，本票沿用，未深究。

---

## 12. 复现

`@bash
export QWEN25_WEIGHTS_DIR='/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen'

# 1) 层数预算（不需要权重，秒级）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t09' && /usr/bin/python3 tools/build_decoder_layer.py --predict"

# 2) 生成 tflite + golden + model.json（含先登记的容差）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t09' && export QWEN25_WEIGHTS_DIR='$QWEN25_WEIGHTS_DIR' && \
  /usr/bin/python3 tools/build_decoder_layer.py --out ops/decoder_layer_t8_fp16"

# 3) 接线自检（设备无关；第一次上板之前做）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t09' && export QWEN25_WEIGHTS_DIR='$QWEN25_WEIGHTS_DIR' && \
  /usr/bin/python3 tools/build_decoder_layer.py --execcheck --out ops/decoder_layer_t8_fp16"

# 4) 编译（锁外）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t09' && \
  ./tools/compile.sh ops/decoder_layer_t8_fp16/model.tflite ops/decoder_layer_t8_fp16/model.dla"

# 5) 板端（持锁；前后各一次 healthcheck，两轮 = 噪声底）
export NPU_LOCK='/d/deepseek_ workspace/mtk/.npu-board.lock'
./tools/devlock.sh run -- bash -c './tools/healthcheck.sh --quick
  ./tools/e2e.sh ops/decoder_layer_t8_fp16 --repeat 3
  cp ops/decoder_layer_t8_fp16/out_device.bin ops/decoder_layer_t8_fp16/out_r1.bin
  ./tools/e2e.sh ops/decoder_layer_t8_fp16 --repeat 3
  cp ops/decoder_layer_t8_fp16/out_device.bin ops/decoder_layer_t8_fp16/out_r2.bin
  ./tools/healthcheck.sh --quick'

# 6) 诊断（锁外）：设备 vs golden / 设备语义模拟 / 反事实 / 噪声底
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t09' && export QWEN25_WEIGHTS_DIR='$QWEN25_WEIGHTS_DIR' && \
  /usr/bin/python3 tools/build_decoder_layer.py --diag --out ops/decoder_layer_t8_fp16"

# 7) 二分半图（同 2)–5)，换 --variant attn_half|ffn_half，输出到 ops/_decoder_bisect/<v>）
`@

**可复现性**：`.tflite` 与 `golden/*.bin` 由同一 builder 重建（同一 seed、同一夹具）**逐字节相同**（实测：
`model.tflite` sha256 `6d2e10c7…`、`golden/expected.bin` `91330e5d…`、`golden/input.bin` `50496119…`，重建两次一致）；
`.dla` **不可复现**、只作产物标识（NOTES §4、`docs/OP-pv-o.md` §2.4）。

---

## 13. 交接

* **新增**：`tools/build_decoder_layer.py`（builder + `--predict` / `--execcheck` / `--analyze` / `--diag` +
  `--variant attn_half|ffn_half` 二分）、`ops/decoder_layer_t8_fp16/`、
  `ops/_decoder_bisect/{attn_half,ffn_half}/`（二分证据）、本文。
* **未改公共文件**：`e2e.sh` / `compare.py` / `compile.sh` / `devlock.sh` / `healthcheck.sh` /
  `build_ops.py`（只 import）、五个组件 builder、`.scratch/` 全部原样。
* **给票 10 的直接输入**：① 61 层整块 = 1 步（层数不是障碍）；② 每层 28.5 MiB 静态权重 ⇒ 24 层 ≈ 684 MiB，
  与 NOTES §6 的 ~273 MB 空闲 RAM 冲突，**这是票 10 的核心问题**；③ 本层 `compare.txt` 就是「逐层对拍」的
  单层基线：多层判据沿用 `atol=rtol=4·2⁻¹¹·max|out|` 形式，但 `max|out|` 与 floor 必须**每层重算**
  （夹具里有 `layer1.*`，换权重即可）；④ 任何新拼装图先跑 `--execcheck`。
* **给 Lead**：本票没有需要 Lead 改公共工具的地方。`docs/EXPERIMENTS.md` / `CAPABILITY.tsv` 的登记留给 Lead；
  建议一行：`DECODER_LAYER(T8, fp16, 61 层) ✅ 1 step / 无 bridge / 28.5 MiB 静态 / 板端 9.5 ULP（下限 9.25）`。

---

## 14. T=128 prefill 档（2026-11-13；spec 的「T=128 紧随其后并列为必过项」）

> 范围：同一张 1 层整块、同一套冻结夹具（`layer0.*`）、同一台设备、同一把板端锁，
> 只把序列长度从 8 换成 **128**。交付：`ops/decoder_layer_t128_fp16/` + `tools/build_decoder_layer.py --T` + 本节。
> 这一节之前，仓库所有的「整块」证据都是 T=8；spec 明令 **T=128 通过前不得声称真实 prefill 可用**（User Stories 21/22）。

### 14.1 做法：一个开关，不是一份新代码

票 09 的 builder 里 `T` 是模块常量。这次把它变成 `--T`（默认 8）经 `configure()` 重新绑定，
凡是**依赖 T 的东西**都跟着变，且只有这些东西：

| 跟着 T 变 | 位置 | T=128 的取值 |
|---|---|---|
| 图输入/输出形状 | `g.act("x", (T,D))`、golden 形状 | `[128,896]` fp16（229 376 B） |
| RoPE sin/cos 常量表 | `tables()`，`[T,heads,32]` ×2 ×(q,k) | 4 张：`128×14×32`、`128×2×32` |
| 因果掩码常量 | `causal_mask()`，`[14,T,T]` | `14×128×128` fp16 = 458 752 B |
| KV 复制 / BMM 形状 | `kv14()`、`scores[14,T,T]`、`PV[14,T,64]` | `[14,128,128]` / `[14,128,64]` |
| numpy golden | `forward()` 全程 | 114 688 个 fp16 code |
| 掩码常数 M | `derive_mask(S)`（本层实测 S 决定） | 见 14.2 |
| 输入 token | `token_ids(T)` | 128 个**互不相同**的 embedding 行 |

**token id 规则**（写进 `model.json` 的 `input_tokens`，T≠8 才写，T=8 的产物因此逐字节不变）：
前 8 个沿用票 03/06/09 的 `[9707, 11, 1879, 828, 11780, 13, 220, 16]`；其余按
`id(i) = (41 + 1103·i) mod 151936`（1103 与词表大小互素、`1103·119 < 151936`，所以 128 个 id
互不相同，也都不撞前 8 个）。**用 128 个不同的行**而不是把 8 行复制 16 份，是为了让 128 个位置真的不同，
否则掩码/softmax 的结论会被「每 8 行一模一样」这种退化结构污染。

**T=8 的产物逐字节不变**（这是硬要求，实测）：`model.tflite` sha256 `6d2e10c7c9abd21999482eb2d8ff7d05
d3941e2e81596f6ea175ea3bde76dc9f`、`golden/expected.bin` `91330e5d…`、`golden/input.bin` `50496119…`、
`model.json` 与提交里的**逐字节相同**（用 `--out /tmp/...` 重建后 `diff` 验证，`c702006`）。

### 14.2 事前登记（写于任何板端动作之前，2026-11-13）

**掩码必须重新量**（票 05 §11.2 的硬要求）。本层实测：

    max|scores| = 5165.76  ->  S = max|scaled score| = 645.72      (T=8 时是 578.52)
    有效区间 M ∈ [2S+104, 65504-S] = [1395.4, 64858.3]
    => 沿用 M = -4096（2^12，fp16 精确）；余量：距下界 2.94×，距上界 15.8×
    饱和性实测：M = -4096 vs -8192 的 fp16 概率 code 差异 = 0 / 229376

即 **T 变长后 S 只从 578.5 涨到 645.7，−4096 仍在区间中部**，不需要换常数；但这是**量出来的**，不是沿用。
（如果 T 再大 3 倍、S 到 ~2000 以上，`derive_mask()` 会自动抬到下一个 2 的幂；区间都容不下时它**直接报错**而不是硬塞。）

**容差（已写进 `model.json`，跑之前）**：`max|out| = 0.6270`，该量级 fp16 ULP = 4.883e-4：

* **设备无关下限** floor = `1.6972e-02` = **34.75 ULP**（numpy 逐层模拟：每个张量边界落一次 fp16、层内 fp32）；
* 登记 `atol = rtol = max(4·2⁻¹¹·max|out|, 4·floor) = 6.7871e-2`（`4·2⁻¹¹·max|out|` = 1.22e-3，被 `4·floor` 支配）；
* `threshold@max|out| = atol + rtol·max|out| = 0.1105`；
* `tolerance_revision` 与推导理由写在 `model.json`；**没有在跑之后改过**。

**这个阈值为什么这么大、它还算不算判据**——必须说清楚，因为 **0.0679 比 NOTES §4 的 2e-2 宽**：

* floor 的 99% 来自一个阶段：`probs`（sim vs golden 在这一阶段就差 0.133，即 136 ULP）。原因不是实现，
  是 **fp16 存不下这些分数**：本层分数被 `q/k/v_proj.bias` 抬到 ±5165，fp16 在该量级的粒度是 **4.0**，
  过 `MUL(1/8)` 后仍是 **0.5**；而 softmax 对 0.5 的敏感度在最平的 row 上就是 ~12%。
  实测 1792 个 (head,row) 里 **11.6%** 的 row 出现 >1e-2 的概率差异。
* 于是**真正的判据是「设备 vs 逐层 fp16 模拟」**（`--diag` 的第二步），它与 golden 的差只是「fp16 存储的必然」。
  这一条与票 09 §2.3 的口径一致：整块判据用登记阈值，实现正确性用 dev-vs-sim。

**预测（跑之前写的，逐条在 14.4 复核）**

| # | 预测 | 若不成立意味着 |
|---|---|---|
| P1 | 板端 **PASS**，`max_abs` 落在 floor 附近（30–45 ULP 量级），不是 1e-1 量级 | 远大于 floor ⇒ 接线/布局错（NOTES §10.2 那一类），不是精度 |
| P2 | 噪声底 = **0**（同 `.dla` 同输入两轮，114 688 个 code 逐位相同） | 阈值按 3–5× 噪声底重定，但**必须在跑之前**重登记 |
| P3 | **dev vs 逐层 fp16 模拟** ≪ dev vs golden（前者个位数 ULP） | 若两者同量级 ⇒ 设备行为与「边界落 fp16」模型不符，floor 的推导要重做 |
| P4 | 层数 **61**（与 T=8 相同），1 个执行步，无 bridge | 层数变多 ⇒ 某个算子按 T 被切成多段（例如 TRANSPOSE/SPLIT 分块） |
| P5 | 常驻内存仍是 **~29 MiB 静态**、Temp 从 14K 涨到 ~1.2M | Temp 撑爆 ⇒ 需要重新考虑 T 的上限 |
| P6 | `probs` 的 argmax 与 golden 一致（掩码位恒为 0） | 不一致 ⇒ 掩码或 softmax 在长 row 上失效 |
| P7 | 相对误差 `max_abs/max|out|` 在 1e-2 量级，**不是** 1e-3 | 若真到 1e-3，说明 T=128 反而更准 —— 那要先怀疑 golden 写错了 |

### 14.3 编译（与 T=8 的差异只在内存，不在层数）

`ops/decoder_layer_t128_fp16/model.compile.log` 原文摘要：

    EXIT=0   Target: MDLA_5_0   ===Graph[0] has 1 Execution Step(s)===
    61 个原生层（层清单与 T=8 逐项相同，顺序也相同）
    DRAM:  Input 224K  Output 224K  Temp 1.2M  Static 29M  Code 31K  Total 30M
    L1:    2.8M（--l1-size-kb=3072 的 hint 之内）
    SIZE=30609269   SHA256=0592b8ea3bd46a34...

**层数为什么不变（61 → 61）**：T 只改**维度**，不改算子数。整块里没有任何算子按 T 被切段
（TRANSPOSE/SPLIT/CONCAT/BMM/SOFTMAX 在 `[14,128,128]` 上仍各是 1 层）。这与「层数随 T 增长」的
直觉相反，也是这次值得记下来的一条：**层数预算只跟图的拓扑有关，不跟 T 有关**；T 影响的是常驻内存。

**内存为什么涨**：Temp 从 14K 涨到 1.2M（`scores`/`probs` 各 458 K + `mlp_y` 1.2 M 的池化），
Input/Output 从 14K 涨到 224K（`128×896×2`）。Static 几乎不变（29.4 MiB 权重 + 0.94 MiB RoPE 表与掩码）。
**离上限都还很远**：Temp 1.2 M ≪ 可用 DRAM，也没有把单步撑破。

### 14.4 板端（`compare.txt`）与噪声底

持锁跑两轮（每轮 `e2e --repeat 3`，前后各一次 `healthcheck --quick`，两次 healthcheck 都 PASS）：

    op        : DECODER_LAYER_T128_FP16 {'T':128, 'D':896, 'Hq':14, 'Hkv':2, 'HD':64, 'I':4864}
    expected  : 114688 vals   device: 114688 vals   logical shape=(128, 896)
    max_abs   : 1.733e-02        rmse: 1.671e-03        rel_l2: 1.311e-02
    nan/inf   : exp 0/0  dev 0/0  range: exp [-0.5786,0.627]  dev [-0.5791,0.6304]
    tol       : atol=0.0678711 rtol=0.0678711   ->   PASS

| 量 | 值 | 口径 |
|---|---|---|
| `max_abs` | **1.7334e-2** | = **35.50 ULP** of `max|out|=0.6270`；相对 **2.765e-2** |
| `rmse` / `rel_l2` | 1.671e-3 / **1.311e-2** | |
| 逐位相同 code | 4185 / 114688 | |
| **噪声底** | **0.0000e+00**（0/114688） | 同 `.dla` 同输入两轮 ⇒ 硬件确定性（与票 09 相同） |
| **设备 − 下限** | **+0.75 ULP** | `|dev−golden| − |fp16sim−golden| = 35.50 − 34.75` |
| dev vs fp16 sim | 8.728e-3 = **17.88 ULP**，26990/114688 逐位相同 | 见下 |
| 反事实判别力 | 丢掩码 **7.02×**、丢 RoPE **6.79×**、丢 bias **5.36×**、丢残差 **5.36×**、`eps=0` **0.15×** | 相对登记阈值 |

**噪声底 = 0 ⇒ 阈值不能按「3–5× 噪声底」取**（NOTES §4 的退化分支）。这里的退化与票 09 不同：
票 09 是「floor 本身只有 1e-3 相对、噪声为 0」，这里是 **floor 很大（34.75 ULP）而噪声为 0**。
所以阈值由 floor 撑起（`4×floor`），而**判定实现正确与否靠下面这条更强的对拍**：

> **设备与「逐层 fp16 舍入」模拟的差 = 17.88 ULP，小于该模拟与 golden 的差（34.75 ULP）。**
> 即：设备相对于「理想 fp16 设备」的偏离，比「理想 fp16 设备」相对于 fp32 golden 的偏离还小。
> 实现没有额外引入误差，剩下的差全部由 fp16 存储决定。与 NOTES §10.1「设备误差只应比下限多 1–2 ULP」一致
> （实测 **+0.75 ULP**）。

`noise_floor.json`（`ops/decoder_layer_t128_fp16/noise_floor.py` 生成）里存了完整账：
`noise_floor_value = 0.0`、`device_vs_golden.max_abs_in_ulp = 35.5`、
`fp16_sim_vs_golden.max_abs_in_ulp = 34.75`、`device_minus_floor_in_ulp = 0.75`，以及逐阶段 floor 表
（`probs` 136.2 ULP 是 floor 的唯一主要来源；`n1` 0.5 ULP、`scores` 1.0 ULP）。

### 14.5 预测复核（逐条）

| # | 预测 | 实际 | 结论 |
|---|---|---|---|
| P1 | PASS，`max_abs` 落在 floor 附近（30–45 ULP） | PASS，35.50 ULP（floor 34.75） | ✅ |
| P2 | 噪声底 = 0 | 0.0000e+00，0/114688 | ✅ |
| P3 | dev vs 逐层 fp16 模拟 ≪ dev vs golden（前者个位数 ULP） | **17.88 ULP vs 35.50 ULP**，约一半，不是个位数 | ❌ **预测偏乐观**，见下 |
| P4 | 层数 61、1 步、无 bridge | 61、1 步、无 bridge | ✅ |
| P5 | 静态 ~29 MiB、Temp → ~1.2M | Static 29M、Temp 1.2M | ✅ |
| P6 | `probs` argmax 与 golden 一致 | golden vs fp16 模拟：**1779/1792 = 99.27%**；掩码位概率恰为 0 | 部分 ✅（13 个 row 翻转） |
| P7 | 相对误差在 1e-2 量级 | 2.765e-2 | ✅ |

**P3 为什么错，以及为什么它不影响结论**：17.88 ULP 不是「实现错」，是**近并列 row 的 softmax 把
任何一点算术差放大**。同一批 row（top1 与 top2 的 scaled 分数相差不到一个 fp16 步长 0.5）在
golden↔sim 之间就已经翻转了 11.6%；设备与 numpy 的 fp32 累加顺序不同，会在同一批 row 上再翻转一次，
于是 dev-vs-sim 也有 17.9 ULP。判据因此不能是「dev ≈ sim 到个位数」，只能是
「dev 不比 floor 差多少」——**实测 +0.75 ULP，这才是本档真正的结论**。
反过来说：**如果哪天 `device − floor` 变成 10 ULP 以上，那就是真问题**。

### 14.6 适用范围 / 不适用

* **适用**：1 层 decoder 整块、`T=128`、fp16、14Q/2KV GQA、冻结随机权重夹具 `layer0.*`。
  这是「**prefill 的单层几何在 T=128 上可以整块下沉**」的证据。
* **仍然不是**「真实 prefill 可用」：24 层串联、logits、KV cache 都还没在 T=128 上做过（§10 的
  「24 层 ≈ 684 MiB vs 设备空余 RAM」一条都没解决）。spec 的 T=128 门槛是「单层整块必过」，
  这一条现在满足；「真实 prefill」这句话仍然不能说。
* **不适用**：变长序列（T 是静态的，RoPE 表是数据）、多 batch、int8、任何性能结论——顺带记下的
  5.28 ms/inf、176.9 FPS 是 `--repeat 3` 的板端计时，**不是**本 spec 的判据。
* **阈值不能搬**：0.0679（T=128）与 0.0090（T=8）不可互换，也不能搬到别的层；floor 依赖该层的
  `max|out|` 与分数动态范围，每层必须重算（`model.json` 的 `tolerance_basis` 就是给这件事留的痕）。

### 14.7 剩余疑问（UNKNOWN，不圆）

1. **ncc 的 FC batch 只支持 1/2/4/8**（`strings ncc-tflite` 里有 `fc batch 1/2/4/8`）。本图在 T=128
   没有因此受限（FC 自由分块），但**T 继续变大时它会以什么形式暴露，没有测**（例如 T=512 的
   `[512,896]` FC、或 `[512,4864]` 的 MLP）。
2. **`1 Execution Step` 是否等于「真单步」**：日志不给内部 dma/loop 计数；`5.28 ms/inf` 对 1 层
   28.5 MiB 权重（≈3.3 GB/s 有效带宽）偏慢，可能与分块/循环有关，但**没有测**。
3. **24 层在 T=128 下的常驻内存**没有测（每层 29.4 MiB ⇒ 24 层 ≈ 706 MiB 静态 + 每层 1.2 M 中间量）。
4. **`probs` 在近并列 row 上的 ~12% 抖动**是夹具性质（bias 主导到 ±500）还是真实 Qwen 权重也如此，
   未验证；若真实权重也这样，24 层串联时这些抖动会成为误差放大源，**值得在票 10 之前先量**。
5. **掩码常数的 T 依赖**：本层 T=128 的 S=645.7 仍远小于 −4096 的适用上界，所以 M 没换；但
   `S ∝ 分数` 而分数被 bias 主导，**T 增长时 S 会不会突然跳上去没有模型**，只能每层实测（`derive_mask` 会兜住）。

