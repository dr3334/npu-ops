# OP-actreduce —— 激活与归约算子在 MDLA_5_0 上的能力与数值

> 范围：**RELU / LOGISTIC / TANH / SOFTMAX / MEAN / SUM**（单算子，int8 与 fp16）。
> 采集：2026-11-12 · 分支 `feat/actreduce`（worktree `wt-actreduce`）· ncc-tflite **7.3.15**
> · `--arch=mdla5.0 --num-mdla=2 --l1-size-kb=3072 --disallow-bridge`
> · 板：MT6897（Titan 1 / MT8792），`ro.board.platform=mt6897`，`Android 16`，
> fingerprint `HEADWOLF/F8/F8:16/BP2A.250605.031.A3/1766461355:userdebug/release-keys`
> · 全部板端执行持同一把设备锁 `/d/deepseek_ workspace/mtk/.npu-board.lock`。

**一句话结论**：六个算子全部**编译成真正的 MDLA 层**（`ReLULayer` / `SigmoidLayer` / `TanhLayer` /
`SoftMaxLayer` / `MeanLayer` / `SumLayer`），`--disallow-bridge` 下**没有任何 CPU 桥接**；
16/17 个模型在板上跑通且**全部满足事先登记的数值契约**（多数逐位相等）。
**SOFTMAX 是原生 NPU 算子**——注意力的 softmax 不需要回退 CPU。
唯一失败的是**宽度不满足"输入行 16 字节对齐"**的形状（编译期失败，非数值问题），
该规则已用交叉实验确认为**字节**约束而非元素约束，并且能统一解释 `docs/CAPABILITY.md` 里
FULLY_CONNECTED 的 "K 必须是 16 的倍数"。

---

## 0. TL;DR 结果表

| op | 形状 | dtype | 编译 | 发射的层（exec-plan） | 板端 | 数值结果（登记契约） |
|---|---|---|---|---|---|---|
| RELU | [1,64] | fp16 | ✅ exit=0 | `ReLULayer<0>{RELU}` → MDLA_5_0 | ✅ | **逐位相等**（atol=0，max_abs=0） |
| RELU | [1,64] | int8 | ✅ exit=0 | `ReLULayer<0>{RELU}` | ✅ | **逐位相等** 64/64（int8_tol=0） |
| LOGISTIC | [1,64] | fp16 | ✅ exit=0 | `SigmoidLayer<0>{LOGISTIC}` | ✅ | **逐位相等**（max_abs=0，atol=2e-3 只用 0%） |
| LOGISTIC | [1,64] | int8 | ✅ exit=0 | `SigmoidLayer<0>{LOGISTIC}` | ✅ | **逐位相等** 64/64（int8_tol=2 只用 0） |
| TANH | [1,64] | fp16 | ✅ exit=0 | `TanhLayer<0>{TANH}` | ✅ | max_abs=4.883e-4（= 1 个 fp16 码），61/64 逐位相同 |
| TANH | [1,64] | int8 | ✅ exit=0 | `TanhLayer<0>{TANH}` | ✅ | **逐位相等** 64/64（int8_tol=2 只用 0） |
| **SOFTMAX** | [1,8,16] | fp16 | ✅ exit=0 | `SoftMaxLayer<0>{SOFTMAX}` | ✅ | max_abs=2.441e-4（1 个 fp16 码 @p=0.28） |
| **SOFTMAX** | [1,8,16] | int8 | ✅ exit=0 | `SoftMaxLayer<0>{SOFTMAX}` | ✅ | **逐位相等** 128/128 |
| SOFTMAX | [1,1,64] | fp16 | ✅ exit=0 | `SoftMaxLayer<0>` | ✅ | max_abs=6.104e-5（L=64 单头） |
| SOFTMAX | [1,12,128] β=0.125 | fp16 | ✅ exit=0 | `SoftMaxLayer<0>` | ✅ | max_abs=1.526e-5（**β 被正确采纳**） |
| SOFTMAX | [1,4,64] | int8 | ✅ exit=0 | `SoftMaxLayer<0>` | ✅ | 1/256 差 1 码（int8_tol=2，两次运行一致） |
| MEAN | [1,8,16]→[1,8] axis=2 | fp16 | ✅ exit=0 | `MeanLayer<0>{MEAN}` | ✅ | **逐位相等**（8/8） |
| MEAN | [1,8,16]→[1,1,16] axis=1 | fp16 | ✅ exit=0 | `MeanLayer<0>{MEAN}` | ✅ | **逐位相等**（16/16，非末轴） |
| MEAN | [4,16]→[4,1] axis=1 | int8 | ✅ exit=0 | `MeanLayer<0>{MEAN}` | ✅ | **逐位相等** 4/4（int8_tol=1 只用 0） |
| SUM | [1,8,16]→[1,8] axis=2 | fp16 | ✅ exit=0 | `SumLayer<0>{SUM}` | ✅ | **逐位相等**（8/8） |
| SUM | [4,16]→[4,1] axis=1 | int8 | ✅ exit=0 | `SumLayer<0>{SUM}` | ✅ | **逐位相等** 4/4（int8_tol=0） |
| RELU | **[1,13]** | fp16 | ❌ exit=1 | 层已映射但需桥接 | – | **编译期失败**，见 §5.1 |

产物：`ops/<op>_<shape>_<dtype>/{model.json, model.compile.log, compare.txt, e2e.json}`，
模型由 `tools/build_actreduce.py` 可复现生成（`--list` 列出全部 17 个 spec）。

---

## 1. 方法与证据链

### 1.1 模型生成（`tools/build_actreduce.py`）

复用 `tools/build_ops.py` 的 flatbuffers 脚手架，按 TFLite 规范而不是按编译器喜好构造：

- 单输入单输出；`RELU/LOGISTIC/TANH` 是**无 options** 的独立 op，
  `SOFTMAX` 用 `SoftmaxOptions{beta}`，`MEAN/SUM` 用 `ReducerOptions{keep_dims}` + **常量 axis 张量**；
- int8 量化按 **TFLite op 规范**：RELU in/out 同 scale+zp0；LOGISTIC out `1/256, zp=-128`；
  TANH out `1/128, zp=0`；SOFTMAX out `1/256, zp=-128`；MEAN/SUM out scale == in scale（无 requant）；
- `model.tflite` 的输入张量**不带 data buffer**（AGENTS.md §7.3）。

### 1.2 golden 与被测实现独立（AGENTS.md R3）

`golden(expected.bin)` 由 `build_actreduce.py` 自己算，**不经过 ncc、不经过 TFLite interpreter、不经过设备 runtime**：

- 浮点：从**实际存下来的** fp16 输入值升到 float64 计算 → 只做一次「舍入到存储类型」；
- 整数：int64 精确累加/精确超越函数 → 再按登记的量化规则取码。

### 1.3 先登记容差，再上板（AGENTS.md §1 / R1）

容差写在 `SPECS` 里，随模型一起落到 `model.json`（`int8_tol` 或 `atol`/`rtol` + `tolerance_rationale`），
**在任何一次编译/板端执行之前**就已确定，事后没有改动：

| op | dtype | 登记契约 | 理由（事前） |
|---|---|---|---|
| RELU | fp16 | atol=0, rtol=0 | `max(x,0)` 在 fp16 下无损 |
| RELU | int8 | int8_tol=0 | in/out 同 scale、zp=0 ⇒ 码值就是 `max(q,0)` |
| LOGISTIC | fp16 | atol=2e-3, rtol=1e-2 | ≈4 个 fp16 ULP @0.5（u=4.9e-4），容忍 LUT 近似 |
| LOGISTIC | int8 | int8_tol=2 | 定点 LUT vs 精确 sigmoid，2/256 码 |
| TANH | fp16 | atol=2e-3, rtol=1e-2 | 同上，输出域 [-1,1] |
| TANH | int8 | int8_tol=2 | 同上 |
| SOFTMAX | fp16 | atol=2e-3, rtol=1e-2 | 期望 fp32 内部计算 + fp16 存储 |
| SOFTMAX | int8 | int8_tol=2 | 定点 softmax 的码值差 |
| MEAN/SUM | fp16 | atol=2e-3 / 0.1 | MEAN 无放大；SUM=16 项、\|sum\|≤64 → ULP=0.0625，2 码 |
| MEAN | int8 | int8_tol=1 | 只有 requant 边界 |
| SUM | int8 | int8_tol=0 | out scale == in scale ⇒ 精确整数和 + 饱和 |

`tools/compare.py` 现在优先读取 `model.json` 的 `atol/rtol`（与 `int8_tol` 同等待遇），
因此 `compare.txt` 就是**按登记契约**判定的结果，而不是它的内置默认值。

### 1.4 板端串行 + 状态门（AGENTS.md R5 / §3.4）

- 所有 adb 操作都在 `tools/devlock.sh run -- ...` 里，锁路径与其它 worktree 相同；
- 12 个主算子前后各跑一次**已知正常程序**（`ops/fc_1x64x64_int8`，即 `healthcheck.sh` 的 `GOOD_OP`）：
  预门 `exact YES 64/64`、后门 `exact YES 64/64`；
- 每批结束扫 `dmesg`：`apusys/mdla timeout|fail|error` 命中 **0** 条。
  ⇒ 这些 kernel 没有留下状态污染（§3.4 的"会污染下一次执行"不成立）。
- 注：门的证据只在本文记录，`ops/fc_1x64x64_int8/` 的跟踪文件已在提交前还原，
  以免把别人的算子产物混进本分支。

### 1.5 五次结论分开（AGENTS.md R7）

编译成功 / ioctl 成功 / 数值正确 / 设备状态正常 / 性能改善，本文分开写：
本文**没有**性能结论，表里出现的 `ms/inf` 只表示"跑起来了"的量级
（各批观测区间 0.4–1.6 ms，含进程内提交与等待开销，批次间抖动明显，不可用于比较算子快慢）。

---

## 2. 逐算子：假设 → 预测 → 做法 → 结果 → 适用范围 → 剩余疑问

### 2.1 RELU（对照组）

- **假设 H1**：MDLA 有 elementwise 激活通路，RELU 会编译成独立层且数值精确。
- **预测**（跑之前写）：`ReLULayer` on `MDLA_5_0`，int8 逐位相等、fp16 逐位相等。
- **做法**：`[1,64]` int8（full-range 码，含 −128/127 两个端点）与 fp16（含 −6/0/+6）；
  golden = `max(x,0)`。
- **结果**：✅ 编译 `ReLULayer<0>{RELU}`；板端 int8 **64/64 逐位**、fp16 **max_abs=0**。
  宽度 [1,13] **编译失败**（见 §5.1），是形状边界而不是数值问题。
- **适用性**：RELU 可以作为独立 NPU 算子使用（不必依赖 conv/FC 的 fused activation）。
  输入最内维字节数须为 16 的倍数（§3）。
- **剩余疑问**：`RELU6`/`LEAKY_RELU`/`ELU` 未测；与 conv/FC 的 fused 形式未对比。

### 2.2 LOGISTIC

- **假设 H1**：LOGISTIC 走 MDLA 的 sigmoid/LUT 通路，fp16 精度在几个 ULP 内；
  H2：int8 定点实现与"精确 sigmoid + 取码"差 1–2 码。
- **预测**：`SigmoidLayer`；fp16 误差 ≪2e-3；int8 |Δ|≤2。
- **做法**：`[1,64]`；fp16 输入 uniform(−8,8) 并钉住 −8/0/+8；int8 输入 scale=1/16、zp=0（±8 量程），
  输出按 TFLite 规范 `1/256, zp=-128`。
- **结果**：✅ `SigmoidLayer<0>{LOGISTIC}`；**fp16 max_abs = 0（逐位相等！）**、**int8 64/64 逐位相等**。
  设备实现比事前预测的"LUT 近似"更准：fp16 下与 float64→fp16 的 golden 完全重合。
- **适用性**：可直接用于门控/swish 类结构（SiLU = x·logistic(x) 需要一个 MUL，见 §7）。
- **剩余疑问**：极端输入（|x|>8）的饱和行为只由端点 −8/+8 抽样覆盖；`fp16 下 x=±30` 未测。

### 2.3 TANH

- **假设 H1**：TANH 走 MDLA 的 `TanhLayer`，fp16 精度几个 ULP 内；int8 差 1–2 码。
- **预测**：`TanhLayer`；fp16 max_abs ≤2e-3；int8 |Δ|≤2。
- **做法**：`[1,64]`，fp16 输入 uniform(−6,6) 钉住 ±6/0；int8 scale=1/16、输出 `1/128, zp=0`。
- **结果**：✅ `TanhLayer<0>{TANH}`；fp16 **max_abs=4.883e-4 且只出现在 3/64 个元素**，
  该值恰好等于 `|y|≈1` 处的 1 个 fp16 码（2⁻¹¹）⇒ 设备算的是"正确值舍入到相邻码"；
  int8 **64/64 逐位相等**。
- **适用性**：可作为 NPU 独立算子；GELU(tanh 近似) 需要额外的 MUL/ADD 组合。
- **剩余疑问**：TANH 与 LOGISTIC 的**融合**（Sigmoid 后接乘法）未测。

### 2.4 SOFTMAX（本任务的关键问题）

- **竞争假设**：
  - **H1**：SOFTMAX 是原生 MDLA 层（`SoftMaxLayer`），能直连、无需桥接；
  - **H2**：MDLA 无 softmax 原语，ncc 会插入 CPU 桥接，`--disallow-bridge` 下编译失败
    （这是"attention 必须回退"的情形）。
- **预测（H1 成立时）**：exec-plan 出现 `SoftMaxLayer<0>{SOFTMAX}` 且 `Target: MDLA_5_0`，
  编译 exit=0；fp16 数值在几个 ULP 内，int8 码值差 ≤2。**H2 成立时**：
  `Bridging is disallowed but gap is detected between Input and MDLA_5_0`。
- **做法**：4 个独立模型：`[1,8,16]` fp16 / int8、`[1,1,64]` fp16、`[1,12,128]` fp16 **β=0.125**。
  golden 在 float64 下做 `exp(x−max)` 归一化，再舍入到存储类型；int8 按 TFLite 规范 `1/256, zp=−128`。
- **结果**：**H1 成立，H2 被否定。**
  - `SoftMaxLayer<0>{SOFTMAX}` on `MDLA_5_0`，4 个模型全部 exit=0、无桥接；
  - fp16：max_abs = 2.441e-4（`[1,8,16]`，出现在 p=0.2844 处，正好 1 个 fp16 码）、
    6.104e-5（`[1,1,64]`）、1.526e-5（β=0.125）；
  - int8：`[1,8,16]` **128/128 逐位相等**；`[1,4,64]` 1/256 差 1 码（两次独立运行完全一致）；
  - **β 被采纳**：β=0.125 的模型只有按 β=0.125 归一化才能对到 1.5e-5；
    若设备忽略 β（β=1），分布会尖锐得多，误差量级 ≥1e-2。
- **适用性**：**attention 的 softmax 可以留在 NPU 上**（本实验覆盖 3D `[1,H,L]` 的 fp16 与 int8，
  L ∈ {16,64,128}，β ∈ {1, 0.125}）。
- **剩余疑问**：
  1. 4D `[1,H,L,L]` 批量注意力头未测（本实验只到 3D，见 §7）；
  2. 大 L（512/1024，LLM 上下文）未测，LUT 分档可能带来不同误差；
  3. **masking**：fp16 `−inf` 输入（因果掩码）是否输出 0 未测；int8 无法表达 −inf，需要专门的"加偏置再 softmax"方案；
  4. int8 在 L=64 出现 1 码差异，原因（累加顺序 vs 取整）未定位。

### 2.5 MEAN

- **竞争假设**：H1 归约层能直连 MDLA；H2 归约需要 CPU 桥接（"gap detected"）。
- **预测**：H1 → `MeanLayer<0>{MEAN}`，exit=0；数值上 fp16 逐位、int8 差 ≤1 码。
- **做法**：`[1,8,16]` axis=2 keep_dims=false（fp16）、`[1,8,16]` axis=1 keep_dims=true（fp16，非末轴）、
  `[4,16]` axis=1 keep_dims=true（int8，out scale == in scale）。axis 是**常量张量**。
- **结果**：H1 成立。`MeanLayer<0>{MEAN}`，三个模型板端**逐位相等**（8/8、16/16、4/4）。
- **踩过的坑（重要失败知识）**：第一次编译出的 `.dla` **能编译但不能执行**，见 §5.2 —— 是**我们的编码错误**
  （把常量 axis 张量写进了 subgraph inputs），不是 NPU 能力问题。修正后一次通过。
- **适用性**：可作为 NPU 独立算子；axis 必须是编译期常量；非末轴归约也可用。
- **剩余疑问**：`keep_dims` 与任意 axis 组合未穷举；4D/更大张量的归约未测；`MEAN` 的 `int8` 输出
  requant（out scale ≠ in scale）未测。

### 2.6 SUM（附加）

- **假设**：SUM 与 MEAN 同源（`SumLayer`），int8 在 out scale == in scale 时可做**精确整数和**。
- **预测**：`SumLayer<0>{SUM}`；fp16 逐位、int8 逐位（无 requant）。
- **做法**：`[1,8,16]` axis=2 fp16；`[4,16]` axis=1 int8（码值 ±7 ⇒ |sum|≤112，不饱和、不 requant）。
- **结果**：✅ `SumLayer<0>{SUM}`；两者**逐位相等**。
- **适用性**：可用；`REDUCE_SUM`（TFLite builtin `SUM` = 74）被 importer 接受。
- **剩余疑问**：out scale ≠ in scale 的 requant 版未测；`SUM` 在 int8 下的溢出语义（饱和 vs 回绕）未单独构造。

---

## 3. 统一边界规则：**输入张量「最内维 × 元素字节」必须是 16 的倍数**

### 3.1 现象

RELU `[1,13]` fp16 的 exec-plan 里**已经**出现 `ReLULayer<0>{RELU}`，但编译器随后判失败：

```
Bridging is disallowed but gap is detected between Input and MDLA_5_0
ERROR: Fail to compile ops/relu_1x13_fp16/model.tflite
```

即"层能映射，但输入到层之间需要一段 CPU/桥接才能对齐"。为找边界做了两轮**只编译不上板**的扫描
（脚本见 `tools/build_actreduce.py` 的 spec + 临时扫描脚本，产物在 `/tmp/scan*`，不进仓库）。

### 3.2 扫描结果（✅=exit 0 / ❌=上面那条 gap 错误）

| RELU fp16 宽度 | 1–7 | 8 | 9–15 | 16 | 18 | 20 | 24 | 32 | 48 | 52 | 56 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 输入行字节 | 2–14 | 16 | 18–30 | 32 | 36 | 40 | 48 | 64 | 96 | 104 | 112 | 128 |
| 编译 | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |

| SOFTMAX fp16 `[1,1,n]` | 8 | 12 | 16 | 20 | 24 | 32 | 40 | 48 | 64 | 100 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 输入行字节 | 16 | 24 | 32 | 40 | 48 | 64 | 80 | 96 | 128 | 200 | 256 | 512 |
| 编译 | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |

| MEAN fp16 `[1,8,w]`（axis=2） | w=12 | 13 | 14 | 15 | 16 | 17 | 24 |
|---|---|---|---|---|---|---|---|
| 输入行字节 | 24 | 26 | 28 | 30 | 32 | 34 | 48 |
| 编译 | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |

### 3.3 判定实验：是"元素数"还是"字节数"？（AGENTS.md §3.1 拆耦合变量）

| 对照 | 元素数 | 字节数 | 编译 | 判别 |
|---|---|---|---|---|
| fp16 w=8 | 8 | **16** | ✅ | 同为 8 个元素 |
| int8 w=8 | 8 | **8** | ❌ | ⇒ 不是"元素数" |
| int8 w=16 | 16 | **16** | ✅ | 同为 16 字节 |
| int8 w=48 | 48 | **48** | ✅ | 48 = 3×16 |
| int8 w=40 / 44 / 56 | – | 40 / 44 / 56 | ❌❌❌ | 非 16 倍数 |
| fp16 w=52 | – | 104 | ❌ | 104 非 16 倍数 |
| MEAN fp16 `[1,8,12]` | – | **总 192 B（是 16 倍数），行 24 B** | ❌ | ⇒ 看**每行**而不是总大小 |

⇒ 约束是 **`innermost_dim × bytes_per_element % 16 == 0`**，并且作用在**输入张量**上。
输出行**不受**此约束：`mean_4x16_int8` / `sum_4x16_int8` 的输出是 `[4,1]`（行宽 **1 字节**）仍能编译并跑通。

### 3.4 这条规则统一解释了已有的 FC 结论

`docs/CAPABILITY.md` 记录的 "int8 FULLY_CONNECTED 的 K 必须是 16 的倍数"
（K=8❌ 16✅ 24❌ 32✅ 40❌ 48✅ 56❌ 64✅ 72❌ 80✅ 112✅ 128✅ …）与本规则**完全一致**：
int8 下 1 字节/元素，K 的字节数就是 K。也就是说，之前那条"FC 的 K 要 16 对齐"并不是 FC 的数学约束，
而是**同一条输入行 16 字节对齐**规则在 FC 上的投影（FC 的输入行宽 = K）。
`112 % 16 == 0`、`72 % 16 == 8` 两组数据正好卡在规则两侧，是这条解释的强证据。

### 3.5 适用范围与未证实部分

- 覆盖：2D `[1,W]`、3D `[1,H,W]` 的 RELU/SOFTMAX/MEAN；int8 与 fp16；`mdla5.0`；ncc 7.3.15。
- **未证实**：4D 张量、`NHWC` 直连布局、非连续/转置输入、batch>1、以及 16 是否为唯一模数
  （只测了 16 的倍数与几个非倍数；没有测"32 对齐是否更宽松"这类反例）。
  ⇒ 按 AGENTS.md R1，本规则目前只在本节的形状范围内成立，超出请当作 UNKNOWN。

---

## 4. 复现

```bash
cd "/d/deepseek_ workspace/mtk/wt-actreduce"
export MSYS_NO_PATHCONV=1

# 1) 生成模型 + golden（WSL 里的 /usr/bin/python3，需要 tflite flatbuffer 包）
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-actreduce" && \
  /usr/bin/python3 tools/build_actreduce.py --list'
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-actreduce" && \
  /usr/bin/python3 tools/build_actreduce.py softmax_1x8x16_fp16 \
      --out ops/softmax_1x8x16_fp16/model.tflite'

# 2) 编译（记录 exec-plan / target / exit / sha256 到 model.compile.log）
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-actreduce" && \
  ./tools/compile.sh ops/softmax_1x8x16_fp16/model.tflite ops/softmax_1x8x16_fp16/model.dla'

# 3) 板端执行 + 对拍（持同一把设备锁）
NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" \
  ./tools/devlock.sh run -- ./tools/e2e.sh ops/softmax_1x8x16_fp16
cat ops/softmax_1x8x16_fp16/compare.txt
```

---

## 5. 失败知识（必须保留，后续 Agent 不要重复）

### 5.1 `relu_1x13_fp16` —— 编译期失败（形状边界，保留为反例）

```
===Graph[0] has 1 Execution Step(s)===
Graph[0] Step[0]
ExecutionStep
 ├ StepId: 0
 ├ Target: MDLA_5_0
 └ Subgraph:
   ├ ReLULayer<0>{RELU}
   └ Output: OpResult (external)

Bridging is disallowed but gap is detected between Input and MDLA_5_0
ERROR: Fail to compile ops/relu_1x13_fp16/model.tflite
EXIT=1
```

解读：**层已经映射到 MDLA_5_0**，失败发生在"输入→层"的连边需要桥接。
在本仓策略（`--disallow-bridge`）下这就是**该形状不可用**；去掉该 flag 会得到一个
"NPU 层 + CPU 转换"的混合图，那 **不是** 独立 NPU 算子（AGENTS.md R6）。
该目录没有 `compare.txt`：没有 `.dla` 就没有板端执行。原因见 §3。

### 5.2 MEAN/SUM 曾"能编译、不能执行" —— 我们自己的编码错误

第一版 `.dla` 编译 exit=0、层是 `MeanLayer<0>{MEAN}`，但板端：

```
$ ./bin/neuronrt -d -m hw -a dla/mean_1x8x16_fp16.dla
Input :
	Handle = 0, <1 x 1 x 8 x 16>, size = 256 bytes
	Handle = 1, is an unused input.
Output :
	Handle = 0, <1 x 1 x 1 x 8>, size = 16 bytes
The number of input files should be 2
Inference fail
```

给第 2 个 handle 喂文件后：

```
Cannot get input[1] size.
Inference fail
```

根因：`tools/build_actreduce.py` 当时把 **op 的输入列表**（`[input, axis]`）同时当成
**subgraph 的 inputs**，于是常量 axis 张量被声明成"模型输入"，而模型输入又带 data buffer
——正是 AGENTS.md §7.3 那个陷阱。ncc 把它变成一个"没有 size 的 input handle"，
runtime 因此永远拒绝执行。改成 subgraph inputs = `[input]`、op inputs = `[input, axis]` 后一次通过。

**教训**：`能编译` ≠ `能执行`（R7）。看到 `.dla` 的 I/O handle 数不对，先怀疑自己的 flatbuffer，
用 `neuronrt -d` 看 I/O 表是最快的判别手段。

### 5.3 工具链问题（已由 Lead 修复，记录以便追溯）

1. `tools/e2e.sh` 曾把 `WSLROOT` 硬编码成 `/mnt/d/.../npu-ops`，在任何 worktree 里第 6 步对拍都会：
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'ops/relu_1x64_fp16/model.json'
   ```
   （推/跑/回读都成功，只有对拍在错的 cwd）。现改为从自身 checkout 推导 `WSLROOT`。
2. `tools/compare.py` 曾只对整数算子读 `model.json:int8_tol`，浮点算子用内置默认值
   (fp16 0.02)，导致"登记的容差"和"实际判定"不一致。现在 `atol/rtol` 也优先读 `model.json`。
3. worktree 的 `.sh` 曾被 checkout 成 CRLF，WSL 里 `bash script.sh` 直接
   `/usr/bin/env: 'bash\r': No such file or directory`；已加 `.gitattributes`（`text eol=lf`）。

### 5.4 已知不可用（前次侦察，未重复测试）

`SIN` / `COS` / `EMBEDDING_LOOKUP` 被 importer 拒绝、`GATHER` 目标端不支持 —— 本次**没有**重测。

---

## 6. 可复现性说明（新发现，影响记录规范）

- `model.tflite`：**逐字节确定**（同一 spec 重复生成 sha256 相同，已核验）。
- `model.dla`：**不是逐字节可复现**——同一 `model.tflite` 连续编译 3 次得到 3 个不同 sha256
  （尺寸相同：2584 B）。因此 `model.compile.log` 里的 `SHA256` 只是"这一次产物的指纹"，
  **不能**当作"构建输入相同则 sha 相同"的判据。
  ⇒ 复现应比对 **(model.tflite sha256 + ncc 版本 + 编译 flag)**，见 AGENTS.md §4 的记录要求，
  这里补一条限制：`.dla` sha256 不能用于跨次构建的相等性判断。

---

## 7. 未覆盖 / 开放问题（明说，不猜）

1. **4D 注意力**：只测到 3D `[1,H,L]`；`[B,H,L,L]`、batch>1 未测。
2. **大 L**：L 只到 128（fp16）/64（int8）；LLM 的 512–1024 未测，LUT 分档误差未知。
3. **masking**：fp16 `−inf` 掩码、int8 的"加负偏置再 softmax"方案都未测。
4. **算子融合**：本实验全部是**单算子独立图**。真实模型里 activation 会被 conv/FC 吸收
   （fused activation），本实验**不能**说明融合后仍走同一路径。
5. **fp32**：按既有结论不可直连（需要桥接），本次没测。
6. **能力边界**：只扫了"最内维宽度"一个维度（外加 batch/height 固定），组合空间未扫。
7. **`RELU6` / `LEAKY_RELU` / `ELU` / `PRELU` / `L2_NORMALIZATION`** 未测。
8. **性能**：本文没有任何性能结论；`ms/inf` 只是量级。
9. `docs/CAPABILITY.md` 里"FC 的 K 必须 16 的倍数"可以升级成 §3 的统一规则，
   但那是公共文档，留给 Lead 合并时决定。
