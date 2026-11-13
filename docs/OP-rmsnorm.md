# OP-rmsnorm —— RMSNorm 及其原语 RSQRT 在 MDLA_5_0 上的能力与数值

> 范围：**RSQRT（原语探针）** 与 **RMSNorm 组合图**
> `y = x * rsqrt(mean(x², axis=-1) + eps) * gamma`（逐 token，hidden = 64）
> · 分支 `feat/rmsnorm`（worktree `wt-rmsnorm`）· ncc-tflite **7.3.15**
> · `--arch=mdla5.0 --opt=3 --num-mdla=2 --l1-size-kb=3072 --disallow-bridge`（`tools/compile.sh` 恒定开启）
> · 板：MT6897（Titan 1 / MT8792），`ro.board.platform=mt6897`，`Android 16`，
>   fingerprint `HEADWOLF/F8/F8:16/BP2A.250605.031.A3/1766461355:userdebug/release-keys`
> · 所有板端执行持同一把设备锁 `/d/deepseek_ workspace/mtk/.npu-board.lock`
> · 生成器：`tools/build_rmsnorm.py`（`--list` 列出 11 个 spec，`model.tflite` 字节可复现）

**一句话结论**：**RSQRT 是原生 MDLA_5_0 算子**（fp16 与 int8 都编译成 `RSqrtLayer`，无桥接）；
**整条 RMSNorm 链（MEAN → ADD eps → RSQRT → MUL x → MUL gamma）在单个 execution step 里全部落在
MDLA_5_0 上**，`[1,64]`/`[8,64]` × `int8`/`fp16` 四个主模型板端全部通过事先登记的数值契约
（fp16 差 ≤2 ULP，int8 逐位或 ±1 码）。
因此 **“RMSNorm 原生跑在 NPU 上”是可行的，不需要 CPU 侧 rsqrt**。
两个必须记住的边界：**fp16 下 eps 常数只有 ~0.6% 的表示精度**，
**int8 下 eps 根本不可表示（会被常量折叠掉，ADD 层直接消失）**。

---

## 0. TL;DR 结果表

| op | 形状 | dtype | 编译 | 发射的层（exec-plan） | 板端 | 数值 vs **事前登记**契约 |
|---|---|---|---|---|---|---|
| **RSQRT** | [1,64] | fp16 | ✅ exit=0 | `RSqrtLayer<0>{RSQRT}` → MDLA_5_0 | ✅ | **max_abs=9.77e-4 = 1 个 fp16 ULP**，48/64 逐位（atol=1e-2，只用 10%） |
| **RSQRT** | [1,64] | int8 | ✅ exit=0 | `RSqrtLayer<0>{RSQRT}` | ✅ | **64/64 逐位**（int8_tol=2，只用 0） |
| **RMSNORM** | [1,64] | fp16 | ✅ exit=0 | `ElementWiseMul<0>`→`MeanLayer<1>`→`ElementWiseAdd<2>`→`RSqrtLayer<3>`→`ElementWiseMul<4>`→`ElementWiseMul<5>`（**同一 step**） | ✅ | max_abs=1.953e-3（**2 ULP**），20/64 逐位，atol=1e-2 → **PASS** |
| **RMSNORM** | [8,64] | fp16 | ✅ exit=0 | 同上 6 层 | ✅ | max_abs=1.953e-3（2 ULP），291/512 逐位 → **PASS** |
| RMSNORM | [8,64] | fp16，eps=1.0 | ✅ exit=0 | 同上 6 层 | ✅ | max_abs=1.953e-3，260/512 逐位 → **PASS**（eps 加法确实执行，见 §2.4） |
| **RMSNORM** | [1,64] | int8 | ✅ exit=0 | Mul→Mean→**RSqrt→Mul→Mul**（**ADD 被折叠，层号跳到 `<3>`**） | ✅ | **64/64 逐位**（int8_tol=4，只用 0） |
| **RMSNORM** | [8,64] | int8 | ✅ exit=0 | 同上 5 层 | ✅ | 495/512 逐位，**max\|Δ\|=1 码**（17/512，tol=4，只用 25%）→ **PASS** |
| RMSNORM | [8,8] | fp16 | ✅ exit=0 | 6 层（行=16 B，**16 字节规则的阳性对照**） | ✅ | max_abs=9.77e-4（1 ULP），44/64 逐位 → **PASS** |
| RMSNORM | [8,4] | fp16 | ❌ **exit=1** | 层已映射（6 层照样打印）但 **IN_GAP** | – | 阴性对照：行=8 B < 16 B，见 §4.1 |
| RMSNORM | [1,64] | fp32 | ❌ **exit=1** | 无（**importer 阶段**就失败） | – | `ERROR: Fail to process the given model`，见 §5.1 |
| RSQRT | [1,64] | fp32 | ❌ **exit=1** | `RSqrtLayer<0>` 已打印，随后 **IN_GAP** | – | 判别实验：fp32 卡在**边界张量**，不是卡在算子，见 §5.2 |

`IN_GAP` = `Bridging is disallowed but gap is detected between Input and MDLA_5_0`。

产物：`ops/<spec>/{model.json, model.compile.log, compare.txt, e2e.json}`
（`.tflite`/`.dla`/`golden/` 是可再生成物，按 `.gitignore` 不入库；`model.tflite` 字节可复现，已核验）。

> 本文**没有性能结论**。日志里的 `ms/inf` 只表示“跑起来了”（本批 0.45–1.23 ms/inf，含进程内提交与
> 等待开销，批次间抖动大，不能用来比较算子快慢）。

---

## 1. 方法与证据链

### 1.1 事前登记容差（AGENTS.md §1 / R1）

全部写进 `tools/build_rmsnorm.py` 的 `SPECS`，随模型落到 `model.json` 的 `atol/rtol` 或 `int8_tol`
以及 `tolerance_rationale`，**在任何一次编译/板端执行之前就固定**，事后没有改过：

| op | dtype | 登记契约 | 事前理由 | 实际用量 |
|---|---|---|---|---|
| RSQRT | fp16 | atol=1e-2, rtol=1e-2 | 分支 brief 的统一 fp16 契约（≈20 个 fp16 ULP @1.0），容忍“设备内部精度与我的 float32 golden 不同” | max_abs/atol = **9.8%** |
| RSQRT | int8 | int8_tol=2 | 定点表查找 vs 精确 rsqrt，预留 2 码 | **0** |
| RMSNORM | fp16 | atol=1e-2, rtol=1e-2 | 6 个算子的链，其中 rsqrt 的输入是累加出来的均值 ⇒ 允许误差略大于单算子 | **19.5%** |
| RMSNORM | int8 | int8_tol=4 | 4 个取整/requant 级（x·x /32、MEAN /64、RSQRT 表、x·r /16、y·γ /64）各 ≤0.5 码，再加 rsqrt 表 ±1 码被 MUL-by-x 放大 ≤2 码 ⇒ 1+1+1+2 = 4 码（s_y=1/32 ⇒ 0.125 绝对值） | **25%** |

`tools/compare.py` 优先读 `model.json` 里的这两个字段，所以 `compare.txt` 判定用的就是**登记的**契约，
不是它的内置默认值。

### 1.2 golden 与被测实现独立（R3）

`tools/build_rmsnorm.py` 自己算 golden，**不经过 ncc、不经过 TFLite interpreter、不经过设备 runtime**：

- fp16：从**实际存下来的** fp16 输入升到 **float32** 算 `x² → mean → +eps → 1/sqrt → ·x → ·γ`，
  只在最后舍入一次到存储类型；同时额外算一份 float64 版本用于量化“golden 自身的精度噪声”——
  四个 fp16 模型的 `golden_f32_vs_f64_max_abs` 全部为 **0.0**（即 golden 的 float32/float64 两条路径
  落到同一个 fp16 码），所以板端差异不可能是 golden 的舍入造成的。
- int8：**从码值出发**，按每级的 power-of-two 乘子 + round-half-away-from-zero 逐级推进
  （`x·x /32`、`MEAN /64`、rsqrt 表、`x·r /16`、`y·γ /64`），**不含任何设备侧代码**。
  量化尺度在模块 docstring 里一次写清（zp 恒为 0，所有尺度都是 2 的幂 ⇒ 乘子是精确的 2 的幂，
  没有“乘子近似”这一层不确定性）。
- **常数用的是“实际存进模型的值”**：fp16 的 `eps=1e-5` 存成 fp16 是 **1.0014e-05**（次正规，~0.6% 精度），
  golden 用的就是 1.0014e-05，`model.json` 里 `eps` / `eps_stored` 两个字段都留着；γ 同理。

### 1.3 板端串行 + 状态门（R5 / §3.4）

- 所有 adb 操作都在 `tools/devlock.sh run -- ...` 内，锁路径与其它 worktree 相同；
- 一批 8 个算子前后各跑一次已知正常程序（`healthcheck.sh` 的 `GOOD_OP=fc_1x64x64_int8`）：
  **pre `HEALTHCHECK: PASS` / post `HEALTHCHECK: PASS`**；
- 每批结束扫 `dmesg`（`apusys.*(timeout|fail|error)|mdw.*(timeout|fail)`）：命中 **0** 条；
- **重复性**：`rmsnorm_8x64_fp16` 与 `rmsnorm_8x64_int8` 各独立跑两次，`out_device.bin`
  **逐字节相同**（int8 仍是同样的 17/512 差 1 码）⇒ 差异是**确定性的取整路径**，不是噪声。

### 1.4 五次结论分开（R7）

| 结论 | 本文对应章节 | 状态 |
|---|---|---|
| 编译成功 | §0 表 + `model.compile.log`（EXIT/SIZE/SHA256） | 8/8 主模型 ✅，3 个对照按预测失败 |
| ioctl / 执行成功 | `e2e.json` + `compare.txt`（device 输出文件长度对得上） | ✅ |
| 数值正确 | §2 各小节 | ✅ 全部满足登记契约 |
| 设备状态正常 | §1.3 状态门 + dmesg + 重复性 | ✅ |
| 性能改善 | —— | **无结论**（不做） |

---

## 2. 逐项：假设 → 预测 → 做法 → 结果 → 适用范围 → 剩余疑问

### 2.1 RSQRT（原语探针，问题 1）

- **竞争假设**
  - H1（原生）：MDLA 有 rsqrt 原语，`RSQRT` 直接映射成层；
  - H2（桥接）：MDLA 没有 rsqrt，ncc 插入 CPU 桥接，`--disallow-bridge` 下编译失败。
- **事前静态证据（指向 H1）**：`ncc-tflite` 二进制里有
  `MTKEXT_RSQRT` / `RSqrt: ParamSetup fails.` / `RSqrt: TileSetup fails.` / `XFL_RSQRT_QUANT`，
  以及 `libmvpu_isa.so` 导出指令 **`v_rsqrt_f32`**（另有 `v_rsqrt.f32` 的 LIR 符号与
  aarch64 NEON 的 `frsqrte/frsqrts` 引用）。所以事先把 H1 写成了主预测。
- **预测**（跑之前写）：fp16 → `RSqrtLayer` on `MDLA_5_0`、exit=0、误差 ≤1 fp16 ULP；
  int8 → 可能受 `XFL_RSQRT_QUANT` 支持，但若是 requant 路径可能 IN_GAP（预测较弱，记 2 个可能）。
- **做法**：`[1,64]`；fp16 输入 `U(0.25,4)` 并钉住 0.25 / 1.0 / 4.0；int8 输入 scale 1/16、码值
  `[8,127]`（x∈[0.5,7.94]），输出 scale 1/64、zp 0（避 `mean→0` 时的除零）。
- **结果**：**H1 成立，H2 否定。**
  - fp16：`RSqrtLayer<0>{RSQRT}` → `Target: MDLA_5_0`，DRAM input/output 各 128 B；
    `max_abs=9.766e-04`，**恰好 1 个 fp16 ULP**，48/64 逐位（也就是设备与
    `1/sqrt(x)` 的正确舍入基本一致，差异只在舍入方向）；
  - int8：`RSqrtLayer<0>{RSQRT}`，**64/64 码值逐位相等**（我的 golden 是
    `clip(round(rsqrt(q·s_in)/s_out))`，即设备的定点表恰好等价于“先精确 rsqrt 再取码”）。
- **适用范围**：mdla5.0 + ncc 7.3.15；`[1,64]` 一维宽度 64；fp16 / int8；输入行 128 B / 64 B。
- **剩余疑问**：宽度扫描未做（只测 64）；`x→0`（rsqrt→∞）与 `x<0`（NaN）**未测**，真实 RMSNorm 靠
  `+eps` 规避，但 int8 下 eps 会被折叠（§2.5），此时若某行 `mean=0` 设备行为**未知**；
  未测 `SQRT + RECIPROCAL` / `DIV` 作为回退（**本分支不需要**，因为 RSQRT 原生可用）。

### 2.2 RMSNorm 组合图（fp16，问题 2 的主线）

- **假设**：这 6 个算子能组成**单个** MDLA execution step，且不需要任何 CPU 桥接。
- **预测**（跑之前写）：5–6 层全部落在 `MDLA_5_0`、exit=0、`max_abs ≤ ~2e-3`。
  同时写下了三个真实风险：
  (a) `MUL [T,64] × [T,1]` 需要广播，可能被判成需要 CPU relayout → IN_GAP；
  (b) `MUL(x,x)` 把同一个张量列两次作输入，importer 可能拒绝；
  (c) 中间张量 `[T,1]` 的**最内行只有 2 字节**，若中间量也是 DRAM 边界张量，就会踩 16 字节规则。
- **做法**：
  ```
  input ──► MUL(x,x) ──► MEAN(axis=1, keep_dims) ──► ADD(eps[T,1]) ──► RSQRT ──┐
    │                                                                          │
    └────────────────────────────► MUL(广播 x·r) ──► MUL(gamma[64]) ──► output ◄┘
  ```
  eps 常数写成**满形状 `[T,1]`** 而不是标量，让 ADD 完全不需要广播（只有两个 MUL 必须广播，
  而那是 RMSNorm 的数学本身：一个行标量乘整行）；γ 用 `[64]` 常数（真实模型的形状）。
- **结果**：**三个风险全部没有发生。**
  - 6 层按顺序落在 `MDLA_5_0`，**同一个 `Graph[0] Step[0]`**（`docs/CAPABILITY.md` 关心的
    “层间是否被折叠/是否需要桥接”在这里是：不折叠、不桥接）；
  - 广播 MUL 被接受（`ElementWiseMulLayer`，无 relayout 层）；
  - `MUL(x,x)` 同张量双输入被接受；
  - 中间 `[T,1]`（2 字节行）**完全没问题** ⇒ 16 字节规则约束的是 **DRAM 边界张量**，
    不是层间中间量（§4 给阳性/阴性对照）；
  - 数值：`[1,64]` max_abs=**1.953e-3 = 2 ULP**（20/64 逐位）；`[8,64]` max_abs=1.953e-3、
    291/512 逐位。`range` 也对得上（exp/dev 上下界差 ≤0.002）。
- **机制（部分解释，不是全部）**：把 golden 改成“**每层边界都舍入到 fp16、层内 float32 累加**”后，
  与设备的逐位相符数从 291/512 升到 **392/512**，`[8,64]` 的 221 个不一致里有 **123 个**被解释；
  但 `[1,64]` 反而变差（20/64 → 3/64）。⇒ **“边界 fp16、层内 fp32” 是主要机制但不是全部**，
  MDLA 内部的累加顺序/精度仍是 **UNKNOWN**（见 §7）。
- **适用范围**：`T ∈ {1,8}`、`H=64`、fp16、mdla5.0/ncc 7.3.15；**不覆盖** hidden=896 的真实形状。

### 2.3 int8 组合图

- **假设**：整链 int8 可编、可跑；误差按“每级 ≤1 码”预算。
- **预测**：层表与 fp16 相同；数值不期望逐位，登记 `int8_tol=4`。
- **做法**：全链 zp=0、尺度全取 2 的幂（`s_x=1/16, s_sq=s_m=s_m2=1/8, s_r=s_y1=s_y=1/32, s_g=1/64`），
  乘子因此是精确的 2 的幂；输入码限制在 `|code|≤32`（|x|≤2），使 `q_x²/32 ≤ 32` 不饱和。
- **结果**：
  - `[1,64]`：**64/64 码值逐位相等**（tol=4，实际用 0）；
  - `[8,64]`：495/512 逐位，**17 个元素差 1 码且全部是 +1**（tol=4，实际用 1，即预算的 25%）。
  - **重要结构发现**：int8 的 exec-plan 是 `Mul<0> → Mean<1> → RSqrt<3> → Mul<4> → Mul<5>`，
    **层号从 1 跳到 3，`ElementWiseAddLayer<2>` 消失了** —— 因为 `eps=1e-5` 在 `s_m2=1/8` 下
    取码为 **0**，`ADD 0` 被常量折叠掉。这正是事前预测的那一条，也是 int8 路径的硬限制（§2.5）。
- **那 17 个 +1 码的定位（做了什么、排除了什么）**：
  | 假设 | 检验 | 结果 |
  |---|---|---|
  | 取整规则不同（half-up / half-even / half-to-zero） | 换四种取整重算 | 更差（482 / 479 / 472 of 512）⇒ **排除** |
  | 最后两个 MUL 在设备上融合（没有中间 requant） | 算“融合版” | 385/512 ⇒ **明显更差，排除** |
  | 某个行级中间量（q_m 或 q_r）差 1 码 | 逐行对 q_m、q_r 各 ±1 搜索 | 每一行的最优解都是 **±0**（60–64/64）⇒ **排除** |
  ⇒ 差异是**行内逐元素**的、确定性 +1 码，不等于任何已知的取整/融合规则。
  剩余候选（rsqrt 表分档、末级乘子的定点实现细节）**未证实 ⇒ 标 UNKNOWN**；
  要定死需要把中间量暴露成 graph output 的新探针（见 §7）。
- **量化误差 vs 设备误差（分开报）**：`[8,64]` 上
  `max|ideal_float − golden·s_y| = 0.071`（纯量化误差），
  `max|ideal_float − device·s_y| = 0.071`（设备只多了 ≤1 码 = 0.031，且没让最大值变大）。
  ⇒ **int8 路径的误差几乎全部来自量化本身，设备不额外贡献可测误差。**
- **适用范围**：`T ∈ {1,8}`、`H=64`、上面那套尺度；**换尺度（尤其 `s_sq`）必须重算预算**，
  因为它直接决定 `q_x²` 是否饱和、以及 MEAN 的 `/64` 落在哪。

### 2.4 eps 可观测性消融（单变量：只改 eps）

- **问题**：真实 `eps=1e-5` 时 `eps/mean(x²) ≈ 3e-6`（本批数据的逐行 `mean(x²)` 见
  `model.json`/日志：2.23–3.50），ADD eps 在数值上是**看不见**的。若编译器把 ADD 丢掉，
  带 `eps=1e-5` 的测试**照样会 PASS** —— 一个假的“通过”。
- **做法**：同形状、同数据、同 γ，只把 `eps` 从 `1e-5` 换成 **1.0**（此时 eps 占分母 ~13–30%）。
- **预测**：如果 ADD 真被执行，`eps=1.0` 的模型仍应 PASS；若被折叠，误差会 >1e-1 直接失败。
- **结果**：**PASS**（max_abs=1.953e-3、260/512 逐位）。离线算出的
  `max|y(eps=1) − y(eps=0)| = 0.374`，比登记 atol 大 37 倍 ⇒ **ADD eps 确实执行了**，
  `[1,64]`/`[8,64]` 在 fp16 下的 eps 通路可信。
- **适用范围**：只证明 fp16 的 ADD 通路；**int8 的 ADD 恰好相反——被折叠**（§2.5）。这是
  §2.4 存在的意义：不加这组对照，int8 的“ADD 消失”就会被误读成“编译器总是把 eps 丢掉”。

### 2.5 eps 在两种存储下的真实命运（本分支最有用的两条工程结论）

| dtype | eps 常数实际存成 | ADD 层 | 含义 |
|---|---|---|---|
| fp16 | `fp16(1e-5) = 1.0014e-05`（次正规，~7 bit 有效位） | ✅ 保留 | eps 可用；但**精度只有 ~0.6%**。真实 Qwen2.5 的 `eps=1e-6` 存成 fp16 是 `9.54e-07`（~6% 误差）。只要 `mean(x²)` 不是极小时它无所谓；**靠 eps 保稳定性时要知道它只被存成 6% 精度的数**。 |
| int8 | `round(1e-5 / 0.125) = 0` | ❌ **被常量折叠，层消失** | int8 的 mean 码空间分辨率是 1 码 = 0.125，`eps/mean` 需要 ≥ 1/128 才可表示；真实的 `1e-5` 比 LSB 小 4 个数量级。⇒ **int8 RMSNorm 在“mean 很小”的行上等价于没有 eps**，此时若 `mean→0`，`rsqrt(0)` 的设备行为**未测/未知**（§7）。要让 int8 保住 eps，只能改结构（例如对 mean 用更细的尺度、或把 eps 折进 rsqrt 的输入偏移），不能只调一个 scale 就同时保住动态范围。 |

### 2.6 唯一一次“层打印了但编译失败”的教训（复述 CAPABILITY §6）

`rmsnorm_8x4_fp16_row8b` 的 exec-plan **完整打印了 6 个层**（含 `RSqrtLayer<3>`），紧接着才是
`Bridging is disallowed but gap is detected between Input and MDLA_5_0` ——
**判断成败只能看 `EXIT=`**，不能看层名。这次是同一个陷阱第二次出现（第一次是 `relu_1x13_fp16`）。

---

## 3. 复现

```bash
cd "/d/deepseek_ workspace/mtk/wt-rmsnorm"
export MSYS_NO_PATHCONV=1

# 1) 生成模型 + golden（WSL 里的 /usr/bin/python3，需要 tflite flatbuffer 包）
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-rmsnorm" && \
  for s in $(/usr/bin/python3 tools/build_rmsnorm.py --list); do \
    /usr/bin/python3 tools/build_rmsnorm.py "$s" --out "ops/$s/model.tflite"; done'

# 2) 编译（exec-plan / target / exit / sha256 落到 model.compile.log）
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-rmsnorm" && \
  ./tools/compile.sh ops/rmsnorm_8x64_fp16/model.tflite ops/rmsnorm_8x64_fp16/model.dla'

# 3) 板端执行 + 对拍（持同一把设备锁）
NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" \
  ./tools/devlock.sh run -- ./tools/e2e.sh ops/rmsnorm_8x64_fp16
cat ops/rmsnorm_8x64_fp16/compare.txt
```

`model.tflite` 与 `golden/` 都是**字节可复现**的（同 spec 重建 sha256 相同，已核验：
`rmsnorm_8x64_fp16` = `1fa0575df976ec72…`）；`.dla` **不是**，只能标识产物（AGENTS.md §4）。

---

## 4. 对齐规则：16 字节规则的阳性/阴性对照

### 4.1 现象

| spec | 形状 | 最内行字节 | 编译 | 证据 |
|---|---|---|---|---|
| `rmsnorm_8x8_fp16` | [8,8] | 8×2 = **16 B** | ✅ exit=0 | `ElementWiseMul<0> … ElementWiseMul<5>`，板端 PASS（1 ULP） |
| `rmsnorm_8x64_fp16` | [8,64] | 64×2 = **128 B** | ✅ exit=0 | + int8 版 64 B 也 ✅ |
| `rmsnorm_8x4_fp16_row8b` | [8,4] | 4×2 = **8 B** | ❌ exit=1 | 层表已打印，随后 `IN_GAP` |

⇒ 本组合图**复现**了 `docs/CAPABILITY.md §6` 的规则（`innermost_dim × bytes % 16 == 0`，作用在
**输入**张量上），并给出 T=8、6 层链的新数据点（8 B 失败 / 16 B 通过，只差一个宽度）。

### 4.2 对规则的**收窄**（建议 Lead 合并进 CAPABILITY.md）

组合图里的 `mean`/`rsqrt` 中间张量是 `[T,1]`，最内行**只有 2 字节**，但编译、执行、数值全正常。
⇒ 这条 16 字节规则约束的是**图边界（DRAM 上的 input/output）张量**，
**不**约束同一 execution step 内的层间中间量。
（旁证：`model.compile.log` 的 DRAM 表里 Temp 列为 0，中间量在片内。）

---

## 5. 失败知识（保留原始行，后续 Agent 不要重复）

### 5.1 fp32 组合图：importer 阶段就失败（连层都没打印）

```
$ ./tools/compile.sh ops/rmsnorm_1x64_fp32/model.tflite ...
ERROR: Fail to process the given model
EXIT=1
SIZE=0
```

### 5.2 fp32 单算子 RSQRT：层能映射，卡在**边界**

```
   ├ RSqrtLayer<0>{RSQRT}
   └ Output: OpResult (external)

Bridging is disallowed but gap is detected between Input and MDLA_5_0
ERROR: Fail to compile ops/rsqrt_1x64_fp32/model.tflite
EXIT=1
```

判别意义：fp32 **不是**“某个算子不支持”，单算子 RSQRT 的 fp32 版本层是映射成功的，失败发生在
“Input → MDLA_5_0”的连边 ⇒ **fp32 的 DRAM 边界张量需要桥接**（与 `docs/CAPABILITY.md §5` 的
E-004 一致）。而 6 层 fp32 图连 importer 都过不去（`Fail to process the given model`），
**具体是哪个算子/哪种组合触发 importer 拒绝，未做逐算子二分 ⇒ UNKNOWN**（fp32 反正不可直连，
本分支没有继续投入）。

⇒ 结论：**mdla5.0 上没有可用的 fp32 直连路径**，本文登记的 `atol=1e-5` fp32 契约**永远无法被行使**；
fp16 的 `1e-2` 是当前唯一可用的浮点契约，实测误差 2 ULP（≈2e-3），比契约好 5 倍。

### 5.3 其它（不重复踩）

- int8 `eps` 的折叠（§2.5）不是 bug，是量化分辨率的必然结果；看到 “ADD 层消失”不要以为 importer 出错。
- 不要把 `[T,1]` 中间量误判成 16 字节规则的反例（§4.2）。

---

## 6. 交付物清单

| 文件 | 说明 |
|---|---|
| `tools/build_rmsnorm.py` | 11 个 spec 的生成器（模型 + golden + model.json）；SPECS 里含**事前**预测与容差 |
| `ops/rsqrt_1x64_{fp16,int8,fp32}/` | 原语探针（fp32 为预期的编译失败档） |
| `ops/rmsnorm_{1x64,8x64}_{fp16,int8}/` | 四个主模型 |
| `ops/rmsnorm_8x64_fp16_eps1/` | eps 可观测性消融 |
| `ops/rmsnorm_8x8_fp16/` | 16 字节规则阳性对照（行 16 B） |
| `ops/rmsnorm_8x4_fp16_row8b/` | 16 字节规则阴性对照（行 8 B，预期的编译失败档） |
| `ops/rmsnorm_1x64_fp32/` | fp32 组合图（预期/实际均为编译失败档） |
| `docs/OP-rmsnorm.md` | 本文 |

`tools/run_all.sh` 会把 `model.compile.log` 里带 `EXIT=1` 的目录自动当 **XFAIL**（复核“仍然编译失败”），
所以三个预期失败档不会污染回归。

---

## 7. 未覆盖 / 开放问题（明说，不猜）

1. **hidden=896（真实模型）未测**：只测了 64 与 8。896×2 = 1792 B 行，16 字节规则满足；
   但 MDLA 的分块/L1 行为在大 H 下可能不同（`DRAM Usage` 的 Temp 列是否仍为 0 未知）。
2. **更大 T**（prefill 的 128/512 token）未测；本分支只到 T=8。
3. **rsqrt 的输入极值未测**：`x→0`（溢出到 inf）与 `x<0`（NaN）。fp16 有 eps 托底；
   **int8 下 eps 被折叠，`mean=0` 的行行为未知** —— 这是 int8 路径唯一没被覆盖的“会炸”的点。
   实际部署时应保证输入尺度使 `mean ≥ 1 码`（本实验用 `|code|≤32` 保证 `mean` 码 ≥ 10）。
4. **fp16 的机制只解释了一半**：“每层边界舍入到 fp16 + 层内 fp32 累加”能解释 `[8,64]` 的
   123/221 个不一致，但 `[1,64]` 反而变差 ⇒ 设备内部的累加顺序/精度 **UNKNOWN**。
   要定死：把 `mean`/`rsqrt`/`x_norm` 依次暴露成 graph output 的新探针（本分支没做）。
5. **int8 的 17/512 +1 码**：四种取整规则、MUL 融合、行级中间量 ±1 全部被排除（§2.3），
   仍未定位 ⇒ UNKNOWN。定位同上，需要中间量探针。
6. **与 FC/Conv 的融合**：真实 decoder block 里 RMSNorm 的输出会喂进 FC（`gamma` 甚至可以被吸进
   权重）。本文全部是**独立图**，融合后是否仍走同一路径、numerics 是否变化 **未测**。
7. **其它 dtype/量化方案**：`int16` 激活、per-channel γ、`s_sq` 取别的值都没扫；
   int8 对尺度的敏感性没有做网格（只测了一套尺度）。
8. **回退路径**：`SQRT + RECIPROCAL` / `DIV` 未测（RSQRT 原生可用，本分支不需要）。
9. **性能**：本文无任何性能结论。

---

## D=896（Qwen2.5-0.5B 真实宽度）：三种均值实现对照（2026-11-13）

起因：`bestkojima/mtk-npug-handoff` 的 `frontend.norm()` 用 **静态 FC 权重算均值**
（`W[32,D]` 全部 = 1/D → `[T,32]` → CONCAT 复制 → 裁剪）来避开窄输出；源码还注明
"第二个 `[32→D]` 扩张 FC 在真实 D2048 图上触发过 APUSYS I/O 错误"。
我们在自己的栈（TFLite → ncc → MDLA）上把这三种实现都造了出来（`tools/build_rms896.py`）。

| 变体 | 均值实现 | 编译 | 层数 | 板端 |
|---|---|---|---|---|
| **A（我们原来的）** | `MEAN(axis=-1, keep_dims)` + ADD eps | ❌ `ElementWiseAdd: input types are incompatible` | 1 | — |
| **A0（去 eps）** | 同上，去掉 ADD | ❌ `RSqrt: output shape is incompatible` | 1 | — |
| **B（他们的 tiled statistics）** | `FC[32,896]`(W=1/896) → `[T,32]` → **CONCAT ×28** → SLICE 裁剪 → RSQRT | ✅ EXIT=0 | **6** | ✅ |
| **C（他们报告会出错的扩张 FC）** | `FC[32,896]` → `[T,32]` → **FC[896,32] 扩张** → RSQRT | ✅ EXIT=0 | 6 | ❌ 数值错（`rel_l2 0.82`，输出被压到 ±0.78，参考 ±4.4） |

**B 的数值（独立 numpy 参考，fp16）**：

| T | 编译 | 板端 | rel_l2 |
|---|---|---|---|
| 1 | EXIT=0（73.2 KB） | **PASS** | ~5e-4 量级 |
| 8 | EXIT=0（72.6 KB） | **PASS** | 5.76e-04 |
| 128 | EXIT=0（72.6 KB） | **PASS** | ~5e-4 量级 |

### 结论（对我们这条栈的直接指导）

1. **D=896 上，我们原来的 `MEAN` 方案编不过**（先 ADD 类型、后 RSQRT 形状），而
   **FC-tile + CONCAT 能编、能跑、逐层数值正确** —— 这条 workaround 在我们的栈上同样成立；
2. **扩张 FC 在 D=896 就已经数值错误**（他们把现象描述为 D2048 的 I/O 错误，
   我们在 D896 看到的是"能跑但结果被压缩"），所以**不能用**；
3. `MEAN` 在小宽度（D=64/128）可用、在大宽度不可用 —— 这是一个**宽度依赖的 bridge/形状问题**，
   与 `docs/CAPABILITY.md` 的 16 字节行规则不是同一条（这里行宽 1792 B 早就满足）。

### 复现

```bash
/usr/bin/python3 tools/build_rms896.py --variant B --T 8 --out ops/rms896_d896_T8_vB
./tools/compile.sh ops/rms896_d896_T8_vB/model.tflite ops/rms896_d896_T8_vB/model.dla
./tools/e2e.sh ops/rms896_d896_T8_vB
```
