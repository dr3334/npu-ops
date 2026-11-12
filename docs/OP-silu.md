# E-110 — SwiGLU 激活路径：`silu(gate)*up`（组合代价、融合与否、int8 可行性）

> 范围：`feat/silu` worktree（`wt-silu`）。本文只记录**我自己**的实验；`tools/build_silu.py`、
> `ops/{silu,gated,fused,gelu,logistic}_*` 是本分支新增产物。
> 方法约束见 `AGENTS.md`（R1–R10）与 `docs/AGENT-BRIEF.md`。
> 公共索引（`docs/EXPERIMENTS.md`、`PLAN.md`、`CAPABILITY.md`）**未改**，`E-110` 的编号由本文件占用，
> 索引登记留给 Lead（AGENT-BRIEF §2）。

---

## 0. TL;DR

| 变体 | 形状 | dtype | 编译 | 发射的层（exec-plan） | 板端 | 数值（登记契约） |
|---|---|---|---|---|---|---|
| A `silu` | [1,696] | fp16 | ✅ exit=0 | `SigmoidLayer<0>` + `ElementWiseMulLayer<1>` | ✅ | max_abs **3.906e-3** = 1 fp16 码（@\|y\|≈8），rel_l2 2.1e-4 → **PASS** |
| A `silu` | [8,696] | fp16 | ✅ | 同上 | ✅ | max_abs 3.906e-3，rel_l2 2.5e-4 → PASS |
| A `silu` | [1,696] | int8 | ❌ exit=1 | 层已映射但 `Input` 与 MDLA 之间要桥接 | – | **编译期失败**（696 % 16 = 8，见 §6） |
| A `silu` | [8,696] | int8 | ❌ exit=1 | 同上 | – | **编译期失败** |
| A `silu`（补宽） | [1,704] | int8 | ✅ exit=0 | `SigmoidLayer<0>` + `ElementWiseMulLayer<1>` | ✅ | 与 **TFLite 参考模型逐位相同 704/704**；对 exact golden 14/704 差 1 码 → PASS(tol=2，实用 1) |
| A `silu`（fp16 对照） | [1,704] | fp16 | ✅ | 同 A | ✅ | max_abs 3.906e-3 → PASS |
| B `gated` | [1,696] | fp16 | ✅ | `SigmoidLayer<0>` + `ElementWiseMulLayer<1>` + `ElementWiseMulLayer<2>` | ✅ | max_abs 3.906e-3，rel_l2 3.2e-4 → PASS |
| B `gated` | [8,696] | fp16 | ✅ | 同上 | ✅ | max_abs 3.906e-3，rel_l2 3.6e-4 → PASS |
| B `gated` | [1,696] | int8 | ❌ exit=1 | 层已映射，桥接被拒 | – | **编译期失败** |
| B `gated`（补宽） | [1,704] | int8 | ✅ | `Sigmoid` + 2×`ElementWiseMul` | ✅ | 对 TFLite 参考模型 **2/704 差 1 码（舍入 tie）**；对 exact golden 35/704 差 1 码 → PASS |
| C `fused`（2×FC 常量权重 + 三算子） | [1,128]→[1,696] | fp16 | ✅ | `FullyConnected<0>` + `Sigmoid<2>` + `ElementWiseMul<3>` + `FullyConnected<1>` + `ElementWiseMul<4>`（**一个 MDLA step**） | ✅ | max_abs 6.250e-2 = 1 fp16 码（@\|y\|≈64），rel_l2 5.4e-4 → PASS |
| C `fused` | [8,128]→[8,696] | fp16 | ✅ | 同上 | ✅ | max_abs 6.250e-2，rel_l2 4.9e-4 → PASS |
| C `fused` | [1,128]→[1,696] | int8 | ✅ | 同上 | ✅ | 对 TFLite 参考模型 1/696 差 1 码；对 exact golden 41/696 差 1 码 → PASS(tol=3，实用 1) |
| C `fused` | [8,128]→[8,696] | int8 | ✅ | 同上 | ✅ | 对参考模型 19/5568 差 1 码；对 exact golden 321/5568 → PASS |
| D `GELU`（exact/erf） | [1,696] | fp16 | ❌ exit=1 | **无 exec plan**（op 不支持） | – | **编译期失败**，见 §7 |
| D `GELU`（approx=true) | [1,696] | fp16 | ✅ | `GeluV2Layer<0>{GELU}` | ✅ | 对 exact erf **max_abs 1.953e-3**、266/696 有差 → PASS（吃满 atol=2e-3） |
| 对照：`LOGISTIC` 单算子 | [1,696] | fp16 | ✅ | `SigmoidLayer<0>` | ✅ | max_abs **4.883e-4 = 1 fp16 码**，14/696 有差 |

**四句话结论**

1. **组合本身不花钱**：`silu = LOGISTIC → MUL` 编译成 2 个原生层、**1 个 MDLA 执行 step**、
   `Temp = 0`（中间张量不落 DRAM）。板端输出**恰好等于** `fp16(x · 设备自己算出的 sigmoid 码)`（0/696 差异）
   ⇒ 组合没有引入任何额外舍入，误差 100% 来自 sigmoid 层自身的 1 码（LUT）误差。
2. **编译器不做层融合**：gated/fused 图里每个 op 都是独立层（code 段按层线性增长：
   Logistic 336 B、每层 Mul 336 B、FC 见下表），fusion 只发生在"整图打包成 1 个执行 step"这一级，**不是**把
   sigmoid 吸进 FC 的 fused activation（TFLite 的 FC fused activation 也没有 SILU 档位）。
3. **int8 的真实障碍是宽度不是精度**：696 宽的**模型输入**行 696 B，不满足 16 字节行规则 ⇒
   编译期失败（错误原文只提 `Input`）；补到 **704** 即可编译，且 int8 数值**可逐位预测**（silu 704/704 与 TFLite 参考模型一致）。
   边界张量补宽 8 列、误差 1 码（= out_scale = 1/16 = 0.0625）是 int8 的全部代价。
4. **原生 SILU 在这个 schema 里不存在**，`GELU` 存在但 exact 形式被 ncc 拒绝，只有
   `approximate=true`（或 `--fast-gelu-approximate` / `--opt-aggressive`）才能落到 `GeluV2Layer`，
   且精度只到 1.953e-3（1 fp16 码量级、部分点超过 1 码）。

**推荐（详见 §8）**：块内用 **fp16**、`silu`+`mul` 保持为 **LOGISTIC + MUL + MUL 三算子、与两个投影层同一个图**；
int8 只在"把 696 补到 704（模型输入、`down_proj` 的 K、权重行一起补）"可接受时使用，否则 int8 走不通。

---

## 1. 证据绑定（R1）

| 项 | 值 |
|---|---|
| 仓库/分支 | `npu-ops` worktree `wt-silu`，分支 `feat/silu`，基线 commit `387bd71`，本实验 commit `eba50c1`（预测+编译矩阵）/ 本文所在 commit |
| 编译器 | `ncc-tflite` **Version: 7.3.15**（NeuroPilot SDK basic 7.0.8 包内），`--arch=mdla5.0 --opt=3 --disallow-bridge --num-mdla=2 --l1-size-kb=3072` |
| 设备 | adb serial `0123456789ABCDEF`；`ro.board.platform=mt6897` / `ro.hardware=mt8792`；Android 16；fingerprint `HEADWOLF/F8/F8:16/BP2A.250605.031.A3/1766461355:userdebug/release-keys`；SELinux Enforcing，root |
| 板端执行器 | 厂商 `neuronrt -m hw -a <dla> -i … -o … -c 3`（**不是**自建提交器） |
| 板端串行 | 整批（预检 → 11 次推理 → 复检 → dmesg）在**一次** `NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" ./tools/devlock.sh run -- …` 内完成；另两次（`silu_1x696_fp16` 首发验证、`logistic_1x696_fp16` 对照）各自单独持锁 |
| 模型生成 | `tools/build_silu.py`（新增；复用 `tools/build_ops.py` 的 flatbuffers 脚手架，**未改**公共工具），seed `20261112` |
| 生成/编译/板端日期 | 2026-11-12 23:41–23:52 (+08:00) |

每个 op 目录的 `model.compile.log` 含 cmd 行、ncc 版本、date、`Target: MDLA_5_0`、DRAM 表、`EXIT=`、`SIZE=`、
`SHA256=`；`compare.txt` 是 `tools/compare.py` 原始输出；`e2e.json` 是 `tools/e2e.sh` 的机器可读收尾行。

`.dla` 指纹（sha256 前 16 位，**只标识某次产物**，不是可复现性判据 —— AGENTS §4 / OP-elementwise §1）：

| op | sha256 | | op | sha256 |
|---|---|---|---|---|
| silu_1x696_fp16 | `21e392e6bd685139` | | fused_1x128x696_fp16 | `8eaac0a550caad5c` |
| silu_8x696_fp16 | `0c556969167c812a` | | fused_8x128x696_fp16 | `efe9502af17712d1` |
| silu_1x704_fp16 | `e6733cdeac84c56e` | | fused_1x128x696_int8 | `4d254dd7962f3326` |
| silu_1x704_int8 | `91dd21602927ba5c` | | fused_8x128x696_int8 | `70a4faf126e65be0` |
| gated_1x696_fp16 | `05110e3ce1e04b9e` | | gelu_1x696_fp16_tanh | `5eb4523eb893dbcf` |
| gated_8x696_fp16 | `c81b0e5628d0f94c` | | logistic_1x696_fp16 | `0f400eb8fd6d6306` |
| gated_1x704_int8 | `50d715c683597053` | | | |

设备状态门（AGENTS §3.4）：整批前后各跑一次 `tools/healthcheck.sh --quick`（已知正常算子 `fc_1x64x64_int8`）
→ **PRE PASS / POST PASS**；批后 `dmesg | tail -60` 里 `apusys.*(timeout|fail|error)|mdw.*(timeout|fail)`
命中 **0** 行。⇒ 这批 kernel 没有留下状态污染。

---

## 2. 方法

### 2.1 四个模型族（`tools/build_silu.py`）

| 族 | 图 | 模型输入 | 输出 |
|---|---|---|---|
| A `silu` | `t=LOGISTIC(x)`；`y=MUL(x,t)` | 1 个 `[T,W]` | `[T,W]` |
| B `gated` | `t=LOGISTIC(gate)`；`s=MUL(gate,t)`；`y=MUL(s,up)` | 2 个 `[T,W]`（gate, up） | `[T,W]` |
| C `fused` | `gate=FC(x,Wg)`；`up=FC(x,Wu)`；再接 B 的三个算子 | 1 个 `[T,K]`（K=128），权重为常量 | `[T,W]` |
| D `gelu` | 原生 `GELU`（`GeluOptions{approximate}`） | 1 个 `[T,W]` | `[T,W]` |

`W=696` 是 Qwen2.5-0.5B 的真实 FFN 宽度；`T∈{1,8}`；C 的 `K=128` 是残差流宽度。
模型输入张量**不带 data buffer**（AGENTS §7.3），常量权重/偏置带 buffer。

### 2.2 int8 量化契约：为什么每个 requant 乘子都是 2 的幂

TFLite 的 int8 算子先算 `(a-azp)*(b-bzp)`，再乘 `real_multiplier = sa*sb/so`、加 output zero point。
本实验**主动选** `so = sa*sb·2^k`，使乘子退化成"移位"，这样唯一的舍入就是移位本身的 1 次舍入，
结果在码域可预测（否则舍入规则变成实现细节，`int8_tol` 只能靠"参考实现与设备相同"来假通过 —— R3）：

| 变体 | x / gate | up | sigmoid | 中间 `s` | 输出 | 乘子 |
|---|---|---|---|---|---|---|
| A silu | 1/16, zp 0 | – | 1/256, **zp −128** | – | 1/16, zp 0 | `2^-8` |
| B gated | 1/16, zp 0 | 1/128, zp 0 | 1/256, zp −128 | 1/16, zp 0 | 1/8, zp 0 | `2^-8` ×2 |
| C fused | 1/16, zp 0（权重 1/127） | 同 x 的 FC 输出 1/16 | 1/256, zp −128 | 1/16, zp 0 | 1/2, zp 0 | FC `1/127`（**非** 2 的幂）+ `2^-8` + `2^-7` |

- sigmoid 的 `1/256, zp=-128` 是 TFLite LOGISTIC 规范（与 OP-actreduce 一致）；
- A 的输出 scale 取 1/16（= x 的 scale）：`silu(x) ≤ x`（x>0），所以量程够；x=8 端点会被**钳到 127**（golden 也钳）；
- B 输出取 1/8：`|silu(gate)*up| ≤ 8`（gate≤8、\\|up\\|≤1），1 码 = 0.125；
- C 输出取 1/2：两个 ±8 张量之积需要更宽量程（≤64），这是 **int8 在本结构上精度差的根源**，不是为了凑通过；
- C 的 FC 输出 scale **刻意定成 1/16**（让 gate/up 与 A/B 同一契约），因此 FC 的 requant 乘子是 1/127，
  非 2 的幂 ⇒ 这一级只能靠 OP-MATRIX §3 记录的"两段舍入"规则预测（实测见 §5.3）。

### 2.3 独立 golden（R3）+ 三种参考模型

`tools/build_silu.py` 自己算参考值，**不经过 ncc / TFLite interpreter / 设备 runtime**：

- `golden/expected.bin`（**主 golden，`compare.py` 用这个**）：从**存下来的**输入值出发，在 float64 里做精确
  数学（`silu`、`gelu`=erf 精确式），最后**只做一次**舍入到存储类型 / 编码到 int8 输出 scale；
- `golden/expected_alt.bin`（假设 1）：**TFLite 参考实现**会算出的结果 —— fp16 是"每个张量都存成 fp16"，
  int8 是"逐算子量化 + 31 bit 乘子两段舍入（`SaturatingRoundingDoublingHighMul` + `RoundingDivideByPOT`）"；
- `golden/expected_alt2.bin`（假设 2，仅 2 的幂乘子图）：后端把 2 的幂乘子实现成**单次移位**
  （`RoundingDivideByPOT` 一次）而不是 high-mul+shift。

`python3 tools/build_silu.py --analyze ops/<name>` 会把设备 dump 与三者逐一比对 —— 这就是"组合到底实现了
哪个舍入模型"的判据（结果见 §5、§6），也避免把"参考实现和设备一起错"当成通过。

### 2.4 先登记容差（R1，跑之前已写进 `model.json`）

| 族 | dtype | 登记契约 | 事前理由 |
|---|---|---|---|
| A/B/C/D | fp16 | `atol=2e-3, rtol=1e-2` | 组合图每个中间张量都落 fp16，误差应是"输出量级处的几个 fp16 ULP"；rtol 覆盖大 \|y\|，atol 覆盖小 \|y\| |
| A | int8 | `int8_tol=2` | 解析界：(a) sigmoid 量化到 1/256 → 经 `x·sig` 传播 ≤ `|x|/512` = 0.0156 real = **0.25 码**；(b) 每次移位舍入 ≤0.5 码；(c) golden 自身舍入 0.5 码 ⇒ 1.25 → 取 2 |
| B | int8 | `int8_tol=2` | (a)+(两个 MUL 各 0.5)+(c) = 1.75 → 2 |
| C | int8 | `int8_tol=3` | B 的 1.75 + FC 的 1/127 非幂乘子（≥1 码） = 2.75 → 3 |

**实测都只用到 ≤1 码**（§5.2、§5.3）：登记值偏保守，事后**没有**调整。

---

## 3. 结果 A：`silu(x) = x·sigmoid(x)`

**竞争假设**
- H1：`LOGISTIC` 与 `MUL` 都能落 MDLA，组合后仍是"两层/一步"，误差只由两个 fp16 张量决定；
- H2：组合会引入额外舍入（例如 MUL 在更低精度里算），误差大于单算子；
- H3：链式图触发桥接，`--disallow-bridge` 下编译失败。

**预测（跑之前写在 `model.json:prediction`）**：H1 → `SigmoidLayer` + `ElementWiseMulLayer`、`Target: MDLA_5_0`、
exit=0、fp16 在几个 ULP 内；H2 → 误差明显大于 1 码；H3 → `Bridging is disallowed …`。

**结果（五个结论分开，R7）**
1. **编译**：[1,696] / [8,696] fp16 **exit=0**，`SigmoidLayer<0>{LOGISTIC}` + `ElementWiseMulLayer<1>{MUL}`，
   1 个 Execution Step，`Target: MDLA_5_0`；DRAM 1x696 = Input 1.4K/Output 1.4K/**Temp 0**/Code 672/Total 3.4K，
   L1 4.0K。int8 在 696 宽度**编译失败**（§6），补宽到 [1,704] 后 exit=0。
2. **提交**：`neuronrt` 正常返回，`-c 3` 三次推理无错误输出。
3. **数值**：fp16 `max_abs=3.906e-3`、`rmse=6.8e-4`、`rel_l2=2.1e-4`，**PASS**（`3.906e-3 = 8·4.883e-4`，
   即 x=8 端点处"1 个 sigmoid fp16 码 × x"）；int8 [1,704] 对 exact golden **14/704 差 1 码**、
   对 **TFLite 参考模型 704/704 逐位相同**。
4. **设备状态**：整批前后 healthcheck PASS，dmesg 0 命中（§1）。
5. **性能**：仅记录，不作结论 —— 1x696 `1.23 ms/inf`、8x696 `0.61 ms/inf`、1x704 int8 `1.74 ms/inf`
   （`-c 3` 平均，含进程内提交/等待开销，无控温、无对照实现）。

**机制：组合的代价是 0（本实验最强的一条证据）**
把**同一个** `input.bin` 喂给"只有 LOGISTIC 的模型"（`ops/logistic_1x696_fp16`，对照实验）：

| 量 | 结果 |
|---|---|
| `LOGISTIC` 单算子 vs `fp16(exact sigmoid)` | **14/696 不一致**，max\|d\| = 4.883e-4 = **1 个 fp16 码**（@0.5 附近 2⁻¹¹） |
| `silu` 设备输出 vs `fp16(x · 设备 sigmoid 码)` | **0/696 不一致** |
| `silu` 设备输出 vs `fp16(x · 精确 sigmoid)` | 13/696 不一致 |

⇒ **中间张量确实是 fp16，MUL 是正确舍入的 fp16 乘法；组合的全部残差 = sigmoid 层自己的 1 码误差**。
sigmoid 单算子在 696 宽度上不是逐位精确（14/696 差 1 码），这与 OP-actreduce 在 [1,64] 上观察到的"逐位相等"
**不矛盾**：宽度/输入样本不同，且本实验的容差内仍然 PASS。这条差异只影响"逐位"这个宣称，不影响可用性。

**适用范围**：fp16、`[1,696]`/`[8,696]`/`[1,704]`、无广播、`mdla5.0`、ncc 7.3.15。
**剩余疑问**：sigmoid 的 1 码差异是 LUT 分档还是别的机制（UNKNOWN，见 §9）；`T` > 8、多 batch 未测。

---

## 4. 结果 B：门控 epilogue（`gate`、`up` 两个模型输入）

**假设/预测**：同 A，但多一个 MUL；预测 3 层、1 step、`Temp 0`；数值上 `s` 也会落一次 fp16。

**结果**
1. **编译**：fp16 [1,696] / [8,696] exit=0，`Sigmoid` + `Mul` + `Mul`，1 step；DRAM 1x696 = Input 2.7K/Output 1.4K/
   **Temp 0**/Code 1.0K；8x696 Input 21K（两个 `[8,696]` 输入）/Output 10K/Code 736。
   int8 [1,696] **编译失败**，[1,704] exit=0（Static 1.0K/Code 1.3K/Total 4.4K）。
2. **提交**：正常；`-c 3`。
3. **数值**：fp16 `max_abs=3.906e-3`（1x696 rel_l2 3.2e-4；8x696 3.6e-4）→ PASS。
   int8 [1,704]：对 exact golden 35/704 差 1 码；对**TFLite 参考模型只有 2/704 差 1 码**。
4. **设备状态**：同 §3。
5. **性能**：1x696 0.59 ms/inf、8x696 0.73 ms/inf、1x704 int8 1.30 ms/inf（仅记录）。

**机制（第二个 MUL 也是正确舍入）**：用 `LOGISTIC` 对照跑出的设备 sigmoid 码，按
`y = fp16(fp16(gate·s_dev)·up)` 预测 → **0/696 不一致**。⇒ 两个 MUL 都是正确舍入的 fp16 乘法，
中间张量 `s` 落一次 fp16；组合仍无额外舍入。

**int8 的 2 个残差点是舍入 tie，不是语义差异**：`idx 51/680`，参考模型 −1 码、设备 0 码
（即"半码处"设备向零取整、TFLite 参考向远离零取整）。⇒ int8 的 3 算子链在**非 tie 点逐位可预测**。

**适用范围**：同 §3，两输入同形状无广播；`up` 量程限定在 ±1（scale 1/128）。

---

## 5. 结果 C：常量权重 + 融合变体（`[T,128]` 单输入 → `[T,696]`）

**问题**：把 `GATE_PROJ`/`UP_PROJ` 作为常量权重放进**同一个图**，编译器会不会把 sigmoid/乘法融进 FC？
会不会额外花 DRAM/step？

**假设/预测**：H1 不融合 → 5 个层、1 个 step、`Temp 0`、code 段 ≈ 两个 FC + 3 个逐元素层的线性叠加；
H2 融合 → 层数 < 5（例如 `Sigmoid` 被 FC 吸收或两个 MUL 合成一个）。

**结果：H1 成立，H2 否决。**
1. **编译**：fp16/int8、T=1/8 全部 exit=0。exec-plan：
   ```
   ├ FullyConnectedLayer<0>{FULLY_CONNECTED}
   ├ SigmoidLayer<2>{LOGISTIC}
   ├ ElementWiseMulLayer<3>{MUL}
   ├ FullyConnectedLayer<1>{FULLY_CONNECTED}
   ├ ElementWiseMulLayer<4>{MUL}
   ```
   **一个 Execution Step**；`Temp 0`；静态权重 352K（fp16，两份 `[696,128]`）/177K（int8）。
   code 段：fp16 1.7K、int8 2.0K —— 与"每层独立 code"的线性叠加吻合（对比：单 LOGISTIC 336 B、单 MUL 336 B），
   **没有任何层被合并**（TFLite 的 FC `fused_activation` 也没有 SILU 档位）。
2. **提交**：正常；`-c 3`。
3. **数值**：fp16 T=1 `max_abs=6.250e-2`、`rel_l2=5.4e-4`；T=8 `max_abs=6.250e-2`、`rel_l2=4.9e-4` → PASS。
   `6.250e-2` 恰好是 \|y\|≈64 处的 **1 个 fp16 码**；对"每张量落 fp16"模型 685/696 逐位相同。
   int8：对 TFLite 参考模型 1/696（T=1）、19/5568（T=8）差 1 码 —— 差异点同样在 FC 1/127 乘子的舍入边界上，
   **不是**结构性错误（最大差 1 码）。
4. **设备状态**：同 §3。
5. **性能**：fp16 T=1 0.98 ms/inf、T=8 1.64 ms/inf；int8 T=1 1.17 ms/inf、T=8 1.12 ms/inf（仅记录）。

**"不融合"的代价与收益**：层数 5 而不是 1；但**整图 1 次提交、`Temp = 0`、L1 只用 8.0K**，
因此"常量权重 + epilogue 同图"并不会多付一次 DRAM 往返 —— 融合的收益（少一次 submit、中间张量不落 DRAM）
在**不融合**的情况下也已经拿到了，因为 ncc 把它们排进同一个 MDLA step。

---

## 6. 边界：16 字节输入行规则 vs 696，以及 int8 的连锁约束

### 6.1 现象（编译期，逐字）

`ops/silu_1x696_int8/model.compile.log`（`Sigmoid+Mul` 都已被识别、target 已是 MDLA_5_0，失败在切分阶段）：

```
===Graph[0] has 1 Execution Step(s)===
Graph[0] Step[0]
ExecutionStep
 ├ StepId: 0
 ├ Target: MDLA_5_0
 └ Subgraph:
   ├ SigmoidLayer<0>{LOGISTIC}
   ├ ElementWiseMulLayer<1>{MUL}
   └ Output: OpResult (external)

Bridging is disallowed but gap is detected between Input and MDLA_5_0
ERROR: Fail to compile ops/silu_1x696_int8/model.tflite
EXIT=1
SIZE=0
```

`gated_1x696_int8` 同一条错误（3 层版本）。

### 6.2 判定（与 OP-actreduce §3 的规则一致，这里是它的新证据）

| 对照 | 最内层行字节 | 编译 | 说明 |
|---|---|---|---|
| fp16 `[1,696]` | 696×2 = **1392** = 87×16 ✅ | OK | fp16 天然满足 |
| fp16 `[1,704]` | 1408 = 88×16 ✅ | OK | – |
| int8 `[1,696]` | **696** = 43×16 + 8 ❌ | **FAIL** | 模型**输入**行触发桥接（错误原文：`gap … between Input and MDLA_5_0`） |
| int8 `[1,704]` | 704 = 44×16 ✅ | OK | **最小补宽 = 704**（输入与输出同为 704，未单独验证"只补输入、输出保持 696"） |
| int8 `[1,128]`→696（fused） | 输入 128 ✅；696 宽张量**只在图内部** | OK | ⇒ **规则只作用于 DRAM 边界张量，不作用于中间张量** |

**这条对照是关键**：同一个 696 宽张量，作为**模型输入**（A/B）失败，作为**图内部张量**（C 的 gate/up/s）却
可以编译、可以跑。⇒ 约束在"边界"上，与 OP-matrix §2.1 的表述一致，并排除了"任何 696 宽张量都不行"。

### 6.3 int8 的连锁约束：`down_proj` 的输入也是 696

MLP 里 `gate/up` 的输出 `[T,696]` 正是 `down_proj`（FC，`K=696`）的输入。补宽只补 epilogue 不够：

| 编译探针（`/tmp/fck`，`tools/build_ops.py fc --m 8 --k K --n 128 --dtype int8`） | 结果 |
|---|---|
| `K=696` int8 FC | **exit=1**：`Bridging is disallowed but gap is detected between Input and MDLA_5_0` |
| `K=704` int8 FC | exit=0（size 93805 B） |

⇒ **int8 路线要端到端成立，必须把 696 全链路补到 704**：`gate_proj`/`up_proj` 的输出列数（= `down_proj` 的 K）
与权重行一起补 8 列。补法（零权重列）本身未测，属 UNKNOWN；用现成 `fc_8x128x696_int8`（`K=128,N=696`）是
**能编过**的，所以"投影能出 696 列"没问题，卡点在**下游 FC 的输入行宽**。

---

## 7. 结果 D：原生 `SILU` / `GELU`（不假设，直接测）

### 7.1 这个 schema 里**没有** `SILU`

`tflite`（flatbuffers schema 包，`/home/dr/.local/lib/python3.10/site-packages/tflite`）里
`BuiltinOperator` 共 209 个名字，`GELU = 150` 存在，`SILU` **不存在**（`getattr(S.BuiltinOperator,'SILU') → None`）。
⇒ 在 ncc 7.3.15 + 本 schema 下**无法表达**"原生 SILU"，`silu` 只能写成 `LOGISTIC + MUL`（本实验 A/B/C 的做法）。

### 7.2 `GELU`：exact 被拒，approximate 才能落 NPU

`ops/gelu_1x696_fp16/model.compile.log`（逐字，很有信息量）：

```
OP[0]: GELU
 ├ MDLA: Not support ERF approximation, please set the 'approximate' parameter to true for TANH approximation. Or try --opt-aggressive or --fast-gelu-approximate for SIMOID approximation
 ├ EDPA: unsupported operation
ERROR: Cannot find an execution plan because of unsupported operations
ERROR: Fail to compile ops/gelu_1x696_fp16/model.tflite
EXIT=1
```

| 变体 | 编译 | 发射的层 | 板端 | 数值 |
|---|---|---|---|---|
| `approximate=false`（erf） | ❌ exit=1 | 无 | – | – |
| `approximate=true`（tanh 近似） | ✅ exit=0 | `GeluV2Layer<0>{GELU}`，Static 64 B、L1 4.0K（**带 LUT**） | ✅ | 对 exact erf **max_abs 1.953e-3**，266/696 有差（185 个 >1 码）→ PASS |
| 同上模型 + `--fast-gelu-approximate` | ✅ exit=0 | `GeluV2Layer<0>`（Static 0、L1 0、code 336，**无 LUT**） | 未上板 | – |
| 同上模型 + `--opt-aggressive` | ✅ exit=0 | `GeluV2Layer<0>`（同上） | 未上板 | – |

**GELU 的精度结论（分开写）**：`approximate=true` 的板端输出与 **exact erf** 的最大差 1.953e-3，恰好是
\|y\|≈2–4 处的 1 个 fp16 码；但偏差点分布在正尾（device 更接近 `x`）与负尾小值（device 给 `-0.0`，精确值 −6e-4）。
另测：与"教科书 tanh 近似"比 max 1.953e-3、rmse 2.7e-4；与"sigmoid 近似 `x·σ(1.702x)`"比 max **2.1e-2**
⇒ sigmoid 近似被否决，**内部到底用哪种近似是 UNKNOWN**，可用的结论只有经验误差界 **≤1.96e-3**（吃满 atol）。
**如果后续模型需要 GELU，按 2e-3 的预算规划，不要按"精确 erf"规划。**

---

## 8. 推荐：块（block）内 MLP 的写法

依据只有本实验测过的部分（`[T,128] → [T,696]`、T∈{1,8}、mdla5.0、ncc 7.3.15）：

### 8.1 dtype：**fp16**，不是 int8

| 判据 | fp16 | int8 |
|---|---|---|
| 696 宽编译 | ✅（1392 B 行，87×16） | ❌ 需补到 704（模型输入 + `down_proj` 的 K 一起补） |
| 数值 | max 1 个 fp16 码（组合附加误差 **0**） | 1 码 = out_scale（silu 1/16 → **0.0625 绝对**；epilogue 1/8 → 0.125） |
| 可预测性 | 组合 = `fp16(x·设备sigmoid码)`，逐位可复现 | silu 与 TFLite 参考模型逐位相同；3 算子/FC 版本在 tie 点差 1 码 |
| 额外成本 | 输入/输出各 2× 字节 | 省一半带宽，但要补 8 列（1.1%）且要改权重与下游算子 |

⇒ **fp16 是推荐路径**。int8 只有在"整条 696 链路都补到 704"可接受时才用；否则多算子链会卡在 `down_proj` 的 K。

### 8.2 图结构：`LOGISTIC + MUL + MUL` 三算子，与两个投影层**放同一个图**

- 不要指望融合：ncc **不融合**（层数 5 保持 5，code 段线性叠加），TFLite FC 也没有 SILU 档位；
- 但**同图**已经把"融合的收益"拿到手了：**1 个 execution step、`Temp = 0`**（中间 `gate/up/s` 全在 L1，
  L1 用量 8.0K），不需要为了省一次提交去做任何特殊处理；
- 若把 epilogue 拆成单独的 `.dla`，在块内就要多一次提交 + 一次 DRAM 往返（每层 `[T,696]` fp16 = 1392 B/T），
  因此**推荐整块一个图**（本实验的 C 变体就是这种形态，已编译 + 板端通过）；
- 权重放常量 buffer（本实验 C）已验证可编译可执行；若权重需要运行时切换，走 E-005 的动态权重路径（本实验未测）。

### 8.3 具体形状

```
输入 x:[T,128] fp16
  FC(x, W_gate[696,128]) -> gate [T,696]
  FC(x, W_up  [696,128]) -> up   [T,696]
  sig = LOGISTIC(gate)                      # 1 个 fp16 码误差来源（唯一）
  y   = MUL(MUL(gate, sig), up)             # 0 额外舍入（已证）
输出 y:[T,696] fp16  -> down_proj FC(K=696)
```
`T=1` 与 `T=8` 都通过；696 列在 fp16 下不需要补宽。`down_proj` 的输入行宽 = 696×2 = 1392 ✅。

---

## 9. 失败清单与"没有证明的事"

### 9.1 编译期失败（本实验，全部保留）

| 模型 | 错误 | 原因 |
|---|---|---|
| `silu_1x696_int8`、`silu_8x696_int8`、`gated_1x696_int8` | `Bridging is disallowed but gap is detected between Input and MDLA_5_0` | int8 边界张量行 696 B 非 16 倍数（§6） |
| `gelu_1x696_fp16`（erf） | `OP[0]: GELU … Not support ERF approximation …; EDPA: unsupported operation; ERROR: Cannot find an execution plan` | MDLA 只支持近似的 GELU（§7） |
| int8 FC `K=696`（探针，未入仓库） | `Bridging is disallowed …` | 下游 FC 输入行宽同样受限（§6.3） |

### 9.2 运行期失败

无。13 次板端推理（A×4、B×3、C×4、D×1、对照 LOGISTIC×1）全部正常返回；`compare.txt` 13/13 PASS。

### 9.3 明确**未**证明（UNKNOWN，不要当结论用）

1. **sigmoid 1 码差异的机制**：LUT 分档？输入样本？宽度相关？本实验只知道"696 宽、14/696 差 1 码"。
   与 OP-actreduce 在 [1,64] 上的"逐位精确"不冲突，但**没有**做区分实验（例如同输入不同宽度扫一遍）。
2. **GELU 内部用的哪种近似**：只测到误差界 1.95e-3；sigmoid 近似被否决，tanh 近似无法与 LUT 区分。
   `--fast-gelu-approximate` / `--opt-aggressive` 两个变体**只编译未上板**。
3. **int8 的 704 补宽做法**：本实验只是把宽度整体改成 704（数据也重采），**没有**测"696 有效 + 8 列补零"
   的等价性，也没测补零对 `down_proj` 精度的影响。
4. **`T` 更大 / 真 batch**：只测 T∈{1,8}；`T=32/128`、paged/prefill 形态未测。
5. **多块串联**（attention + MLP 一整层一个图）、`gate/up` 来自不同 scale 的 int8 图、per-channel 量化：未测。
6. **性能**：只有 `ms/inf` 量级（0.6–1.7 ms，模型太小、含进程开销、无控温、无对照），**没有任何**
   "融合更快/更慢"的结论。
7. **自建提交器**：本实验全部用厂商 `neuronrt`；`npu_binop` 框架对 `LOGISTIC`/`MUL` 的覆盖未做（另属 OWN-SUBMITTER）。
8. **fp32**：未测（按既有结论需桥接）。

---

## 10. 复现

```bash
cd "/d/deepseek_ workspace/mtk/wt-silu"
export MSYS_NO_PATHCONV=1

# 0) 列出/生成模型 + golden（WSL 里的 /usr/bin/python3，需要 tflite flatbuffer 包）
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-silu" && \
  /usr/bin/python3 tools/build_silu.py --list'
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-silu" && \
  /usr/bin/python3 tools/build_silu.py silu_1x696_fp16 --out ops/silu_1x696_fp16/model.tflite'

# 1) 编译（exec plan / target / exit / sha256 进 model.compile.log；--disallow-bridge 已固定在 compile.sh）
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-silu" && \
  ./tools/compile.sh ops/silu_1x696_fp16/model.tflite ops/silu_1x696_fp16/model.dla'

# 2) 板端 + 对拍（两输入模型由 e2e.sh 的 golden/input1.bin 自动带上）
NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" ./tools/devlock.sh run -- \
  ./tools/e2e.sh ops/silu_1x696_fp16
cat ops/silu_1x696_fp16/compare.txt

# 3) 判"设备实现的是哪个舍入模型"
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-silu" && \
  /usr/bin/python3 tools/build_silu.py --analyze ops/silu_1x696_fp16'
```

批量复跑（13 个已通过板端的 op）：

```bash
for n in silu_1x696_fp16 silu_8x696_fp16 silu_1x704_fp16 silu_1x704_int8 \
         gated_1x696_fp16 gated_8x696_fp16 gated_1x704_int8 \
         fused_1x128x696_fp16 fused_8x128x696_fp16 fused_1x128x696_int8 fused_8x128x696_int8 \
         gelu_1x696_fp16_tanh logistic_1x696_fp16; do
  NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" ./tools/devlock.sh run -- ./tools/e2e.sh ops/$n
done
```

回归判据：`.dla` **不是**字节可复现的（AGENTS §4），一致性只比 **数值**（`compare.txt`），
`model.compile.log` 里的 `SHA256/SIZE` 只作产物标识。

---

## 11. 交接（R8）

- **新增**：`tools/build_silu.py`（16 个 spec，`--list` 可列；`--analyze` 做三模型比对）、
  `ops/{silu,gated,fused,gelu,logistic}_*`（各含 `model.json`、`model.compile.log`、`compare.txt`、`e2e.json`，
  gelu 额外保留两份 flag 编译日志 `flag_fastgelu.compile.log` / `flag_aggressive.compile.log`）、本文。
- **未改公共工具**：`tools/e2e.sh`、`compile.sh`、`compare.py`、`devlock.sh`、`healthcheck.sh`、`probe/`、
  `PLAN.md`、`README.md`、`docs/EXPERIMENTS.md`、`docs/CAPABILITY.md` 全部原样。
- **需要 Lead 处理的事**：
  1. `docs/EXPERIMENTS.md` 增加 `E-110` 索引（本文件占号，未改公共文件）；
  2. `docs/CAPABILITY.tsv` / `OP-MATRIX.md` 可登记三行：`SILU(LOGISTIC+MUL)`/`GATED_SILU_MUL`（fp16 ✅）、
     `FUSED_FC_SILU_MUL`（fp16/int8 ✅，不融合但 1 step）、`GELU(approximate=true)`（✅ 但误差 1.95e-3）；
  3. `tools/rebuild_ops.sh` / `run_all.sh` 认领新 op 时，生成器是 `tools/build_silu.py`（不是 `build_ops.py`）；
     否则 `model.tflite` 被删后 `e2e.sh` 会去调 `build_ops.py` 而失败；
  4. 若要并入 `run_all.sh` 回归：12 个可跑的 op 里 `*_1x696_int8` 三个 + `gelu_1x696_fp16` 是**预期编译失败**（XFAIL）类。
- **下一位 Agent 的下一步（按价值排序）**：
  1. sigmoid 1 码差异的机制（同输入扫 64/128/256/512/696/704 宽度 → 是否与宽度/分档有关）；
  2. int8 "696 有效 + 补零到 704" 的等价性，以及它如何一路传到 `down_proj`（改 `fc_8x128x696_int8` 的 N 与权重行）；
  3. 整层（attention+MLP）一个图的 `Temp`/L1 预算，验证"同图省一次提交"在真实规模下的收益；
  4. `--fast-gelu-approximate` / `--opt-aggressive` 的板端数值（本实验只编译未上板）。
