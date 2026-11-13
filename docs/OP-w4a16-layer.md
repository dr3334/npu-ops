# OP-w4a16 —— 4W16A（int4 权重 / **int16 激活**）：质量决定性实验

> 票：`.scratch/quant-runtime/issues/04d-4w16a-whole-layer.md`；spec `.scratch/quant-runtime/spec.md`；
> ADR `docs/adr/0005-quantization-tier-target.md`（含两次实测订正：4-bit per-channel 达不到 0.9999；
> 质量天花板在**激活**不在权重）。
> 分支 `impl/q04d-4w16a`，worktree `wt-q04d`；权重夹具 `npu-ops/weights/qwen25_frozen`（票 02 冻结，只读引用）。
> 编译器 ncc-tflite **7.3.15** / `--arch=mdla5.0 --opt=3 --disallow-bridge --num-mdla=2 --l1-size-kb=3072`；
> 设备 mt6897 / Android 16（`0123456789ABCDEF`），板端全部持 `/d/deepseek_ workspace/mtk/.npu-board.lock`
> （会话前后各一次 `healthcheck --quick`，两次都 PASS）。
> 交付：`tools/build_w4a8.py`（新增 `--a-bits 8|16`、`--seam f16|native|native_sym`、15 个 `p16_*` 探针、
> requant 溢出护栏）、`tools/compare_i16.py` + `tools/e2e_i16.sh`（int16 对拍，见 §12）、
> `ops/w4a16_{fc_proj,rmsnorm,swiglu,bmm_qk,bmm_pv}/`、`ops/decoder_layer_t16_w4a16/`、
> `ops/decoder_layer_t16_w4a16_native/`、`ops/_w4a16_probe/*`、本文。
> 公共工具（e2e/compare/compile/devlock/build_ops/quant_contract）**一行未改**。

---

## 0. TL;DR

| 项 | 4W8A（票 04） | 8W8A（票 04c） | **4W16A（本票）** |
|---|---|---|---|
| 整层 device vs 独立 fp64 链 | 0.9091 | 0.9649 | **0.9510**（golden 0.95093） |
| 只量化权重（fp64 链，无激活误差） | 0.9510 | 0.99985 | **0.95096**（int4 权重） |
| 段级 `scores → probs`（整层内，golden） | 0.9541 | 0.9606 | **0.99918** |
| 段边界 `scores → probs`（device，喂精确 q/k） | — | — | **0.9999993**（native 缝）/ 0.9999846（fp16 缝） |
| 原生层数 / 执行步 | 61 / 1 | 61 / 1 | **61 / 1**（fp16 缝）或 **59 / 1**（native 缝） |
| `.dla` | 7,634,441 | 15,089,161 | **7,727,093**（+1.2%） |
| 板端单层（同会话 `-c 30`，中位） | 1.40 ms | 2.23 ms | **2.15 ms**（native 2.02 ms） |

**一句话回答票面的唯一问题**：把激活换成 int16 后，**激活侧的 0.96 天花板消失了**——`scores → probs`
在段边界上从 0.9606 抬到 **0.9999993**（device），整层内也从 0.9606 抬到 **0.99918**（> 0.999）。
但**整层到不了 0.9999**：它落在 **0.9510**，与「只量化 int4 权重、激活保持 fp64」的 **0.95096 完全重合**
（差 4e-5，在噪声内）。也就是说 **4W16A 的整层质量 = int4 per-channel 权重本身的质量**，
再往上加激活精度一点用都没有；剩下的 0.049 缺口 100% 在**权重**（TFLite 只有 per-channel scale）。
**int16 激活不需要 fp16 缝**：int16 掩码 SOFTMAX 直接编得过（3 层），整层从 61 层降到 **59 层**，
质量与 fp16 缝持平（0.95095 vs 0.95100）。

---

## 1. 票面要回答什么

1. 组件级 FC / SwiGLU / BMM-QK / BMM-PV 的 4W16A 版：device vs 量化 golden（码级）+ vs fp64 链 cosine；
2. 整层 4W16A（T=16）：层数 / 执行步 / Static / 噪声底 / 码级 / cosine；
3. **关键对照**：`scores → probs` 段是否从 0.96 抬到 ≥0.999；整层是否 ≥0.9999；
4. int16 的代价：`.dla` 体积、Static、板端单层时间（对 4W8A 1.44 / 8W8A 2.17 ms）；
5. int16 下是否仍需要 fp16 缝（若某个 seam 编不过，写清位置与层数代价）。

**不做**：24 层、KV cache、decode、group-wise scale、真实权重（仍用冻结随机夹具）。

---

## 2. 事前登记（都在第一次板端运行之前写进 `model.json`）

| 项 | 登记值 | 依据 |
|---|---|---|
| 组件 `int8_tol`（码） | **1** | 票面「算子级 = device == 量化 golden（码级 0/≤1）」 |
| 整层 `int8_tol`（码） | **2** | 与票 04/04c 同一数值、同一理由（int16 域的 2 码 = int8 域的 0.008 码，没有放宽；未事后调整） |
| 设备忠实 | ｜cos(dev) − cos(golden)｜ ≤ 0.003 | ADR 0005 订正第 1 条 |
| 质量判据 | `cosine_min = 0.9999`（device vs 独立 fp64 链） | ADR 0005 |

**三个 FAIL 原样保留**（都是登记值下的 FAIL，没有改阈值）：
整层 0.9510 ≪ 0.9999；`swiglu` 0.96826 < 0.9999；`rmsnorm`/`swiglu`/`softmax` 的码级超 1 码（见 §4.3，原因是器件 LUT）。

---

## 3. 做法：只加两个开关，不重写 harness

票 04c 已把权重位宽参数化成 `--w-bits 4|8`；本票加 `--a-bits 8|16`，并把「激活」这一侧全部改成随位宽走的：

| 部位 | int8（04/04c） | **int16（本票）** |
|---|---|---|
| 激活张量类型 | INT8，`s = max|v|/127` | **INT16**，`s = max|v|/32767` |
| 权重 | int4 packed 2 codes/byte，per-channel | 同一份代码/同一份夹具（**未动**） |
| bias | int32 per-channel，`s[n] = s_a·s_w[n]` | 同（未动） |
| requant | `quant_contract.py` 两步整数舍入，per-channel 乘子 | 同（未动） |
| LOGISTIC / SOFTMAX 定点约定 | 输出 `1/256`、zp `-128` | 输出 **`1/65536`、zp `-32768`**（TFLite int16 约定）；对称变体 `1/32767`、zp 0 |
| 掩码 SOFTMAX 缝 | 必须走 fp16（int8 域不可表示 + int8 SOFTMAX 编不过） | **可以不走**（见 §4/§10） |
| 图结构 | — | 逐算子同构；只有 seam 可切换（`--seam`） |

**唯一被顺手修掉的缺陷（本票自己的 bug，不是仓库的）**：int16 的累加器可以到 ~1e9，而
`ratio ≈ 3e-4` 会让 gemmlowp 的 left shift 达到 +12，`acc<<12` 再乘 m0 **溢出 int64**。
numpy 溢出、器件也溢出**同一个值** ⇒ device vs golden 仍然「码级 0–1 一致」而**两边都算错**
（`ops/_w4a16_probe/p16_native_seam` 第一版：段 cosine 0.39）。
现在每次 requant 都先过 `_requant_check()`（`acc<<shift` 必须 < 2^62/m0，否则**构建期报错**），
并把 1/8、1.0 两个常量改成「码 1 + 真实尺度」（累加器只有 int16 量级，ratio ≈ 0.5–1.3，无需左移）。
**教训**：device == golden 不等于「算对了」，两者可以一起溢出；护栏必须建在构建期。

---

## 4. int16 算子接受度（15 个探针，先编后跑）

### 4.1 编译接受度（全部 EXIT=0、无 bridge）

| 探针 | 算子 | 层数 | 结论 |
|---|---|---|---|
| `p16_fc` | FULLY_CONNECTED v9（int4 权重 per-channel + int32 bias） | 1 | ✅ |
| `p16_bmm_qk` / `p16_bmm_pv` | BATCH_MATMUL（`adj_y` / P·V 形状） | 1 / 1 | ✅ |
| `p16_mul` / `p16_add` / `p16_sub` | MUL / ADD / SUB（不同尺度的加） | 1 | ✅ |
| `p16_mean` / `p16_rsqrt` / `p16_logistic` | MEAN / RSQRT / LOGISTIC | 1 | ✅ |
| `p16_shapeops` | RESHAPE+SPLIT+CONCAT+TRANSPOSE | 5 | ✅ |
| `p16_mul_one` | MUL 常量 1.0（`1/65536,zp-32768` → `1/32767,zp0`） | 1 | ✅ |
| `p16_dequant` | DEQUANTIZE int16 → fp16 | 1 | ✅ |
| `p16_seam` | DEQ + MUL + ADD(mask) + SOFTMAX + QUANTIZE（fp16 缝） | 5 | ✅ |
| `p16_native_seam` | ADD(mask −4096 原生域) + MUL(1/8) + SOFTMAX int16（对称输出） | 3 | ✅ |
| `p16_native_seam_tflite` | 同上，SOFTMAX 输出 `1/65536,zp-32768` + MUL 1.0 | 4 | ✅ |

**结论：int16 激活在 MDLA5.0 上没有出现任何 dtype 缝的编译障碍**。特别是**掩码 SOFTMAX 可以直接 int16**
（int8 在票 04 里是「Large scale quant Softmax may have accuracy issue」+ 无执行计划），
因为 int16 的分数域里 −4096 掩码只占 −28500 码（±32767 内）。

### 4.2 板端码级（device vs 量化 golden）

| 探针 | 逐位 | max 码 | mean 码 | 判据 |
|---|---|---|---|---|
| `p16_bmm_qk` | **0/512** | 0 | 0 | PASS |
| `p16_bmm_pv` | 2047/2048 | 1 | 0.00 | PASS |
| `p16_mul` | **0/1024** | 0 | 0 | PASS |
| `p16_add` | **0/1024** | 0 | 0 | PASS |
| `p16_sub` | 1023/1024 | 1 | 0.00 | PASS |
| `p16_mean` | 10/16 | 1 | 0.38 | PASS |
| `p16_mul_one` | **0/1024** | 0 | 0 | PASS |
| `p16_shapeops` | **0/1024** | 0 | 0 | PASS |
| `p16_dequant`（fp16 输出） | — | rel_l2 2.0e-4 | — | PASS |
| `p16_rsqrt` | 111/1024 | **23** | 3.18 | **FAIL**（见 §4.3） |
| `p16_logistic` | 161/1024 | **21** | 5.44 | **FAIL**（见 §4.3） |
| `p16_seam`（fp16 缝） | 3501/3584 | 2000 | 2.83 | FAIL（softmax LUT） |
| `p16_native_seam` | 3411/3584 | 151 | 1.04 | FAIL（softmax LUT） |

### 4.3 RSQRT / LOGISTIC / SOFTMAX 的器件定点表精度（本票的新量化事实）

* **RSQRT / LOGISTIC 是定点表（LUT），分辨率是「绝对」的 ~2e-4 满量程**：int8 域里表现为 1–2 码
  （票 04 的 4 码预算就是从这来的），到 int16 域里同一张表就变成 **15–23 码**（mean 3.2 / 5.4）。
  相对误差仍是 4.6e-4 / 6.4e-4，**对 cosine 完全无影响**（rmsnorm 的 cosine = 1.0000000），
  但「码级 ≤1」在 int16 域对这两个算子**结构上做不到**。→ 登记值 1 保留，FAIL 保留。
* **SOFTMAX 内部也是定点表**（概率按 ~2^-8–2^-9 分辨率算），所以「小概率元素」的码级差可以到 151。
  它落在「同一行里占比很小」的那些元素上（例如 0.0000 vs 0.004），绝对误差 ~2e-3，cosine 不受影响。

---

## 5. 组件证据（组一，器件）

| op | 形状 | 编译 | device vs 量化 golden | vs 独立 fp64 链 |
|---|---|---|---|---|
| `ops/w4a16_fc_proj` | q_proj `[16,896]→[16,896]`，W int4 `[896,896]` | EXIT=0、1 层、1 步、Static 408K、`.dla` 425,409 B | 9867/14336 逐位、**max 1 码**、mean 0.31 → **PASS** | **0.9999934 PASS** |
| `ops/w4a16_rmsnorm` | `[16,896]→[16,896]` 5 层链 | EXIT=0、5 层、1 步、Static 60K、8,457 B | 9617/14336、max **15**、mean 0.49 → **FAIL** | **1.0000000 PASS** |
| `ops/w4a16_swiglu` | gate/up/down 三个 4W16A FC（I=4864） | EXIT=0、6 层、1 步、Static 6.4M、6,662,051 B | 369/14336、max **54**、mean 11.62 → **FAIL** | **0.9682552 FAIL** |
| `ops/w4a16_bmm_qk` | `[14,16,64]×[14,16,64]→[14,16,16]`，adj_y | EXIT=0、1 层、1 步、Static 0、9,971 B | 3583/3584 逐位、**max 1** → **PASS** | **1.0000000 PASS** |
| `ops/w4a16_bmm_pv` | `[14,16,16]×[14,16,64]→[14,16,64]` | EXIT=0、1 层、1 步、Static 0、16,775 B | 14334/14336 逐位、**max 1** → **PASS** | **1.0000000 PASS** |

**四条读法**

1. **FC / BMM 在 int16 域里是逐位或 1 码**（fc_proj 的 31% 值差 1 码是 per-channel 乘子的两次舍入边界，
   与 int8 档同源；bmm 两段几乎全逐位）。票面「码级 0/≤1」在这四个算子上成立（rmsnorm/swiglu 例外，理由 §4.3）。
2. **SwiGLU 的 cosine 0.9683 不是激活的错**：它里面 3 个 FC 的权重是 int4（每矩阵 14–16%），
   4W8A 的 swiglu 是 0.9674、8W8A 是 0.99917。**4W16A ≈ 4W8A**（差 0.0009）⇒ swiglu 这一段由权重定，
   激活从 int8 换 int16 一点没变（因为 4-bit 权重已经把它压住了）。
3. **bmm 两段（无权重）在 int16 下 cosine = 1.0000000**（int8 时是 0.9999566 / 0.9999431）：
   Q·Kᵀ 的 1 码从 37.08（int8）变成 0.1437（int16），256× 精细。
4. **rmsnorm 的 cosine = 1.0000000 而码级 max 15**：这两个数不矛盾，见 §4.3（LUT 的绝对分辨率）。

---

## 6. 整层证据（组二）

### 6.1 编译

| 图 | 层数 | 执行步 | Input/Output | Temp | Static | Code | Total | `.dla` |
|---|---|---|---|---|---|---|---|---|
| `decoder_layer_t16_w4a16`（fp16 缝） | **61** | **1** | 28K/28K | 16K | 7.3M | 43K | 7.4M | **7,727,093 B** |
| `decoder_layer_t16_w4a16_native` | **59** | **1** | 28K/28K | 16K | 7.3M | 43K | 7.4M | 7,727,235 B |
| `_timing_w4a8`（基线） | 61 | 1 | 14K/14K | 7.0K | 7.2M | 33K | 7.3M | 7,634,421 B |
| `_timing_w8a8`（基线） | 61 | 1 | 14K/14K | 7.0K | 14M | 33K | 14M | 15,089,141 B |

* 单执行步、无 bridge（`--disallow-bridge` 恒开）；两条基线在本会话重建后与票 04/04c 的
  `.dla`/码级**逐位复现**（4W8A 12314/14336 max 38；8W8A 12660/14336 max 36）⇒ 参数化没有动 int8 档。

### 6.2 板端（`compare.txt`，持锁）

| 图 | 逐位 | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | max | mean | median |
|---|---|---|---|---|---|---|---|---|---|
| **4W16A（fp16 缝）** | 0.84% | 2.5% | 4.2% | 6.0% | 9.5% | 17.3% | **581** | **61.43** | 38 |
| **4W16A（native 缝）** | 1.13% | 3.6% | 6.2% | 8.6% | 13.1% | 24.5% | **807** | **44.98** | 25 |
| 8W8A（票 04c） | 11.7% | 32.5% | 48.4% | 59.2% | 73.3% | 90.1% | 36 | 4.21 | 3 |
| 4W8A（票 04） | 14.1% | 38.2% | 54.8% | 65.7% | 77.9% | 91.2% | 38 | 3.83 | 2 |

**读法**：int16 的码级比 int8 **差 15 倍**，但这只是「1 码变细了 256 倍」的另一种说法——
1 码 = 1.547e-5（输出 max 0.5069），mean 61 码 = **9.4e-4 绝对 = 0.19% 相对**，
而 8W8A 的 mean 4.21 码 = 1.68e-2 绝对 = 3.3% 相对。**码级不可跨位宽比较**，跨档只能看 cosine。

### 6.3 噪声底与设备忠实

| 量 | fp16 缝 | native 缝 |
|---|---|---|
| 噪声底（同 `.dla` 同输入两轮） | **0 码**（`out_device.bin` 逐字节相同） | **0 码** |
| `cos(dev, fp64)` | **0.9510043** | 0.9509451 |
| `cos(量化 golden, fp64)` | 0.9509311 | 0.9509562 |
| ｜cos(dev) − cos(golden)｜ | **7.3e-5** ≤ 0.003 ⇒ **设备忠实** | 1.1e-5 |

---

## 7. 段级表（票面要求「段级 cosine 表要全」）

golden vs 独立 fp64 链（device-free，`--upto <stage>`；device 与 golden 的差 ≤ 7.3e-5，见 §6.3）：

| stage | 4W8A（票 04） | 8W8A（票 04c） | **4W16A（本票）** | 变化 |
|---|---|---|---|---|
| n1（RMSNorm#1，无权重） | 0.9995702 | 0.9995702 | **1.0000000** | 激活 256× 精细 |
| q3（FC+RoPE） | 0.9999334 | 0.9999410 | **0.9999933** | |
| scores（BMM，无权重） | 0.9999013 | 0.9999110 | **0.9999949** | 1 码 37.08 → 0.1437 |
| **probs（缝 + 掩码 softmax）** | **0.9541** | **0.9606418** | **0.9991772** | **+0.0386** |
| attn（P·V） | 0.9504 | 0.9651807 | **0.9898655** | |
| r1（o_proj + 残差） | 0.9717 | 0.9836251 | **0.9905675** | |
| n2（RMSNorm#2） | 0.9716 | 0.9835984 | **0.9906752** | |
| mlp_y | 0.9231 | 0.9669002 | **0.9614877** | |
| down | 0.9111 | 0.9669223 | **0.9497289** | |
| **out** | 0.9130 | **0.9673771** | **0.9509311** | |

**段边界单独测**（`ops/_w4a16_probe/`，只做 `scores → probs`、喂精确的 int16 q/k、device 出数）：

| 实现 | 层数 | device cos vs fp64 | golden cos | ｜Δcos｜ |
|---|---|---|---|---|
| 4W8A / 8W8A（int8 分数域） | 5 | 0.9541 / 0.9606 | — | — |
| **4W16A + fp16 缝** | 5 | **0.9999846** | 0.9999780 | 7e-6 |
| **4W16A + native int16 缝** | **3** | **0.9999993** | 0.9999997 | 4e-7 |

**两条结论**

1. **票面的关键问题成立**：`scores → probs` 从 0.9606（int8）抬到 **0.9999993**（int16，段边界、device），
   在整层里也到 **0.99918**（> 0.999）。掉码不再从 probs 开始——probs 现在是全层最干净的段之一。
2. 整层内的 probs 只到 0.99918（而不是 0.999999）是因为**它的输入 scores 已经带了 int4 权重的误差**
   （scores = 0.99999，softmax 把它放大约 10×）。**这是权重误差经过 softmax 的放大，不是激活的错。**

---

## 8. 误差归因：整层 = int4 权重天花板（本票的结论）

| 量 | 值 | 说明 |
|---|---|---|
| 只量化 int4 权重、激活 fp64（`--wq-chain`） | **0.9509612** | 权重的天花板 |
| 只量化 int8 权重、激活 fp64 | 0.9998451 | 票 04c 复现 |
| **4W16A 整层 device** | **0.9510043** | 与 int4 权重天花板差 **+4.3e-5** |
| 4W16A 整层里「激活侧」的贡献 | ≈ 0 | 由上面两行直接读出 |

**结论**：int16 激活把**激活侧**的误差压到了不可测量的程度（< 1e-4），整层质量**恰好等于**
int4 per-channel 权重的质量。ADR 0005 第二次订正说「天花板在激活」——**在 int8 激活下成立；
把激活换成 int16 之后天花板搬到了权重侧**，而权重侧在 TFLite 只有 per-channel scale 的情况下
就是 0.951（每矩阵 14–16%）。所以：
* 「再加激活精度」对本层**没有收益**（已验证：int16 vs int8 在 swiglu 上几乎没动，0.9683 vs 0.9674）；
* 要越过 0.9999 只能动权重方案（group-wise scale / hessian / fp16 权重），**这是 04b 的结论，不是位宽问题**。

---

## 9. int16 的代价（组三）

| 量 | 4W8A | **4W16A（fp16 缝）** | 4W16A（native 缝） | 8W8A |
|---|---|---|---|---|
| `.dla` | 7,634,421 | **7,727,093（+92,672 B，+1.21%）** | 7,727,235（+1.22%） | 15,089,141 |
| Static | 7.2M | **7.3M** | 7.3M | 14M |
| Temp / Input+Output | 7.0K / 14K+14K | **16K / 28K+28K** | 16K / 28K+28K | 7.0K / 14K+14K |
| Code | 33K | **43K** | 43K | 33K |
| 层数 | 61 | **61** | **59** | 61 |
| 板端单层（同会话交替，`-c 30`，3 轮） | 1.296 / 1.776 / 1.402 → **中位 1.40 ms** | 1.898 / 2.237 / 2.146 → **中位 2.15 ms** | 2.021 / 2.222 / 1.858 → **中位 2.02 ms** | 2.226 / 2.287 / 1.411 → 中位 2.23 ms |

* **内存代价极小（+1.2%）**：4W16A 的权重还是 int4，int16 只影响激活张量（T×D = 14 KB → 28 KB）与
  int16 kernel 的 Code（+10 KB）。**这是 4W16A 相对 8W8A 的真正优势：质量 0.951 vs 0.965 差 1.4 个点，
  内存差 2 倍（7.7 MB vs 15.1 MB）。**
* **时间 +45%~55%**（1.40 → 2.02–2.15 ms），与 8W8A 的 2.23 ms 同一档：int16 的 elementwise 层更贵。
  **注意噪声**：同一份 `.dla` 在本会话内可以测出 1.41 / 2.23 ms 两个值（r3 的 8W8A），
  跨会话漂移比档位差还大（票 04c 已记录）。上表是同一把锁、同一会话、A/B/C/D 交替 3 轮的中位数，
  只能读成「4W8A 最快；8W8A / 4W16A 同档，比它慢 1.4–1.6×」。

---

## 10. 是否需要 fp16 缝（票面 checklist 最后一项）

**不需要。** 三条证据：

1. **编得过**：int16 掩码 SOFTMAX 直接是 3 层（ADD + MUL + SOFTMAX），`--disallow-bridge` 下 EXIT=0；
   若 SOFTMAX 输出用 TFLite 的 `1/65536, zp-32768` 再加 1 层 MUL 回对称域，是 4 层。
   对比 fp16 缝的 5 层（DEQ + MUL + ADD + SOFTMAX + QUANTIZE）。**整层 61 → 59 层。**
2. **算得对**：native 缝的段边界 cosine **0.9999993**，比 fp16 缝的 0.9999846 **更好**（fp16 的 11 bit
   尾数在 −4096 掩码附近只剩 2.0 的分辨率，int16 在原生分数域是 0.1437）。
3. **整层等价**：0.9509451（native）vs 0.9510043（fp16 缝）——差 6e-5，同档。

为什么 int8 做不到而 int16 可以：掩码 −4096 在 int8 分数域要 2^-11 的相对分辨率（不可表示），
在 int16 域只占 −4096/0.1437 = **−28500 码**（±32767 内），而且 1/8 的缩放可以**放在掩码之后**
（先加掩码、再乘 1/8、再 softmax；顺序反了掩码就装不下，这是本票实现里的一个坑，注释写在
`_seam_native()` 里）。

---

## 11. 与厂商 4W16A 的对照口径（票面要求写清）

* 厂商数字是 **device token match 23/268**（端到端、268 个 prompt 的首 token 命中），
  我们的数字是 **单层输出 vs 独立 fp64 链的 cosine**（T=16、冻结随机夹具、无 tokenizer）。
  **两者不是同一个量**，不能直接换算；能对照的只有「同一格式下质量离 fp16 有多远」这件事：
  厂商 23/268 ≈ 8.6% 命中说明 4W16A 在他们的端到端上**远不是 fp16 等价**，与我们测到的
  「4W16A 整层 cosine 0.9510、天花板是 int4 权重 0.95096」在**方向与量级上一致**（都远低于 fp16）。
* 我们的 0.9510 是**该层在这套夹具上的性质**，不含 24 层误差累积、不含 KV cache、不含采样；
  端到端 token 命中率只能在上层（24 层 + decode）测，本票不做。
* **可比的部分是格式**：4W16A 的 `.dla` 只比 4W8A 大 1.2%，而 8W8A 是 2×——这解释了厂商为什么
  把 `sym4W_sym16A` 作为出厂档（内存省、质量由权重定，激活再精细也没用）。

---

## 12. 工具侧的交付与需要 Lead 决定的事

* **新增**（不改公共工具）：`tools/compare_i16.py`（int16 对拍：码级分布 + 浮点分支）、
  `tools/e2e_i16.sh`（`e2e.sh` 的同流程，第 6 步换成 `compare_i16.py`）。
* **需要 Lead 做的一行公共改动**：`tools/compare.py` 的 `DT` 表加上 `"int16": np.int16`
  （现在 `DT["int16"]` 会 KeyError）。本票按 AGENTS 8 没有动它，用 `compare_i16.py` 替代；
  两者输出契约一致（`int8_tol` = 码级容差，PASS/FAIL 同义）。
* **新增 builder 能力**：`--a-bits 8|16`、`--seam f16|native|native_sym`、15 个 `p16_*` 探针、
  `_requant_check()`（构建期拒绝会溢出 int64 的 requant）。int8 档的输出**逐位不变**
  （`--w-bits 8` 重建 `w8a8_bmm_qk/model.json` 只差 `builder` 字符串；两档 `.dla`/码级与票 04/04c 复现）。

---

## 13. 负结果 / UNKNOWN（不圆）

1. **整层码级压不到 1 码**：mean 61（fp16 缝）/ 45（native），max 581/807。来源仍是 elementwise
   段的舍入模型（票 04 §11.2 的「把 elementwise 尺度改成 2 的幂倍数」**仍未做**）+ int16 的
   LUT（RSQRT/LOGISTIC/SOFTMAX）绝对分辨率。**没有事后放宽阈值**。
2. **`p16_fc` 探针有 1/1024 的 839 码差**（golden 32767 vs device 31928），出现在**恰好满量程**的那个元素；
   真实组件 `w4a16_fc_proj` 是 0/1 码，未复现。器件在「输出顶到 ±32767」时的饱和规则 **UNKNOWN**。
3. **native 缝为何不比 fp16 缝快**：59 层 < 61 层，但板端 2.02 vs 2.15 ms（同档，差在噪声内），
   与「层数少就快」的直觉不符；没有做逐层 profile。**UNKNOWN**。
4. **时间方差**：同一 `.dla` 在本会话内 1.41–2.29 ms/inf；没有做时钟/热态控制，中位数只作档位比较。
5. **只测了 T=16**：T=8 的 P·V 仍被 16 B 行规则挡住（票 04 结论，本票未重测）；decode 的 T=1 形态未碰。

---

## 14. 复现

    export QWEN25_WEIGHTS_DIR="/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen"

    # int16 算子接受度（15 个探针，先编后跑）
    wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-q04d' && \
      for p in p16_fc p16_bmm_pv p16_bmm_qk p16_mul p16_add p16_sub p16_mean p16_rsqrt \
               p16_logistic p16_shapeops p16_seam p16_dequant p16_mul_one \
               p16_native_seam p16_native_seam_tflite; do \
        /usr/bin/python3 tools/build_w4a8.py \$p --a-bits 16 --out ops/_w4a16_probe/\$p && \
        ./tools/compile.sh ops/_w4a16_probe/\$p/model.tflite ops/_w4a16_probe/\$p/model.dla; done"
    export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
    ./tools/devlock.sh run -- ./tools/e2e_i16.sh ops/_w4a16_probe/p16_native_seam

    # 组件与整层（两种缝）
    wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-q04d' && \
      /usr/bin/python3 tools/build_w4a8.py layer --w-bits 4 --a-bits 16 --seam f16 \
        --out ops/decoder_layer_t16_w4a16 && \
      ./tools/compile.sh ops/decoder_layer_t16_w4a16/model.tflite ops/decoder_layer_t16_w4a16/model.dla"
    ./tools/devlock.sh run -- ./tools/e2e_i16.sh ops/decoder_layer_t16_w4a16

    # 质量口径（device vs 独立 fp64 链）与分段
    wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-q04d' && \
      /usr/bin/python3 tools/build_w4a8.py --a-bits 16 --diag --out ops/decoder_layer_t16_w4a16 && \
      for st in n1 q3 scores probs attn r1 n2 mlp_y down out; do \
        /usr/bin/python3 tools/build_w4a8.py layer --w-bits 4 --a-bits 16 --upto \$st --out /tmp/stg; done"

    # 归因：只量化权重的 fp64 链
    wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-q04d' && \
      /usr/bin/python3 tools/build_w4a8.py --wq-chain"

**可复现性**：同参数重建 `model.tflite` 与 `golden/*.bin` 逐字节稳定（与票 04/04c 一致）；
`.dla` 不可复现（仓库既有约定），只作产物标识。

---

## 15. 票面 checklist 对照

| 票面要求 | 状态 | 证据 |
|---|---|---|
| 沿用 `tools/build_w4a8.py`，加 `--a-bits 8\|16`：激活与输出 int16、bias 仍 int32 per-channel、op version 取接受的档 | ✅ | §3；FC v9、BMM v1 全部 EXIT=0 |
| 组件级：FC / SwiGLU / BMM-QK / BMM-PV 的 4W16A 版（板端 vs 量化 golden；vs fp64） | ✅ 数据齐，2 过 2 不过 | §5：FC/BMM-QK/BMM-PV 码级 PASS + cosine ≥ 0.9999；SwiGLU 0.96826 FAIL（int4 权重）、rmsnorm 码级 FAIL（LUT） |
| 整层 4W16A（T=16）：层数/执行步/Static/噪声底/码级/cosine | ✅ | §6：61 层 / 1 步 / 7.3M / 噪声 0 / mean 61 max 581 / **0.9510** |
| 关键对照：`scores → probs` 是否从 0.96 抬到 ≥0.999；整层是否 ≥0.9999 | ✅ | §7：段边界 **0.9999993**、整层内 **0.99918**（≥0.999 ✅）；整层 **0.9510 < 0.9999 ✗**，缺口全在 int4 权重（§8） |
| 记录 int16 的代价：`.dla`、Static、板端单层时间 | ✅ | §9：+1.2% / 7.3M / 2.02–2.15 ms（对 4W8A 1.40、8W8A 2.23，同会话） |
| 若某个 seam 编不过，记录该 seam 与 fp16 替代的位置与层数代价 | ✅ **不需要** | §10：int16 掩码 SOFTMAX 编得过；61→59 层 |
