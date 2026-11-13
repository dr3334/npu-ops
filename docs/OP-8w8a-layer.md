# OP-8w8a —— 8W8A（int8 权重 / int8 激活）参考档：同一 harness 的中间点

> 票：`.scratch/quant-runtime/issues/04c-8w8a-reference.md`；spec `.scratch/quant-runtime/spec.md`；
> ADR `docs/adr/0005-quantization-tier-target.md`（含 2026-11-13 的实测订正）；上位票证据 `docs/OP-w4a8-layer.md`。
> 分支 `impl/q04c-8w8a`，worktree `wt-q04c`；权重夹具 `npu-ops/weights/qwen25_frozen`（票 02 冻结，`QWEN25_WEIGHTS_DIR` 只读引用）。
> 编译器 ncc-tflite **7.3.15** / `--arch=mdla5.0 --opt=3 --disallow-bridge --num-mdla=2 --l1-size-kb=3072`；
> 设备 mt6897 / Android 16（`0123456789ABCDEF`），板端全部持 `/d/deepseek_ workspace/mtk/.npu-board.lock`（会话前后各一次 `healthcheck --quick`，两次都 PASS）。
> 交付：`tools/build_w4a8.py`（新增 `--w-bits 4|8`、`--weights-report`、`--wq-chain`）、`ops/w8a8_{fc_proj,swiglu,bmm_qk,bmm_pv}/`、
> `ops/decoder_layer_t16_w8a8/`、`ops/_w8a8_bisect/*`、本文。公共工具（e2e/compare/compile/devlock/build_ops）**一行未改**。

---

## 0. TL;DR

| 项 | 结果 |
|---|---|
| 8W8A 整层（T=16，沿用票 04 的 seam） | **EXIT=0、61 层、1 个执行步、无 bridge**；Static **14M**（4W8A 是 7.2M）、`.dla` **15,089,161 B**（4W8A 7,634,441） |
| 板端噪声底 | **0 码**（同一 `.dla` 同输入两轮，`out_device.bin` 逐字节相同；两档都测了） |
| 码级判据（device vs 量化 golden，登记 `int8_tol=2`） | **FAIL**：12660/14336 差（88.3%），均值 4.21 码、中位 3、**max 36** |
| 质量判据（ADR 订正后：device vs 独立 fp64 链 cosine） | **FAIL**：**0.9649298**（golden 自身 0.9673771）——差 **0.035**，不达 0.9999 |
| 设备是否忠实（`\|cos(dev) − cos(golden)\|`） | **忠实**：0.0024 ≤ ADR 订正后的 0.003 |
| 四个组件的码级 | **全部逐位**：FC 0/14336、SwiGLU 0/14336、BMM-QK 0/3584、BMM-PV 0/14336（票面预期「0 码」成立） |
| 四个组件的 cosine（vs fp64 链） | FC **0.9999718** PASS / BMM-QK **0.9999566** PASS / BMM-PV **0.9999431** PASS / SwiGLU **0.9991673** **FAIL**（差 0.0008） |
| 每矩阵权重量化误差 | int4 **14.02–15.83%** → int8 **0.772–0.873%**（**18×**）；激活 int8 per-tensor 仍 **2.39%** |
| 误差是谁的 | **激活**：只量化权重（fp64 链）到 **0.99985**；整层 0.9674 的缺口全在激活侧，且从 `probs` 一段开始掉 |
| 与 4W8A 的差 | 内存 **2×**（`.dla` +7,454,720 B = 参数量/2）、时间 **+25%~+51%**、整层 cosine 0.9091 → **0.9649** |

**一句话**：8W8A 在这条栈上**编得出、跑得动、组件逐位、噪声为 0**，权重误差比 4-bit 小 18 倍；但整层 cosine 只从 0.9091 抬到 **0.9649**，**仍不达 0.9999**。天花板不在权重（权重单独已到 0.99985），而在 **int8 激活**——具体是 `scores`/`probs` 这一段（1 码 = 37，softmax 近并列 row 翻转）。所以「把权重升到 8 bit」解决不了质量判据。

---

## 1. 票面要回答什么

1. 组件级：FC / SwiGLU / BMM-QK / BMM-PV 的 8W8A 版，板端 device vs 量化 golden（预期 0 码）+ vs fp64 链 cosine；
2. 整层 8W8A（沿用 T=16 的 seam）：层数 / 执行步 / Static 字节 / 噪声底 / 码级误差 / cosine；
3. 每矩阵权重量化误差（与 4-bit 的 14–16% 直接对比）；
4. 与 4W8A 的差值：`.dla` 体积、静态内存、板端单层前向时间；
5. 结论：8W8A 是否达到算子级 cosine ≥ 0.9999，或至少落在可接受区间；2× 内存 / 速度差是否值得。

**不做**：24 层、KV cache、decode、group-wise scale、混合精度整网（留给后续票）。

---

## 2. 事前登记（写进 `model.json`，都在第一次板端运行之前）

| 项 | 登记值 | 依据 |
|---|---|---|
| 组件 `int8_tol` | **1** | ADR 0005 的「\|Δ\| ≤ 1 码」；票面另外要求「预期 0 码」，0 码由 `compare.py` 的 `exact=` 行直接给，不靠 PASS 推断 |
| 整层 `int8_tol` | **2** | 与票 04 同一理由、同一数值：图里含同一条 int8 RMSNorm 链（RSQRT 定点表，`docs/OP-rmsnorm.md` §2.3 预算 4 码，票 04 实测 max 2 码）+ 同一条 fp16↔int8 缝 |
| 质量判据 | `cosine_min=0.9999`（device vs 独立 fp64 链） | ADR 0005 原文 |
| 设备忠实判据 | `\|cos(dev) − cos(golden)\| ≤ 0.003` | ADR 0005 2026-11-13 订正第 1 条（4-bit 档的算子级判据改为这一条） |

**没有事后调阈值**：组件的 1 码与整层的 2 码都是构建期写进 `model.json` 的；整层实测 max 36 码、cosine 0.9649，两个 FAIL 都原样保留。

---

## 3. 做法：一个开关，不重写

票 04 的 builder `tools/build_w4a8.py` 只加了一个 `--w-bits 4|8`：

| 部位 | int4（票 04） | int8（本票） |
|---|---|---|
| 权重码 | `[-8,7]`，`s[n] = max\|W[n,:]\| / 7` | `[-127,127]`，`s[n] = max\|W[n,:]\| / 127` |
| 落盘 | 2 codes/byte，低 nibble 在前（TFLITE INT4） | 原始 int8（TENSOR INT8），`quantized_dimension=0` |
| bias | int32 per-channel，`s[n] = s_a·s_w[n]` | 同（不变） |
| 激活 | int8 per-tensor，`s = max\|v\|/127` | 同（不变） |
| requant | `tools/quant_contract.py`（两次整数舍入 + per-channel 乘子） | 同（不变） |
| 图 | — | **一字未改**：同 T=16、同 seam、同 op 顺序、同 op version（FC v9） |

两侧共用的只有一件事被统一了：权重码除以**哪个**尺度。见 §10.2（票 04 的两份旧代码一份除 float32、一份除 float64，本票统一成 float64）。

---

## 4. 组件证据（组一）

| op | 形状 | 编译 | 板端 device vs 量化 golden | vs 独立 fp64 链 |
|---|---|---|---|---|
| `ops/w8a8_fc_proj` | q_proj `[16,896]→[16,896]`，W int8 `[896,896]` | EXIT=0、1 层、1 步、Static 792K、`.dla` 815,765 B | **0/14336 逐位** | cosine **0.9999718**、rel_l2 0.751% |
| `ops/w8a8_swiglu` | `[16,896]→[16,896]`，gate/up/down 三个 8W8A FC（I=4864），6 层 | EXIT=0、6 层、1 步、Static 12M、`.dla` 13,149,437 B | **0/14336 逐位** | cosine **0.9991673**、rel_l2 4.105% |
| `ops/w8a8_bmm_qk` | `[14,16,64]×[14,16,64]→[14,16,16]`，adj_y | EXIT=0、1 层、1 步、Static 0、`.dla` 9,969 B | **0/3584 逐位** | cosine **0.9999566**、rel_l2 0.933%（1 码 = 37.08，max\|ref\| 4710） |
| `ops/w8a8_bmm_pv` | `[14,16,16]×[14,16,64]→[14,16,64]` | EXIT=0、1 层、1 步、Static 0、`.dla` 16,773 B | **0/14336 逐位** | cosine **0.9999431**、rel_l2 1.066% |

**四条读法**

1. **8 个 int8 权重张量（896/896/896/896/4864/4864/896 个 per-channel 尺度）在板上是逐位的**：FC 与 SwiGLU 都 0 码。票面「预期 0 码」成立，且**不是靠 +1 码的容差蒙过去的**（`exact: YES`）。
2. **BMM 两段没有权重**（输入是 int8 激活），所以 4W8A 与 8W8A 的这两个 op **golden 逐字节相同**，`.tflite` 只差描述字符串（568 B / 560 B 两档一样大）。8W8A 的「BMM 8-bit」在语义上就是「不变」，这一点在表里如实标。
3. `swiglu` 的 cosine（0.99917）与码级逐位**不矛盾**：码级说「设备 = 我的量化模型」，cosine 说「量化模型离 fp64 多远」。0.99917 的缺口是**激活**（该 op 里 3 个 FC + 2 个 int8 MUL 各自的 per-tensor 量化），不是权重（§6 的 int8 权重只 0.78%）。
4. `fc_proj` 的 0.9999718 与票 04 的 0.9999647 几乎一样 —— **这一格是 bias 主导的**（q_proj 有 bias，把相对误差摊薄），别拿它当「8-bit 权重比 4-bit 好很多」的证据；纯权重那部分是 0.990 → 0.99997（§6）。

---

## 5. 整层证据（组二）

### 5.1 编译（`ops/decoder_layer_t16_w8a8/model.compile.log`）

    ===Graph[0] has 1 Execution Step(s)===    Target: MDLA_5_0
    61 个原生层（＝票 04 的 4W8A 整块 61 层，逐层同构）
    DRAM: Input 14K  Output 14K  Temp 7.0K  Static 14M  Code 33K  Total 14M
    EXIT=0   SIZE=15089161

* **1 个执行步、无 bridge**（`--disallow-bridge` 恒开）；
* **Static 14M vs 4W8A 的 7.2M**：整层权重 14,909,440 参数，int4 = 7,454,720 B、int8 = 14,909,440 B；两个 `.dla` 的体积差正好是 **+7,454,720 B**（见 §8），说明 int8 权重真的按 8 bit 落盘、int4 真的按 4 bit。

### 5.2 板端（`compare.txt`，持锁）

    op        : DECODER_LAYER stage=out {...}  dtype=int8
    expected  : 14336 vals   device: 14336 vals   logical shape=(16, 896)
    exact     : NO   mismatched=12660/14336
    contract  : int8_tol=2  max_abs_diff=36  fraction_off=88.3092%
    FAIL

码差分布（|Δ| in codes，`ops/decoder_layer_t16_w8a8`，与 4W8A 同表并列）：

| 图 | 逐位 | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | max | mean | median |
|---|---|---|---|---|---|---|---|---|---|---|
| **8W8A** | **11.7%** | 32.5% | 48.4% | 59.2% | 73.3% | 90.1% | 99.1% | **36** | **4.21** | 3 |
| 4W8A | 14.1% | 38.2% | 54.8% | 65.7% | 77.9% | 91.2% | 98.8% | 38 | 3.83 | 2 |

### 5.3 噪声底与「设备是否忠实」

| 量 | 值 |
|---|---|
| 噪声底（同 `.dla` 同输入两轮，14336 值） | **0 码**（`out_device.bin` 逐字节相同；4W8A 同样为 0） |
| `cos(dev, fp64 链)` | **0.9649298**（rel_l2 26.64%） |
| `cos(量化 golden, fp64 链)` | **0.9673771**（rel_l2 25.75%） |
| `cos(dev, 量化 golden)` | 0.9824443 |
| `\|cos(dev) − cos(golden)\|` | **0.0024** ≤ 0.003 ⇒ **ADR 订正后的算子级判据成立**（设备忠实） |
| 1 码（输出域） | 3.991e-3（`max\|out\| = 0.507`） |

**读法**：噪声底 0 ⇒ 12660 处码差是**确定性的模型差**，不是抖动；而设备离真理（0.9649）与我的量化模型离真理（0.9674）只差 0.0024 ⇒ 设备没有额外引入可测误差，**88% 的码差来自「逐算子 ±1 码级舍入模型」，与 4-bit 档同源**（票 04 的 4W8A 也是同一量级：0.9091 vs 0.9134，差 0.0032）。

---

## 6. 每矩阵权重量化误差（组三）

`/usr/bin/python3 tools/build_w4a8.py --weights-report`（同一份夹具、同一套 per-channel 对称方案，只换码域）：

| 矩阵 | int4 rel_l2 | int4 cos | **int8 rel_l2** | **int8 cos** | 参数 |
|---|---|---|---|---|---|
| Wq | 14.2036% | 0.9900681 | **0.7833%** | 0.9999693 | 802,816 |
| Wk | 14.0255% | 0.9903056 | **0.7720%** | 0.9999702 | 114,688 |
| Wv | 14.4058% | 0.9897750 | **0.7932%** | 0.9999685 | 114,688 |
| Wo | 14.1047% | 0.9902029 | **0.7768%** | 0.9999698 | 802,816 |
| Wg | 14.1210% | 0.9901812 | **0.7781%** | 0.9999697 | 4,358,144 |
| Wu | 14.0784% | 0.9902377 | **0.7764%** | 0.9999699 | 4,358,144 |
| Wd | 15.8318% | 0.9876993 | **0.8727%** | 0.9999619 | 4,358,144 |
| 激活 n1（int8 per-tensor，参照） | — | — | **2.3944%** | — | 14,336 |

* int4 的七个数字与票 04 §6.3 **逐位复现**（0.1420/0.1403/0.1441/0.1410/0.1412/0.1408/0.1583）⇒ 参数化没有动 4-bit 的量化数学。
* **int8 权重误差 0.77–0.87%，比 int4 小 18 倍**；但**激活的 2.39% 现在是最大单项**——这一条决定了下面 §7 的归因。

---

## 7. 误差归因：权重还是激活（票面的关键问题）

### 7.1 只量化权重（激活保持 fp64）

`/usr/bin/python3 tools/build_w4a8.py --wq-chain`：把独立 fp64 链跑在「反量化后的权重」上（激活精确），与纯 fp64 链对比：

    weights-only int4: cosine(out)=0.9509612  rel_l2(out)=31.90%
    weights-only int8: cosine(out)=0.9998451  rel_l2(out)=1.76%

⇒ **权重侧的天花板是 0.99985**（int8）。整层实际是 0.9674，与 0.99985 的 0.032 缺口**全部来自激活**。

### 7.2 设备无关的分段（golden vs fp64 链，`--upto`）

| stage | 4W8A（票 04） | **8W8A（本票）** |
|---|---|---|
| n1（RMSNorm#1，无权重） | 0.9995702 | **0.9995702**（同一段，逐位相同） |
| q3（FC+RoPE） | 0.9999334 | **0.9999410** |
| scores（BMM，无权重） | 0.9999013 | **0.9999110** |
| **probs（缝 + 掩码 softmax）** | **0.9541** | **0.9606418** |
| attn（P·V） | 0.9504 | **0.9651807** |
| r1（o_proj + 残差） | 0.9717 | **0.9836251** |
| n2（RMSNorm#2） | 0.9716 | **0.9835984** |
| mlp_y | 0.9231 | **0.9669002** |
| down | 0.9111 | **0.9669223** |
| out | 0.9130 | **0.9673771** |

**两条结论**

1. **掉码仍然从 `probs` 开始**（0.99991 → 0.9606），而 `probs` 这一段**不经过任何权重**（BMM 输出 → DEQUANTIZE → ×1/8 → +mask → SOFTMAX）。8W8A 与 4W8A 在这一段几乎同值（0.9606 vs 0.9541）⇒ 这是**激活**侧的性质：`scores` 的 1 码 = 37.08（max 4710，相对 0.79%），softmax 对「差 < 1 码」的近并列 row 会翻转 argmax（票 04 实测 224 行里 17 行翻）。
2. 权重从 4→8 bit 在**中间段**确实有大收益（mlp_y 0.9231 → 0.9669、down 0.9111 → 0.9669），但整层被 probs 那一段的 0.96 天花板压住，**最终只到 0.9674**。

---

## 8. 与 4W8A 的差值（组四）

两档用**同一个 builder、同一次会话、同一把锁**测，除权重格式外一切相同：

| 量 | 4W8A | **8W8A** | 差 |
|---|---|---|---|
| `.dla` 体积 | 7,634,441 B | **15,089,161 B** | **+7,454,720 B（1.976×）** = 14,909,440 参数 ÷ 2 |
| Static（编译日志） | 7.2M | **14M** | 2× |
| 原生层数 / 执行步 | 61 / 1 | **61 / 1** | 无 |
| 板端单层前向（同会话交替 3 对，`-c 30`） | 中位 **1.44 ms/inf** | 中位 **2.17 ms/inf** | **+1.5×**（3 对：1.643/1.333/1.437 vs 2.155/2.213/2.171） |
| 同上，`-c 20` 单发 | 1.3209 ms/inf（745 FPS） | 1.83605 ms/inf（536 FPS） | +1.39× |
| 同上，另一会话 `-c 20` | 1.33135 ms/inf（733 FPS） | 1.66000 ms/inf（593 FPS） | +1.25× |
| 整层 cosine（device vs fp64） | 0.9090539 | **0.9649298** | +0.056 |
| 整层 rel_l2 | 43.45% | **26.64%** | −16.8 pt |
| 整层码级（device vs golden） | 12314/14336、max 38 | 12660/14336、**max 36** | 同档（都是逐算子舍入模型，与档位无关） |

**时间的口径提醒**：跨会话重复测到 8W8A 1.66–2.21 ms/inf、4W8A 1.32–1.64 ms/inf，**会话间的时钟/热态漂移比档位差还大**。同会话交替（A/B/A/B/A/B）的 3 对是唯一能对比的读法，结论是 **1.3–1.5×**，不是 2×——说明这一层不是纯权重带宽受限（Temp/L1 那部分 61 层的工作量与档位无关）。

**值不值**：内存确实 2×，时间 +25%~+50%，整层 cosine 0.9091 → 0.9649（rel_l2 43%→27%）。**若判据是 0.9999，两者都不值**——因为换成 8-bit 权重后天花板已经移到激活侧（§7），再往上加权重精度没有任何收益；真正能动的是 attention 的分数/softmax 段（ADR 目标形态「attention 4W16A + MLP 4W8A」正是这个意思）。

---

## 9. 与票面 checklist 对照

| 票面要求 | 状态 | 证据 |
|---|---|---|
| 组件级 FC / SwiGLU / BMM-QK / BMM-PV 的 8W8A 版，device vs 量化 golden（预期 0 码） | ✅ **四个全 0 码** | §4（`exact: YES`，非容差内通过） |
| 组件级 vs fp64 链 cosine | ✅ 数据齐（3 过 1 不过） | §4：FC 0.9999718、BMM-QK 0.9999566、BMM-PV 0.9999431 过；SwiGLU 0.9991673 不过（差 0.0008） |
| 整层 8W8A（T=16）：层数/执行步/Static/噪声底/码级/cosine | ✅ 全给 | §5：61 层 / 1 步 / 14M / 噪声 0 / 均值 4.21 max 36 / 0.9649 |
| 每矩阵量化误差（对照 4-bit 14–16%） | ✅ | §6：int4 14.02–15.83% → int8 0.772–0.873% |
| `.dla` 体积 / 静态内存 / 板端单层前向时间 | ✅ | §8：+7,454,720 B / 7.2M→14M / 1.3–1.5× |
| 结论：是否达到算子级 cosine ≥ 0.9999，2× 值不值 | ✅ | §4（组件 3/4 过）、§5（整层 **0.9649，差 0.035**）、§8（不值，天花板在激活） |

---

## 10. 对票 04 证据的两处订正（都留痕，不删旧结论）

### 10.1 组件表里 `bmm_qk` 的 cosine

`docs/OP-w4a8-layer.md` §4 写 `bmm_qk`「cosine 0.9999013」，但该 op 的 `model.json`（提交在仓库里）是 **0.9999565708**。0.9999013 是**整层 `scores` stage** 的值（那一路的 q/k 来自 int8 FC+RoPE 链，与组件直接吃「精确 q3 的 per-tensor 量化」不同输入）。本票按 artifact 记 **0.9999566**，并保留票 04 的 stage 表原值。

### 10.2 4-bit 的两份权重码算法被统一了（整层/`swiglu` 的证据有 ~1e-4 位移）

票 04 的代码里，权重码除以**哪个**尺度有两种写法：`sw4()`（组件 `fc_proj`）用 float64 尺度，`q_fc_tables()`（整层与 `swiglu`）用 float32 尺度。本票统一成 float64（`sw_weights()`），于是：

| 4W8A 证据 | 票 04（提交值） | 本票重建 | 位移 |
|---|---|---|---|
| `ops/w4a8_fc_proj` cosine | 0.9999647316522977 | **0.9999647316522977** | 0（原本就是 float64 路径） |
| `ops/w4a8_swiglu` cosine | 0.967380449092259 | **0.9673534027823805** | −2.7e-5 |
| `ops/decoder_layer_t16_w4a8` cosine(golden) | 0.9129615320000575 | **0.9133737549356803** | +4.1e-4 |
| 同上 码级（device vs golden） | 12522/14336、max 41 | **12314/14336、max 38** | 权重码在半个 LSB 边界上少量翻转，golden 与图同时变 |

两个 4W8A 组件（`fc_proj`、`swiglu`）都在本会话**重跑并刷新了 `compare.txt`**：仍然 **0/14336 逐位**，所以票 04 的码级结论不变。整层的质量结论也不变（0.9130 → 0.9134，仍在 0.91 档）。**本票两档全部数字都出自这一套统一代码**，比较是同口径的。

---

## 11. 适用范围 / 不适用

**适用**

1. **int8 per-channel 权重（8 个张量、896/4864 个尺度）在 FC 上逐位**：`w8a8_fc_proj` 与 `w8a8_swiglu` 各 0/N 码，`.dla` 1 个执行步；
2. **8W8A 整层仍是 61 层 / 1 个执行步 / 无 bridge**，噪声底 0 —— 档位不改变执行形态；
3. **`.dla` 体积差 = 参数量/2**（+7,454,720 B）：int8/int4 的落盘是真的，不是编译侧假象；
4. **8W8A 的整层质量天花板由 int8 激活决定**（权重单独 0.99985，整层 0.9649，probs 段从 0.9999 掉到 0.9606）；
5. 组件级判据：8-bit 档的 FC / BMM 段确实能逐位 + 过 0.9999；SwiGLU 过不了（0.99917），原因是它含 3 个 FC + 2 个 int8 MUL 的激活量化叠加。

**不适用**

* 24 层 / KV cache / decode（下一票）；
* 真实权重档（本票用冻结随机夹具；真实 `k_proj.bias` 更大、S 更大，阈值要重算）；
* 性能结论（1.3–1.5× 只在本图、本 T=16、本会话成立；T=1 的 decode 形态完全不同）；
* 「8W8A 质量可用」——**本票证明的恰恰是「升权重档没用」**（§7）。

---

## 12. 剩余疑问（UNKNOWN，不圆）

1. **probs 的近并列翻转能否用「fp16 分数」消掉**：把 `scores` 段改成 fp16（attention 4W16A / 或 BMM 输出直接 fp16）应当能把 probs 从 0.9606 抬到 ~0.9999+，**未试**——这是下一票的自然实验，也是 ADR 目标形态要的。
2. **权重侧 0.99985 差的那一口气**：per-channel int8 只到 0.99985，group-wise（或 fp16 权重）能不能过 0.9999 **未试**；注意它与激活侧的缺口（0.032）差一个数量级，优先级低。
3. **码级 88% 的差压不到 0**：与 4-bit 档同一量级（12314 vs 12660），来源是 elementwise int8 段（MUL/ADD/RSQRT/LOGISTIC/SOFTMAX）的舍入模型，我只对齐了 FC/BMM 的两次舍入契约。票 04 §11.2 的估算（把 elementwise 输出尺度改成 2 的幂倍数）**仍未做**。
4. **时间方差**：跨会话 8W8A 1.66–2.21、4W8A 1.32–1.64 ms/inf，没有做时钟/热态控制；`-c 20/30` 的均值也不等于稳态吞吐（含首轮装载外的时间）。真实 decode 的每 token 时间是另一件事。

---

## 13. 复现

    export QWEN25_WEIGHTS_DIR="/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen"

    # 组件（8W8A）：build -> compile -> 板端（持锁，前后 healthcheck --quick）
    wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-q04c' && \
      /usr/bin/python3 tools/build_w4a8.py fc_proj --w-bits 8 --out ops/w8a8_fc_proj && \
      ./tools/compile.sh ops/w8a8_fc_proj/model.tflite ops/w8a8_fc_proj/model.dla"
    export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
    ./tools/devlock.sh run -- ./tools/e2e.sh ops/w8a8_fc_proj

    # 整层（T=16，8W8A）与 4W8A 基线（同一条命令换 --w-bits）
    wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-q04c' && \
      /usr/bin/python3 tools/build_w4a8.py layer --w-bits 8 --out ops/decoder_layer_t16_w8a8 && \
      ./tools/compile.sh ops/decoder_layer_t16_w8a8/model.tflite ops/decoder_layer_t16_w8a8/model.dla"
    ./tools/devlock.sh run -- ./tools/e2e.sh ops/decoder_layer_t16_w8a8

    # 设备 dump 对独立 fp64 链（质量口径）
    wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-q04c' && \
      /usr/bin/python3 tools/build_w4a8.py layer --w-bits 8 --diag --out ops/decoder_layer_t16_w8a8"

    # 每矩阵权重量化误差 / 只量化权重的归因
    wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-q04c' && \
      /usr/bin/python3 tools/build_w4a8.py --weights-report && \
      /usr/bin/python3 tools/build_w4a8.py --wq-chain"

    # 分段（掉码定位，device-free）
    wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-q04c' && \
      /usr/bin/python3 tools/build_w4a8.py layer --w-bits 8 --upto probs --out ops/_w8a8_bisect/probs"

    # 噪声底（同 .dla 同输入两轮）与计时
    ./tools/devlock.sh run -- bash -c 'cp ops/decoder_layer_t16_w8a8/out_device.bin /tmp/r1.bin; \
      ./tools/e2e.sh ops/decoder_layer_t16_w8a8; cmp /tmp/r1.bin ops/decoder_layer_t16_w8a8/out_device.bin && echo NOISE=0'
    adb shell "cd /data/local/tmp/npuops && LD_LIBRARY_PATH=\$PWD/lib ./bin/neuronrt -m hw \
      -a dla/decoder_layer_t16_w8a8.dla -i dla/decoder_layer_t16_w8a8.in.bin -o dla/o.bin -c 30"

**可复现性**：`build_w4a8.py layer --w-bits 8` 重建的 `model.tflite` 与 `golden/*.bin` 逐字节稳定（同票 04 的结论）；`.dla` 不可复现（仓库既有约定），只作产物标识。