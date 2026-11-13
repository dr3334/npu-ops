# OP-w4a8 —— 4W8A（int4 权重 / int8 激活）单层 decoder

> 票：`.scratch/quant-runtime/issues/04-w4a8-single-layer.md`；spec `.scratch/quant-runtime/spec.md`；
> ADR `docs/adr/0005-quantization-tier-target.md`。
> 分支 `impl/q04-w4a8-layer`，worktree `wt-q04`；权重夹具 `npu-ops/weights/qwen25_frozen`（票 02 冻结，sha256 校验）。
> 编译器 ncc-tflite **7.3.15** / `--arch=mdla5.0 --opt=3 --disallow-bridge --num-mdla=2 --l1-size-kb=3072`；
> 设备 mt6897 / Android 16（`"0123456789ABCDEF"）`，板端全部持 `/d/deepseek_ workspace/mtk/.npu-board.lock`。
> 交付：`tools/build_w4a8.py`（builder + `--upto` 分段 + `--diag`）、`ops/w4a8_{fc_proj,rmsnorm,swiglu,bmm_qk,bmm_pv}/`、
> `ops/decoder_layer_t16_w4a8/`、`ops/_w4a8_bisect/{q3,scores,probs,attn,r1,down}/`、`ops/_w4a8_probe_*/`、本文。

---

## 0. TL;DR

| 项 | 结果 |
|---|---|
| 整层 4W8A（T=16） | **EXIT=0、61 层、1 个执行步、无 bridge**；静态 **7.2 MiB**（int4 权重真的按 4 bit 落盘）、Temp 7.0K |
| 板端噪声底 | **0 码**（同 .dla 同输入两轮，两次 dump 逐字节相同） |
| 码级判据（device vs 量化 golden） | **FAIL**：12522/14336 差（87.3%），均值 4.13 码、中位 ≤2 码、**max 41 码**（`int8_tol=2` 跑前登记） |
| 质量判据（ADR 0005：cosine ≥ 0.9999） | **FAIL**：device vs 独立 float64 链 **cosine 0.9107 / rel_l2 0.425** |
| 误差是谁的 | **不是实现的**：`cos(dev,fp32)=0.9107` 与 `cos(golden,fp32)=0.9130` 同档 ⇒ 设备忠实实现了这套量化数学；差距来自**量化本身** |
| 五个组件 | FC / SwiGLU / BMM-QK / BMM-PV 板端 **逐位（0 码差）**；RMSNorm int8 **max 2 码**（RSQRT 定点表，仓库文档预算 4 码） |
| 三个负结果 | ① int8 掩码 SOFTMAX **编不过**（无 exec plan）；② int8→int8 `QUANTIZE`（RequantizeLayer）**能编但算错**（最小 5 层图 rel_l2 0.85 vs 0.00009）；③ T=8 的 P·V **被 16 B 行规则挡住**（EXIT=1 bridge） |
| 与 ADR 0005 的关系 | 本票**实测把 ADR 的担心变成数字**：4-bit 每通道权重每矩阵 14% 相对误差，int8 分数在 softmax 上放大 —— 这正是 ADR 选「attention 4W16A + MLP 4W8A」的理由 |

**一句话**：4W8A 单层在这条栈上**编得出来、跑得动、1 个执行步、噪声为 0**，五个组件里四个板端逐位；但**数值达不到 ADR 的两条判据**——不是接线错，而是 int4 每通道权重（14%/矩阵）与 int8 分数（softmax 近并列 row）本身的代价，逐段量化见 §6。

---

## 1. 票面要回答什么

1. int4 权重按 TFLite 约定打包（2 个/字节、低 nibble 在前）、per-channel scale、op version 取 ncc 接受的那几档；
2. 逐算子先过编译：FC / BMM / CONCAT / SPLIT / SOFTMAX + RMSNorm（eps 在低比特下会被折掉）；
3. 上板对拍：算子级 `|Δ| ≤ 1 码` + `cosine ≥ 0.9999`，**逐层记录哪一段开始掉码**；
4. 记录 16 B 行规则在 int8 激活下的约束；
5. 噪声底与阈值按上一轮方法**先登记后跑**。

**不做**：24 层、KV cache、decode、性能结论、混合精度整网（留给下一票）。

---

## 2. 事前登记（写于本票任何编译/板端动作之前）

| # | 预测 | 结果 |
|---|---|---|
| P1 | 4W8A 单层能编、1 个执行步、无 bridge，层数与 fp16 的 61 同档（±5） | ✅ 61 层 / 1 步（§5） |
| P2 | int4 权重在设备上**真的只占 4 bit**（静态体积 ≈ 参数量/2） | ✅ 7.2 MiB vs fp16 29.4 MiB（§5） |
| P3 | 噪声底 = 0（仓库所有链都是逐字节相同） | ✅ 0 码（§5.3） |
| P4 | 组件级码差 ≤1 码；RMSNorm 例外（RSQRT 定点表，docs/OP-rmsnorm.md §2.3 预算 4 码） | ✅ 四个组件 0 码；RMSNorm **max 2 码**（§4） |
| P5 | 整层**达不到** ADR 的 cosine 0.9999（4-bit 每通道权重的固有噪声） | ✅ 实测 0.9107，且归因见 §6 |
| P6 | 16 B 行规则在 int8 下仍然生效（T=8 的 probs 行宽 8 B） | ✅ 复现 EXIT=1（§7.3） |
| P7 | int8 下的 eps 会被常量折叠（ADD 层消失），golden 必须同步 eps=0 | ✅ 5 层、ADD 不见；影响 <1 码（§8） |

**容差登记（跑之前写进 model.json）**

* 组件：`int8_tol=1`（ADR 0005 的「1 码」），`quality_criterion.cosine_min=0.9999`（ADR 原文）；
* 整层：`int8_tol=2`，理由写进 `LAYER_TOL`（图里含 int8 RMSNorm 链 + 1 条 fp16↔int8 缝），**在整层第一次板端运行之前**就写进 model.json；
* **没有任何一次「看到 FAIL 再调阈值」**：RMSNorm 组件在 `int8_tol=1` 下实测 max 2 码，本票**保留这个 FAIL**（ops/w4a8_rmsnorm/compare.txt），没有改登记重跑。整层的 `int8_tol=2` 是登记在前、并且**实测仍然 FAIL**（max 41 码），也没有改。

---

## 3. 做法

### 3.1 图的形状（tools/build_w4a8.py，T=16）

与 fp16 整块同构（docs/OP-decoder-layer.md §3.1），只把 dtype 换成 int4 权重 / int8 激活，并按下面的实测边界改了两处：

    x[16,896] int8 --RMSNorm×5(eps 折掉)--> n1
      --4W8A FC q/k/v(+int32 per-channel bias)--> RESHAPE[T,H,64]
      --RoPE(split-half, cos/sin 量化成 int8 常量)--> 4×MUL + SUB + ADD + CONCAT
      --SPLIT+CONCAT 2->14--> TRANSPOSE[1,0,2]
      --BMM(adj_y) scores[14,16,16] int8
      --DEQUANTIZE--> fp16 MUL(1/8) --+mask--> SOFTMAX --> QUANTIZE(1/127)     <- 唯一的 dtype 缝
      --BMM(P,V) [14,16,64] int8 --TRANSPOSE--> RESHAPE[16,896]
      --4W8A FC(o_proj, 零 bias) --ADD(x)--> r1
      --RMSNorm×5--> n2 --4W8A FC gate/up--> LOGISTIC --> MUL --> MUL --> 4W8A FC down --ADD(r1)--> out

**为什么 T=16 而不是 T=8**：int8 激活下 P·V 的 p 是 `[H,T,T]`，T=8 ⇒ 行宽 **8 B**，被 16 B 行规则挡住（§7.3 有本票的复现）。T=16 ⇒ 16 B，编译通过。fp16 的 T=8 结论不受影响（fp16 每行 16 B）。

**为什么 softmax 走 fp16 缝**：两条独立理由（§7.1、§7.2）：int8 掩码 SOFTMAX 编不过；且 −4096 的掩码在 int8 分数域（1 码 = 37）里根本不可表示。ADR 0005 明确允许「每条 dtype 缝显式写 DEQUANTIZE（+1 层/缝）」。

### 3.2 量化契约

| 部位 | 约定 | 依据 |
|---|---|---|
| 激活（全图） | per-tensor 对称 int8，s = max|v|/127，zp=0 | 与仓库全部 int8 op 一致 |
| 权重 | int4 对称 **per-output-channel**，s[n] = max|W[n,:]|/7，codes ∈ [−8,7] | probe-w4a8 板端逐位证据 |
| 打包 | 2 codes/byte，**低 nibble 在前** | probe-w4a8：反序 63/64 错 |
| bias | **int32 + per-channel**，s[n]=s_a·s_w[n] | ncc: Bias shape should be per channel quant |
| 输出 | **只能 per-tensor**（per-channel 输出被拒：Output cannot be per-channel quantization） | probe pc_out |
| op version | FC **v9**（{1,3,4,5,6,7,9} 可用；canonical v13 加载期被拒） | probe |
| requant | 两次整数舍入 + 整乘子 M0（tools/quant_contract.py）；per-channel 逐通道套用 | Q02 + 本票 |
| 精度承诺 | **不承诺逐位**（ADR 0005）；码级 |Δ|≤tol + cosine | ADR |

**per-channel 乘子的新证据**：Q02 的逐位结论只覆盖「张量级单乘子」。本票 `w4a8_fc_proj`（896 个 per-channel 乘子、896 个 per-channel bias scale）在板端 **0/14336 逐位** ⇒ **per-channel 乘子在 FC 上也是逐位的**，Q02 的例外清单可以据此收窄（仍不覆盖 conv/avgpool）。

### 3.3 三个 numpy 参考

| 名字 | 作用 |
|---|---|
| **fp32/fp64 链**（`chain_f32`） | 与 flatbuffer 写出代码零共享；**质量判据**用它对（cosine / rel_l2） |
| **量化 golden**（每个 helper 同时算码值并发射算子） | 设备语义模型（两次舍入契约、per-channel 乘子、fp16 缝逐边界舍入）；**码级判据**用它（tools/compare.py） |
| `--diag` | 板端 dump 对 fp32 链的 cosine/rel_l2（独立口径，不受 golden 影响） |

> 说明（R3 边界）：码级 golden 与图形发射共用一套 helper 的*循环结构*，但**数学是 numpy 独立实现的**；真正的独立参考是 fp32 链，它就是 §6 里所有 cosine 的来源。

---

## 4. 组件证据（每个组件一个 op 目录，各自编译 + 上板）

| op | 形状 | 编译 | 板端（device vs 量化 golden） | 质量（vs fp32 链） |
|---|---|---|---|---|
| `ops/w4a8_fc_proj` | q_proj `[16,896]→[16,896]`，W int4 `[896,896]` | EXIT=0、1 层、1 步、Static 400K | **0/14336 逐位** | cosine **0.9999647**、rel_l2 8.40e-3 |
| `ops/w4a8_rmsnorm` | `[16,896]→[16,896]`，5 层 int8 | EXIT=0、5 层、1 步、Static 1.9K | **861/14336**，max **2 码** → **FAIL**（登记 1 码） | cosine 0.9995702、rel_l2 2.93e-2 |
| `ops/w4a8_swiglu` | `[16,896]→[16,896]`，gate/up/down 全 4W8A，6 层 | EXIT=0、6 层、1 步、Static 6.3M | **0/14336 逐位** | cosine **0.9673804**、rel_l2 2.62e-1 |
| `ops/w4a8_bmm_qk` | `[14,16,64]×[14,16,64]→[14,16,16]`，adj_y | EXIT=0、1 层、1 步 | **0/3584 逐位** | cosine 0.9999013（分数 max 4710，1 码 = 37） |
| `ops/w4a8_bmm_pv` | `[14,16,16]×[14,16,64]→[14,16,64]` | EXIT=0、1 层、1 步 | **0/14336 逐位** | cosine 0.9999431 |

**四条结论**

1. **int4 权重 + int8 激活在 FC 上是逐位的**（不是「1 码」而是 0 码）；SwiGLU 三个 FC 拼起来也逐位 ⇒ 4W8A 的**实现是干净的**。
2. `swiglu` 的 cosine 只有 **0.967** 而码级逐位 —— 两者不矛盾：码级说的是「设备 = 我的量化模型」，cosine 说的是「量化模型离 fp32 多远」。**4-bit 权重的代价就在这个 gap 里**（§6）。
3. `rmsnorm` 是唯一码级不逐位的组件：RSQRT 是定点表（docs/OP-rmsnorm.md §2.1 实测 int8 ±1 码），又被后一级 `MUL(x, rsqrt)` 放大，实测 max 2 码。仓库对该链登记的预算是 **4 码**，本票登记 1 码 ⇒ **保留 FAIL**，不重登记。
4. `bmm_qk` 的 1 码 = 37（分数 max 4710）：这条 op 的**相对**误差 1.15e-2 —— 正是它后面 softmax 掉码的根源（§6、§7.1）。

---

## 5. 整层证据

### 5.1 编译（ops/decoder_layer_t16_w4a8/model.compile.log）

    ===Graph[0] has 1 Execution Step(s)===    Target: MDLA_5_0
    61 个原生层（fp16 整块也是 61 层）
    DRAM: Input 14K  Output 14K  Temp 7.0K  Static 7.2M  Code 33K  Total 7.3M
    EXIT=0   SIZE=7634440

* **1 个执行步、无 bridge**（`--disallow-bridge` 恒开）；
* **Static 7.2 MiB**：int4 权重合计 ≈14.9M 参数 / 2 = 7.45 MB ⇒ 权重**真的只占 4 bit**（fp16 整块是 28.5 MiB）；
* 层数 61 = fp16 整块的 61；probs 段由「int8: MUL+ADD+SOFTMAX（3 层）」换成「DEQUANTIZE+MUL+ADD+SOFTMAX+QUANTIZE（5 层）」，而 int8 路径少掉的 RMSNorm eps ADD（2 层）正好抵掉（**巧合，不是设计**）。

### 5.2 板端（compare.txt，持锁两轮）

    op        : DECODER_LAYER stage=out {'T': 16, 'D': 896, 'H': 14, ...}  dtype=int8
    expected  : 14336 vals   device: 14336 vals   logical shape=(16, 896)
    exact     : NO   mismatched=12522/14336
    contract  : int8_tol=2  max_abs_diff=41  fraction_off=87.3465%
    FAIL

码差分布（|Δ| in codes）：

| ≤0 | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | max | mean |
|---|---|---|---|---|---|---|---|---|
| 12.6% | 33.9% | 50.0% | 61.3% | 74.9% | 90.6% | 98.7% | **41** | **4.13** |

### 5.3 噪声底与设备 vs 真理

| 量 | 值 |
|---|---|
| 噪声底（同 .dla 同输入两轮） | **0 码**（14336/14336 逐字节相同）⇒ 硬件确定性，与仓库其余证据一致 |
| `cos(dev, fp32 链)` | **0.910659** |
| `cos(量化 golden, fp32 链)` | **0.912962** |
| `cos(dev, 量化 golden)` | 0.982818 |
| 1 码（输出域） | 3.99e-3（out max|v| = 0.507） |

**读法**：设备与独立 fp32 链的距离（0.9107）和我的量化模型与 fp32 链的距离（0.9130）**只差 0.0023**。⇒ 设备没有额外引入可测误差；87% 的码差来自「量化 golden 与设备在**每算子 ±1 码级舍入规则**上的差异」，在 ~30 个算子后被放大到均值 4 码。

---

## 6. 哪一段开始掉码（票面第 3 条）

### 6.1 设备无关（golden vs fp32 链，逐 stage）

| stage | max abs ref | 1 码 | cosine | max_abs | 误差（码） |
|---|---|---|---|---|---|
| n1（RMSNorm#1） | 1.051 | 8.35e-3 | 0.9995702 | 1.59e-2 | 1.9 |
| q3（FC+RoPE） | 25.55 | 0.201 | 0.9999334 | 0.387 | 1.9 |
| scores（BMM） | 4710 | 37.1 | 0.9999013 | 117 | 3.2 |
| **probs（softmax）** | 1 | — | **0.9541** | 0.985 | — |
| attn（PV） | 0.1468 | 1.16e-3 | 0.9504 | 0.135 | 116 |
| r1（o_proj+残差） | 0.0871 | 6.86e-4 | 0.9717 | 2.34e-2 | 34 |
| n2（RMSNorm#2） | 3.674 | 0.030 | 0.9716 | 0.998 | 33 |
| mlp_y（SwiGLU 中段） | 1.186 | 9.73e-3 | 0.9231 | 0.615 | 63 |
| down | 0.513 | 3.98e-3 | 0.9111 | 0.293 | 74 |
| **out** | 0.507 | 3.99e-3 | **0.9130** | 0.297 | 75 |

**掉码从 probs 开始**（0.9999 → 0.954），不是从 FC。原因：分数张量的 1 码 = 37（max 4710），除以 √d 后仍是 **4.6**，而 softmax 对「差 < 1 码」的 row 是**近并列**——这些 row 的 argmax 会翻。

### 6.2 板端分段（同一把锁、同一次会话）

| 图（--upto） | 层数 | 执行步 | device vs 量化 golden |
|---|---|---|---|
| q3（FC+RoPE） | 16 | 1 | **1 码（186/14336，1.3%）PASS** |
| scores（+复制+TRANSPOSE+BMM） | 33 | 1 | **1 码（593/3584，16.5%）PASS** |
| probs（+缝+掩码+softmax） | 37 | 1 | rel_l2 0.193（fp16 输出，近并列 row 抖动，码级不可比） |
| attn（+PV+回排） | 47 | 1 | 29.2% 差、max 64 码 |
| r1（+o_proj+残差） | 49 | 1 | 58.6%、max 26 |
| down（+RMSNorm#2+SwiGLU） | 60 | 1 | 87.5%、max 42 |
| out（+残差） | **61** | 1 | 87.3%、max 41 |

⇒ **PASS 的是 q3 / scores（≤1 码），从 probs 起掉码**；之后每过一个 FC 就多一层 ±1 码级舍入差（29% → 59% → 88%），这与 §5.3「设备 vs 真理 = golden vs 真理」一致：**不是某一段接错**。

### 6.3 误差归因（数据，不是猜测）

| 来源 | 量 | 证据 |
|---|---|---|
| int4 每通道权重 | **每个矩阵 rel_l2 14.0–15.8%**（Wq 0.1420、Wk 0.1403、Wv 0.1441、Wo 0.1410、Wg 0.1412、Wu 0.1408、**Wd 0.1583**），cos 0.988–0.990 | 直接对矩阵算 |
| 同上，传递到投影 | 无 bias 的 MLP：gate 与 fp32 差 **14.2%**；mlp_y 21%；down **26%** | §6.1 |
| int8 激活（per-tensor） | n1 量化误差 rel_l2 **2.39%**（max 1.05、rms 0.100） | 直接算 |
| 单独消融 | 「只量化权重」rel 14.3%、「只量化激活」rel 0.36%（q/k/v 的 **bias 主导**，把相对误差摊薄） | 同一组输入 |
| int8 分数 | 1 码 = 37（max 4710）；probs argmax 与 fp32 比 **17/224 row 翻转**（7.6%） | §6.1 |
| softmax 近并列 | 仓库 fp16 T=128 也见过同样现象（11.6%，docs/OP-decoder-layer.md §14.4） | 一致性检查 |

**结论**：4-bit 每通道权重的 14% 是**主项**（MLP 无 bias，藏不住），int8 分数是**次项**但直接决定 attention 的 argmax。这正是 ADR 0005 选「attention 4W16A + MLP 4W8A」的定量依据；也说明 ADR 的「算子级 cosine ≥ 0.9999」在 **TFLite 只支持 per-channel（没有 group-wise）** 的前提下，对 4-bit 权重**不可能**达到（本票最好的一格 fc_proj 0.999965 是 bias 主导的假象，纯权重那部分是 0.990）。

---

## 7. 三个负结果（都做了最小复现）

### 7.1 int8 掩码 SOFTMAX 编不过

    OP[37]: SOFTMAX
     ├ MDLA: Large scale quant Softmax may have accuracy issue. Try --opt-aggressive?
     ├ EDPA: unsupported operation
    ERROR: Cannot find an execution plan because of unsupported operations

即使 --opt-aggressive 也不一定救得回来（本票没有开：图是研究形态，先保可复现）；更重要的是**掩码本身不可表示**：−4096 在 1 码 = 37 的分数域里要 −111 码，超过 int8 的一半量程，所以「int8 分数 + int8 掩码 ADD + int8 softmax」这条路在数值上也不成立。⇒ 走 ADR 允许的 fp16 缝（DEQUANTIZE / MUL / ADD / SOFTMAX / QUANTIZE，5 层，1 个执行步）。

### 7.2 QUANTIZE（int8→int8 RequantizeLayer）**能编但算错**

最小复现（op 目录，同一批分数码、同一掩码）：

| 图 | 层表 | 板端 vs golden |
|---|---|---|
| ops/_w4a8_probe_seam（含 REQUANTIZE，比值 1） | Requantize → Dequantize → Add → SoftMax | **rel_l2 0.850** |
| ops/_w4a8_probe_seam_alt（含 REQUANTIZE，比值 0.93） | 同上 | **rel_l2 0.843** |
| ops/_w4a8_probe_seam_noreq（去掉 REQUANTIZE，把尺度直接写进输入张量） | Dequantize → Add → SoftMax | **rel_l2 9.25e-5 PASS** |
| ops/_w4a8_probe_dequant / _dequant_add | Dequantize（+Add 常数 −4096） | **逐位 0** |

现象：加了 QUANTIZE 之后 softmax 输出只剩 1/k（k=可见位置数）= **输入被压成常数**。比值 1 与 0.93 都坏 ⇒ 不是「乘子为 1 的饱和」，而是**这条 RequantizeLayer 在本栈上不按 TFLite 语义工作**（机制 **UNKNOWN**，没有反汇编 .dla）。⇒ 本票的图**不用** int8→int8 requant；1/√d 改在 fp16 段里乘。**给后续票的警告**：ops/*_int8 里凡是显式写 QUANTIZE 的链都要重新验一遍（仓库现有的 int8 链多是「同尺度直连」，不受影响）。

### 7.3 16 B 行规则在 int8 激活下仍然生效

| 图 | 内层行宽 | 编译 |
|---|---|---|
| ops/_w4a8_probe_pv_t8：P·V，p[14,8,8] | **8 B** | **EXIT=1**：Bridging is disallowed but gap is detected between Input and MDLA_5_0 |
| ops/w4a8_bmm_pv：P·V，p[14,16,16] | 16 B | EXIT=0、1 层、1 步、板端逐位 |
| 仓库旧证据 ops/bmm_pv_t8_int8 | 8 B | 同上 EXIT=1（本节是同一失败的独立复现） |

⇒ 与 fp16 一样，**输入侧最严**：小行宽形状（T=8 的 [H,8,8]、4864 以外的窄宽度）在 int8 下会 bridge，不是「int8 更省所以更宽松」。本票整层因此取 **T=16**。

---

## 8. eps 在 int8 下被折掉（ADR 0005 要求写进记录）

* int8 RMSNorm 的 exec plan 是 **MUL → MEAN → RSQRT → MUL → MUL 5 层，ADD(eps) 直接消失**（ops/w4a8_rmsnorm；与 docs/OP-rmsnorm.md §2.5 一致：round(eps/s) 在保住动态范围的任何尺度下都是 0）；
* 因此量化 golden 在 RMSNorm 里**用 eps = 0**（与图同语义），fp32 参考仍用真实的 **1e-6**；
* **定量的口径差**：本层 mean(x²) ≈ 2.3e-4，eps/(2·mean) ≈ 2.2e-3 ⇒ rms 变 0.11%，n1 = 1.05 时约 **0.25 码**，小于 1 码 ⇒ **在本层不可判别**（与 docs/OP-decoder-layer.md §7 的 fp16 结论同向，但这里的原因是量化分辨率而不是 fp16 尾数）；
* **「折进下游 FC 的 bias」没有做，且做不到**：eps 的效果是**每行一个乘性因子** 1/sqrt(mean+eps)，不是加性常数，静态 bias 表示不了。可做的只有「golden 同步用 eps=0」（本票做了）+ 记录 <1 码的影响。这一条是对任务书措辞的**订正**，不是偷懒。

---

## 9. 与票面 checklist 对照

| 票面要求 | 状态 | 证据 |
|---|---|---|
| int4 打包（2/byte、低 nibble 前）、per-channel scale、op version | ✅ | §3.2；probe-w4a8 的 nibble 反序对照 63/64 错；FC v9；w4a8_fc_proj 板端逐位 |
| 逐算子先过编译（FC/BMM/CONCAT/SPLIT/SOFTMAX+RMSNorm） | ✅（SOFTMAX 例外） | §4 五个组件；SOFTMAX 的 int8 版**编不过**（§7.1），改 fp16 缝 |
| 上板对拍：|Δ|≤1 码 + cosine ≥ 0.9999；逐层记录掉码 | ⚠️ **码级与质量都没过** | 组件 4/5 逐位、RMSNorm 2 码；整层 max 41 码 / cosine 0.9107；掉码从 probs 开始（§6） |
| 记录 16 B 行规则在 int8 下的约束 | ✅ | §7.3（T=8 EXIT=1 / T=16 PASS，两处独立复现） |
| 噪声底与阈值先登记后跑 | ✅ | §2（组件 int8_tol=1、整层 int8_tol=2 均在跑前写入 model.json；RMSNorm 的 FAIL 保留未改） |

**票面第 3 条如实标 ⚠️**：本票把它拆成「实现是否忠实」（设备 vs 量化模型：逐段都能对齐到 1–2 码，见 §6.2 的 q3/scores）与「量化档本身够不够准」（远不够，§6.3）。后者是**档位的性质**，不是本票能修的。

---

## 10. 适用范围 / 不适用

**适用**

1. **4W8A 能在 mdla5.0 上编成 1 个执行步**（61 层、7.2 MiB 静态 int4 权重、无 bridge），板端噪声底 0；
2. **int4 per-channel 权重 + int32 per-channel bias 的 FC 是逐位的**（896 个 per-channel 乘子，0/14336）；
3. **int8 RoPE（MUL/SUB/ADD + int8 cos/sin 常量）可用**：q3 段 max 1 码；
4. **int8 BMM（Q·Kᵀ adj_y、P·V）逐位**，形状 [14,16,64]/[14,16,16]；
5. **16 B 行规则在 int8 下依旧生效**（输入侧最严）；T=16 是这套几何的最小可用 T；
6. **int8 掩码 softmax 不可用**（§7.1/7.2）；混合精度缝必须显式 DEQUANTIZE（+1 层/缝）；
7. **码级判据只在「设备 vs 同一量化模型」之间有意义**；跨档一律用码级/余弦（ADR 0005）。

**不适用**

* 24 层 / KV cache / decode（下一票）；
* 真实权重档（本票用冻结随机夹具；真实 k_proj.bias 更大，S 更大，阈值要重算）；
* 性能：fc_proj 0.864 ms/inf 等只作记录，不作结论；
* 「4W8A 精度可用」——**本票证明的恰恰相反**（§6.3）。

---

## 11. 剩余疑问（UNKNOWN，不圆）

1. **RequantizeLayer（int8→int8 QUANTIZE）为什么算错**：只做到「比值 1 与 0.93 都坏、去掉就对」，没有反汇编 .dla，也没有试 --opt-aggressive；
2. **码级 87% 差能不能压到 0**：差的是每算子 ±1 码的舍入模型（40 个算子里我只对齐了 FC/BMM 的两次舍入契约）。仓库既有做法是「要素级乘子取 2 的幂，退化成纯移位」（docs/OP-silu.md §2.2），本票没做这轮重排；估算：把所有 elementwise 输出尺度改成 2 的幂倍数后，**除 probs 的近并列 row 外**可以逐位；
3. **近并列 row 的翻转率**：本层 probs 17/224 row 翻转（vs fp32）。这是夹具性质（bias 主导分数）还是真实 Qwen 权重也如此，未验证 —— 若真实权重也这样，它会成为 24 层串联的主要误差源（与 docs/OP-decoder-layer.md §14.7 第 4 条同一条）；
4. **group-wise int4**：TFLite schema 没有 group 尺度，只有 per-channel；厂商 shared_weights.bin 是 group≈256（docs/OP-shared-weights.md §4.4 的体积口径）。要在本栈上做到 5% 级的 4-bit 噪声，只能自造 group 量化（例如把 K 切成 8 段各自一个 per-channel FC），**未试**；
5. **scores 的 int8 表示**：1 码 = 37 是掉码起点。把 1/√d 折进 q 投影对 *int8* 无帮助（相对分辨率恒为 1/127），但对 fp16 有用（docs/OP-decoder-layer.md §3.1）；真正的修法是 **attention 走 4W16A**（ADR 的目标形态），**留给下一票**。

---

## 12. 复现

    export QWEN25_WEIGHTS_DIR='/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen'

    # 组件：build -> compile -> board（持锁，前后 healthcheck）
    wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-q04' && \
      /usr/bin/python3 tools/build_w4a8.py fc_proj --out ops/w4a8_fc_proj && \
      ./tools/compile.sh ops/w4a8_fc_proj/model.tflite ops/w4a8_fc_proj/model.dla"
    export NPU_LOCK='/d/deepseek_ workspace/mtk/.npu-board.lock'
    ./tools/devlock.sh run -- ./tools/e2e.sh ops/w4a8_fc_proj

    # 整层（T=16）
    wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-q04' && \
      /usr/bin/python3 tools/build_w4a8.py layer --out ops/decoder_layer_t16_w4a8 && \
      ./tools/compile.sh ops/decoder_layer_t16_w4a8/model.tflite ops/decoder_layer_t16_w4a8/model.dla"
    ./tools/devlock.sh run -- ./tools/e2e.sh ops/decoder_layer_t16_w4a8

    # 分段（掉码定位）
    wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-q04' && \
      /usr/bin/python3 tools/build_w4a8.py layer --upto attn --out ops/_w4a8_bisect/attn"

    # 设备 dump 对独立 fp32 链（质量口径）
    wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-q04' && \
      /usr/bin/python3 tools/build_w4a8.py layer --diag --out ops/decoder_layer_t16_w4a8"

**可复现性**：同一 builder 重建 ops/w4a8_fc_proj/model.tflite 与 golden/expected.bin **逐字节相同**（sha256 37901fcf… / c5991b8e…，实测两次）；.dla 不可复现（仓库既有约定），只作产物标识。
