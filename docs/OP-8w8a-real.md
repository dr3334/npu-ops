# OP-8w8a-real —— 8W8A 用**真实 Qwen2.5-0.5B 权重**在设备上复核：夹具不悲观，真权重更差

> 票：`.scratch/quant-runtime/issues/04e-8w8a-real.md`；spec `.scratch/quant-runtime/spec.md`；
> ADR `docs/adr/0005-quantization-tier-target.md`（三次实测订正）；上位票证据
> `docs/OP-8w8a-layer.md`（04c）、`docs/OP-groupwise-scale.md`（04b）、`docs/OP-w4a8-layer.md`（04）。
> 分支 `impl/q04e-8w8a-real`，worktree `wt-q04e`。编译器 ncc-tflite **7.3.15**
> （`--arch=mdla5.0 --opt=3 --disallow-bridge --num-mdla=2 --l1-size-kb=3072`）；
> 设备 mt6897 / Android 16（`0123456789ABCDEF`），板端全部持 `/d/deepseek_ workspace/mtk/.npu-board.lock`
> （每次板端块前后各一次 `healthcheck --quick`，全部 PASS）。
> 真权重 `/mnt/d/ai/Qwen2.5-0.5B-Instruct/model.safetensors`（bf16，**只读**，不入库、不复制）；
> 夹具 `weights/qwen25_frozen`（票 02，本 worktree 未物化，用 `QWEN25_WEIGHTS_DIR` 指向主检出的只读副本）。
> 交付：`tools/build_w4a8.py` 新增 `--weights {fixture,real}` / `--input {same,fixture,real}` /
> `--stages` / `--device-report`（**图、缝、T、量化规则一字未改**），`ops/decoder_layer_t16_w8a8_real`、
> `ops/_w8a8_real_bisect/{scores,probs,wreal_ifix,wfix_ireal}`、本文。公共工具一行未改。

---

## 0. TL;DR

**结论：8W8A 在真权重下达不到判据，而且比夹具更差（整层 0.8801 vs 0.9649）。ADR 第三次订正里
「夹具比真权重苛刻、真权重 0.9999157」那句话是**权重单独**的口径（激活是 float64），不能推广到整档。
所以主档不能锁 8W8A。**

| 量（同一 harness、同一会话、同一判据） | 夹具权重 | **真权重** |
|---|---|---|
| 权重 | `weights/qwen25_frozen`（票 02） | `Qwen2.5-0.5B-Instruct` layer0（bf16） |
| 输入 | 夹具 embed 行 | 真 embed 行（**同一批 token id**） |
| 每矩阵 int8 权重量化误差 | 0.772–0.873% | **0.828–1.140%** |
| 只量化权重（fp64 链，激活精确） | 0.9998451 | **0.9999157**（与 04b 离线值逐位相同） |
| 整层量化 golden vs fp64 | 0.9673771 | **0.8737825** |
| **整层 device vs fp64** | 0.9649298 | **0.8801239** |
| 设备忠实度 `abs(Δcos(dev,golden))` | 0.0024 PASS | **0.0063 FAIL**（>0.003） |
| 码级 device vs 量化 golden | 12660/14336、max 36 | **13328/14336、max 120** |
| `scores` device vs fp64（分段图） | 0.9999119 | **0.9996434** |
| **`probs` device vs fp64（分段图）** | 0.9580252 | **0.7676888**（golden 0.7046499） |
| `out` device vs fp64（整层） | 0.9649298 | **0.8801239** |
| `.dla` | 15,089,161 B | 15,089,171 B（+10 B 是元数据，权重都是 14,909,440 B） |
| Static / 层数 / 执行步 | 14M / 61 / 1 | 14M / 61 / 1 |
| 板端单层（`-c 30`，同会话交替） | 1.87–2.28 ms/inf | 1.71–2.20 ms/inf |
| 噪声底（同 `.dla` 同输入两轮） | **0 码** | **0 码** |
| 判据（ADR 0005：device vs fp64 ≥ 0.9999） | FAIL（差 0.035） | **FAIL（差 0.120）** |

**一句话**：夹具不是悲观估计。真权重把 layer0 的激活动态范围抬到约 3 倍（`|n1|max` 1.05 → 2.92），
per-tensor int8 激活的相对误差 2.39% → **5.95%**，于是 8W8A 的整层 cosine 从 0.965 掉到 **0.880**，
`probs` 段从 0.958 掉到 **0.768**。**权重从来不是瓶颈（int8 per-channel 0.83–1.14%、weights-only 0.999916）；
瓶颈是 int8 激活，而真权重的激活更难量化。**

---

## 1. 票面要回答什么

1. 真权重 8W8A 单层（T=16 seam）设备 vs 独立 fp64 链：cosine / rel_l2 / 分段（`scores`/`probs`/`out`）；
2. **同一 harness、同一量化规则**下夹具权重的对照数字（票面说这是最重要产出）；
3. 每矩阵权重 int8 量化误差（对照夹具的 0.772–0.873%）；
4. `.dla` 体积、Static、板端单层时间；以及「真权重下 `probs` 是否还掉到 0.96」的直接对照。

**不做**：24 层 / KV cache / decode、4W16A（04d 的分支，本 worktree 未合，见 §10）、
分组 scale（04b 已结，不是答案）、厂商 PTQ（hessian/AdaRound 不在本栈内，我们只要一个可控的 per-channel 参考）。

---

## 2. 事前登记（都在第一次板端运行之前写进 `model.json`）

| 项 | 登记值 | 依据 |
|---|---|---|
| 整层 `int8_tol` | **2** | 与 04/04c 同值同理由：图里含同一条 int8 RMSNorm 链（RSQRT 定点表预算 4 码）+ 同一条 fp16↔int8 缝 |
| 质量判据 | `cosine_min=0.9999`（device vs 独立 fp64 链） | ADR 0005 原文 |
| 设备忠实判据 | `abs(cos(dev) − cos(golden)) ≤ 0.003` | ADR 0005 2026-11-13 订正第 1 条 |
| seam 段（`--upto probs`，fp16 输出） | `atol=1e-3`、`rtol=1e-2` | 输出在 fp16 一侧：1 ULP@1.0 = 9.77e-4；golden 在同一个位置舍入，只有 fp16 表达式顺序可能不同 |
| 噪声底 | 实测，不预设 | 同 `.dla` 同输入两轮逐字节比较 |

**没有事后调阈值**：上面每个值都是构建期写进 `model.json` 的；实测出来的两个 FAIL（真权重整层 0.8801、
码级 13328/14336）原样保留。

---

## 3. 做法：只换「权重从哪来」和「输入从哪来」，图不动

`tools/build_w4a8.py` 加了两个开关和两个报告：

| 部位 | 夹具（04c） | 真权重（本票） |
|---|---|---|
| 权重来源 | `qwen25_weights.get()` | `model.safetensors` 的 `model.layers.0.*`（bf16 → float32；30 行只读 reader，无 torch/safetensors 包） |
| 输入行 | 夹具 `embed.weight` 的 token 行 | 真 `model.embed_tokens.weight` 的**同一批** token id（`TOKEN_IDS` + 算术补位，共 16 行） |
| 权重量化 | per-channel 对称 `s[n]=max(abs W[n,:])/127` | **同一条规则** |
| 激活 | int8 per-tensor `s=max(abs v)/127` | 同 |
| bias / requant | int32 per-channel / 两次整数舍入（`tools/quant_contract.py`） | 同 |
| 图 / 缝 / T / op version | — | **一字未改**（tflite 都是 15,309,888 B，只差常量字节） |

`--input` 是给 §6 的消融用的（默认 `same`＝跟权重走）。`--stages` 出一张 device-free 分段表，
`--device-report` 把 device dump 与独立 fp64 链、量化 golden 三方并排（含 fp16 段）。

---

## 4. 先验一通：真权重通道复现了 04b 的 0.9999157

这是本票的**有效性检查**——如果真权重读法/输入选取与 04b 不同，下面所有数字都不可比：

    weights-only int4 (真权重): cosine(out)=0.9733031  rel_l2=23.1595%
    weights-only int8 (真权重): cosine(out)=0.9999157  rel_l2= 1.3003%   ← ADR 第三次订正那个数
    weights-only int8 (夹具)  : cosine(out)=0.9998451  rel_l2= 1.7604%   ← 04c 的 §7.1 那个数

两个数都在本会话逐位复现（0.9999157 / 0.9998451），所以：**「真权重离线 0.9999157」是
「激活精确（float64）、只量化权重」的一档**。把激活也降到 int8（也就是 8W8A 这一档）之后，
真权重是 0.8738，不是 0.9999。

---

## 5. 真权重 8W8A：设备 vs fp64，分段表

### 5.1 整层（`ops/decoder_layer_t16_w8a8_real`）

    编译 : EXIT=0、61 个原生层、1 个执行步、无 bridge、Static 14M、.dla 15,089,171 B
    板端 : exact NO  mismatched=13328/14336  int8_tol=2  max_abs_diff=120  FAIL
    噪声 : 0 码（同 .dla 同输入两轮，out_device.bin 逐字节相同）

| 量 | device vs fp64 | golden vs fp64 |
|---|---|---|
| cosine | **0.8801239** | 0.8737825 |
| rel_l2 | 191.24% | 99.36% |
| max_abs | 3.0903（= 129.2 码） | 3.0178 |

|Δcos(device, golden)| = **0.0063** > 0.003 ⇒ ADR 那条「设备忠实」判据在真权重下**不成立**（见 §8）。

### 5.2 分段表（三段都上了设备：`--upto scores` / `--upto probs` / 整层）

| stage | 夹具 device | **真权重 device** | 夹具 golden | **真权重 golden** |
|---|---|---|---|---|
| `scores`（BMM，无权重） | 0.9999119 | **0.9996434** | 0.9999110 | 0.9993445 |
| `probs`（缝 + 掩码 softmax，无权重） | 0.9580252 | **0.7676888** | 0.9606418 | 0.7046499 |
| `out`（整层） | 0.9649298 | **0.8801239** | 0.9673771 | 0.8737825 |

**「真权重下 probs 是否还掉到 0.96」的直接回答：不掉到 0.96，是掉到 0.77（device）/ 0.70（golden）。**
方向与 ADR 第三次订正的猜测相反。

### 5.3 device-free 的十段（`--stages`，golden vs fp64）

| stage | 夹具 | **真权重** | 备注 |
|---|---|---|---|
| n1（RMSNorm#1） | 0.9995702 | 0.9979915 | 输入 int8 per-tensor：2.39% → **5.95%** |
| q3（FC+RoPE） | 0.9999410 | **0.9749383** | 真权重的 q/k/v 输出动态范围大（max 25.6→80.6），int8 RoPE 乘损 |
| scores | 0.9999110 | 0.9993445 | 1 码 = 37.1 → **98.5** |
| probs | 0.9606418 | **0.7046499** | softmax 近并列 row 翻转 **25/224 → 171/224** |
| attn（P·V） | 0.9651807 | 0.7449579 | |
| r1（o_proj+残差） | 0.9836251 | 0.9343622 | 残差把 attn 的坑填了一部分 |
| n2 | 0.9835984 | 0.9189672 | |
| mlp_y | 0.9669002 | **0.8165787** | 真权重 MLP 激活的 outlier 最重（\|mlp_y\|max 1.19→6.36） |
| down | 0.9669223 | 0.8781212 | |
| out | 0.9673771 | **0.8737825** | |

**读法**：夹具下掉码的唯一起点是 `probs`；真权重下**有两个起点**——`q3`（0.9999→0.975）与
`probs`（0.9999→0.705），外加 `mlp_y`（0.967→0.817）。这三处全部是 int8 激活侧，权重侧
（weights-only 0.999916）一点都没参与。

---

## 6. 消融：是权重把质量拉低的，不是输入

`--weights` 与 `--input` 正交，四种组合都建了图、都上了设备（同 harness、同会话）：

| 权重 | 输入 | golden vs fp64 | **device vs fp64** | \|Δcos\| | 整层码级 |
|---|---|---|---|---|---|
| 夹具 | 夹具 | 0.9673771 | **0.9649298** | 0.0024 | 12660/14336、max 36 |
| 夹具 | 真 | 0.9758986 | **0.9663124** | 0.0096 | 13307/14336、max 35 |
| 真 | 夹具 | 0.8399401 | **0.8919894** | 0.0520 | 11960/14336、max 126 |
| 真 | 真 | 0.8737825 | **0.8801239** | 0.0063 | 13328/14336、max 120 |

**结论：权重是主因（夹具→真权重 在同一输入下把 cosine 打掉 0.10–0.13），输入不是**
（真输入反而略好：0.9674→0.9759 golden）。这与「夹具的 embed 行输入造成极端分数分布」的猜想相反：
真权重的 embed 行 + 真权重的 layer0 权重一起，把激活 outlier 推得更高：

| 权重 / 输入 | `max abs x` | `max abs n1` | n1 int8 相对误差 | `max abs scores` |
|---|---|---|---|
| 夹具 / 夹具 | 0.070 | 1.051 | 2.39% | 4,710 |
| 夹具 / 真 | 0.088 | 0.947 | 2.16% | 4,749 |
| 真 / 夹具 | 0.070 | 1.344 | 3.21% | 9,309 |
| 真 / 真 | 0.088 | **2.916** | **5.95%** | **12,515** |

这一列（`|n1|max`）就是整层 cosine 的解释变量：per-tensor 的 int8 尺度由**最极端的一个值**决定，
outlier 每翻一倍，绝大多数元素的码数就掉一半。

---

## 7. 每矩阵权重 int8 量化误差（票面第 3 项）

`--weights-report`，**同一个 per-channel 对称方案**，只换数据：

| 矩阵 | 夹具 int4 | 夹具 **int8** | 真权重 int4 | 真权重 **int8** |
|---|---|---|---|---|
| Wq | 14.2036% | **0.7833%** | 18.6542% | **1.0325%** |
| Wk | 14.0255% | **0.7720%** | 14.9875% | **0.8278%** |
| Wv | 14.4058% | **0.7932%** | 17.9465% | **0.9880%** |
| Wo | 14.1047% | **0.7768%** | 20.4656% | **1.1311%** |
| Wg | 14.1210% | **0.7781%** | 15.3322% | **0.8465%** |
| Wu | 14.0784% | **0.7764%** | 15.4860% | **0.8546%** |
| Wd | 15.8318% | **0.8727%** | 20.4948% | **1.1396%** |
| 激活 n1（per-tensor int8，参照） | — | 2.3944% | — | **5.9531%** |

* 夹具的 7 个数与 04c §6 **逐位相同**（0.7833/0.7720/0.7932/0.7768/0.7781/0.7764/0.8727），证明参数化没动 4W8A/8W8A 任一档的量化数学；
* 真权重的 int8 误差是夹具的 **1.07–1.30 倍**（0.828–1.140%），int4 是 **1.07–1.56 倍**（14.99–20.49%）：
  真实权重的尾部比高斯夹具重，但**仍然是 1% 量级**，撑不起整层 0.12 的 cosine 缺口；
* **激活的 5.95% 现在是最大单项**，是夹具（2.39%）的 2.5 倍——这就是 §5/§6 的归因。

---

## 8. 设备忠实度：夹具的 0.003 界在真权重下不够用

| 图 | `abs(cos dev − cos golden)` | ADR 0.003 | 方向 |
|---|---|---|---|
| 夹具整层 | 0.0024 | PASS | device 略差于 golden |
| 真权重整层 | **0.0063** | **FAIL** | device **比 golden 更接近 fp64**（0.8801 > 0.8738） |
| 真权重 / 夹具输入 | **0.0520** | **FAIL** | 同上（0.8920 > 0.8399） |
| 夹具 / 真输入 | 0.0096 | FAIL | device 略差（0.9663 < 0.9759） |
| 真权重 `probs` 段 | **0.0630** | FAIL | device 好于 golden（0.7677 > 0.7046） |

两条事实：

1. **噪声底是 0 码**，所以这些码差都是确定性的模型差，不是抖动；
2. 真权重的动态范围一大，**我的量化 golden 在 elementwise int8 段偏悲观**（device 反而更接近 fp64）：
   golden 把每个 elementwise 算子的输出都按 per-tensor int8 尺度重新舍入一次，设备用的是定点累加，
   在 `max|out|` 3.0（1 码 = 2.39e-2，夹具只有 3.99e-3）这种量级下两者的差别被放大。
   ⇒ **在真权重上判断「设备有多准」应读 device 列，不能读 golden 列**；ADR 那条 0.003 的界是在夹具量级上校的，
   真权重下需要重定（建议：以 `1 − cos(golden)` 为尺度，或直接以 device 列为准）。

本票**没有**据此调任何阈值，两个 FAIL 都原样留着。

---

## 9. 体积 / Static / 时间（票面第 4 项）

| 量 | 夹具 | 真权重 |
|---|---|---|
| `model.tflite` | 15,309,888 B | 15,309,888 B（只差常量内容） |
| `model.dla` | 15,089,161 B | 15,089,171 B |
| 差 | **+10 B**（元数据；int8 权重本体都是 14,909,440 B = 参数量 × 1 B） | |
| Static（编译日志） | 14M | 14M |
| 原生层 / 执行步 | 61 / 1 | 61 / 1 |
| 板端单层 `-c 30`（同会话交替 3+4 对） | 2.2777 / 2.04003 / 2.16653 / 2.2474 / 1.86963 / 2.16153 / 1.95957 ms | 1.8345 / 1.77053 / 1.71013 / 1.91337 / 1.99927 / 2.1991 / 2.08147 ms |
| 分段图 `.dla` | `scores` 977,459 B、`probs` 988,083 B | `scores` 977,469 B、`probs` 988,093 B |

**时间的读法**：两个源是**同一张图的同一份字节**（只差常量值），所以任何差别都不是档位差。
14 个 `-c 30` 采样全部落在 **1.71–2.28 ms/inf**，交替 A/B 的 7 对里真权重有时快有时慢（第一块 3 对真权重偏快，
第二块 4 对又反过来）⇒ **差异是会话/热态漂移，不是数据依赖**，与 04c §8 的结论一致（8W8A ≈ 2.17 ms/inf）。

---

## 10. 与票面 checklist 对照

| 票面要求 | 状态 | 证据 |
|---|---|---|
| 取 layer0 的 q/k/v/o/gate/up/down，per-channel max-abs 量化到 int8，记录 scale 规则 | ✅ | §3（对称、per-output-channel：`s[n]=max|W[n,:]|/127`；bias int32 per-channel） |
| 输入用真实 token 的 embed 行，记录选择 | ✅ | §3（`model.embed_tokens.weight` 的同一批 16 个 token id，与夹具同 id） |
| 设备跑 8W8A 单层（T=16 seam），与独立 fp64 链比 cosine/rel_l2，分段 scores→probs→out | ✅ | §5（三段都上了设备，三个量都给了） |
| 同一 harness 再跑夹具一版，做成对照表 | ✅ | §0 总表、§5.2 分段、§5.3 十段 |
| 结论：真权重下是否 ≥0.9999 | ✅ **否**：0.8801（差 0.120） | §0/§5 |
| 每矩阵 int8 量化误差、`.dla` 体积、板端时间 | ✅ | §7、§9 |
| 额外（票面没要求，但决定下一步）：真权重 vs 真输入的消融 | ✅ | §6 |

---

## 11. 结论与建议

1. **主档不能锁 8W8A。** 真权重下整层 device cosine **0.8801**，比夹具的 0.9649 还差 0.085；
   两个分量（`scores` 0.99964、`probs` 0.7677）里 `probs` 是主坑，且 `q3`、`mlp_y` 也各加了一层。
2. **ADR 第三次订正的推断要改口径**：那句「真权重 0.9999157、夹具 0.9649，所以夹具更苛刻」比较的是
   **只量化权重的真权重** 与 **整条 int8 链的夹具**。同口径下真权重是 0.8738（golden）/ 0.8801（device），
   夹具才是偏乐观的那一个。已在 ADR 追加第四次订正。
3. **瓶颈仍然是激活位宽**：int8 per-channel 权重 0.83–1.14%，weights-only 0.999916；而 per-tensor int8 激活
   在真权重上 5.95%。⇒ 8W8A 提高权重位宽没有意义，**只有 16 位激活才可能过 0.9999**，这与 ADR 定的
   「主档 = 4W16A」方向一致，也说明 04d 那条线是唯一剩下的路。
4. **给 04d 的提醒（重要）**：真权重把 `scores` 推到 1.25e4（夹具 4.7e3）。当前的 seam 已经是 fp16，
   而 fp16 在 `scores/8` ≈ 1.5e3 处 1 ULP ≈ **0.98**——softmax 的输入被量化到 1.0 的台阶上（这正是
   `probs` 0.77 的一部分来源，golden 里就能看到，不需要设备）。所以 16 位激活如果只是「int16 表示」，
   仍可能被分数段的绝对 ULP 卡住；**04d 的 4W16A 必须在真权重上复测**，夹具数字不足以定档。
5. **夹具继续当回归基线**（它对设备忠实度、码级、体积、时间都给得干净），但**质量结论一律以真权重为准**——
   这一条建议写进 spec 的量化判据。

---

## 12. UNKNOWN（不圆）

1. **真权重下 device 比 golden 更准**（0.8920 vs 0.8399 最有代表性，差 0.052）。我只对齐了 FC/BMM 的两次
   舍入契约，elementwise int8（MUL/ADD/RSQRT/LOGISTIC）仍是「每步 round 到 per-tensor int8」的模型；
   真权重动态范围大时它偏悲观。因此**设备真实天花板可能略高于 0.88**，但没有做逐段设备对拍，
   无法给出上界。这与 04c §12.3（码级 88% 的差压不到 0）是同一个未解项。
2. **输入只用了 16 条 token 的 embed 行**（票面允许）。layer0 的真实输入确实就是 embedding 行，
   但真实 prompt 的 hidden（多 token、连续文本）分布可能不同，可能更容易或更难。未试。
3. **时间没有时钟/热态控制**，只做了同会话交替；1.71–2.28 的区间里没有可归因于数据或档位的成分。
4. **真权重的 int4 数字**（14.99–20.49%）只作为参照列出；4-bit 在真权重上是否也同样不可用，属 04/04b 的延伸，
   本票未在设备上复测。
5. **04d（4W16A）的数字不在本 worktree**：`--a-bits` 尚未合入，我没法做「真权重 4W16A」的对照，
   所以「8W8A vs 4W16A 二选一」我只给出 8W8A 这一侧的判决（不可用）。

---

## 13. 复现

```bash
export QWEN25_WEIGHTS_DIR="/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen"
export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
WT="/mnt/d/deepseek_ workspace/mtk/wt-q04e"

# 1) 真权重整层（T=16，8W8A）与夹具对照：build 在 WSL 里，编译放锁外
wsl.exe -e bash -lc "cd '$WT' && /usr/bin/python3 tools/build_w4a8.py layer --w-bits 8 --weights real --out ops/decoder_layer_t16_w8a8_real"
wsl.exe -e bash -lc "cd '$WT' && /usr/bin/python3 tools/build_w4a8.py layer --w-bits 8 --weights fixture --out ops/decoder_layer_t16_w8a8"
wsl.exe -e bash -lc "cd '$WT' && ./tools/compile.sh ops/decoder_layer_t16_w8a8_real/model.tflite ops/decoder_layer_t16_w8a8_real/model.dla"
wsl.exe -e bash -lc "cd '$WT' && ./tools/compile.sh ops/decoder_layer_t16_w8a8/model.tflite ops/decoder_layer_t16_w8a8/model.dla"
./tools/devlock.sh run -- ./tools/e2e.sh ops/decoder_layer_t16_w8a8_real
./tools/devlock.sh run -- ./tools/e2e.sh ops/decoder_layer_t16_w8a8

# 2) 分段（scores / probs）与消融（权重 × 输入四组合）
wsl.exe -e bash -lc "cd '$WT' && /usr/bin/python3 tools/build_w4a8.py layer --w-bits 8 --weights real --upto scores --out ops/_w8a8_real_bisect/scores"
wsl.exe -e bash -lc "cd '$WT' && /usr/bin/python3 tools/build_w4a8.py layer --w-bits 8 --weights real --upto probs  --out ops/_w8a8_real_bisect/probs"
wsl.exe -e bash -lc "cd '$WT' && /usr/bin/python3 tools/build_w4a8.py layer --w-bits 8 --weights real    --input fixture --out ops/_w8a8_real_bisect/wreal_ifix"
wsl.exe -e bash -lc "cd '$WT' && /usr/bin/python3 tools/build_w4a8.py layer --w-bits 8 --weights fixture --input real    --out ops/_w8a8_real_bisect/wfix_ireal"
# 每张图按 1) 的方式 compile + e2e

# 3) 数：device 与 fp64 三方并排 / 分段表 / 每矩阵权重误差 / 权重单独
wsl.exe -e bash -lc "cd '$WT' && /usr/bin/python3 tools/build_w4a8.py --weights real    --device-report ops/decoder_layer_t16_w8a8_real ops/_w8a8_real_bisect/scores ops/_w8a8_real_bisect/probs"
wsl.exe -e bash -lc "cd '$WT' && /usr/bin/python3 tools/build_w4a8.py --weights fixture --device-report ops/decoder_layer_t16_w8a8 ops/_w8a8_bisect/scores ops/_w8a8_bisect/probs"
wsl.exe -e bash -lc "cd '$WT' && /usr/bin/python3 tools/build_w4a8.py --stages --weights real"
wsl.exe -e bash -lc "cd '$WT' && /usr/bin/python3 tools/build_w4a8.py --weights-report --weights real"
wsl.exe -e bash -lc "cd '$WT' && /usr/bin/python3 tools/build_w4a8.py --wq-chain --weights real"

# 4) 噪声底（同 .dla 同输入两轮）与计时（同会话交替 A/B，-c 30）
./tools/devlock.sh run -- bash -c 'cp ops/decoder_layer_t16_w8a8_real/out_device.bin /tmp/r1.bin; ./tools/e2e.sh ops/decoder_layer_t16_w8a8_real; cmp /tmp/r1.bin ops/decoder_layer_t16_w8a8_real/out_device.bin && echo NOISE=0'
adb shell "cd /data/local/tmp/npuops && LD_LIBRARY_PATH=\$PWD/lib ./bin/neuronrt -m hw -a dla/decoder_layer_t16_w8a8_real.dla -i dla/decoder_layer_t16_w8a8_real.in.bin -o dla/o.bin -c 30"
```

**可复现性**：`--weights real --out ...` 重建的 `model.tflite` 与 `golden/*.bin` 逐字节稳定（同 04/04c）；
`.dla` 不可复现（仓库既有约定），只作产物标识。
---

## 订正（2026-11-13，票 04f 在跑 8W16A 时发现，**本票的真权重数字暂不可引用**）

`tools/build_w4a8.py` 的 RoPE 段有一个真 bug：`q_concat` 的 golden 把 `lo`/`hi` 两半按**同一个 scale** 拼接，而两半各自 `pick_scale`。真权重下 `|hi|max=80.59`、`|lo|max=51.13`（比值 1.576），拼出来的 `hi` 半被低估 1.576 倍；夹具下这个比值只有 1.0003，所以**夹具的 04c/04d 数字没被污染，本票（真权重）的 0.8801 被污染了**。

证据（真权重 8W16A，`--upto q3` 上板）：
- 现行 golden `cos(q3, fp64) = 0.9746969`；两半各用自己的 scale 则是 **0.9999999**；
- 设备语义实测 = 「把 hi requant 进 lo 的 scale 再按 int16 截断」，与模型逐码差 ≤4 码（cosine 0.9933635 vs 0.9933636）⇒ **设备按 TFLite 语义是对的，是 golden 没 requant**；但声明的输出 scale 装不下 hi 的 outlier，0.22% 元素被截断。

修掉后（离线）q3 段：8W16A **0.9999999**、4W16A 0.9999847、8W8A 0.9996304、4W8A 0.9996204。

因此：**本票「真权重比夹具更差（0.88 vs 0.96）」的结论很可能是这条 bug 造成的，不是档案性结论**。整层数字由票 04f 修复后重测；在那之前，请不要引用本票的 0.8801 作为「真权重下 8W8A 的真实质量」，也不要用它去定档。

仍然成立的部分：8W8A 的每矩阵权重误差、`.dla` 体积、板端时间、噪声底 0、以及「真权重的 |n1|max 确实比夹具大（1.05→2.92）」这条观察。
**修复后回填（2026-11-13，票 04f）**：本票的 8W8A 真权重整层读数是 **0.8801**（RoPE scale bug 未修时）。修复后同一格是 **0.8971**——方向不变（真权重仍比夹具的 0.9649/0.9685 差约 0.068），但绝对值变了。正文数字**保留原值**（不改写历史），以本条为准。
