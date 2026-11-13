# OP-groupwise —— 分组 scale（group-wise）能否在本栈表达

> 票 `.scratch/quant-runtime/issues/04b-groupwise-scale-feasibility.md`；spec `.scratch/quant-runtime/spec.md`；
> ADR `docs/adr/0005-quantization-tier-target.md`（含 2026-11-13 实测订正）；上位票证据 `docs/OP-w4a8-layer.md`。
> 分支 `impl/q04b-groupwise`，worktree `wt-q04b`。编译器 ncc-tflite **7.3.15**（`--arch=mdla5.0 --opt=3 --disallow-bridge`），
> 设备 mt6897 / Android 16，板端全部持 `/d/deepseek_ workspace/mtk/.npu-board.lock`。
> 交付：`tools/build_q04b.py`（曲线 / 分组 FC / 分组整层 / 两个 schema 探针）、
> `ops/q04b_fc_g{2,4,8}/`、`ops/q04b_layer_g{2,4,8}/`、`ops/q04b_layer_real_g{1,4,8}/`、
> `ops/q04b_probe_{scale2n,axis1}/`、`ops/q04b_errcurve/`（三份曲线原文 + json）、本文。

---

## 0. TL;DR —— 三句话

1. **属性级不可表达**：TFLite 的 `QuantizationParameters` 只有一个 `quantized_dimension`，ncc 7.3.15 也没有任何
   分组/blockwise 开关。两个探针把这条路封死：给 896 宽的 int4 权重声明 **1792 个 scale** → 加载期
   `Fail to process the given model`；把 FC 权重的 `quantized_dimension` 改成 1（按**输入**通道）→ 同样在加载期被拒。
2. **分解级可表达，且设备逐位**：把 K 切成 G 段、每段一个 per-channel int4 FC、再相加，等价于
   「每个 (输出通道, K 段) 一个 scale」，即 group size = K/G 的分组量化。`G=2/4/8` 全部 **EXIT=0 / 1 个执行步**，
   板端 **0/14336 逐位**（`int8_tol=1` 跑前登记）。代价：每矩阵 `2G−1` 层，整层实测 **61 → 83 / 119 / 191**
   （1.36× / 1.95× / 3.13×）。
3. **但它救不了 4-bit**：(a) 权重误差确实降（真实 Qwen2.5-0.5B 权重，per-channel **17.6%** → G=8 **13.2%** →
   G=128（group 7）**7.6%**）；(b) **int8 输出/累加舍入把增益吃掉了** —— 每个部分和都要落回 int8，
   整层 4W8A 的 cosine 在 G=1/4/8 上是 **0.866 / 0.820 / 0.856**（真实权重）与 **0.913 / 0.901 / 0.921**（夹具），
   **没有一个方向能靠近 ADR 的 0.9999**；(c) 厂商 bin 的体积口径反推 group ≈ **256**（不是 ADR 猜的 32–64），
   对应我们的 G≈3.5、权重误差 ~15%，**与 per-channel 同量级** —— 所以「分组 scale」不是 14% 与厂商 4-bit 差距的来源。

**给票 05 的结论**：分组 scale **不能**用作 4-bit 的救命方案（要 4.5× 层数才拿到 7.6%，且层指标不动）。
若 04c 的 8W8A 达到判据（本票离线已见 int8 per-channel 权重、同一条链、真实权重：层 cosine **0.9999157**），
**主档应取 8W8A**，4-bit 只在「能证明不掉质量」的位置上做局部使用。

---

## 1. 票面要回答什么

| 票面条目 | 结论 | 证据 |
|---|---|---|
| 查 TFLite schema / ncc：`quantized_dimension` 之外有无分组量化的表达 | **没有**（属性级）；但可用算子**分解**等价表达 | §2 两个探针 + `--help` / `--show-mtkext-ops` |
| 厂商 MTKEXT 的 `--import_qparams_config_file` 能否导入我们自造的图 | **不能**（它是 MTK 转换器*输出*侧工具，输入不是 TFLite；qparams 网格仍是 per-channel） | §2.3 |
| ① K 切 G 段、每段一次 FC 再求和 | **可行、设备逐位**；给出「误差 vs 层数」曲线 | §3、§4、§5 |
| ② N 切块、每块独立 per-channel | 对 K 方向的动态范围**没有用**（每块仍是整行一个 scale），未立项 | §5.4 |
| 用厂商 bin 体积反推分组大小，作为目标误差参照 | group ≈ **256**（scale+zp 则 512） | §6 |
| 结论 + 最小复现（编译 + 板端各一例） | 见 §0、§9 | `ops/q04b_fc_g4`（编译+板端）、`ops/q04b_probe_scale2n`（编译失败） |

**不做**：24 层、KV cache、decode、混合精度整网；也不做 8W8A 的板端结论（那是 04c）。

---

## 2. 属性级表达性：三条路，全部封死

### 2.1 探针 A：给 896 宽的权重声明 1792 个 scale（`ops/q04b_probe_scale2n`）

最便宜的分组尝试：张量保持 2-D `[896,896]`，只把 `scale` 数组给成 `2N`（想表达「每个通道两个 scale」）。

    # ops/q04b_probe_scale2n/model.compile.log
    ERROR: Fail to process the given model          # 加载/解析期，不是后端期
    EXIT=1  SIZE=0

### 2.2 探针 B：把 FC 权重的 `quantized_dimension` 改成 1（`ops/q04b_probe_axis1`）

schema 上唯一另一个轴。若能编，至少能得到「每列一个 scale」（跨行共享，不是分组），也能据此判定还剩几个自由度。

    # ops/q04b_probe_axis1/model.compile.log
    ERROR: Fail to process the given model
    EXIT=1  SIZE=0

⇒ **FC 的 int4 权重只接受 `quantized_dimension=0` + `len(scale)==N`**。分组 scale 在属性层无处安放。

### 2.3 工具侧：ncc 与 MTK 转换器

| 检查项 | 结果 |
|---|---|
| `ncc-tflite --help`（7.3.15 全量） | **没有任何** group / blockwise / qparams 开关；只有 `--opt-*`、`--rewrite-pattern`、`--extract-static-data` 这类 |
| `ncc-tflite --show-mtkext-ops` | MTKEXT_* 只是同一批算子的自定义包装（`MTKEXT_FULLY_CONNECTED` 等 60 余个），**没有**分组量化算子 |
| `--import_qparams_config_file`（`skills/mtk-llm-pipeline/references/02-quantization-matrix.md:19`） | 属于 **MTK 转换器**（本机源码 `.tmp/ext/conv/mtk_converter/python/converters/base_converter.py:702`）。它是**产生** TFLite 的一端（`convert_to_tflite`），不接受 TFLite 作为输入 ⇒ **无法把 qparams 导进我们自造的图** |
| 厂商的质量旋钮在哪 | 同一模块的 `transformer*.so` 里有 `use_hessian_opt`、`hessian_opt_block_size`、AdaRound、Bias Correction（`strings` 实测）⇒ 他们 4-bit 好，靠的是**转换器内部的 hessian/AdaRound 舍入**，不是 TFLite 里多出来的 scale 维度 |
| 缺的环节 | ① 转换器的输入前端（torch/onnx 之类的浮点图），我们没有；② 即便进了他们的前端，qparam 网格仍是 per-channel —— 分组 scale 只存在于**他们自己烘进权重/图的那份产物**里 |

⇒ 属性级「不能」；要分组 scale 只能**改变图的形状**（§3）。

---

## 3. 分解：K 切 G 段 = 每 (n, 段) 一个 scale

`FULLY_CONNECTED` 的权重 `[N,K]` 只沿 dim0 有 per-channel scale。把 K 切成 G 段后，**每段各自一个 FC**：

    x[T,K] --SPLIT axis=1--> x_g[T,K/G]                    (G 份，整层里按输入张量复用一次)
          每段: x_g · W[:, g*KG:(g+1)*KG]^T + bias_g   -> y_g[T,N] int8（自己的 per-channel scale）
          y = ((y_0 + y_1) + y_2) + ...                  (G-1 个 ADD)

每个 `y_g`（即每个 FC）都有**自己的** per-channel int4 scale ⇒ 全局就是 `scale[n, g]`，group size = K/G。
这就是 GPTQ/厂商那套「分组 scale」在 TFLite 里的等价形态。

实现（`tools/build_q04b.py`）：`group_fc_tables()` 生成 `[G,N,K/G]` 的 int4 码与 `[G,N]` 的 scale；`QLG.q_fc()`
在 `s_w.ndim==2` 时走分组路径；bias 只在第 0 段非零（按第 0 段的 `s_a·s_w[0]` 取整，避免 G 份各舍一次）；
每段的输出尺度取该段自己的 `max/127`，ADD 链按 gemmlowp 两次舍入契约落码。整层的分组版是用同一套 helper
**替换**票 04 的 FC 发射与量化表（`bw.QL` / `bw.q_fc_tables` 两个接缝），其余算子一字未改。

**层数**（实测，`model.compile.log`）：

| G | group(K=896) | 每矩阵层数 | 整层层数 | 执行步 | Static | .dla |
|---|---|---|---|---|---|---|
| 1（票 04） | 896 | 1 | **61** | 1 | 7.2 MiB | 7.63 MB |
| 2 | 448 | 3 | **83** | 1 | 7.3 MiB | 7.73 MB |
| 4 | 224 | 7 | **119** | 1 | 7.4 MiB | 7.91 MB |
| 8 | 112 | 15 | **191** | 1 | 8.4 MiB | 9.04 MB |

组件图的层表也对得上（`ops/q04b_fc_g4/model.compile.log`）：`SliceLayer{SPLIT} → FC×4 → ADD×3`，**1 个执行步**。
注意 SPLIT 在整层里被**复用**（q/k/v/o 共用一次对 n1 的切分，gate/up 共用一次对 n2 的），所以不是严格的 `7×(2G−1)`。

---

## 4. 判据与阈值登记（跑之前）

| 图 | 登记 | 说明 |
|---|---|---|
| `q04b_fc_g2/g4/g8` | `int8_tol=1`、`cosine_min=0.9999` | ADR 0005 的算子级原文；**写进 model.json 之后才编译/上板** |
| `q04b_layer_g2/g4/g8` | `int8_tol=2` | 沿用票 04 的 `LAYER_TOL`（图里含 int8 RMSNorm 定点 RSQRT 链 + 一条 fp16↔int8 缝），已在票 04 登记并保留 FAIL |
| 探针 | 不设判据 | 探针的产出是**编译日志**，不是数值 |

**没有任何一次「看到 FAIL 再调阈值」**：本票唯一的 FAIL 是整层的码级/质量判据，如实保留；组件全部 PASS。

**开工时的预测（与结果对照，全部记下来）**：

| # | 预测 | 结果 |
|---|---|---|
| P1 | 属性级分组 scale 不存在（schema 只有 `quantized_dimension`） | ✅ 两个探针都在加载期被拒（§2） |
| P2 | 但用 K 切段可以等价表达，且设备逐位 | ✅ 0/14336，`G=2/4/8` 全 PASS（§5.1） |
| P3 | 真实权重的分组收益比夹具大（真实权重有离群值） | ✅ 17.6%→7.6%（G=128）；夹具 14.4%→7.3% |
| P4 | 分组能把 4-bit 的层 cosine 从 0.91 拉到接近 0.9999 | ❌ **被推翻**：权重只量化时 0.9733→0.9943（G=128），整条 int8 链上 **0.866/0.820/0.856**（G=1/4/8）——没有改善 |
| P5 | 层数代价 ≈ 每矩阵 G−1 | ⚠️ 实测每矩阵 `2G−1` 层（G 个 FC + G−1 个 ADD），整层 1.36×/1.95×/3.13× |

P4 被推翻是**本票最重要的产出**：拆开看，权重误差确实降了（P3 ✅），但**每个部分和都要落回 int8**，
这一步的舍入把增益吃掉（§5.3 有分解证据）。


---

## 5. 证据

### 5.1 组件：分组 FC 在板端逐位（`ops/q04b_fc_g{2,4,8}/compare.txt`）

    op        : FULLY_CONNECTED T=16 K=896 N=896 G=4 group_size=224   dtype=int8
    exact     : YES   mismatched=0/14336
    contract  : int8_tol=1  max_abs_diff=0  fraction_off=0.0000%
    PASS

`G=2`、`G=8` 同为 **0/14336**。板端时间（`neuronrt -c 3`，只作记录）：G=2 1.142 ms、G=4 0.797 ms、G=8 1.068 ms /
inf（T=16 的 896×896 单矩阵；int8 域，不与 fp16 票的数字混用）。**噪声底不重复测**：同 `.dla` 同输入的确定性
已在票 04 与仓库其余链上多次实测为 0，本票三个组件同机制。

### 5.2 整层：编得出来，但数值不动（`ops/q04b_layer_g4`）

    ===Graph[0] has 1 Execution Step(s)===   Target: MDLA_5_0   119 层   Static 7.4M   EXIT=0
    compare.txt : mismatched=13193/14336   max_abs_diff=41   int8_tol=2   FAIL
    --diag      : device vs 独立 fp64 链   cosine 0.9096674   rel_l2 4.3160e-01
    量化 golden vs 同一 fp64 链           cosine 0.9013246   rel_l2 4.5101e-01

→ 设备仍然**忠实**（device 甚至比 golden 略好 0.0084），所以 0.90 这个数字是**量化档本身**的，与票 04 的
0.9130/0.9107 同档。码级 92%/41 码比票 04（87%/41）略差：G−1 个 ADD 每个都多一层 ±1 码级舍入模型的对齐误差
—— 同一个原因（§5.3d）。

### 5.3 增益去哪了：权重降了，整条链没降

**（a）权重误差**（numpy，只重量化；真实权重与冻结夹具两套）

| G | group | 真实 Qwen2.5-0.5B，7 矩阵均值 | 冻结夹具（高斯）均值 |
|---|---|---|---|
| 1 | 896 | **17.62%** | 14.40% |
| 2 | 448 | 16.06% | 13.60% |
| 4 | 224 | 14.58% | 12.76% |
| 8 | 112 | 13.19% | 11.86% |
| 16 | 56 | 11.82% | 10.90% |
| 32 | 28 | 10.45% | 9.84% |
| 64 | 14 | 9.06% | 8.67% |
| 128 | 7 | **7.56%** | 7.34% |

（真实权重逐矩阵 per-channel：Wq 18.65、Wk 14.99、Wv 17.95、Wo **20.47**、Wg 15.33、Wu 15.49、Wd **20.49**；
分布 max/std = 10.6–42.8、kurtosis 3.4–22.8 —— 4-bit 吃的是尾巴，所以分组对**真实权重**更有效，但也没有质变。）

**（b）权重只量化时的整层 cosine**（真实权重、真实 embedding 行作输入、其余 fp64）

| G | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| out cosine | 0.9733 | 0.9780 | 0.9818 | 0.9847 | 0.9869 | 0.9895 | 0.9922 | **0.9943** |
| probs cosine | 0.9989 | 0.9985 | 0.9992 | 0.9992 | 0.9994 | 0.9996 | 0.9996 | 0.9998 |

**（c）同一条链把 int8 激活 + 每个部分和的 requant 也算进去**（真实权重，`ops/q04b_layer_real_g*`）

| G | 整层层数 | 4W8A 整层 golden vs fp64 |
|---|---|---|
| 1 | 61 | 0.8659 |
| 4 | 119 | **0.8199** |
| 8 | 191 | 0.8555 |

夹具上同方向（票 04 的 0.9130 → G=2 0.9192 / G=4 0.9013 / G=8 0.9206）。⇒ **分组不改善整层指标**。

**（d）为什么**：每个部分和都必须落成 int8（`FULLY_CONNECTED` 的输出只有 int8 一种落法，本栈没有 int32 输出，
也**不能**用 int8→int8 的 `QUANTIZE` 做 requant —— 票 04 §7.2 已证它算错）。于是每个 `y_g` 被舍入一次、
每个 ADD 再舍入一次，量级 ≈ √G/2 码 × 输出尺度；而权重误差在 G=8 只改善 25% 相对。两条曲线在 **G≈16 交叉**，
之后继续加 G 是净亏（下表：Wo 0.146→0.316、Wd 0.157→0.191 在 G≥32 反而变差）。

**（e）FC 级（真实权重、真实上游激活、含 int8 累加），`fcsim_real` 原文见 `ops/q04b_errcurve/fcsim_real.txt`**

| 矩阵 | G=1 | G=4 | G=8 | G=16 | G=32 | G=64 |
|---|---|---|---|---|---|---|
| Wv 投影 rel_l2 | 0.2449 | 0.1978 | 0.1890 | **0.1752** | 0.1753 | 0.1903 |
| Wo | **0.1463** | 0.1809 | 0.2194 | 0.2714 | 0.3160 | 0.3923 |
| Wg | 0.1119 | 0.0995 | 0.0936 | **0.0914** | 0.0946 | 0.1076 |
| Wu | 0.1758 | 0.1548 | 0.1469 | **0.1404** | 0.1404 | 0.1547 |
| Wd | 0.1572 | **0.1460** | 0.1488 | 0.1639 | 0.1913 | 0.2358 |

最好的格子是 G=16（group 56）拿到 4–20% 相对改善、层数 3.4×；`Wo`/`Wd` 这类离群最重的矩阵反而**变差**。

### 5.4 方向②（N 切块）为什么没做

把 N 切成块、每块独立 per-channel，等价于在 N 方向多几个 scale —— 而 per-channel 本来就是「每个 N 一个」，
切块只会更差（每块的行最大值不会比整行小）。对 K 方向的动态范围**零作用**，所以本票只在文档里给出这个判断，
未建图。要真做 N 向分组，代价与 K 向相同（每块一次 FC + CONCAT），收益方向却相反，无意义。

---

## 6. 厂商的 group 到底多大：**≈256**，不是 32–64

ADR 0005 订正 §2 用「0.5075 B/param」反推「每 32–64 个权重一个 scale」。按体积精确算：

    参数 493,961,216（24 层 357,826,560 + lm_head 136,134,656）
    纯 4-bit            = 246,980,608 B
    厂商 bin             = 250,737,664 B    差 3,757,056 B = 参数的 0.76%
    若每个 group 一个 fp16 scale：g = 2 × 493,961,216 / 3,757,056 ≈ 263  ->  g = 256（含对齐）
    若 scale+zp 各 fp16（4 B）：   g ≈ 526                                   ->  g = 512

⇒ 厂商 group 是 **256（或 512）**，比 ADR 的估计**粗 4–8 倍**。落到我们的口径：K=896 时 G≈3.5，即上表 G=4 那一格
—— **权重误差 14.6%（真实权重）**，与 per-channel 的 17.6% 同量级。

**这直接改写 ADR 0005 的第 2 条**：厂商 4-bit 的端到端 23/268 token match，与他们 group≈256 的 4-bit 权重误差
（~15%）是一致的；**分组 scale 不是「14% vs 厂商」的差距来源**。他们真正多出来的东西在转换器里：
`use_hessian_opt` / AdaRound / Bias Correction（§2.3 在模块里实测到这些开关）。

---

## 7. 结论（落到票面要求的三选一）

**「不能」用于达到判据；「能」但代价与收益不可比。** 分开写清：

1. **属性级**：不能。TFLite schema 一个 `quantized_dimension`，ncc 无分组开关，两个探针在**加载期**被拒（§2）。
2. **分解级**：能，且设备逐位（0/14336，`G=2/4/8`）。最小复现 = `ops/q04b_fc_g4`（编译日志 + compare.txt 在仓库里）。
3. **代价**：每矩阵 `2G−1` 层（G FC + G−1 ADD）；整层 **61 → 83(1.36×) / 119(1.95×) / 191(3.13×)**，
   全部仍是 **1 个执行步**（层数不是执行步瓶颈，票 09 已证；瓶颈是常驻内存 7.2→8.4 MiB）。
4. **收益**：权重误差 17.6% → 13.2%(G=8) → 7.6%(G=128=group 7)；但 **int8 部分和舍入** 让整层指标持平甚至
   变差 ⇒ **对 ADR 判据没有任何帮助**；代价最大的那档（G=128，≈4.5× 层数）也只有权重-only 的 0.9943。

**给票 05 的建议**：主档取 **8W8A**（本票离线已见：同一条链、真实权重、int8 per-channel 权重 → 层 cosine
**0.9999157**、rel_l2 1.3e-2、probs cosine 0.9999960；请 04c 用板端钉死）。4-bit 只在能证明不掉质量的位置
局部使用；若要继续压 4-bit，方向是**舍入策略**（GPTQ/AdaRound，改码不改图、零层数代价），不是分组 scale。

---

## 8. 旁证：舍入策略 vs scale 粒度（含一条方法论教训）

`tools/build_q04b.py gptq_probe` 在真实权重上跑了一个 8 行校准 / 8 行留出的小 GPTQ（per-channel int4，零额外层）：

| 矩阵 | RTN（留出） | GPTQ（留出） | GPTQ（在自己的校准行上评） |
|---|---|---|---|
| Wq | 0.0517 | 0.0477 | 0.0030 |
| Wo | 0.1289 | 0.1027 | 0.0054 |
| Wv | 0.2011 | 0.1878 | 0.0156 |
| Wd | 0.1617 | 0.1627 | 0.0052 |

留出集上只有 0–20% 改善；而「在自己的校准行上评」能到 0.003 —— **那是陷阱**（16 行能拟合 896 列）。
**方法论教训（建议进 NOTES）**：任何校准类量化实验，**校准集与评估集必须分开**，否则会得到 10× 的假改善。
（本票样本量只有 8 行，远小于真正的 GPTQ 用法；上表只作方向参考，不当结论。）

---

## 9. 复现

    export QWEN25_WEIGHTS_DIR="/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen"   # 夹具（高斯）
    # 真实权重（只读）: /mnt/d/ai/Qwen2.5-0.5B-Instruct/model.safetensors   (可用 QWEN25_REAL_SAFETENSORS 覆盖)

    # 三条离线曲线（无板端、无 flatbuffer） + 一条旁证
    /usr/bin/python3 tools/build_q04b.py errcurve        # 夹具：权重误差 vs G + 整层（权重-only）
    /usr/bin/python3 tools/build_q04b.py errcurve_real   # 真实权重：分布 + 权重误差 + 整层（权重-only）
    /usr/bin/python3 tools/build_q04b.py fcsim_real      # 真实权重：含 int8 累加的 FC 级误差
    /usr/bin/python3 tools/build_q04b.py gptq_probe      # 舍入策略旁证（校准/留出）
    # 原文已存 ops/q04b_errcurve/{errcurve_fixture,errcurve_real,fcsim_real}.txt 与 *.json

    # 组件：建图 → 编译 → 板端（持锁，前后 healthcheck）
    /usr/bin/python3 tools/build_q04b.py fc_g --group 4 --out ops/q04b_fc_g4
    ./tools/compile.sh ops/q04b_fc_g4/model.tflite ops/q04b_fc_g4/model.dla
    export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
    ./tools/devlock.sh run -- ./tools/healthcheck.sh --quick
    ./tools/devlock.sh run -- ./tools/e2e.sh ops/q04b_fc_g4
    ./tools/devlock.sh run -- ./tools/healthcheck.sh --quick

    # 整层（分组）+ 真实权重版（离线设备模型，不上板）
    /usr/bin/python3 tools/build_q04b.py layer --group 4 --out ops/q04b_layer_g4
    /usr/bin/python3 tools/build_q04b.py layer --group 1 --real --out ops/q04b_layer_real_g1

    # 两个封路探针
    /usr/bin/python3 tools/build_q04b.py schema_scale2n --out ops/q04b_probe_scale2n
    /usr/bin/python3 tools/build_q04b.py schema_axis1   --out ops/q04b_probe_axis1

**可复现性**：同一 builder 重建 `ops/q04b_fc_g4/model.tflite` 与 `golden/*.bin` 逐字节相同（脚本无随机源；
权重来自冻结夹具或只读的真实权重文件）；`.dla` 按仓库既有约定不作复现判据。

---

## 10. 适用范围 / 不适用

**适用**

1. 想在 TFLite + ncc 栈上得到「每 (输出通道, K 段) 一个 scale」= **K 切段分解**，这是唯一形态；
2. 该形态**设备逐位**（0/14336），实现细节已在 `tools/build_q04b.py` 落地（bias 只放第 0 段、每段自己的输出尺度、ADD 链两次舍入）；
3. 分组 scale 的**权重误差**曲线（夹具与真实权重两套）与**层数代价**（`2G−1`/矩阵，整层 61→83/119/191）；
4. 厂商 4-bit 的 group ≈ **256**（体积口径），因此他们的 4-bit 权重误差与我们 per-channel 同量级。

**不适用**

* 指望分组 scale 让 4-bit 达到 cosine 0.9999（§5.3 证明任何 G 都不行）；
* 把本票的时间数字当性能结论（只有单矩阵 ms/inf，未做 24 层/时序）；
* 真实权重的**板端**对拍（`ops/q04b_layer_real_*` 只有离线设备模型，没有 `.dla`）；
* 8W8A 的结论（04c）。

---

## 11. 剩余疑问（UNKNOWN，不圆）

1. **ncc 为什么在加载期拒掉两个探针**：只拿到 `Fail to process the given model`（没加 `--verbose`），没有区分
   「scale 数组长度与轴长不符」与「FC 不接受 dim≠0」—— 对结论无影响（都不可表达），但机制没钉死。
2. **有没有第三条路**：`--rewrite-pattern=?` 在本版会直接打印整份 `--help`（该用法不认 `=?`），所以「ncc 内部
   有没有把 FC 拆成分组 FC 的 pattern」只是**没被证伪**，不是被证伪。要查得靠 `--verbose` 或反汇编。
3. **部分和舍入能否更省**：本票用「每段自己的输出尺度 + 逐级 ADD（各自尺度）」。还能试「所有段共享一个输出尺度」
   （码域精确相加，但每段更粗）与「两两树形 ADD」；没做这轮扫描。估算三者的误差同阶（√G/2 码），故未优先做。
4. **真实权重的整层板端**：`q04b_layer_real_*` 是 numpy 设备模型；要上板得再编 3 张图（每张 ~2 min）+ 3 次持锁运行，
   按「结论已定」的性价比跳过。
5. **GPTQ 在正确用法下能把 4-bit 压到多少**：本票只有 8 行校准 + 简化实现（留出集 0–20%）。厂商转换器里有
   AdaRound / Bias Correction / hessian 三件套，真实收益没测 —— 那是**下一票**的问题，不是本票的结论。
