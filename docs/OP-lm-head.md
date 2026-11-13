# OP-lm-head —— LM head + logits（Qwen2.5-0.5B · T=8 · fp16 · tied 输出投影）

> 票：`.scratch/qwen25-attention/issues/08-lm-head-logits.md`（blocked by 02）
> 分支 `impl/08-lm-head` · worktree `wt-t08` · op 目录 `ops/lm_head_t8_fp16/`
> 图：`x [8,896] fp16` → **FULLY_CONNECTED（权重 = 夹具 `embed.weight`，tied，只物化一份）** → `logits [8,151936] fp16`
> 权重来源：`weights/qwen25_frozen`（票 02，`weights_digest=898fa3195d627a4c…`，加载器 `tools/qwen25_weights.py`）

---

## 0. TL;DR

（实验完成前留空；见 §5 结果）

---

## 1. 这一票要回答什么

1. 输出投影与词嵌入是**同一个矩阵**（tied），不得物化两份（否则白多约 260 MiB）；
2. vocab=151936 的 logits 与独立 numpy golden 的 atol/rtol 对上；
3. **top-1 判据不启用**，并写清它冻结到真实权重阶段（ADR 0003）；
4. 记录输出张量尺寸与**设备→host 搬运成本**（实测 push/pull 时间）；
5. 记录本算子的**实测噪声底**。

判据口径照 `NOTES.md` §4：同一 `.dla` 同一输入连跑两次求 `max|r1-r2|` = 噪声底；阈值取噪声底 3–5 倍且 **< 2e-2**；
噪声底为 0 时退化为 fp16 末位量级（约 1e-3 相对），并写明退化理由。

---

## 2. 事前登记：假设与预测（**写于建图/编译/上板之前**，2026-11-13 10:2x）

### 2.1 设计假设

| # | 假设 | 依据 |
|---|---|---|
| A1 | 图就是**单个 FULLY_CONNECTED**（不做权重转置）：`out[m,v] = Σ_k x[m,k]·W[v,k] + b[v]`，`W` 布局 `[vocab, hidden]` 与 `embed.weight` 的 `[151936,896]` 一致 | TFLite FC 的 `weights_format=DEFAULT` 就是 `[N,K]`；`build_ops.py` 已用同一语义 |
| A2 | **tied**：图里 `embed.weight` 只出现一次（一个常量缓冲 272,269,312 B）；`lm_head.weight` 是加载器别名，不二次物化 | `tools/qwen25_weights.py` 的 `get("lm_head.weight") is get("embed.weight")`（票 02 §6.4 第 5 条） |
| A3 | FC **必须有第三个输入 bias**（ncc 加载期硬要求，NOTES §5 坑 1）；真实 LM head **无 bias** → 喂全零 `[151936] fp16`（303,872 B） | NOTES §5-1 |
| A4 | 输入 `x` 取夹具 **`layer0.attn.o_proj.weight` 的 8 行**（0..7）按**行 RMS 归一**：模拟 final RMSNorm 输出（行 RMS=1），确定性、不引入额外 RNG；于是 logits 尺度 = `‖x‖·std(w) ≈ sqrt(896)×0.015249 ≈ 0.456`，与真模型同量级 | 票 02 §4 实测 `embed.weight std=0.015249`；**为什么不用 embed 行见 §2.4** |
| A5 | 上板走厂商 runtime `./bin/neuronrt -m hw`（`tools/e2e.sh`），**不涉及**自建提交器，因此 `docs/ISSUE-const-fc.md` 的"常量权重 FC 地址异常"不适用于本票 | e2e.sh 第 4 步 |

### 2.2 预测（每条都给证伪条件）

| # | 预测 | 若被推翻意味着 |
|---|---|---|
| P1 | `.tflite` ≈ 272,269,312（权重）+ 303,872（零 bias）+ <16 KB（张量表/头）≈ **272.58 MB**；与 embed 字节数之差 < 0.5 MB | 差值 > 0.5 MB ⇒（a）物化了第二份权重，或（b）形状声明与真实数组不一致（NOTES §5-2 的加载期失败模式） |
| P2 | `qw.get("lm_head.weight") is qw.get("embed.weight")` 在 builder 里断言通过 | tied 契约失效（夹具别名机制坏了） |
| P3 | **编译（锁外）**：`EXIT=0`、`Target: MDLA_5_0`、**1 个 execution step**、**1 个 `FullyConnectedLayer`**、无 bridge、`.dla` 大小 ≈ `.tflite`（±2%）、耗时 < 300 s | 出现 `Fail to process the given model`（加载/解析期）/ `Bridging is disallowed` / `Cannot find an execution plan` / `no .dla produced` ⇒ 单体 272 MB 常量 FC 不可编译，转 §3.2 分块 |
| P4 | **push**：272.6 MB 经 `adb push` 约 **8–30 s**（20–35 MB/s）；设备 `/data` 剩 6.4 GB，占 <5%，不是瓶颈 | 远超 30 s ⇒ 记实测值，作为"单体路线不可接受"的证据之一 |
| P5 | **加载 + 运行**：`neuronrt` 能加载 272 MB `.dla` 并跑；单次推理 **10–100 ms**（权重从 DRAM 扫一遍：1.09 GMAC + 272 MB 搬运） | 加载期 OOM / `Fail to load` / 超时 ⇒ 单体路线失败（设备当前 `MemAvailable≈6.1 GB`，空闲 RAM 不是必然瓶颈——**这次实测与 NOTES §6 的 ~273 MB 不同，见 §5**） |
| P6 | **输出尺寸**：`8×151936×2 = 2,430,976 B = 2.43 MB`；`adb pull` < 1 s | 尺寸不符 ⇒ 输出声明/keep_num_dims 写错 |
| P7 | **数值**：`max\|logit\| ≈ 5σ·0.456 ≈ 2.3`；噪声底 `max\|r1-r2\| = 0`（fp16 FC 在本仓库已多次逐位复现，见 `ops/fc_8x128x128_fp16/compare.txt` 的 `max_abs 0.000e+00`）；板端 vs golden `max_abs ≤ 1 ULP`（`\|logit\|≈2.3` ⇒ ULP=9.8e-4） | 噪声底非 0 或 max_abs 超出 1 ULP ⇒ 需查明是累加精度（fp16 vs fp32 累加）还是形状映射错误 |
| P8 | 若 P3/P5 失败，**失败点在加载期或搬运耗时**，不在数值；退路 = **4 分块 FC（N=37984）+ CONCAT(axis=1)**，总权重字节不变（仍一份）、输出仍是 `[8,151936]` | 若连分块也在 importer 期失败 ⇒ 需要报到 Lead（可能是 ncc 对大常量缓冲的硬上限） |

### 2.3 阈值登记（**先登记再测**，含 Lead 11:00 补充的 NOTES §10.1 例外规则）

- 判据公式：`|dev - exp| <= atol + rtol*|exp|`。
- **走 §10.1 例外分支**（不是 §4 的 `<2e-2` 直给）：本算子输出（logits）不是 O(1) 量级——预测
  `max|logit| ≈ ‖x‖·std(w) ≈ 29.93×0.015249×5σ ≈ 2.3`，所以按例外规则：
  `atol = rtol = k · 2^-11 · max|exp|`，k = 4。`max|exp|` 由 **golden**（设备无关，跑之前就已定稿）算出，
  builder 把它写进 `model.json` 的 `atol`/`rtol` + `tolerance_revision`（policy / reason / formula 三段留痕）。
- **上限自检**：只要 `max|exp| < 10.24`，阈值就 `< 2e-2`（`atol + rtol·max|exp| = 8·2^-11·max|exp|`），
  两条口径同时满足；`model.json` 里写了 `under_2e-2_cap: true/false`。
- 噪声底的作用：预测噪声底 **0**（硬件确定性 + 同一 `.dla`）。若实测非 0，阈值取
  `max(§10.1 值, 5×噪声底)`，并**在下一次确认跑之前**重登记（`tolerance_revision.revision += 1` + 理由），
  报告里同时给出"旧登记值 + 第一次跑的实际数值"。**绝不**为了让 compare 变绿而在事后调阈值。
- 报告口径：**永远同时给相对量** `max_abs / max|exp|` 与 `rel_l2`（`compare.py` 已打印 `rel_l2`）。
- **top-1 / argmax 判据不启用**（ADR 0003）：随机权重下 FP16 末位噪声就能翻近似平局的 top-1，argmax 只会产生假警报；
  它冻结到真实权重阶段（最后一档）才启用。本票只对 logits 向量本身做 atol/rtol 对拍。
- **§10.2 也适用**：`EXIT=0 + 单执行步 ≠ 数学正确`。本票的"编译侧结论"一律以板端对拍为准；特别地，
  FC 的 `weights_format=DEFAULT [N,K]` 若被误当 `[K,N]`，编译与单执行步都不会报警，只有板端数值会露馅。

### 2.4 登记修订 #1（**跑之前**，仅依据设备无关的 golden；2026-11-13 10:5x）

第一次按 A4（原版：输入 = `embed.weight` 的 8 行按行 RMS 归一）算 golden，实测

```
golden: abs_max=13.679688  min=-2.066406  max=13.679688  std=0.457509
```

`max` 与 `std` 差 **30 倍** ⇒ 这不是分布尾，而是**结构性的自点积**：
`x_t = embed[t]/rms_t ⇒ logits[t, t] = x_t·embed[t] = 896·rms_t ≈ 896×0.01527 ≈ 13.68`。
逐行验证：8 行对应的 8 个对角元都悬在 13.6–13.7，其余 121 万个数 `|·| ≤ 2.07`。

**后果（这就是要改的原因）**：§10.1 的例外阈值是 `4·2^-11·max|out|`，一个 30σ 的结构性尖峰会把它从
`4.5e-3` 抬到 **2.67e-2**（`> 2e-2` 上限，且相对口径 2.7% 比 fp16 的 1–2 ULP 松 6 倍）。
**尖峰是"这个输入"的属性，不是算子的属性**，拿它当容差标尺等于自己放宽判据。

**修订**：输入改为**与权重去相关**的夹具张量行 —— `layer0.attn.o_proj.weight` 的行 0..7，仍按行 RMS 归一。
理由：(a) 仍完全来自冻结夹具、无额外 RNG；(b) 行 RMS=1 与真模型 final RMSNorm 输出同量级；
(c) 与 `embed.weight` 无共享行 ⇒ 输出回到"1.2M 个近似独立样本"的形态，`max|out|` 由分布尾（5σ）决定。
**预测**：`max|logit| ≈ 5×0.456 ≈ 2.3` ⇒ 阈值 `4·2^-11·2.3 ≈ 4.5e-3`（相对 1%，`< 2e-2` 上限）。

> 留痕：本条修订发生在**任何编译/上板之前**，只用了 golden（设备无关）。A4/P7 已按修订更新；
> 原 embed 输入方案的数值原样保留在上面（失败/放弃的证据也保留，仓库规矩）。
> 附带说明：那次测量顺带证明了"tied"在数值上的可辨识性 —— 若后续要给"同一矩阵"出**板端**结构证据，
> 把输入换回 embed 行、检查 `logits[t,t] ≈ 896·rms_t` 即可（本票不做，见 §6 剩余疑问）。

---