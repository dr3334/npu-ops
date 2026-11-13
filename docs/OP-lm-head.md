# OP-lm-head —— LM head + logits（Qwen2.5-0.5B · T=8 · fp16 · tied 输出投影）

> 票：`.scratch/qwen25-attention/issues/08-lm-head-logits.md`（blocked by 02）
> 分支 `impl/08-lm-head` · worktree `wt-t08` · op 目录 `ops/lm_head_t8_fp16/`
> 图：`x [8,896] fp16` → **FULLY_CONNECTED（权重 = 夹具 `embed.weight`，tied，只物化一份）** → `logits [8,151936] fp16`
> 权重来源：`weights/qwen25_frozen`（票 02，`weights_digest=898fa3195d627a4c…`，加载器 `tools/qwen25_weights.py`）

---

## 0. TL;DR

| 项 | 结果 |
|---|---|
| **op / 图** | `ops/lm_head_t8_fp16`：`x[8,896] fp16` → **4×FC（N=37984，W = `embed.weight` tied，b = 0）** → **CONCAT(axis=1)** → `logits[8,151936] fp16` |
| **单体 FC（N=151936）** | ❌ **编译期硬上限**（不是内存问题）：`MDLA: Dimension should be <= 65535. Operand: 1 got <151936 x 896>`，`EXIT=1`、无 `.dla`（`fail_monolithic.compile.log`） |
| **边界探针** | `FC 8×896×65535` **EXIT=0**；`8×896×65536` **EXIT=1** ⇒ 上限恰为 **65535**；`CONCAT→[8,151936]` 与 `RESHAPE→[8,151936]` 都能编译（§4） |
| **分块后编译** | `EXIT=0`、`Target: MDLA_5_0`、**1 个执行步 / 5 层**（4×FullyConnected + 1×Concat）、**无 bridge**、`.dla` **272,274,707 B**、29.8 s |
| **板端** | `compare.txt` **PASS**：`max_abs 9.766e-04`、`rmse 6.245e-06`、`rel_l2 1.368e-05`；相对口径 `max_abs/max|exp| = 4.44e-4`；**99.77% 逐位相同**、99.98% 在 1 ULP 内 |
| **噪声底** | **0.000e+00**（3 次运行两两比较，121.5 万个数全等）⇒ 硬件确定性；阈值沿用**预登记**的 `atol=rtol=4.29153e-3`（`4·2^-11·max|exp|`，≈4.4 ULP @ max） |
| **tied 只物化一份** | 加载器 `lm_head.weight is embed.weight` 断言通过 + 模型常量字节 = **272,269,312（权重）+ 303,872（零 bias）恰好一份**（`tied_materialised_once: true`） |
| **传输成本** | `.dla` push **272.27 MB / 7.78 s（35.0 MB/s）**；输入 14 KB / 87 ms；**输出 pull 2.43 MB / 215 ms** |
| **推理耗时** | **8.29–9.66 ms/inf**（T=8、vocab 151936、repeat 3）≈ **113–116 FPS** |
| **top-1** | **不启用**（ADR 0003）：随机权重下无语义、1 ULP 即翻盘；冻结到真实权重阶段 |


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

---

## 3. 做法

### 3.1 权重：tied，只物化一份

```python
w = qw.get("embed.weight")      # [151936,896] fp16, sha256 由加载器校验
assert qw.get("lm_head.weight") is w     # tied 别名：同一对象，不重复占内存
```

- 夹具 `weights_digest=898fa3195d627a4c…`、`embed.weight sha256=40cdb3488dedade2…`（`build.log` 第一段）。
- 模型里**只有一份**权重常量：`constant_bytes = 272,573,184 = 272,269,312 + 4×37,984×2`
  （后者是 4 段全零 bias，合计 N 个 fp16 = 303,872 B）。builder 若发现常量字节 ≠ 一份权重 + 零 bias
  会直接 `SystemExit`（`one-weight-copy check`）。
- `weights_format=DEFAULT` ⇒ 权重按 `[N,K]` 解释，与 `embed.weight` 的 `[vocab,hidden]` 完全一致，
  **不需要也不允许转置**（转置是编译期无法发现、只有板端数值会露馅的静默错误，见 §2.3 末条）。

### 3.2 图：为什么必须分块（§4 有完整证据）

单体 `FC(8×896×151936)` 在 **importer/编译期**就被拒：

```
OP[0]: FULLY_CONNECTED
 ├ MDLA: Dimension should be <= 65535. Operand: 1 got <151936 x 896>.
 ├ MDLA: Dimension should be <= 65535. Result : 0 got <8 x 151936>.
 ├ EDPA: unsupported operation
ERROR: Cannot find an execution plan because of unsupported operations
EXIT=1
```

这是**维度**上限，与 272 MB 的权重体积无关（"内存/搬运会不会撞墙"这个担心因此没有成为瓶颈）。
`ceil(151936/65535) = 3`，而 3 不整除 151936 ⇒ **4 块（N=37984）**；4 块 FC 的输出
`[8,37984]` 经 `CONCAT(axis=1)` 拼成 `[8,151936]`。**同一张图、权重仍只一份、输出形状不变**。

### 3.3 输入与 golden（golden 独立性 = 硬规则 R3）

- 输入 `x` = 夹具 `layer0.attn.o_proj.weight` 的行 0..7，按**行 RMS 归一**（实测行 RMS=0.99999–1.00001），
  理由与"为什么不用 embed 行"见 §2.4。
- `golden/expected.bin` = numpy `float32` matmul（按词表分块 8192 行，峰值 RSS ~100 MB）再转 fp16：
  `exp = fp16(x_fp32 @ W_fp32ᵀ + 0)`。与 flatbuffer 构造**零共享代码**。
- 输出统计：`max|exp| = 2.197266`、`std = 0.456503`、无 NaN/Inf —— 与预测 P7 的 `≈2.3` 吻合。

### 3.4 容差（预登记，见 §2.3 / §2.4）

`atol = rtol = 4 × 2^-11 × max|exp| = 4.29153e-3`（写入 `model.json`，含 `tolerance_revision`
的 policy/reason/formula/under_2e-2_cap 四段）。第一次上板就用这个值，**没有事后调整**。

注意方向：这个阈值（相对 **1.95e-3**）比 NOTES §4 的 flat `2e-2`（在 `max|exp|=2.197` 处等于 9.1e-3 相对）
**更紧 4.7 倍**，不是放宽 —— 目的是把判据压到 fp16 末位水平（4 ULP），而不是把红线挪到容易过。
实测 `max_abs=9.766e-4` 只用到合约的 **1/14**。

### 3.5 上板流程（严格按锁）

```bash
export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
# 锁外：builder + ncc（不碰板子）
/usr/bin/python3 tools/build_lm_head.py --chunks 4          # model.tflite + golden/
./tools/compile.sh ops/lm_head_t8_fp16/model.tflite ops/lm_head_t8_fp16/model.dla
# 持锁：healthcheck → e2e×2（噪声底）→ 计时 push/run/pull → healthcheck
./tools/devlock.sh run -- ./ops/lm_head_t8_fp16/board_run.sh
```

---

## 4. 编译结果（锁外）

### 4.1 单体 vs 分块

| 模型 | 结果 | 证据 |
|---|---|---|
| 单体 `FC 8×896×151936` | **EXIT=1**，`Dimension should be <= 65535`（Operand 与 Result 各一条），`SIZE=0`、无 `.dla` | `ops/lm_head_t8_fp16/fail_monolithic.compile.log` |
| **分块 ×4 + CONCAT** | **EXIT=0**，`1 Execution Step`、`Target: MDLA_5_0`、5 层（`FullyConnectedLayer<0..3>` + `ConcatLayer<4>`）、无 bridge、`DRAM Static 259M / Total 261M`、`SIZE=272,274,707`、29.8 s | `ops/lm_head_t8_fp16/model.compile.log` |

`.dla` 与 `.tflite` 体积比 = 272,274,707 / 272,574,824 = **0.9989**（预测 P3 的 ±2% 成立）。

> **.dla 不可字节复现：同一 tflite 连编多次 SIZE/SHA256 会差几个字节，以该目录 model.compile.log 为准；回归只比数值**

### 4.2 边界探针（`ops/lm_head_t8_fp16/probe/`，全部 compile-only）

| 探针 | 形状 | EXIT | 结论 |
|---|---|---|---|
| `fc_65535` | FC 8×896×**65535** | **0**（1 step，`FullyConnectedLayer`，DRAM 113M） | 65535 可编 |
| `fc_65536` | FC 8×896×**65536** | **1**（`Dimension should be <= 65535. Operand: 1 got <65536 x 896>`） | **上限恰为 65535** |
| `fc_37984` | FC 8×896×37984 | 0 | 分块用的形状可编 |
| `concat_151936` | CONCAT(axis=1) 4×[8,37984] → **[8,151936]** | **0**（1 step，`ConcatLayer`，DRAM 4.6M） | **CONCAT 可以产出 151936 宽的张量** |
| `concat_65535` / `concat_65536` / `concat_75968` | CONCAT → [8,65535]/[8,65536]/[8,75968] | 0 / 0 / 0 | concat 不受 65535 限制 |
| `reshape_151936` | RESHAPE [32,37984] → **[8,151936]** | 0（1 step） | 备用路线也不需要 |
| `reshape_big3d` | RESHAPE [32,37984] → [8,4,37984] | 0 | 对照 |

⇒ **65535 是 FULLY_CONNECTED 张量上的限制（机制 UNKNOWN），不是设备/后端对张量维度的全局限制。**

---

## 5. 板端结果（持锁，`board_run.sh`）

### 5.1 数值（`compare.txt`，`atol=rtol=4.29153e-3` 是预登记值）

```
op        : FULLY_CONNECTED+CONCATENATION {'M': 8, 'K': 896, 'N': 151936}  dtype=fp16
expected  : 1215488 vals   device: 1215488 vals   logical shape=(8, 151936)
max_abs   : 9.766e-04
rmse      : 6.245e-06
rel_l2    : 1.368e-05
nan/inf   : exp 0/0  dev 0/0
range     : exp [-2.121,2.197]  dev [-2.121,2.197]
tol       : atol=0.00429153 rtol=0.00429153
PASS
```

**误差结构**（`noise_floor.py` + `probe/error_anatomy.py` → `noise_floor.json`）：

| 指标 | 值 |
|---|---|
| `max_abs / max|exp|` | **4.444e-04**（= 0.5 ULP @ `max|exp|=2.197`，该处 ULP=1.953e-3） |
| 逐位相同 | **99.7698%**（1,215,488 个数中 2,798 个不同） |
| 在 1 ULP 内 | **99.9839%** |
| `|exp| ≥ 0.01` 的元素里最大 ULP 比 | **1.00**（即"最多差 1 个 fp16 末位"） |
| `|exp| < 0.01` 的近零元素（n=21,180）最大绝对误差 | **7.63e-06**（相对看着大，绝对可忽略） |

⇒ 设备与 fp32 参考的差**不是算法差**，是 fp16 输出的**末位舍入**（与 `docs/OP-MATRIX.md` §3
"fp16 层普遍差 1 个 code"一致）。

### 5.2 噪声底（NOTES §4 协议）

| 比较 | `max|r1-r2|` | 不同元素数 | 结论 |
|---|---|---|---|
| `out_r1` vs `out_r2`（两次独立 e2e，同一 `.dla` + 同一输入） | **0.000e+00** | **0 / 1,215,488** | 确定性 |
| `out_r1` vs `out_r3`（第三次，计时块那次） | 0.000e+00 | 0 / 1,215,488 | 确定性 |
| `out_r2` vs `out_r3` | 0.000e+00 | 0 / 1,215,488 | 确定性 |

- **实测噪声底 = 0**（与 P7 一致）：同 `.dla` 同输入完全确定，**没有任何 run-to-run 抖动**。
- 阈值按"噪声底 3–5 倍"在 0 上无法定义，因此按 §4 的**退化规则**取 fp16 末位量级；本票采用
  **§10.1 的例外写法** `atol=rtol=4·2^-11·max|exp| = 4.29e-3`（≈4.4 ULP @ max，相对 ~1e-3），
  且**在跑之前登记**。实测 max_abs 只有合约的 **1/14**（14.05× headroom）。

### 5.3 输出尺寸与搬运成本（`transfer.json`）

| 项 | 字节 | 实测 | 速率 |
|---|---|---|---|
| 输入 `x` | 14,336（8×896×2） | push **87 ms** | — |
| 模型 `.dla` | **272,274,707**（272.27 MB） | push **7,780 ms** | **35.0 MB/s** |
| **输出 `logits`** | **2,430,976（2.43 MB）** | pull **215 ms** | 11.3 MB/s（adb 自报 24.7–26.7 MB/s） |
| 一次推理 | — | **8.29–9.66 ms**（repeat 3：24.87 / 25.39 / 28.98 ms） | 113–116 FPS |

- 输出 `8×151936×2 = 2,430,976 B`，与预测 P6 完全一致；`golden/expected.bin` 同尺寸。
- 272 MB 权重的成本集中在**加载一次**；推理热路径里没有 host 参与（整图 1 个 MDLA 执行步），
  每 token 只需搬 2.43 MB 输出。
- 设备状态：`/data` 6.4 → **6.1 GB**（占用 272 MB），`MemAvailable` **6.06 GB → 6.09 GB**
  —— **本机空闲 RAM 远大于 NOTES §6 记的 ~273 MB**，所以"内存不足"在本票没有出现；
  真正的拦路虎是 §4.2 的**编译期维度上限**。这条对票 10 同样有用：别按 273 MB 预判。

### 5.4 前后体检

`./tools/healthcheck.sh --quick`：实验前 **PASS**、实验后 **PASS**（`fc_1x64x64_int8` 已知-good op 两次都过）。

---

## 6. 预测复核（§2.2 的 P1–P8）

| # | 预测 | 结果 | 说明 |
|---|---|---|---|
| P1 | `.tflite` ≈ 272.58 MB，与 embed 差 < 0.5 MB | ✅ | 单体 272,573,776 B；分块 272,574,824 B；与 embed 差 305,512 B |
| P2 | tied 别名同一对象 | ✅ | `build.log` 的 `same object=True`；常量字节恰一份 |
| P3 | 单体编译 EXIT=0、1 步、1 层、无 bridge、`.dla`≈`.tflite`、<300 s | ❌ **前四条被推翻** | 单体在编译期被 `Dimension <= 65535` 拒（**不是**内存、不是 bridge、不是加载期）；分块后 EXIT=0 / 1 步 / 5 层 / 无 bridge / 0.9989 体积比 / 29.8 s |
| P4 | push 272.6 MB 需 8–30 s（20–35 MB/s） | ✅（偏快） | **7.78 s / 35.0 MB/s** |
| P5 | 能加载并跑，10–100 ms/次 | ✅（偏快） | 实测 **8.29–9.66 ms/inf**；无 OOM、无加载失败 |
| P6 | 输出 2,430,976 B；pull < 1 s | ✅ | 2,430,976 B；**215 ms** |
| P7 | 噪声底 0；`max_abs ≤ 1 ULP`；`max|logit|≈2.3` | ✅ | 噪声底 0；0.5 ULP @ max、逐元素 ≤1 ULP；`max|exp|=2.197` |
| P8 | 若单体失败，失败点在**加载期或搬运耗时**，退路 = 4 分块 + CONCAT | ⚠️ 退路成立、**失败点判错** | 分块路线**成立**（编译 + 板端都过）；但失败点是**编译期维度上限**，与 RAM/push 无关 |

---

## 7. 与票面 checklist 对照

| 票面要求 | 状态 | 证据 |
|---|---|---|
| 复用词嵌入矩阵作为输出投影（tied），不额外物化一份 | ✅ | `lm_head.weight is embed.weight` 断言（`build.log`）；`constant_bytes = 272,269,312 + 303,872`、`tied_materialised_once: true`；`.tflite` 272.57 MB（两份会 >544 MB） |
| 输出 vocab=151936 的 logits，与 golden 的 atol/rtol 对上 | ✅ | `compare.txt` **PASS**；`max_abs 9.766e-04 / rel_l2 1.368e-05`；阈值 4.29153e-3（预登记） |
| **top-1 判据不启用**，记录冻结到真实权重阶段 | ✅ | §2.3 末条 + `model.json.tolerance_revision.note`：ADR 0003 —— 随机权重下 top-1 无语义、1 ULP 翻盘；真实权重档才启用 |
| 记录输出张量尺寸与设备→host 搬运成本 | ✅ | §5.3：2,430,976 B（2.43 MB）、pull **215 ms**；模型 push 7.78 s / 35 MB/s；推理 8.29–9.66 ms |
| 记录本算子的实测噪声底 | ✅ | §5.2：**0.000e+00**，3 组两两比较 0/1,215,488 不同（`noise_floor.json`） |

票面之外的附加交付（对 09/10 有用）：`fail_monolithic.compile.log`（失败证据保留）、
`probe/`（65535 边界探针，可复现）、`board_run.sh`（可复跑的持锁上板脚本）、
`transfer.json`（搬运成本）、`build.log`（tied/体积/容差自检）。

---

## 8. 适用范围 / 不适用

**适用**
- 任何 `x[T,896] fp16` 的 LM head：`T` 只需与 builder 的 `M` 一致（本票 T=8）；vocab=151936、tied 权重、
  `atol=rtol=4.29e-3`（对 `max|logit|≈2.2` 的量级成立；换输入量级要重新登记容差）。
- 票 09/10 的整块集成：LM head 可直接接在 final RMSNorm 之后（输入即 `[T,896]`，行 RMS≈1 时本契约刚好匹配）。
- "65535 维度上限 / 4 分块 + CONCAT"的结论对任何 `N > 65535` 的 FC 都成立。

**不适用**
- **任何语义/精度结论**（top-1、困惑度、生成质量）：随机夹具权重下 logits 只是同一份数值的搬运结果（ADR 0003）。
- **真实权重档的特性**：bf16→fp16 舍入、真权重 272 MB 的 tied 常量、vocab 极值尾巴（`docs/OP-weights.md` §11）。
- **其它 M/K/N**：换形状要重新探 65535 与 16 字节行规则（本票 `K=896`、`N_i=37984` 都满足）。
- **多 token 批 / 动态 T**：本图 T=8 静态。
- **自建提交器路径**：本票全程厂商 `neuronrt`；`docs/ISSUE-const-fc.md` 的常量 FC 地址异常不在本票范围。

---

## 9. 剩余疑问（UNKNOWN，不圆）

1. **65535 限制的确切机制 UNKNOWN**：探针证明 FC 在 65536 被拒、65535 通过，而 CONCAT/RESHAPE 可以产出
   151936 宽的张量。是这条"operand/result 维度"检查只挂在 FC 上，还是挂在某类 MDLA layer 上？
   未做更细 bisect（如把 FC 表达成 3-D `[1,151936,896]` 权重能否绕过）。**不要**把它当成
   "设备不能处理大张量"的通用结论。
2. **0.23% 元素差 1 ULP 的来源**：MDLA 的 fp16 累加/舍入模式（fp32 或 fp16 累加、round-to-nearest 或截断）
   未定；只知"逐元素 ≤ 1 ULP、逐位相同 99.77%"。不影响本契约，但真实权重档若要更紧口径，需按层 dump 才能答。
3. **热路径上的权重常驻**：本次每次都从头 push 272 MB `.dla`；真实部署里模型应常驻，否则每 token
   7.8 s 的搬运不可接受。`.dla` 常驻时的加载时间/内存占用未单独测（本票只测"push + 一次加载 + 3 次推理"）。
4. **真权重档的 tied 一致性**：随机夹具下两名字是同一份数组；真实 `tie_word_embeddings=true` 的导出
   若被拆成两份，本票的"只物化一份"证据链要重做（口径：改 `manifest.tied` + 重新校验常量字节）。
5. **"对角指纹"未在板端启用**（§2.4）：若要把"设备上确实用同一个矩阵"做成板上结构证据，可把输入换回
   embed 行并检查 `logits[t,t] ≈ 896·rms_t`（预期 13.6–13.7）。本票没做（要多一次编译 + 上板，
   且会让 max-based 容差放宽 6 倍），留给真实权重档或票 10 一次性做。
6. **T 变化时 CONCAT 的开销**：本票 T=8，concat 只搬 2.43 MB；T=128 时是 38.9 MB，`DRAM Temp`
   与耗时的变化未测（票 10 若做 prefill 需要看）。

---

## 10. 复现（全部命令）

```bash
# 0) 夹具（332 MB，约 9 s；已有产物时跳过，或 export QWEN25_WEIGHTS_DIR=...）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/npu-ops' && /usr/bin/python3 tools/gen_qwen25_weights.py"

# 1) 建图 + golden（锁外，约 5 s，峰值 RSS 1.1 GB）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/npu-ops' && /usr/bin/python3 tools/build_lm_head.py --chunks 4"

# 2) 编译（锁外，约 30 s）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/npu-ops' && ./tools/compile.sh ops/lm_head_t8_fp16/model.tflite ops/lm_head_t8_fp16/model.dla"

# 3) 边界探针（锁外，compile-only，约 1 min）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/npu-ops' && bash ops/lm_head_t8_fp16/probe/run_probes.sh"

# 4) 上板（持锁；healthcheck → e2e×2 → 计时 → healthcheck，约 30 s）
export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
./tools/devlock.sh run -- ./ops/lm_head_t8_fp16/board_run.sh

# 5) 噪声底与误差结构
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/npu-ops' && /usr/bin/python3 ops/lm_head_t8_fp16/noise_floor.py"
```