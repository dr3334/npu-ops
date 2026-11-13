# OP-weights —— 冻结随机权重夹具（Qwen2.5-0.5B 真实形状 · fp16）

> 范围：**权重夹具本身**（生成 / 冻结 / 校验 / 下游加载接口）。**本票不动板端、不产生 \`ops/\` 目录**，
> 因此没有 \`model.compile.log\` / \`compare.txt\` / 噪声底——那三样属于消费这些权重的算子票（03–10）。
> 本票的判据是：**同一 seed 两次生成逐字节相同**、**每份权重 sha256 可被双方独立校验**、**生成分布的尺度对齐真实权重实测统计量**。
>
> · 分支 \`impl/02-frozen-weights\`（worktree \`wt-t02\`）
> · 生成器 \`tools/gen_qwen25_weights.py\` · 加载器 \`tools/qwen25_weights.py\`
> · 产物 \`weights/qwen25_frozen/\`（\`manifest.json\` 入库；\`*.npz\` 按 \`.gitignore\` 不入库，可复现）
> · seed \`20261113\` · dtype \`float16\` · 19 个张量（一层 decoder ×2 + tied embed/LM head）
>
> **一句话结论**：夹具已冻结并自校验通过；生成尺度的来源**不是**配置里的 \`initializer_range=0.02\` 代理口径，
> 而是本机**真实 Qwen2.5-0.5B-Instruct 权重（bf16）的逐张量实测 (mean, std)**——本机存在真实权重，实测已取得，
> 票面预留的 "UNKNOWN 实测 std 未取得" 分支**不适用**。

---

## 0. TL;DR

| 项 | 结果 |
|---|---|
| 张量数 / 总字节 | 19 个 / 约 332 MB（embed 单独 259.6 MiB，tied，只物化一份） |
| 生成分布 | $N(\mu_t, \sigma_t)$，$(\mu_t,\sigma_t)$ = **真实权重逐张量实测** |
| 实测来源 | \`/mnt/d/ai/Qwen2.5-0.5B-Instruct/model.safetensors\`，bf16，sha256 \`fdf756fa…87fb7fe\` |
| 同 seed 两次生成 | **19/19 张量 sha256 相同**（§6.1） |
| 磁盘产物 vs manifest | **19/19 校验通过**（§6.2） |
| 单比特篡改检测 | 通过（§6.3） |
| 真实权重只做一次性检查（ADR 0003）的风险 | 已复核并收紧：统计量取自真实权重，但**分布 ≠ 数值**，最后一档仍不可省（§8.2） |

---

## 1. 这一票要回答什么

把"**权重从哪来**"这个变量从整条数值对拍链里彻底摘掉，使票 03–10 的任何失配都只能指向实现。具体要求（票面）：

1. 固定 seed 生成，同一 seed 两次生成逐字节相同；
2. 分布为正态，尺度对齐真实权重的实测 std（并记录实测值）；
3. 覆盖单层全部张量：q/k/v（**含 bias**）/o、gate/up/down、两个 RMSNorm 增益、embed/LM head；
4. 每份权重记 sha256，板端与 CPU 侧能各自校验同一份；
5. 生成脚本与 seed 入库，权重产物不入 git；
6. 记录"真实权重只在最后一档做一次性检查"这一取舍及其风险。

张量清单（含 \`layer1.*\`，供票 10 的两层集成用）：

| 夹具名 | shape | dtype | 对应真实张量 |
|---|---|---|---|
| \`embed.weight\`（tied，别名 \`lm_head.weight\`） | [151936, 896] | fp16 | \`model.embed_tokens.weight\` |
| \`layerN.attn.q_proj.weight\` / \`.bias\` | [896, 896] / [896] | fp16 | \`model.layers.N.self_attn.q_proj.*\` |
| \`layerN.attn.k_proj.weight\` / \`.bias\` | [128, 896] / [128] | fp16 | 同上 \`k_proj\` |
| \`layerN.attn.v_proj.weight\` / \`.bias\` | [128, 896] / [128] | fp16 | 同上 \`v_proj\` |
| \`layerN.attn.o_proj.weight\`（**无 bias**） | [896, 896] | fp16 | 同上 \`o_proj.weight\` |
| \`layerN.mlp.gate_proj.weight\` | [4864, 896] | fp16 | \`mlp.gate_proj.weight\` |
| \`layerN.mlp.up_proj.weight\` | [4864, 896] | fp16 | \`mlp.up_proj.weight\` |
| \`layerN.mlp.down_proj.weight\` | [896, 4864] | fp16 | \`mlp.down_proj.weight\` |
| \`layerN.input_layernorm.weight\` | [896] | fp16 | \`input_layernorm.weight\` |
| \`layerN.post_attention_layernorm.weight\` | [896] | fp16 | \`post_attention_layernorm.weight\` |

（N ∈ {0, 1}；两个层**同形状、不同数值**。）

---

## 2. 事前登记：假设与预测（写于生成之前，2026-11-13）

设计已定：\`numpy.random.PCG64\`，**逐张量独立派生 seed**（\`sha256(f"{seed}:{name}")[:8]\`），
正态样本用 **Box–Muller 从 \`Generator.random_raw\` 的 uint64 原始流**变换得到（不走 \`standard_normal\` 的
ziggurat，因为 numpy 对 \`Generator\` 的分布方法**不给跨版本稳定性承诺**，而 PCG64 的原始比特流是承诺稳定的）。
每个张量按行分块生成、边生成边 accumulate float64 统计量并增量喂 sha256。

| # | 预测 | 若预测不成立意味着 |
|---|---|---|
| P1 | 两次独立生成（两个目录），19/19 张量 sha256 相同，\`weights_digest\` 相同 | seed 派生或分块有隐藏状态泄漏（如全局 RNG、分块边界依赖） |
| P2 | 两个目录的 \`.npz\` **文件** sha256 **不同**（zip 成员头带 \`date_time\`），因此校验一律对"数组字节"而非容器字节 | 若相同则 numpy 写 zip 用了固定时间戳——结论不变，但要在本文更正这条事实 |
| P3 | 生成 std 与实测 std 的相对偏差落在有限样本噪声量级：N=128/896 的张量 ≤ 7%，N≥114688 的 ≤ 0.7%，embed ≤ 0.05% | 采样器或缩放写错（如 std 用了方差、或分块错位） |
| P4 | 所有值有限、\`abs_max < 65504\`；\`k_proj.bias\` 的 \`abs_max\` 预计 100–160（实测 std≈32×高斯尾 ≈4.5σ） | fp16 溢出风险，必须换缩放或改口径 |
| P5 | 把任一 fp16 值改动 1 个末位，加载器 sha256 校验必须**失败**（而不是静默通过） | 校验形同虚设 |
| P6 | 删掉一个 \`.npz\` 后 \`verify_all\` 报错并指出缺哪个张量，不静默降级 | 下游会在缺权重时跑出莫名其妙的数值 |

**为什么用"逐张量独立 seed"而不是"一条流顺序切分"**：票 10 之后若要扩到 24 层，只需在清单尾部追加
\`layer2..23\`，已有张量的 sha256 **不变**（下游已引用的哈希不会失效）。顺序流做不到这一点。

---

## 3. 探索性观测：本机是否存在真实 Qwen2.5 权重

> **诚实标注**：这一节是**探索性扫描**，跑之前没有登记预测（任务前提是"本机无真实权重、hf 不可达"）。
> 它是事实采集，不是实验；本文 §2 的预测针对的是**生成实验**，那几个预测确实写在生成之前。

任务前提"本机无真实权重"经实测**不成立**：

- \`curl -m 8 https://huggingface.co\` → HTTP \`000\`（网络确实不可达，这一点与本票前提一致）；
- 但本机磁盘上有真实权重。扫描命令与命中（\`find /mnt/d -maxdepth 6/7\`）：

| 路径 | 说明 |
|---|---|
| **\`/mnt/d/ai/Qwen2.5-0.5B-Instruct/\`** | **本次采用的实测源**（含 \`config.json\`、\`model.safetensors\` 988 MB、\`embedding_fp16.bin\`） |
| \`/mnt/d/n2_ai/edge-ai/models/qwen2.5-0.5b-instruct/\` | 同型号的另一份拷贝（未采用，避免口径分裂） |
| \`/mnt/d/trans_ops_env/qwen3_from_scratch/models/Qwen2.5-0.5B/\` | 同型号的第三份拷贝（未采用） |
| \`/mnt/d/ai/modelscope_qwen3_0.6b_safetensors/\` | Qwen3-0.6B，**不是**本 spec 的模型，不采用 |

\`config.json\` 与 spec 的假设逐项吻合：\`hidden_size=896\`、\`num_attention_heads=14\`、
\`num_key_value_heads=2\`、\`intermediate_size=4864\`、\`vocab_size=151936\`、
\`tie_word_embeddings=true\`、\`rms_norm_eps=1e-6\`、\`rope_theta=1e6\`、\`initializer_range=0.02\`、
\`torch_dtype=bfloat16\`。

权重文件：\`sha256=fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe\`，
290 个张量，全部 \`BF16\`；\`model.embed_tokens.weight\` 为 \`[151936, 896]\`，与 LM head 共享（tied）。

---

## 4. 实测统计量（真实权重，float64 分块累加，std 为总体标准差 ddof=0）

测量脚本：\`tools/gen_qwen25_weights.py --measure --safetensors <path>\`（torch + safetensors，逐张量分块转 float64）。

| 夹具名 | 实测 mean | **实测 std** | 实测 abs_max |
|---|---|---|---|
| embed.weight | 5.14e-05 | **0.015249** | 0.201 |
| layer0.attn.q_proj.weight | -1.69e-05 | **0.066741** | 1.227 |
| layer0.attn.q_proj.bias | -0.15199 | **7.8526** | 79.0 |
| layer0.attn.k_proj.weight | 1.35e-04 | **0.080597** | 1.023 |
| layer0.attn.k_proj.bias | 3.8704 | **32.2177** | 130.0 |
| layer0.attn.v_proj.weight | -8.02e-06 | **0.011472** | 0.121 |
| layer0.attn.v_proj.bias | 1.98e-03 | **0.014815** | 0.098 |
| layer0.attn.o_proj.weight | 1.86e-05 | **0.013129** | 0.563 |
| layer0.mlp.gate_proj.weight | -3.19e-05 | **0.023046** | 0.492 |
| layer0.mlp.up_proj.weight | -2.93e-06 | **0.018255** | 0.279 |
| layer0.mlp.down_proj.weight | -3.32e-06 | **0.018158** | 0.305 |
| layer0.input_layernorm.weight | 0.032157 | **0.092409** | 0.727 |
| layer0.post_attention_layernorm.weight | 0.678170 | **0.150932** | 1.992 |
| layer1.attn.q_proj.weight | 7.28e-05 | **0.028723** | 0.346 |
| layer1.attn.q_proj.bias | 0.14251 | **1.16736** | 10.56 |
| layer1.attn.k_proj.weight | 8.78e-06 | **0.040563** | 0.285 |
| layer1.attn.k_proj.bias | 1.6682 | **18.4200** | 147.0 |
| layer1.attn.v_proj.weight | 5.30e-05 | **0.011528** | 0.083 |
| layer1.attn.v_proj.bias | -0.021304 | **0.240763** | 2.703 |
| layer1.attn.o_proj.weight | -1.55e-05 | **0.014867** | 0.206 |
| layer1.mlp.gate_proj.weight | 1.18e-04 | **0.023273** | 0.412 |
| layer1.mlp.up_proj.weight | 3.51e-06 | **0.017048** | 0.273 |
| layer1.mlp.down_proj.weight | 9.16e-06 | **0.016997** | 0.359 |
| layer1.input_layernorm.weight | 0.684293 | **0.322738** | 3.25 |
| layer1.post_attention_layernorm.weight | 1.006254 | **0.251829** | 1.680 |

**三条对下游有直接后果的观测**：

1. **\`q_proj.bias\` / \`k_proj.bias\` 的量级远超权重**（std 7.9 / 32.2，abs_max 79 / 130），而 \`v_proj.bias\`
   与其他 bias 都在 1e-2 量级。这是 Qwen2.5-0.5B 的真实事实，不是测量错误（\`model.layers.0\` 与
   \`model.layers.1\` 独立复现了同一现象）。后果：**attention 的 q/k 会被 bias 主导**，scores 量级因此远大于
   "只按权重 std=0.02 估"的直觉；票 03/04 的 fp16 动态范围必须按这个量级复核，别用 \`initializer_range\` 估。
2. **两个 RMSNorm 增益不是 1.0**（\`layer0.input_layernorm\` 均值 0.032、\`layer1.input_layernorm\` 均值 0.684，
   而 \`layer0.post_attention_layernorm\` 均值 0.678）。若按"增益 ≈ 1"生成，整块的残差/激活动态会与真实模型
   差一个量级。**因此增益也按实测 (mean, std) 生成**，而不是 \`1 + N(0, 0.02)\`。
3. 训练后的权重 std **大于** \`initializer_range=0.02\`（q_proj 0.067、k_proj 0.081、gate/up 0.023）。
   也就是说，**配置里的 0.02 作为代理口径会系统性偏小 1.1×–4×**。票面给的退路（\`0.02\` 代理）在本机不必要。

---

## 5. 做法：生成器与加载器

### 5.1 采样器（可复现性的关键）

- \`rng = numpy.random.Generator(PCG64(seed_t))\`，\`seed_t = int.from_bytes(sha256(f"{seed}:{name}").digest()[:8], "big")\`；
- 正态样本用 **Box–Muller**：\`u = ((raw >> 11) + 0.5) / 2**53\`（\`raw\` 来自 \`Generator.random_raw\`），
  \`z = sqrt(-2 ln u1) · (cos, sin)(2π u2)\`；
- \`x = float16(mean_t + std_t · z)\`，按行分块（默认 4096 行/块）写入最终缓冲区；
- 统计量与 sha256 都在**存储的 fp16 值**上算（下游读到什么，manifest 就描述什么）。

**为什么不用 \`standard_normal\`**：numpy 明确不给 \`Generator\` 的分布方法跨版本稳定性承诺（"the binary stream
may change"），而 \`PCG64\` 的原始 uint64 流是承诺稳定的。Box–Muller 把 numpy 版本的影响限制在 \`log/sqrt/cos/sin\`
上（\`sqrt\` 是 IEEE 精确的，\`log/cos/sin\` 依赖 libm）。**跨机器逐位一致仍不保证**，见 §11 剩余疑问。

### 5.2 产物与清单

\`\`\`
weights/qwen25_frozen/
  manifest.json            # 入库
  <tensor name>.npz        # 一个张量一个 npz，zip 内 key = 张量名；不入 git
\`\`\`

\`manifest.json\` 顶层：\`schema\`、\`seed\`、\`seed_derivation\`、\`bit_generator\`、\`sampler\`、\`dtype\`、\`layout\`、
\`hash_of\`、\`generated_at_utc\`、\`generator_command\`、\`environment\`（python/numpy/platform）、
\`generator\`/\`loader\`（脚本路径 + sha256）、\`model\`（config 摘要 + **实测源 safetensors 路径/sha256/dtype**）、
\`stats_method\`、\`total_bytes\`、\`weights_digest\`、\`tied\`、\`tensors[]\`。
每个张量：\`name/shape/dtype/bytes/sha256/std/mean/abs_max/file/npz_key/aliases/source_tensor/measured\`。

**\`sha256\` 的定义是"数组原始字节"**（C order、little-endian fp16、contiguous），**不是 \`.npz\` 容器字节**——
容器是 zip，成员头带写入时刻的 \`date_time\`，两次生成容器哈希必然不同（P2 已实测确认）。

### 5.3 加载器接口（下游唯一契约）

\`\`\`python
import sys; sys.path.insert(0, "<repo>/tools")
import qwen25_weights as qw

w = qw.get("layer0.attn.q_proj.weight")      # np.float16 [896,896]，只读，加载即校验 sha256
b = qw.get("lm_head.weight")                 # 与 embed.weight 是同一份数组（tied 别名，不重复占内存）
qw.get_fp32("embed.weight")                  # 便捷：float32 副本（numpy golden 用）
qw.verify_all()                              # -> {"ok": bool, "checked": 19, "errors": [...]}
\`\`\`

目录解析顺序：\`参数 path\` → 环境变量 \`QWEN25_WEIGHTS_DIR\` → \`<repo>/weights/qwen25_frozen\`。
下游 worktree 里没有产物时，用 \`QWEN25_WEIGHTS_DIR\` 指向已生成的那份（**或自己跑一次生成器，sha256 相同**）。

---

## 6. 结果（待填）

### 6.1 P1/P2：同 seed 两次生成

### 6.2 磁盘产物 vs manifest

### 6.3 P5/P6：篡改与缺失检测

### 6.4 P3/P4：尺度和 fp16 值域

---

## 7. 与票面 checklist 对照（待填）

---

## 8. 取舍与风险

### 8.1 \`tie_word_embeddings=true\`：只物化一份（约 259.6 MiB / 272,269,312 B）

- **取舍**：\`embed.weight\` 只存一份，\`lm_head.weight\` 作为 \`aliases\` 指向同一份数组/文件。理由：tied 本来就是
  "一个张量两个用途"，物化两份会让"两份是否需要一致"变成一个可以被静默写错的自由度；也省 272 MB 磁盘/内存/push 时间。
- **风险**：(a) 若将来换 untied 权重（tie=false），别名机制会**掩盖**"其实需要两份独立张量"这件事 →
  只允许在真权重档改口径，且必须改 \`manifest.tied\`；(b) 别名使"实现里 LM head 用了转置/不同布局"这类错误
  更难被发现（同一份数组两种用法都合法）；(c) 一份 272 MB 的张量在 \`.tflite/.dla\` 与板端加载里是真实负担
  （\`docs/OP-block.md\` §设备容量：RAM 空闲可能只有 ~273 MB），票 08 必须实测。

### 8.2 "真实权重只在最后一档做一次性检查"（ADR 0003）的风险

- **收益**：随机固定权重让失配只能指向实现——票 03–10 的每一步都对"同一份数值"负责。
- **风险（本次收紧了一半，剩下一半仍在）**：
  1. ~~分布口径靠猜~~ → **已消除**：本票直接从真实权重实测 (mean, std, abs_max) 生成（§4）；
  2. **分布 ≠ 数值**：真实权重的相关结构（头间的 q/k bias 结构、embed 的 token 频率长尾、少数极大值）
     随机权重没有；因此"随机权重 PASS"**不能**推出"真权重下 logits top-5 有交集"；
  3. **真实权重特有的搬运/格式问题推迟到最后**：bf16→fp16 的舍入、safetensors 布局 → 板端权重文件、
     vocab 151936 的 272 MB 权重、tied 张量在 \`.dla\` 里只出现一次——这些在本票里**只被记录，未被验证**；
  4. **最后一档失败时定位面很大**：一次性检查不逐层 dump，失败只能二分（整块 vs 管线）。缓解办法是把真权重
     也包装成同一套张量名，让 §5.3 的加载器接口可以原样消费真权重（本票未实现，见 §11）。
- **结论**：这条取舍仍然成立（对拍要的是"同一份数值"），但不能拿它当"实现对 Qwen2.5"的证据。语义锚点仍然只能落在
  最后一次真权重检查上。

---

## 9. 适用范围 / 不适用

**适用**：票 03–10 的全部数值对拍（板端 + numpy golden 用同一份张量字节）；两层集成的权重来源。

**不适用**：任何语义/精度结论（top-1、困惑度、生成质量）；RoPE 的 sin/cos 表（票 03 自算，不在本夹具内）；
tokenizer / embedding 的语义；24 层中 \`layer2..23\` 的权重（本票只造 2 层，按 §2 的 seed 派生规则可无损追加）；
int8 路径（全 fp16）。

---

## 10. 复现

\`\`\`bash
# 生成（WSL 的 /usr/bin/python3，需要 numpy；只写 weights/qwen25_frozen/）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t02' && /usr/bin/python3 tools/gen_qwen25_weights.py"

# 同 seed 是否逐字节相同（重新生成到内存，与 manifest 里的 sha256 逐一比对）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t02' && /usr/bin/python3 tools/gen_qwen25_weights.py --verify"

# 磁盘产物是否与 manifest 一致（缺文件/字节被改都会报）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t02' && /usr/bin/python3 tools/qwen25_weights.py --verify"

# 重新实测真实权重统计量（torch + safetensors）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t02' && /usr/bin/python3 tools/gen_qwen25_weights.py --measure --safetensors /mnt/d/ai/Qwen2.5-0.5B-Instruct/model.safetensors"
\`\`\`

---

## 11. 剩余疑问（UNKNOWN，不圆）

1. **跨机器/跨 libm 的逐位一致未验证**：Box–Muller 用到 \`log/cos/sin\`，不同 libm/numpy 版本可能差最后一位。
   本机能保证"同 seed 两次生成相同"（§6.1）；**跨机器只保证"sha256 一比对就知道是否相同"**，不保证相同。
   → 实践口径：**共享同一份产物目录**（或同机重生成 + \`--verify\`），不要跨机各生成各的。
2. **真权重档的 dtype 路径未验证**：源是 bf16，夹具是 fp16；"bf16 值 → fp16 存储"的舍入行为在真权重档才第一次
   真正发生（本票只是绕开了它）。
3. **\`k_proj.bias\` std=32 的 fp16 语义**：真权重档下 q/k 由 bias 主导，attention scores 量级比"权重口径"大一个数量级；
   夹具已如实复刻这一量级，但**板端 MDLA 的 fp16 累加器是否在这量级上丢精度，本票无法回答**（要票 03/04 的板端证据）。
4. **\`embed\` 272 MB 的单份物化在板端的真实占用/加载时间未测**（要票 08）。
5. **命名契约的冻结范围**：19 个张量名/shape/dtype/sha256 从本票起视为**冻结**（下游已引用）；追加只允许
   **在清单尾部加新张量**（如 \`layer2.*\`），不得改名或改已生成张量的 seed 派生规则。
