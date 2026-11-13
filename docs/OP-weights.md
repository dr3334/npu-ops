# OP-weights —— 冻结随机权重夹具（Qwen2.5-0.5B 真实形状 · fp16）

> 范围：**权重夹具本身**（生成 / 冻结 / 校验 / 下游加载接口）。**本票不动板端、不产生 `ops/` 目录**，
> 因此没有 `model.compile.log` / `compare.txt` / 噪声底——那三样属于消费这些权重的算子票（03–10）。
> 本票的判据是：**同一 seed 两次生成逐字节相同**、**每份权重 sha256 可被双方独立校验**、**生成分布的尺度对齐真实权重实测统计量**。
>
> · 分支 `impl/02-frozen-weights`（worktree `wt-t02`）
> · 生成器 `tools/gen_qwen25_weights.py` · 加载器 `tools/qwen25_weights.py`
> · 产物 `weights/qwen25_frozen/`（`manifest.json` 入库；`*.npz` 按 `.gitignore` 不入库，可复现）
> · seed `20261113` · dtype `float16` · **25 个张量**（一层 decoder 12 个 ×2 + tied embed/LM head）
> · 生成耗时 8.0 s / 校验 5.5 s（WSL `/usr/bin/python3` + numpy 2.2.6，20 核 / 13 GB）
>
> **一句话结论**：夹具已冻结并自校验通过；生成尺度的来源**不是**配置里的 `initializer_range=0.02` 代理口径，
> 而是本机**真实 Qwen2.5-0.5B-Instruct 权重（bf16）的逐张量实测 (mean, std)**——本机存在真实权重、实测已取得，
> 票面预留的 "UNKNOWN 实测 std 未取得" 分支**不适用**（见 §3）。

---

## 0. TL;DR

| 项 | 结果 |
|---|---|
| 张量数 / 总字节 | **25 个** / **331,918,848 B**（316.6 MiB）；`embed.weight` 单独 272,269,312 B（259.6 MiB），tied，只物化一份 |
| `weights_digest` | `898fa3195d627a4ce055cbdb6fc762a12e74665b431e3ae09b530b4abdad18a4` |
| 生成分布 | $N(\mu_t,\sigma_t)$，$(\mu_t,\sigma_t)$ = **真实权重逐张量实测**（§4） |
| 实测来源 | `/mnt/d/ai/Qwen2.5-0.5B-Instruct/model.safetensors`，bf16，sha256 `fdf756fa…87fb7fe` |
| 同 seed 两次生成（**两个目录**） | **25/25 张量 sha256 相同**、`weights_digest` 相同、**25/25 `.npz` 容器字节也相同**（§6.1） |
| `--verify`（同机重新生成 vs manifest） | **25/25 OK**，5.5 s（§6.2） |
| 磁盘产物 vs manifest（`qwen25_weights.py --verify`） | **25/25 + 1 个别名 PASS**（§6.3） |
| 篡改 1 ULP / 删文件 / 未知张量名 | 三档**全部按预期报错**，不静默（§6.4） |
| 预测复核 | P1/P5/P6 成立，P3 有 1 项轻微越界，P4 部分成立，**P2 被推翻**（§6.5，已更正 §5.2 的事实陈述） |
| 票面 checklist | 6/6（§7） |

---

## 1. 这一票要回答什么

把"**权重从哪来**"这个变量从整条数值对拍链里彻底摘掉，使票 03–10 的任何失配都只能指向实现。票面要求：

1. 固定 seed 生成，同一 seed 两次生成逐字节相同；
2. 分布为正态，尺度对齐真实权重的实测 std（并记录实测值）；
3. 覆盖单层全部张量：q/k/v（**含 bias**）/o、gate/up/down、两个 RMSNorm 增益、embed/LM head；
4. 每份权重记 sha256，板端与 CPU 侧能各自校验同一份；
5. 生成脚本与 seed 入库，权重产物不入 git；
6. 记录"真实权重只在最后一档做一次性检查"这一取舍及其风险。

张量清单（含 `layer1.*`，供票 10 的两层集成用；共 25 个）：

| 夹具名 | shape | dtype | 对应真实张量 |
|---|---|---|---|
| `embed.weight`（tied，别名 `lm_head.weight`） | [151936, 896] | fp16 | `model.embed_tokens.weight` |
| `layerN.attn.q_proj.weight` / `.bias` | [896, 896] / [896] | fp16 | `model.layers.N.self_attn.q_proj.*` |
| `layerN.attn.k_proj.weight` / `.bias` | [128, 896] / [128] | fp16 | 同上 `k_proj` |
| `layerN.attn.v_proj.weight` / `.bias` | [128, 896] / [128] | fp16 | 同上 `v_proj` |
| `layerN.attn.o_proj.weight`（**无 bias**） | [896, 896] | fp16 | 同上 `o_proj.weight` |
| `layerN.mlp.gate_proj.weight` | [4864, 896] | fp16 | `mlp.gate_proj.weight` |
| `layerN.mlp.up_proj.weight` | [4864, 896] | fp16 | `mlp.up_proj.weight` |
| `layerN.mlp.down_proj.weight` | [896, 4864] | fp16 | `mlp.down_proj.weight` |
| `layerN.input_layernorm.weight` | [896] | fp16 | `input_layernorm.weight` |
| `layerN.post_attention_layernorm.weight` | [896] | fp16 | `post_attention_layernorm.weight` |

（N ∈ {0, 1}；两层**同形状、不同数值**。`o_proj` 无 bias、q/k/v 有 bias 这一事实来自 transformers 源码而非
`config.json`——它没有对应键，漏掉会静默改变数值。）

---

## 2. 事前登记：假设与预测（写于生成之前，2026-11-13）

设计已定：`numpy.random.PCG64`，**逐张量独立派生 seed**（`sha256(f"{seed}:{name}")[:8]`），
正态样本用 **Box–Muller 从 PCG64 的 raw uint64 流**变换得到（不走 `standard_normal` 的 ziggurat，因为 numpy 对
`Generator` 的分布方法**不给跨版本稳定性承诺**，而 PCG64 的原始比特流是承诺稳定的）。
每个张量按行分块生成、边生成边按 float64 累加统计量并增量喂 sha256。

| # | 预测 | 若预测不成立意味着 |
|---|---|---|
| P1 | 两次独立生成（两个目录），25/25 张量 sha256 相同，`weights_digest` 相同（登记时误数为 19，实际清单 25 个，见 §6.5） | seed 派生或分块有隐藏状态泄漏（如全局 RNG、分块边界依赖） |
| P2 | 两个目录的 `.npz` **文件** sha256 **不同**（zip 成员头带 `date_time`），因此校验一律对"数组字节"而非容器字节 | 若相同则 numpy 写 zip 用了固定时间戳——结论不变，但要在本文更正这条事实 |
| P3 | 生成 std 与实测 std 的相对偏差落在有限样本噪声量级：N=128/896 的张量 ≤ 7%，N≥114688 的 ≤ 0.7%，embed ≤ 0.05% | 采样器或缩放写错（如 std 用了方差、或分块错位） |
| P4 | 所有值有限、`abs_max < 65504`；`k_proj.bias` 的 `abs_max` 预计 100–160（实测 std≈32 × 高斯尾 ≈4.5σ） | fp16 溢出风险，必须换缩放或改口径 |
| P5 | 把任一 fp16 值改动 1 个末位，加载器 sha256 校验必须**失败**（而不是静默通过） | 校验形同虚设 |
| P6 | 删掉一个 `.npz` 后 `verify_all` 报错并指出缺哪个张量，不静默降级 | 下游会在缺权重时跑出莫名其妙的数值 |

**为什么用"逐张量独立 seed"而不是"一条流顺序切分"**：票 10 之后若要扩到 24 层，只需在清单尾部追加
`layer2..23`，已有张量的 sha256 **不变**（下游已引用的哈希不会失效）。顺序流做不到这一点。

---

## 3. 探索性观测：本机是否存在真实 Qwen2.5 权重

> **诚实标注**：这一节是**探索性扫描**，跑之前没有登记预测（任务前提是"本机无真实权重、hf 不可达"）。
> 它是事实采集，不是实验；§2 的预测针对的是**生成实验**，那几个预测确实写在生成之前。

任务前提"本机无真实权重"经实测**不成立**：

- `curl -m 8 https://huggingface.co` → HTTP `000`（网络确实不可达，与前提一致）；
- 但本机磁盘上有真实权重。扫描命令与命中（`find /mnt/d -maxdepth 6/7 -iname ...`）：

| 路径 | 说明 |
|---|---|
| **`/mnt/d/ai/Qwen2.5-0.5B-Instruct/`** | **本次采用的实测源**（`config.json`、`model.safetensors` 988 MB、`embedding_fp16.bin`） |
| `/mnt/d/n2_ai/edge-ai/models/qwen2.5-0.5b-instruct/` | 同型号的另一份拷贝（未采用，避免口径分裂） |
| `/mnt/d/trans_ops_env/qwen3_from_scratch/models/Qwen2.5-0.5B/` | 同型号的第三份拷贝（未采用） |
| `/mnt/d/ai/modelscope_qwen3_0.6b_safetensors/` | Qwen3-0.6B，**不是**本 spec 的模型，不采用 |

`config.json` 与 spec 的假设逐项吻合：`hidden_size=896`、`num_attention_heads=14`、
`num_key_value_heads=2`、`intermediate_size=4864`、`vocab_size=151936`、
`tie_word_embeddings=true`、`rms_norm_eps=1e-6`、`rope_theta=1e6`、`initializer_range=0.02`、
`torch_dtype=bfloat16`。

权重文件：`sha256=fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`，
290 个张量全部 `BF16`；`model.embed_tokens.weight` 为 `[151936, 896]`，与 LM head 共享（tied）。
生成器把这份来源的路径/sha256/`config.json` 的 sha256 全部写进 `manifest.json` 的
`model.source_weights`，可追溯。

---

## 4. 实测统计量（真实权重，float64 分块累加，std 为总体标准差 ddof=0）

测量入口：`tools/gen_qwen25_weights.py --measure --safetensors <path>`（torch + safetensors，
逐张量分块转 float64；重跑会打印与脚本内嵌 `MEASURED` 的差值）。
实测：用当前脚本重跑 `--measure`，25 项全部 `dmean=0.00e+00, dstd=0.00e+00`（脚本内嵌值与 safetensors 实测逐位一致）。

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

1. **`q_proj.bias` / `k_proj.bias` 的量级远超权重**（std 7.9 / 32.2，abs_max 79 / 130），而 `v_proj.bias`
   与其他 bias 都在 1e-2 量级。这是 Qwen2.5-0.5B 的真实事实，不是测量错误（layer0 与 layer1 独立复现了同一现象）。
   后果：**attention 的 q/k 会被 bias 主导**，scores 量级远大于"按权重 std=0.02 估"的直觉；
   票 03/04 的 fp16 动态范围必须按这个量级复核，别用 `initializer_range` 估。
2. **两个 RMSNorm 增益不是 1.0**（`layer0.input_layernorm` 均值 0.032、`layer1.input_layernorm` 均值 0.684、
   `layer0.post_attention_layernorm` 均值 0.678）。若按"增益 ≈ 1"生成，整块的残差/激活动态会与真实模型
   差一个量级。**因此增益也按实测 (mean, std) 生成**，而不是 `1 + N(0, 0.02)`。
3. 训练后权重的 std **大于** `initializer_range=0.02`（q_proj 0.067、k_proj 0.081、gate/up 0.023）：
   用 0.02 当代理口径会**系统性偏小 1.1×–4×**。票面给的退路在本机不必要。

---

## 5. 做法：生成器与加载器

### 5.1 采样器（可复现性的关键）

- `rng = numpy.random.Generator(PCG64(seed_t))`，`seed_t = int.from_bytes(sha256(f"{seed}:{name}").digest()[:8], "big")`；
- 正态样本用 **Box–Muller**：`u = ((raw >> 11) + 0.5) / 2**53`（`raw` 来自
  `Generator.bit_generator.random_raw`），`z = sqrt(-2 ln u1) · (cos, sin)(2π u2)`；
- `x = float16(mean_t + std_t · z)`，按行分块（默认每块 ≤ 4 Mi 元素）写入最终缓冲区；
- 统计量与 sha256 都在**存储的 fp16 值**上算（下游读到什么，manifest 就描述什么），
  sha256 在生成过程中增量更新，避免再复制一遍 272 MB 缓冲。

**为什么不用 `standard_normal`**：numpy 明确不给 `Generator` 的分布方法跨版本稳定性承诺
（"the binary stream may change"），而 `PCG64` 的原始 uint64 流是承诺稳定的。Box–Muller 把 numpy 版本的影响限制在
`log/sqrt/cos/sin` 上（`sqrt` 是 IEEE 精确的，其余依赖 libm）。**跨机器逐位一致仍不保证**，见 §11。

### 5.2 产物与清单

```
weights/qwen25_frozen/
  manifest.json            # 入库
  README.md                # 入库（怎么生成、怎么用、契约）
  <tensor name>.npz        # 一张量一 npz，zip 内 key = 张量名；不入 git
```

`manifest.json` 顶层：`schema/version/seed/seed_derivation/bit_generator/sampler/dtype/layout/hash_of/
stats_method/generated_at_utc/generator_command/environment/generator/loader/model/tied/tensor_count/
total_bytes/weights_digest/tensors[]`。
每个张量：`name/shape/dtype/bytes/sha256/std/mean/abs_max/file/npz_key/npz_file_sha256/aliases/
source_tensor/measured{mean,std,abs_max,dtype}`。

**`sha256` 的定义是"数组原始字节"**（C order、little-endian fp16、contiguous），**不是 `.npz` 容器字节**——
容器是 zip，其字节取决于 numpy 版本的 zip 写法。实测（§6.1）numpy 2.2.6 的 `np.savez` 把成员
`date_time` 钉成 `(1980,1,1,0,0,0)`，所以本轮连容器也逐字节相同；但这是**实现细节、不是契约**，
因此 `npz_file_sha256` 只作为仪表记录，校验一律走数组字节。

### 5.3 加载器接口（下游唯一契约）

```python
import sys; sys.path.insert(0, "<repo>/tools")
import qwen25_weights as qw

w = qw.get("layer0.attn.q_proj.weight")      # np.float16 [896,896]，只读，加载即校验 sha256
b = qw.get("lm_head.weight")                 # 与 embed.weight 是同一份数组（tied 别名，不重复占内存）
qw.get_fp32("embed.weight")                  # 便捷：float32 副本（numpy golden 用）
qw.verify_all()                              # -> {"ok": True, "checked": 25, "aliases": 1, ...}
```

目录解析顺序：显式参数 → 环境变量 `QWEN25_WEIGHTS_DIR` → `<repo>/weights/qwen25_frozen`。
下游 worktree 里没有产物时，用 `QWEN25_WEIGHTS_DIR` 指向已生成的那份（**或自己跑一次生成器，sha256 相同**）。

---

## 6. 结果

### 6.1 P1/P2：同 seed 两次生成（两个目录）

```bash
/usr/bin/python3 tools/gen_qwen25_weights.py                      # -> weights/qwen25_frozen（本轮：8.0 s）
/usr/bin/python3 tools/gen_qwen25_weights.py --out /tmp/qwen25_frozen_b --quiet   # 第二轮（6.7 s）
# 比对 manifest 的逐张量 sha256
/usr/bin/python3 - <<'PY'
import json
a = json.load(open("weights/qwen25_frozen/manifest.json"))
b = json.load(open("/tmp/qwen25_frozen_b/manifest.json"))
ta = {t["name"]: t["sha256"] for t in a["tensors"]}
tb = {t["name"]: t["sha256"] for t in b["tensors"]}
print("tensor names equal:", sorted(ta) == sorted(tb))
print("tensor sha256 equal: %d/%d" % (sum(ta[k] == tb[k] for k in ta), len(ta)))
print("weights_digest equal:", a["weights_digest"] == b["weights_digest"])
print("total_bytes equal:", a["total_bytes"] == b["total_bytes"])
PY
# 比对 .npz 容器字节
cd weights/qwen25_frozen && sha256sum *.npz | sort > /tmp/a.sha
cd /tmp/qwen25_frozen_b && sha256sum *.npz | sort > /tmp/b.sha && diff /tmp/a.sha /tmp/b.sha
```

结果：

```
tensor names equal: True
tensor sha256 equal: 25/25
weights_digest equal: True 898fa3195d627a4c
total_bytes equal: True 331918848
npz 容器 sha256：25 个文件 diff 为空 -> IDENTICAL
```

→ **P1 成立**。两轮是**两个独立进程、两个不同目录**（不是同一进程跑两次），因此这条证据也覆盖了"进程内全局
RNG 串味"这一类失败模式。
→ **P2 被推翻**：容器字节也相同（原因见 §5.2）。这条不影响契约（契约本来就定义在数组字节上），
但"两次生成容器必然不同"这句话是错的，已更正。

### 6.2 同机重新生成 vs manifest（`--verify`）

```
$ time /usr/bin/python3 tools/gen_qwen25_weights.py --verify
  OK  embed.weight                               40cdb3488dedade2
  ...（25 行）
--verify OK: 25/25 tensors byte-identical on regeneration,
            weights_digest=898fa3195d627a4ce055cbdb6fc762a12e74665b431e3ae09b530b4abdad18a4
real 0m5.5s
```

### 6.3 磁盘产物 vs manifest（`qwen25_weights.py --verify`）

```
fixture: /mnt/d/deepseek_ workspace/mtk/wt-t02/weights/qwen25_frozen
tensors verified : 25/25
aliases verified : 1
total bytes      : 331918848
weights_digest   : 898fa3195d627a4ce055cbdb6fc762a12e74665b431e3ae09b530b4abdad18a4
RESULT: PASS
```

（另测：`alias lm_head.weight` 与 `embed.weight` 返回**同一个对象** —— 见 §6.4 第 5 条。）

### 6.4 P5/P6：篡改、缺文件、未知名（`qwen25_weights.py --selftest`）

自测在临时目录里只放 manifest + 一个 896 元素的张量，永不触碰真夹具：

```
OK   clean copy loads and verifies sha256           no exception
OK   1-ULP mutation is rejected (sha256)            ... sha256 e5fa6cd0f0fcbd0a… != manifest b24191b2589f6662…
OK   missing .npz is rejected (not silent)          missing layer0.attn.v_proj.bias.npz for tensor 'layer0.attn.v_proj.bias'
OK   unknown tensor name is rejected                unknown tensor 'layer0.attn.nope.weight'; known: … (aliases: lm_head.weight)
OK   alias lm_head.weight resolves to embed.weight  no exception
RESULT: PASS
```

→ **P5、P6 成立**。篡改是"把 `layer0.attn.v_proj.bias` 的第 0 个元素改 1 个 fp16 ULP"，
加载器给出的错误里带期望/实际 sha256，能直接定位到是哪个文件被改。

### 6.5 预测复核

| # | 预测 | 结果 | 说明 |
|---|---|---|---|
| P1 | 25/25 sha256 相同 | ✅ 成立 | §6.1（同时容器也相同） |
| P2 | 两目录 `.npz` 容器 sha256 不同 | ❌ **被推翻** | numpy 2.2.6 的 `np.savez` 把 zip 成员 `date_time` 钉成 1980-01-01，容器因此也逐字节相同；契约仍定义在数组字节上，§5.2 已更正 |
| P3 | N=128/896 ≤ 7%、N≥114688 ≤ 0.7%、embed ≤ 0.05% | ⚠️ 基本成立（1 项越界） | 最大偏差 **-7.57%**（`layer0.attn.k_proj.bias`，N=128）；其余 24 项全在界内（N=896 的最大 3.18%，N=114688 的 0.27%，embed 0.003%）。N=128 的有限样本噪声本身是 $1/\sqrt{2N}=6.25\%$，7.57% = 1.2σ，是抽样波动而非缩放错误 |
| P4 | 全有限、`abs_max<65504`、k_proj.bias abs_max 100–160 | ⚠️ 部分成立 | 全有限、最大 `abs_max` 77.8（无溢出）；但 k_proj.bias abs_max 落 77.8，未进 100–160 —— 预测区间是按**真实权重**的 4.04σ 外推的，N=128 时 `max|z|` 的中位数只有 ~2.9σ，**外推错了**（已记入 §11） |
| P5 | 1 ULP 篡改必须失败 | ✅ 成立 | §6.4 |
| P6 | 缺文件必须报错 | ✅ 成立 | §6.4 |

### 6.6 尺度（生成的 std/mean 与实测对照）

25 项相对偏差（生成 std vs 实测 std）：最大 `-7.57%`、次大 `+3.18%`、`+2.75%`、`+2.31%`，
其余 `<2%`；embed `-0.003%`。全部与有限样本噪声量级一致（N=896 → 2.4%，N=128 → 6.25%，
N=4358144 → 0.03%），**没有系统性偏移**（正负各半）。

生成结果一览（manifest 摘录；完整 sha256 见 `weights/qwen25_frozen/manifest.json`）：

| 张量 | shape | bytes | std（生成） | mean（生成） | abs_max（生成） | sha256[:16] |
|---|---|---|---|---|---|---|
| `embed.weight` | 151936×896 | 272269312 | 0.0152491 | 5.128e-05 | 0.09088 | `40cdb3488dedade2` |
| `layer0.attn.q_proj.weight` | 896×896 | 1605632 | 0.0667188 | 2.701e-05 | 0.3433 | `2f44980c8aa70f21` |
| `layer0.attn.q_proj.bias` | 896 | 1792 | 7.79632 | -0.402 | 25.22 | `74fdf6b1080d83f1` |
| `layer0.attn.k_proj.weight` | 128×896 | 229376 | 0.0803831 | -0.0001763 | 0.3538 | `b4cc1bae49eaafff` |
| `layer0.attn.k_proj.bias` | 128 | 256 | 29.7784 | 6.127 | 77.81 | `b028a7fd086cf570` |
| `layer0.attn.v_proj.weight` | 128×896 | 229376 | 0.0114265 | 7.128e-05 | 0.05078 | `8aefa0240b2f3910` |
| `layer0.attn.v_proj.bias` | 128 | 256 | 0.0149563 | 0.001224 | 0.04279 | `b24191b2589f6662` |
| `layer0.attn.o_proj.weight` | 896×896 | 1605632 | 0.0131301 | 2.651e-06 | 0.06403 | `9e1e336283c02472` |
| `layer0.mlp.gate_proj.weight` | 4864×896 | 8716288 | 0.0230507 | -4.119e-05 | 0.1249 | `8de49b84e98bc457` |
| `layer0.mlp.up_proj.weight` | 4864×896 | 8716288 | 0.0182558 | -5.541e-06 | 0.08893 | `4fd7040fd720e7e7` |
| `layer0.mlp.down_proj.weight` | 896×4864 | 8716288 | 0.0181601 | 1.845e-05 | 0.08942 | `7125c503e155a95b` |
| `layer0.input_layernorm.weight` | 896 | 1792 | 0.09395 | 0.03417 | 0.385 | `e8163ff61aeabdb7` |
| `layer0.post_attention_layernorm.weight` | 896 | 1792 | 0.151377 | 0.6854 | 1.185 | `29f21d75ce3c78e1` |
| `layer1.attn.q_proj.weight` | 896×896 | 1605632 | 0.0287037 | -5.27e-06 | 0.1504 | `b3adcd3dcf165ae1` |
| `layer1.attn.q_proj.bias` | 896 | 1792 | 1.19945 | 0.1168 | 4.855 | `1c306f524828ca56` |
| `layer1.attn.k_proj.weight` | 128×896 | 229376 | 0.0404939 | -3.902e-05 | 0.2084 | `69a00fd687eff298` |
| `layer1.attn.k_proj.bias` | 128 | 256 | 19.0047 | 0.5579 | 42.81 | `f3cc7ac5d3b7ae83` |
| `layer1.attn.v_proj.weight` | 128×896 | 229376 | 0.0115309 | 8.942e-05 | 0.05014 | `bcfd98b203c12bba` |
| `layer1.attn.v_proj.bias` | 128 | 256 | 0.246314 | 0.026 | 0.7222 | `2d9666ec844249ab` |
| `layer1.attn.o_proj.weight` | 896×896 | 1605632 | 0.014882 | 8.306e-06 | 0.07501 | `7da6edd2ac784268` |
| `layer1.mlp.gate_proj.weight` | 4864×896 | 8716288 | 0.0232693 | 0.0001243 | 0.1172 | `37eaf7f390976915` |
| `layer1.mlp.up_proj.weight` | 4864×896 | 8716288 | 0.0170502 | 3.972e-06 | 0.08594 | `e71f73600db15bcf` |
| `layer1.mlp.down_proj.weight` | 896×4864 | 8716288 | 0.0169882 | 4.565e-06 | 0.08563 | `b3d423dee62341c6` |
| `layer1.input_layernorm.weight` | 896 | 1792 | 0.328656 | 0.667 | 1.692 | `04cd077280137098` |
| `layer1.post_attention_layernorm.weight` | 896 | 1792 | 0.254421 | 0.9989 | 1.733 | `8cc7f1812638d621` |

---

## 7. 与票面 checklist 对照

| 票面要求 | 状态 | 证据 |
|---|---|---|
| 固定 seed 生成，同一 seed 两次生成**逐字节相同** | ✅ | §6.1 两个目录 25/25 sha256 相同 + 容器相同；§6.2 `--verify` 25/25 |
| 分布为正态，尺度对齐真实权重**实测 std**（并记录实测值） | ✅ | §4 实测表（25 项）；§6.6 偏差全在抽样噪声量级；脚本内嵌 `MEASURED` 可用 `--measure` 复测 |
| 覆盖单层全部张量（q/k/v 含 bias、o、gate/up/down、两个 RMSNorm 增益、embed/LM head） | ✅ | §1 清单 25 个；`o_proj` 确认无 bias；tied LM head 以别名交付 |
| 每份权重记 sha256，板端与 CPU 侧能各自校验同一份 | ✅ | manifest 内 25 条 sha256 + `weights_digest`；加载器加载即校验（§6.3、§6.4） |
| 生成脚本与 seed 入库，权重产物不入 git | ✅ | `tools/gen_qwen25_weights.py`（seed 默认 20261113 写在脚本里）+ `.gitignore` 只新增 `*.npz` |
| 记录"真权重只做一次性检查"的取舍与风险 | ✅ | §8.2 |

**本票不产生** `ops/` 目录、`compare.txt`、`model.compile.log`、噪声底——任务书明确"本票不需要上板、不需要动
`ops/` "。判据由 §6.1–6.4 的确定性/完整性证据承担。

---

## 8. 取舍与风险

### 8.1 `tie_word_embeddings=true`：只物化一份（259.6 MiB / 272,269,312 B）

- **取舍**：`embed.weight` 只存一份，`lm_head.weight` 作为 `aliases` 指向同一份数组/文件。理由：tied 本来就是
  "一个张量两个用途"，物化两份会让"两份是否需要一致"变成一个可以被静默写错的自由度；也省 272 MB 磁盘/内存/push 时间。
- **风险**：(a) 若将来换 untied 权重（`tie=false`），别名机制会**掩盖**"其实需要两份独立张量"这件事 →
  只允许在真权重档改口径，且必须改 `manifest.tied`；(b) 别名使"实现里 LM head 用了转置/不同布局"这类错误
  更难被发现（同一份数组两种用法都合法）；(c) 一份 272 MB 的张量在 `.tflite/.dla` 与板端加载里是真实负担
  （`docs/OP-block.md` 的设备容量一节：RAM 空闲可能只有 ~273 MB），票 08 必须实测。

### 8.2 "真实权重只在最后一档做一次性检查"（ADR 0003）的风险

- **收益**：随机固定权重让失配只能指向实现——票 03–10 的每一步都对"同一份数值"负责。
- **风险（本次收紧了一半，剩下一半仍在）**：
  1. ~~分布口径靠猜~~ → **已消除**：本票直接从真实权重实测 (mean, std) 生成（§4）；
  2. **分布 ≠ 数值**：真实权重的相关结构与极值尾巴随机权重没有（例如 `k_proj.bias` 真权重
     abs_max=130=4.04σ，夹具 77.8=2.6σ）；因此"随机权重 PASS"**不能**推出"真权重下 logits top-5 有交集"；
  3. **真权重特有的搬运/格式问题推迟到最后**：bf16→fp16 的舍入、safetensors 布局 → 板端权重文件、
     vocab 151936 的 272 MB 权重、tied 张量在 `.dla` 里只出现一次——本票**只记录，未验证**；
  4. **最后一档失败时定位面很大**：一次性检查不逐层 dump，失败只能二分（整块 vs 管线）。缓解思路是把真权重
     也包装成同一套张量名，让 §5.3 的加载器接口原样消费真权重（本票未实现，见 §11）。
- **结论**：这条取舍仍然成立（对拍要的是"同一份数值"），但不能拿它当"实现对 Qwen2.5"的证据；语义锚点仍只能落在
  最后一次真权重检查上。

---

## 9. 适用范围 / 不适用

**适用**：票 03–10 的全部数值对拍（板端 + numpy golden 用同一份张量字节）；两层集成的权重来源。

**不适用**：任何语义/精度结论（top-1、困惑度、生成质量）；RoPE 的 sin/cos 表（票 03 自算，不在本夹具内）；
tokenizer / embedding 的语义；24 层中 `layer2..23`（本票只造 2 层，按 §2 的 seed 派生规则可无损追加）；
int8 路径（全 fp16）。

---

## 10. 复现

```bash
# 0) 环境：WSL 的 /usr/bin/python3（有 numpy），路径含空格必须加引号
# 1) 生成（约 8 s，332 MB；只写 weights/qwen25_frozen/）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t02' && /usr/bin/python3 tools/gen_qwen25_weights.py"

# 2) 同 seed 是否逐字节相同（重新生成到内存，与 manifest 里的 sha256 逐一比对）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t02' && /usr/bin/python3 tools/gen_qwen25_weights.py --verify"

# 3) 磁盘产物是否与 manifest 一致（缺文件/字节被改都会报；RESULT: PASS 就是本票的核心判据）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t02' && /usr/bin/python3 tools/qwen25_weights.py --verify"

# 4) 守卫是否真的会响（临时目录，篡改 1 ULP / 删文件 / 未知名）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t02' && /usr/bin/python3 tools/qwen25_weights.py --selftest"

# 5) 重新实测真实权重统计量（torch + safetensors；会打印与内嵌 MEASURED 的差值）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t02' && /usr/bin/python3 tools/gen_qwen25_weights.py --measure --safetensors /mnt/d/ai/Qwen2.5-0.5B-Instruct/model.safetensors"

# 6) 张量清单 / 一览
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-t02' && /usr/bin/python3 tools/qwen25_weights.py --summary"
```

---

## 11. 剩余疑问（UNKNOWN，不圆）

1. **跨机器/跨 libm 的逐位一致未验证**：Box–Muller 用到 `log/cos/sin`，不同 libm/numpy 版本可能差最后一位。
   本机能保证"同 seed 两次生成相同"（§6.1）；**跨机器只保证"sha256 一比对就知道是否相同"**，不保证相同。
   → 实践口径：**共享同一份产物目录**（或同机重生成 + `--verify`），不要跨机各生成各的。
2. **真权重档的 dtype 路径未验证**：源是 bf16，夹具是 fp16；"bf16 值 → fp16 存储"的舍入在真权重档才第一次
   真正发生（本票绕开了它）。
3. **夹具的极值尾巴比真权重短**：N=128 的张量里 `max|z|` ~2.6–3σ（`layer0.attn.k_proj.bias` abs_max 77.8），
   而真权重同张量是 4.04σ（130）。也就是说**夹具不会给 fp16 动态范围施加真权重那种极值压力**；
   真权重档若出现溢出/饱和，夹具档复现不出来。
4. **`k_proj.bias` std=32 的 fp16 语义**：真权重档下 q/k 由 bias 主导，attention scores 量级比"权重口径"大一个数量级；
   夹具已如实复刻这一量级，但**板端 MDLA 的 fp16 累加器是否在这量级上丢精度，本票无法回答**（要票 03/04 的板端证据）。
5. **`embed` 272 MB 单份物化在板端的真实占用/加载时间未测**（要票 08）。
6. **命名契约的冻结范围**：25 个张量名/shape/dtype/sha256 从本票起视为**冻结**（下游已引用）；追加只允许
   **在清单尾部加新张量**（如 `layer2.*`），不得改名或改已生成张量的 seed 派生规则。
7. **真权重能否复用同一套名字**：要让最后一档复用 §5.3 的接口，需要一个 `bf16 safetensors → 同名字典` 的适配层；
   本票未实现（避免超出票面范围），但接口已按此方向设计。
