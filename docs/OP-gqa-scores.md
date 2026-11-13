# OP-gqa-scores —— GQA 打分：图内 KV 复制（SPLIT→CONCAT）+ 3-D BMM

> 票 04（feature `qwen25-attention`）· 分支 `impl/04-gqa-scores` · worktree `wt-t04`
> 交付物：`ops/gqa_scores_t8_fp16/`（+ `tools/build_gqa_scores.py`）· 本文
> 相关：`docs/OP-gqa.md`（E-112，编译边界）、`docs/OP-bmm.md`（3-D vs 4-D）、
> `docs/OP-shapeops.md`（SPLIT/CONCAT 层与 16 B 规则）、`docs/OP-weights.md`（夹具）、
> `docs/OP-rope.md`（split-half 约定）

---

## 1. 这一票要回答什么

真实 Qwen2.5-0.5B 是 **14 个 query head 对 2 个 KV head**（7:1），head_dim=64。编译器**不会**替我们
广播：2 个 KV head 必须在图内显式复制成 14 份，否则形状对不上。要回答的三件事：

1. **复制能不能完全留在图内？** 用已证形状算子（`SPLIT` + `CONCATENATE`）做 2→14 的复制，
   数据一次都不回 host。
2. **打分能不能只用 1 个原生层？** 14 头走 **3-D BMM**（`[H,T,HD]`×`adj_y=1`），
   而不是 4-D（4-D 会被降成 5 层，见 OP-gqa/OP-bmm）或 14 次单头 unroll。
3. **复制的代价有多大？** 层数与执行步数各是多少——用编译日志实测，不引用二手数字。

本票**只做打分**：不除 `1/sqrt(HD)`、不加因果掩码（那是票 05 的范围）。

## 2. 事前登记：假设与预测（写于第一次编译之前）

### 2.1 设计（为什么是这个图）

接口（host 与图之间只有一个 `[T,H,HD]` 契约，与票 03 的 FC 输出行主序布局一致）：

| 张量 | 形状 | 说明 |
|---|---|---|
| `q` | `[8,14,64]` fp16 | 图输入 0；行主序 = FC 输出 `[T, 896]` 的视图 |
| `k` | `[8,2,64]`  fp16 | 图输入 1 |
| `scores` | `[14,8,8]` fp16 | 图输出（3-D，不是 4-D） |

图中的算子序列：

```
SPLIT(k, axis=1, num=2)        -> k0,k1  [8,1,64]
CONCAT([k0]*7, axis=1)         -> ka     [8,7,64]
CONCAT([k1]*7, axis=1)         -> kb     [8,7,64]
CONCAT([ka,kb], axis=1)        -> k14    [8,14,64]        <- 复制完成，仍在图上
TRANSPOSE(k14, perm=[1,0,2])   -> kt     [14,8,64]
TRANSPOSE(q,   perm=[1,0,2])   -> qt     [14,8,64]
BATCH_MATMUL(qt, kt, adj_y=1)  -> scores [14,8,8]         <- 1 个原生层
```

**头映射**：`[k0×7 | k1×7]` 沿 head 轴拼接 = `repeat_interleave(7)`，即 query head `h`
读 KV head `h//7`。这正是真实 GQA 的映射（不是"按 batch 广播"也不是"按块交错"）。

**两个 TRANSPOSE 是必须的**（NOTES §10.2）：把 `[T,H,HD]` 直接喂给 3-D BMM，或者把 `[T,H]`
RESHAPE 成 `[H,T,HD]`，都能编译、也能单执行步，但**数学是错的**（它按错误的轴配对求和）。
布局变化必须是真正的 TRANSPOSE。本票不赌这一条。

### 2.2 q/k 从哪来（票面要求记录的选择）

**选择：夹具（票 02）的 layer0 `q_proj`/`k_proj` + bias + RoPE**，与票 03 同口径，而不是随机张量。

- `x = N(0,1) float32 [8,896]`（numpy `default_rng(20261113)`，确定性）；
- `q = RoPE(q_proj·x + b_q, split-half)` → fp16 `[8,14,64]`；`k` 同理 → `[8,2,64]`；
- RoPE 用 Qwen2/HF split-half 约定（`out_lo = a·cos - b·sin`、`out_hi = b·cos + a·sin`，
  theta=1e6，位置 0..7），与 `docs/OP-rope.md` §2 的更正版一致。

理由：这张夹具的 `q_proj.bias` std≈7.8、`k_proj.bias` std≈32（实测，见 OP-weights §4），
bias 主导 q/k。用随机张量（±1）会把动态范围做假，掩盖 fp16 在大数上的行为；用夹具则让本票
同时成为"bias 量级下 BMM 的 fp16 精度"第一条实测。

**golden 与图输入严格一致**：`golden/input.bin`=`q`、`golden/input1.bin`=`k`
（`tools/e2e.sh` 会把它们按输入顺序推给设备），`golden/expected.bin` 是**对这两个 fp16 输入**
做 numpy float32 收缩再 round 到 fp16 的结果——与 flatbuffer 写出代码零共享（R3）。

### 2.3 预测

| # | 预测 | 若预测不成立意味着 |
|---|---|---|
| P1 | `main` 编译 `EXIT=0`、`--disallow-bridge` 通过、**1 个执行步**；层数 7（SPLIT 1 + CONCAT 3 + TRANSPOSE 2 + BMM 1） | 某个形状算子被降级/折叠/拆分（层清单会给答案） |
| P2 | 复制的层数开销 = `layers(main) - layers(bmm_only)` ≈ **4 层**（SPLIT 1 + CONCAT 3），且不增加执行步 | 复制引入了额外执行步（那才是真正的代价） |
| P3 | **7 输入 CONCAT（输入张量重复 7 次）被 ncc 接受** | 仓库从未测过 ≥3 输入 CONCAT（OP-shapeops Q4）；若被拒，退到二叉 CONCAT 树（7 份 = 4 个 CONCAT/组，层数 4→12） |
| P4 | 板端数值：设备与 fp32-累加 golden 的差 ≈ 0–1 ULP（仓库既证：`bmm_qk_t16_fp16`/`bmm_pv_t8_fp16`/`fc_8x128x128_fp16` 三例 max_abs **=0**），即设备是 fp32 累加后一次 round | 设备若把部分和也 round 到 fp16，误差会到 ~12 绝对（见 §3 的独立下限），那是必须记录的设备事实 |
| P5 | 噪声底 = 0（同一 `.dla` 同一输入连跑两次逐位相同） | 若不为 0，阈值按 NOTES §4 的 3–5 倍噪声底重定（但必须在跑之前重登记） |
| P6 | 兜底更贵：`unroll14` ≥ 15 层、`twox7` ≥ 7 层，都远多于 `main` 的 1 个 BMM 层 | 兜底更便宜的话，Spec 的"纯亏损"结论要改 |
| P7 | 5-D/6-D BMM 被拒，措辞为 `the rank of shape should be in range [0, 4]` | 只引用 E-112 原文，不重跑（票面允许） |

### 2.4 容差登记（跑之前，NOTES §10.1）

`max|scores|` 由 q/k 的量级决定：`max|q|=30.2`、`max|k|=80`（夹具 + RoPE 后的实测值），
打分 `max|out| = 5368` ——**远大于 1**，所以 NOTES §4 的"阈值 <2e-2"不适用。按 §10.1 的例外规则：

- 独立于设备的 fp16 下限（numpy 逐位模拟"每步部分和 round 到 fp16"，K=64）：`max_abs = 12.0`
  （`max_rel = 1.14`，出现在打分接近 0 的抵消位上）；
- 若设备是 fp32 累加（仓库三例 max_abs=0 支持这一读法），下限退化为输出 round 的 1 个 ULP；
- **登记值：`atol = rtol = 4×2⁻¹¹×max|out| = 10.4838`**（≈4 ULP @ 输出量级），
  写在 `model.json` 的 `atol`/`rtol` + `tolerance_revision` + `tolerance_reason`；
- 报告里同时给相对口径（`max_abs/max|exp|`、`rel_l2`）与噪声底；
- **禁止**跑完再调阈值。

---
