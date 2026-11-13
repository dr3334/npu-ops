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

## 3. 做法

### 3.1 生成器

`tools/build_gqa_scores.py`（本票新增；`import build_ops as bo` 只读引用公共 flatbuffer 工具箱，未改它）。

- `main`：上面的 7 个 TFLite 算子；图输入 `q`/`k`，图输出 `scores`；
- 复现性：同一 seed、同一夹具 → `model.tflite` 与三个 `golden/*.bin` **逐字节可复现**
  （连跑两次 `sha256sum -c` 全 OK）；
- 对照/探针变体（全部只差"复制或布局那一块"）：
  `bmm_only`（去掉复制，k 已 14 头）、`nary14`（单条 14 输入 CONCAT）、
  `twox7`（2×7 头兜底）、`unroll14`（14 次单头兜底）、
  `wrong_reshape`（**故意错**：用 RESHAPE 代替 TRANSPOSE 的对照组）、
  `rank5`（5-D BMM，预期被拒）。

### 3.2 板端协议（守则）

编译不碰板子：builder → `ops/<name>/model.tflite`（锁外）→ `./tools/compile.sh`（锁外）→
只有 `./tools/devlock.sh run -- ./tools/e2e.sh ops/<name>` 持锁。板端实验前后各一次
`healthcheck.sh --quick`（两次都 `HEALTHCHECK: PASS`）。`e2e.sh` 在 `model.tflite`/`model.dla`
已存在时只做 push/run/pull/compare，所以 `model.tflite` 必须在进锁之前就有（本票的 op 名不在
`build_ops.py` 的 SPECS 里，e2e 自己 build 会失败）。

---

## 4. 结果

### 4.1 编译侧（`model.compile.log` 原文）

| 变体 | 层清单（顺序照抄） | 层数 | 执行步 | `.dla` 字节 |
|---|---|---|---|---|
| **main**（交付） | Slice, Concat, Slice, Concat, Concat, Transpose, Transpose, **BatchMatMul** | **8** | **1** | 14,906 |
| `bmm_only`（无复制基线） | Transpose, Transpose, BatchMatMul | 3 | 1 | 10,962 |
| `nary14`（单条 14 输入 CONCAT） | Slice, Slice, Concat, Transpose, Transpose, BatchMatMul | 6 | 1 | 14,910 |
| `twox7`（2×7 头兜底） | Slice, Transpose, Slice, Transpose, Slice, Concat, Transpose, BMM, Slice, Concat, Transpose, BMM, Concat | 13 | 1 | 15,576 |
| `unroll14`（14 次单头兜底） | Transpose, 14×Slice, Transpose, Slice, 14×BMM, Concat | 33 | 1 | 10,962 |
| `wrong_reshape`（对照组，故意错） | Slice, Concat, Slice, Concat, Concat, Reshape, Reshape, BatchMatMul | 8 | 1 | 14,454 |

全部 `EXIT=0`、`Target: MDLA_5_0`、`===Graph[0] has 1 Execution Step(s)===`、
`--disallow-bridge` 下**没有任何 bridge**。main 的 DRAM 账：Input 16K / Output 1.8K / Code 8.1K，Total 25K。

**三个可以直接引用的结论**：

1. **KV 复制的层数开销 = 3–5 层，不增加执行步**。
   `main - bmm_only = 5`（= 2 个 `SliceLayer` + 3 个 `ConcatLayer`）；
   把"每组 7 份"和"合并两组"压成**一条 14 输入 CONCAT** 后是 `nary14 - bmm_only = 3`
   （= 2 个 `SliceLayer` + 1 个 `ConcatLayer`）。两种写法都是**单执行步**。
   注意 `SPLIT` 的**每个输出各发一个 `SliceLayer`**（与 `docs/OP-shapeops.md` E-SHP-04 一致），
   所以"SPLIT 只算 1 层"是错的（我事前的 P1 就错在这里）。
2. **打分本身只是 1 个 `BatchMatMulLayer`**：14 头一次算完，3-D `[14,8,64]`×`adj_y=1`。
   没有任何头数拆分、没有 unroll。
3. **兜底是纯亏损，实测确认**：
   - `unroll14` = **33 层**（14 BMM + 16 Slice + 2 Transpose + 1 Concat），是 main 的 4.1 倍；
   - `twox7` = **13 层**（2 个 7 头 BMM + 4 Transpose + 3 Slice + 3 Concat），是 main 的 1.6 倍，
     而且 **BMM 从 1 个变成 2 个**（多一次层启动）；
   - 两者相对"单体 1 个 BMM 层"都是纯亏损，**不采用**。
   - 与二手数字对账：Spec/E-112 记的"2×7 ≈23 层 / 14 次 unroll ≈15 层"与本文实测的 13 / 33 不同，
     原因是**构造与计数口径不同**（E-112 的 15 层只数 BMM+CONCAT 这一中段，不含 SPLIT 出来的
     16 个 Slice 与 2 个 Transpose；2×7 的 23 层应是用了二叉 CONCAT 树）。本表的数字来自本机
     `model.compile.log`，可逐行复核。方向性结论（兜底 ≫ 单体）在两种口径下都成立。

### 4.2 板端（核心判据）

`ops/gqa_scores_t8_fp16/compare.txt`（同一 `.dla`、同一输入，连跑两次，两次都 PASS）：

```
op        : GQA_SCORES {'T': 8, 'Hq': 14, 'Hkv': 2, 'HD': 64, 'kv_repeat': 7, 'adj_y': 1, 'rank': 3}  dtype=fp16
expected  : 896 vals   device: 896 vals   logical shape=(14, 8, 8)
max_abs   : 0.000e+00
rmse      : 0.000e+00
rel_l2    : 0.000e+00
nan/inf   : exp 0/0  dev 0/0
range     : exp [-5368,4988]  dev [-5368,4988]
tol       : atol=10.4844 rtol=10.4844
PASS
```

**设备输出与 fp32-累加 golden 的 896 个 fp16 值逐位相同**（`np.array_equal` 在 `uint16` 视图上
返回 `True`）。相对口径：`max_abs/max|exp| = 0`、`rel_l2 = 0`。
板端耗时（`--repeat 3`，小图、启动主导，仅记录不作性能结论）：1.448 ms/inf（606.6 FPS）、
0.901 ms/inf（995 FPS）两次。

### 4.3 噪声底（NOTES §4 的测法）

同一 `.dla`、同一输入，两次独立 e2e 各存一份（`out_r1.bin` / `out_r2.bin`）：

```
n = 896  noise_floor max|r1-r2| = 0.0
raw fp16 bitwise equal: True          # 两次运行逐位相同
dev vs exp: max_abs=0.000000e+00 rmse=0.000000e+00 rel_l2=0.000000e+00
max|exp| = 5368  -> max_abs/max|exp| = 0.000000e+00
1 ULP @ max|out| = 2.62109   -> dev is 0 ULP away
```

**噪声底 = 0（设备确定性）** → 判据退化为 fp16 末位量级（NOTES §4 的退化分支 + §10.1 的 ULP 规则）。
本轮**保留跑前登记的 `atol = rtol = 4×2⁻¹¹×max|out| = 10.4844`**（≈4 ULP @ 5368），
实测误差 0（0 ULP）——即实测比契约紧 4 个 ULP；事后不因"实测为 0"收窄阈值（收窄虽更严，
但会让"登记-测量"的时序失真）。

### 4.4 对照组：错的布局会被抓出来（证明判据不空转）

逐位相同的 PASS 需要一条"能失败"的对照，否则无法排除"判据形同虚设"。`wrong_reshape` 把两个
TRANSPOSE 换成 RESHAPE（`[8,14,64] -> [14,8,64]` 的平坦重解释，正是 NOTES §10.2 的坑）：

| 变体 | 编译 | 执行步 | 板端 max_abs | rmse | rel_l2 | 判定 |
|---|---|---|---|---|---|---|
| main（TRANSPOSE） | EXIT=0 | 1 | **0** | 0 | **0** | **PASS** |
| wrong_reshape（RESHAPE） | EXIT=0 | 1 | **8.256e+03** | 3.021e+03 | **1.226** | **FAIL** |

即：**编译通过 + 单执行步 + 形状对 = 与数学正确无关**（NOTES §10.2 第二条），本票用同一条板端
判据把两者分开了。

### 4.5 头映射的敏感性（GQA 的 `h//7` 不是随便一种复制）

golden 用的映射是 `head h -> KV head h//7`。把另两种"看起来也合理"的复制方式算成 golden，
它们与本 repo 采用的映射差多远（纯 numpy，不上板）：

| KV 映射 | 与 `h//7` 的 max_abs | rel_l2 |
|---|---|---|
| `h//7`（采用） | — | — |
| `h%2`（两组交错） | 4,986 | 0.638 |
| 两组对调（`h<7 -> kv1`） | 6,449 | 1.051 |

差异比噪声底（0）大 3–4 个数量级，所以"复制对了"这件事是被实测钉住的，不是形状对上就算。

### 4.6 附加数据点：N 输入 CONCAT 在板上可用（`nary14`）

`docs/OP-shapeops.md` 的 Q4 记"3 输入以上 CONCAT 未测"（e2e 只喂两个输入，只能拆图）。本票顺手把它测了：

- `nary14`（14 输入、**同一个张量重复出现 7 次**）：编译 6 层 / 1 执行步 / 无 bridge；
- 板端 `compare.txt` **PASS**，`max_abs = 0`、`rel_l2 = 0`、逐位相同。

→ N 元 CONCAT（含重复输入）的**运行时语义在 fp16 + axis=1 上是正确的**，且 ncc 没有把它拆成 14 层。
（仍未测：多输入**不同 scale**（int8 requant）、axis≠1 的 N 元形态。）

### 4.7 rank 上限 = 4（本机复现，不只引用）

`rank5` 变体（同一 BMM，输入 `[1,1,14,8,64]`）编译：

```
 ├ MDLA: the rank of shape should be in range [0, 4]
ERROR: Cannot find an execution plan because of unsupported operations
ERROR: Fail to compile ops/_gqa_variants/rank5/model.tflite
EXIT=1
```

- 与 E-112 记的原文一致；阶段是**后端期**（`Cannot find an execution plan`），不是加载期；
- 6-D 属同一检查（`[0, 4]` 的上界），本票未再单独造 6-D 模型；
- 因此注意力一律走 **3-D**：`[H,T,HD]` / `[H,T,T]`。

---

## 5. 与票面 checklist 对照

| 票面要求 | 状态 | 证据 |
|---|---|---|
| 2 个 KV head 在图内复制成 14 份（SPLIT + CONCATENATE），数据不回到 host | ✅ | §4.1 层清单里 `SliceLayer`×2 + `ConcatLayer`×3 全在 `Step[0]/MDLA_5_0` 内；图只有 2 个外部输入、1 个输出；无 bridge |
| 3-D BMM（adj_y）产出 `[14,8,8]` 的 scores，与 golden 对上 | ✅ | §4.2 `compare.txt` PASS，896/896 **逐位相同**，`max_abs=0` |
| 记录复制带来的层数开销与执行步数（预期仍是单执行步） | ✅ | §4.1：开销 **5 层**（`main 8 - bmm_only 3`），最省写法 **3 层**（`nary14 6 - 3`）；**执行步仍是 1** |
| 明确记录**不采用**"2×7 头"与"14 次单头 unroll"，并写出层数代价 | ✅ | §4.1 表：`twox7` **13 层 / 2 个 BMM**、`unroll14` **33 层 / H 次启动** vs 单体 **1 个 BMM 层**；并给出与 Spec 二手数字（23/15）的差异说明 |
| 记录 rank 上限是 4：5-D/6-D 会明确报错 | ✅ | §4.7 本机复现 `the rank of shape should be in range [0, 4]`（后端期）；6-D 同检查、引用 E-112 |
| 记录本算子的实测噪声底 | ✅ | §4.3 噪声底 **0**（两次运行逐位相同），实测误差 0 ULP，登记阈值 4 ULP |
| 票面"q 从哪来"要记录选择 | ✅ | §2.2：用夹具 layer0 `q_proj`/`k_proj` + bias + RoPE（与票 03 同口径）；golden 与图输入严格一致 |
| 本票不缩放、不加掩码（票 05 的事） | ✅ | §6 边界；图中没有 MUL-by-const、没有 ADD(mask)、没有 SoftMax |

---

## 6. 适用范围 / 不适用

**适用**

- `ncc-tflite 7.3.15`，`--arch=mdla5.0 --num-mdla=2 --l1-size-kb=3072 --disallow-bridge`（厂商编译路径）；
- fp16；`T=8`、`Hq=14`、`Hkv=2`、`HD=64`（即 Qwen2.5-0.5B 的真实形状）；
- KV 复制用 `SPLIT(axis=1,2)` + `CONCATENATE(axis=1)`，复制比 7，映射 `h//7`；
- 打分走 **3-D** `BATCH_MATMUL` + `adj_y=1`，布局靠**真正的 TRANSPOSE** `[T,H,HD]->[H,T,HD]`；
- 图输入布局是 `[T,H,HD]`（= FC 输出 `[T, H*HD]` 的行主序视图），这是与票 03 的接缝。

**不适用 / 不要引用成结论**

- **不含缩放与掩码**：这里没有 `1/sqrt(64)`，也没有因果掩码（票 05）。scores 的量级（`max|out|=5368`）
  是"未缩放"的，别拿它去和 softmax 之后的东西比；
- **不是性能证据**：只跑了 `--repeat 3`，FPS 是启动主导的小图数字；
- 未测 `T=128`、未测 24 层串联、未测 KV cache（票 09/10 的范围）；
- 未测 int8、未测 `HD≠64`、未测 `Hq/Hkv` 的其它比值（如 4:1、8:1）；
- `nary14`（6 层写法）虽然板端 PASS，但**主交付仍是 `main`**（3 条 CONCAT，与票面描述的"各 CONCAT 7 次"一致）；
  票 09 若要用更省层的写法，请自己重跑一次 e2e（本文件的 `ops/_gqa_variants/nary14/` 已留证据）。

---

## 7. 剩余疑问（UNKNOWN，不圆）

1. **设备是 fp32 累加还是更高精度**：本票只观察到"与 fp32-累加 golden 逐位相同"（0 ULP）。
   这支持"设备按 fp32 累加、只 round 一次输出"，但**无法区分 fp32 与更高精度累加**；
   也无法回答"如果累加器是 fp16，误差会长什么样"（那是 §2.4 的 12.0 绝对上界，本机未出现）。
2. **大数上的抵消位**：scores 里存在若干接近 0 的元素（`min|scores| = 8.1`，而单项乘积量级 ~2.4e3），
   即每元素有 ~300× 的抵消。fp16 输出在这种元素上的**相对**误差天然很大；本票用绝对口径
   （4 ULP @ max|out|）覆盖，没有为"相对小元素"单独定契约。票 05 的 softmax 会把这些元素放大成
   exp 的差，**是否需要用更大 atol 或改成 `max - x` 平移**，留给票 05 实测。
3. **复制在 24 层/更大 T 下的成本未测**：本票只测了 T=8 的层数与执行步；`T=128` 时 `k14` 的
   中间张量是 128×14×64×2 B ≈ 229 KB，是否仍在同一执行步、DRAM 账是否变化，未测。
4. **`SPLIT` 的多输出为什么发 2 个 `SliceLayer`**：观察到但未深究（E-SHP-04 只说"SPLIT 降成 slice"）；
   对层预算的影响已计入（复制开销按 3–5 层记）。
5. **N 元 CONCAT 的 int8/多 scale 形态**：§4.6 只验了 fp16 同 scale；int8 下不同 scale 需要 requant，
   语义未测（OP-shapeops Q9）。

---

## 8. 复现

```bash
# 0) 环境：WSL 的 /usr/bin/python3（numpy + flatbuffers + tflite schema），路径含空格必须加引号
export QWEN25_WEIGHTS_DIR="/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen"

# 1) 生成（锁外）：model.tflite + model.json + golden/{input,input1,expected}.bin
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/npu-ops' && /usr/bin/python3 tools/build_gqa_scores.py"

# 2) 容差下限（锁外，独立于设备）：fp16 顺序累加 vs fp32 golden
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/npu-ops' && /usr/bin/python3 tools/build_gqa_scores.py --tol-analysis"

# 3) 编译（锁外）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/npu-ops' && ./tools/compile.sh ops/gqa_scores_t8_fp16/model.tflite ops/gqa_scores_t8_fp16/model.dla"

# 4) 板端（持锁）：e2e 只做 push/run/pull/compare
export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
./tools/devlock.sh run -- ./tools/e2e.sh ops/gqa_scores_t8_fp16
cp ops/gqa_scores_t8_fp16/out_device.bin ops/gqa_scores_t8_fp16/out_r1.bin
./tools/devlock.sh run -- ./tools/e2e.sh ops/gqa_scores_t8_fp16      # 噪声底第二份
cp ops/gqa_scores_t8_fp16/out_device.bin ops/gqa_scores_t8_fp16/out_r2.bin
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/npu-ops' && /usr/bin/python3 ops/gqa_scores_t8_fp16/noise_floor.py"

# 5) 对照与探针（编译/跑法同上，变体名换一下）
#    bmm_only / nary14 / twox7 / unroll14 / wrong_reshape / rank5
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/npu-ops' && /usr/bin/python3 tools/build_gqa_scores.py --variant twox7"
```

---

## 9. 预测复核

| # | 预测 | 结果 | 说明 |
|---|---|---|---|
| P1 | main `EXIT=0`、单执行步、7 层 | ⚠️ 部分成立 | EXIT=0 与单执行步成立；**层数是 8 不是 7** —— 我把 `SPLIT` 记成 1 层，实际每个输出各发一个 `SliceLayer`（§4.1、§7 Q4） |
| P2 | 复制开销 ≈4 层、不增加执行步 | ⚠️ 数量级成立、数字修正 | 实测 **5 层**（2 Slice + 3 Concat），"单执行步"成立；换 14 输入单条 CONCAT 后是 **3 层** |
| P3 | 7 输入 CONCAT（重复输入）被接受 | ✅ 成立 | main 编译 EXIT=0 且板端逐位 PASS；额外的 `nary14`（14 输入、重复 7 次）也编译 + 板端 PASS（§4.6） |
| P4 | 板端误差 ≈0–1 ULP | ✅ 成立（更好） | 896/896 **逐位相同**（0 ULP，§4.2） |
| P5 | 噪声底 = 0 | ✅ 成立 | §4.3 两次运行逐位相同 |
| P6 | `unroll14` ≥15 层、`twox7` ≥7 层 | ✅ 成立 | 实测 **33 层 / 13 层**（§4.1） |
| P7 | 5-D/6-D 被拒，措辞为 rank ∈ [0,4] | ✅ 成立 | §4.7 本机复现，后端期 |

**事后自评**：唯一的偏差在"层数 7 vs 8"，原因是事前把 `SPLIT` 的层数记少了——这正是为什么
层数必须从 `model.compile.log` 抄而不能推。板端数值、单执行步、兜底更贵三条都按预期成立。

### 附：本票产出的文件

- `tools/build_gqa_scores.py`（生成器 + 6 个变体 + 容差下限分析）
- `ops/gqa_scores_t8_fp16/`：`model.json` / `model.compile.log` / `compare.txt` / `e2e.json`（入库）；
  `model.tflite` / `model.dla` / `golden/*.bin` / `out_*.bin`（按 .gitignore 不入库，可复现）
- `ops/gqa_scores_t8_fp16/noise_floor.{py,txt}`、`mapping_sensitivity.{py,txt}`（噪声底与头映射敏感性的可复算脚本）
- `ops/_gqa_variants/{bmm_only,nary14,twox7,unroll14,wrong_reshape,rank5}/`（对照与探针的
  `model.json` + `model.compile.log`，上过板的还带 `compare.txt` / `e2e.json`）
