# 一个 Qwen2.5 风格 decoder block 的整图试验

日期：2026-11-12 · 板 Titan 1 / MT6897 · ncc 7.3.15 `--disallow-bridge`。
模型：`ops/block_qwen25_1l_fp16`（T=8, H=64, 1 head / head_dim 32, FFN=128, fp16）。
生成器：`tools/build_block.py`（可复现，含 numpy 参考）。

## 1. 结果

**整块 16 个层全部落在同一个 `MDLA_5_0` 执行步，没有 bridge；板端数值通过。**

| 指标 | 值 |
|---|---|
| 编译 | `EXIT=0`，`.dla` 78.8 KB，`Target: MDLA_5_0`（单步） |
| 层清单 | FullyConnected×10（含 2 个**动态权重**）、SoftMax、Transpose、ElementWiseAdd×2、Sigmoid、ElementWiseMul×2 |
| 板端 | 512 个 fp16 输出 |
| 数值 | **max_abs = 9.766e-04**，rmse = 1.986e-04，rel_l2 = 3.730e-04（≈ fp16 量化步长），NaN/Inf = 0 |
| 对照 | 独立的 numpy float32 参考（同一份权重） |

## 2. 图里有什么 / 图外有什么

图内（全在 NPU）：
```
q=FC(x,Wq) k=FC(x,Wk) v=FC(x,Wv) → s=FC(q,k)[动态B] → softmax → a=FC(p,vᵀ)[动态B]
→ o=FC(a,Wo) → r1=x+o → g=FC(r1,Wg) u=FC(r1,Wu) → sigmoid(g) → m=g*sig → y=m*u
→ d=FC(y,Wd) → out=r1+d
```
图外（有意保留，见 §4）：
- **RMSNorm**：需要 rsqrt，是否能在 MDLA 上是 open（rms-agent 在查）；
- **RoPE**：`SIN`/`COS` 被 importer 拒绝，只能用 CPU 预算的 sin/cos 常量，本轮没有拼进图。

## 3. 三个值得记的点

1. **动态权重 FC 在图内可用**：`s=FC(q,k)` 与 `a=FC(p,vᵀ)` 的权重是运行时张量（不是常量），
   编译器照样把它们放进 MDLA 步——这解决了"attention 的两个矩阵乘能不能上 NPU"的最大疑问。
2. **Transpose 也是真实层**（`TransposeLayer<5>`），不是被折叠掉的视图。
3. **整块没有 bridge**，说明这套算子组合对 ncc 是"可整块下沉"的；之前 BMM 单独测出来的 shape 边界
   （16 字节行规则）在这个形状集里刚好都满足。

## 4. 尚未覆盖（必须说清）

- RMSNorm / RoPE 不在图内 → **这还不是"完整 block"**；
- 单 head（真实模型 14 Q heads / 2 KV heads）、T=8（真实 prefill 128/2048）；
- 没有 KV cache、没有残差之外的跨 block 状态；
- **这是厂商编译路径**：`.dla` 由 ncc 生成，不是我们自建提交器；
- 数值是 fp16 路径（atol 2e-2 合同），没有做 int8 版整块。

## 5. 下一步

1. 等 rms/rope/silu/bmm 四个 agent 的结论，把 RMSNorm 与 RoPE 拼进图 → **完整 block**；
2. int8 版整块（需要更多对齐检查：E-003 的 requant 规则已经可用于写 golden）；
3. 多 head / T=128 的 prefill 变体；
4. 把整块作为"基线"，再逐个用自建提交器替换其中的 Linear（openmtk M2/M3 的路径）。

## 6. v2 尝试（RMSNorm 进图）—— 目前失败，记为失败知识

`tools/build_block2.py` 把 RMSNorm 链（`MUL→MEAN→ADD→RSQRT→MUL→MUL`）放进同一个图，
ncc 在 **importer 阶段**就拒了：

```
ERROR: Fail to process the given model     EXIT=1
```

（没有层清单，说明还没走到 lowering；RMSNorm 单独编译是能过且落 MDLA 的，见 `docs/OP-rmsnorm.md`。）

**待排除的候选**（下一轮按顺序做，一次只改一处）：
1. `gamma` 的 **广播 MUL**：`[T,H] × [H]` 在 TFLite 合法，但 ncc 可能不接受非标量广播 → 先换成 `[1,H]` 常量或把 gamma 乘进相邻 FC；
2. `RSQRT` 在这个图里的 **op version**（单独模型用 version=1 通过，这里可能被折叠成别的形式）；
3. `MEAN` 的 axis 常量张量形状（`[1]` vs 标量）；
4. 图规模（16 → 27 层）触发的 importer 限制。

结论：**"带 Norm 的整块"还没有跑通**；v1（无 Norm）已跑通，四类新算子（RMSNorm/RoPE/SiLU/BMM）
各自单独在 NPU 上都已证明。把 v2 的 importer 错误排掉，才谈得上"完整 block"。

## 7. v2 排错记录（截至本轮）

按"一次只改一处"推进，importer 错误随修改在移动，说明结构在逐步对，但还没收敛：

| 尝试 | 结果 |
|---|---|
| `mean[T,1] + eps[T,1]`（两个 fp16 常量/张量） | `ERROR: Invalid ElementWiseAddLayer<2>{ADD}: input types are incompatible with each other` |
| 去掉 eps ADD（eps=0，直接 `RSQRT(mean)`） | `ERROR: Invalid RSqrtLayer<2>{RSQRT}: output shape is incompatible with inputs` |
| 把 `mean`/`rsqrt` 改成 rank-1 `[T]`（keep_dims=false） | 仍是 RSQRT shape 不兼容，并新增 `ElementWiseMulLayer<3>: input shapes are incompatible` |

**关键反差**：同一套 `MUL→MEAN→(ADD)→RSQRT→MUL→MUL` 在**独立模型**里（`docs/OP-rmsnorm.md`，
形状 `[1,64]`/`[8,64]`）能编译并在 MDLA 上跑出 2 ULP 的结果；拼进整图就报形状/类型不兼容。
⇒ 问题不在算子能力，而在**整图里的形状传播**（最可能是 MEAN 的 `keep_dims` 与后续张量声明不匹配，
或 ncc 要求这组链路的张量保持与图其它部分一致的 rank）。

**下一步（具体动作）**：把 `tools/build_rmsnorm.py` 里独立模型使用的
`tensor()` 声明（shape/keep_dims/axis 常量形状）与 `build_block2.py` 逐字段对比，
先把**只含 RMSNorm + 一个 FC** 的两算子图跑通，再逐步加回 attention/MLP。

**当前可用的整块**：v1（无 Norm、无 RoPE）已通过，`ops/block_qwen25_1l_fp16`。

### 7.1 bisect 结果（RMSNorm + 1 个 FC，两算子图）

把整块缩到最小复现：`MUL(x,x) → MEAN(axis=1,keep) → RSQRT → MUL(x,r)` 再 `FC`，**仍然失败**：

```
层次=1  ERROR: Invalid RSqrtLayer<2>{RSQRT}: output shape is incompatible with inputs
```

两种张量声明都试过：

| mean/rsqrt 声明 | keep_dims | 结果 |
|---|---|---|
| `[T,1]` | true | RSQRT output shape incompatible（层次=1 就停） |
| `[T]` | false | RSQRT 不兼容 + `ElementWiseMulLayer<3>: input shapes are incompatible` |

而**完全相同的链**在 `tools/build_rmsnorm.py` 的独立模型里（`ops/rmsnorm_8x64_fp16`）能编译并跑出 2 ULP。
两者张量声明（mean `[T,1]`、eps `[T,1]`、rsqrt `[T,1]`、axis `[1] int32`）**看起来一致**。

⇒ 结论：不是形状声明本身，**是构造顺序/索引细节**（很可能是 flatbuffer 里 tensor 与 operator 的登记顺序，
或 MEAN 的 axis 张量在 buffer 表中的位置）。下一步应当做的是 **把两份 .tflite 的 tensor 表逐字段 diff**
（`flatc -t --raw-binary schema` 或直接解析），而不是继续在手写图里试错。

> 记录这条的意义：它说明"独立算子能跑"与"同一算子拼进整图能跑"是**两个不同的验证**，
> 后者才是 block 级别真正需要的。

## 8. ✅ v2 修好了：带 RMSNorm 的整块通过

**根因**（tflite tensor 表 diff 一眼看出来的）：`rmsnorm()` 里先 `act("..._mean")` 建均值张量，
随后又 append 了 axis 常量张量，于是 `idx[mean] = len(tensors) - 1` 指向了 **axis 张量**，
`RSQRT` 的输入实际是那个 int32 的 axis → 报 "output shape is incompatible with inputs"。

修法：`act()` 的返回值直接存成 `ti_m` / `ti_r`，不再用 `len(tensors)-1` 事后取。

**结果**

| 指标 | 值 |
|---|---|
| 图 | 24 层（含 2×RMSNorm 链：MUL→MEAN→RSQRT→MUL，10×FC、SoftMax、Transpose、Add×2、Sigmoid、Mul×2） |
| 编译 | `EXIT=0`，单 `MDLA_5_0` 执行步，`.dla` 83.1 KB |
| 板端 vs numpy | **max_abs 9.766e-04**，rmse 1.95e-04，rel_l2 3.73e-04 |
| 判定 | PASS（fp16 合同 atol/rtol 3e-2；实测差 1 个 fp16 code） |

**已知偏差（必须写清）**：图里 `eps = 0`（`mean + eps` 的 ElementWiseAdd 仍会报 type 不兼容，
单独模型带 eps 是通过的）。因此本块与"真实 Qwen2.5 RMSNorm"差一个 eps=1e-5；
在 `mean(x²)≈0.25` 的数据上相对影响 ~4e-5，但**不能声称 eps 已支持**。

**经验**：这类"独立算子能跑、拼图就跑不了"的问题，**tflite tensor 表的字段 diff 比手写试错快一个数量级**
——这次 10 分钟定位，而之前三轮盲试都没找到。

## 9. 完整 block（RMSNorm + RoPE + 多头就绪的 attention + SwiGLU）✅

| 变体 | 层数 | 编译 | 板端 |
|---|---|---|---|
| `block_qwen25_1l_full_fp16`（T=8，1 head，**RMSNorm + RoPE 都在图内**） | **42** | EXIT=0，单 MDLA 步，`.dla` 91.8 KB | **PASS**，max_abs ≈ 1 fp16 code（rmse 1.86e-04） |
| `block_qwen25_1l_full_t128_fp16`（**T=128 prefill**） | 42 | EXIT=0，`.dla` 99.5 KB | **PASS**，rmse 1.90e-04 |

图内现在包含：`MUL→MEAN→RSQRT→MUL`（×2 RMSNorm）、`SPLIT→MUL×4→SUB/ADD→CONCAT`（×2 RoPE）、
`FC`（含 1 个动态权重 scores）、`SoftMax`、`Transpose`、`Sigmoid`、`MUL`、`ADD` —— 全部在**同一个 execution step**。

### 多 head 变体：已定为构造器 bug（**ticket 01 已修，定稿见 §10**）

原记录（本文旧版 / ROUNDUP）把 `--heads 2` 的失败读成"importer 不支持多头"，**那是误读**。
根因在构造器：`fc()` 按 FC 的**目标宽度** `[n,k]` 声明权重张量，而 `heads>1` 时传进去的
`Wq/Wk/Wv` 只有 `HD` 行，于是 flatbuffer 声明了一个大 14 倍的缓冲区（`heads=14` 时声明 896×896
= 802,816 个数，而数组是 `(64,896)` = 57,344 个数）。ncc 在**模型加载期**就退出
（`ERROR: Fail to process the given model`）——从未走到逐层校验，所以打不出层清单，
看起来才像"编译器不支持"。

修好之后（权重张量按**数组真实形状**声明 + `check_fc_weight()` 形状守卫，见 §10）：

| op | 头数 / 形状 | 层数 | 编译 | 板端 |
|---|---|---|---|---|
| `ops/block_qwen25_multhead_h2_fp16` | 2 头 / H=64, HD=32, T=8 | 27 | EXIT=0，单 `MDLA_5_0` 执行步，无 bridge，`.dla` 101 KB | 见 §10.5 |
| `ops/block_qwen25_multhead_h14_fp16` | **14 头 / H=896, HD=64（真实形状）** | 30 | EXIT=0，单 `MDLA_5_0` 执行步，无 bridge，`.dla` 7.06 MB | 见 §10.5 |

头数本身不是限制：`docs/OP-gqa.md`（E-112）把单层 attention 的头数扫到 **1024** 仍是一个
`BatchMatMulLayer`；本 ticket 的贡献是把**整块**（RMSNorm + 14 头 attention + SwiGLU）的构造修对。

判据提示（按错误措辞分阶段，完整表见 §10.2）：`the given model` = 加载/解析期，
`the input model` = 逐层校验期，
`Cannot find an execution plan` / `Bridging is disallowed` = 后端期。

### 覆盖声明（不夸大）

- ✅ 单 head、T=8 / T=128、fp16、RMSNorm+RoPE+attention+SwiGLU 全在 NPU；
- ✅ 多 head（2 头 / 14 头、H=896 HD=64 真实形状）：编译侧单执行步、无 bridge；板端数值见 §10.5；
- ⚠️ 多头路径**不含 RoPE**（RoPE 只拼在单头路径里，已在单头整块验证）；GQA 的 KV 显式复制不在本块里（ticket 04）；
  int8 整块未做；无 KV cache；**仍是厂商编译路径**（`.dla` 由 ncc 生成）。

---

## 10. ticket 01：多头块的根因、修法与 h2 / h14 整块（2026-11-13）

> 分支 `impl/01-block-builder`（worktree `wt-t01`）· 生成器 `tools/build_block2.py` ·
> 编译 `ncc-tflite 7.3.15 --arch=mdla5.0 --disallow-bridge --num-mdla=2 --l1-size-kb=3072`（`tools/compile.sh`）。
> 本节是**修法定稿**：§9 的旧段落与 `ROUNDUP-2026-11-13.md` 里"多 head 在 importer 失败 / WIP"的说法**以本节为准**。

### 10.1 问题：一句含混的错误挡住了一整条路线

`tools/build_block2.py --heads 2` 编不过，ncc 只回一行：

```
ERROR: Fail to process the given model
EXIT=1
```

没有层清单、没有张量名，于是被读成"编译器不支持多头 attention"，并据此设计了"14 次单头 unroll"的兜底。
**真正的原因是构造器**：

| heads | `fc()` 声明的权重张量 | 数组真实形状 | 声明 / 实际 |
|---|---|---|---|
| 1（原路径） | `[HD,H]` = `[32,64]` | `(32,64)` | 1×（对） |
| 2 | `[H,H]` = `[64,64]`（4096 个数） | `(32,64)`（2048 个数） | 2× |
| 14 | `[896,896]` = 802,816 个数 | `(64,896)` = 57,344 个数 | **14×** |

flatbuffer 声明"这个缓冲区有 802,816 个数"，实际只写了 57,344 个 ⇒ ncc 在**模型加载期**就退出。
**修法**：权重张量按**数组真实形状**声明，并且形状与 `[n,k]` 不一致时**立刻在构造期 assert**，
错误信息里写清张量名、期望 `[n,k]`、实际 shape（`check_fc_weight()`）——把"加载期的一句含混话"
变成"构造期的一句指名道姓"。

### 10.2 三种错误措辞：先看它卡在哪一期（供后续排障引用）

| ncc 措辞 | 阶段 | 特征 | 本仓库的例子 |
|---|---|---|---|
| `ERROR: Fail to process the given model` | **加载 / 解析期**（flatbuffer → 内部图） | **没有层清单**，也没有张量名 | 本 ticket 的原 bug（声明比数据大）；`ops/rmsnorm_1x64_fp32` |
| `ERROR: Fail to process the input model` + `Invalid XXXLayer<N>{OP}: …` | **逐层校验期**（形状 / 类型传播） | **层清单会打印**，并在出错那层给 `Invalid <Layer>: <原因>`；EXIT=1 | `ops/rms896_d896_T8_vA0`：`Invalid RSqrtLayer<2>{RSQRT}: output shape is incompatible with inputs`；`ops/rms896_d896_T8_vA`：`Invalid ElementWiseAddLayer<2>{ADD}: input types are incompatible` |
| `Cannot find an execution plan` / `Bridging is disallowed but gap is detected …`（`IN_GAP`） | **后端期**（下沉 / 执行步规划） | 层清单**齐全**（每层都有 `XXXLayer`），失败发生在 "Input → MDLA_5_0" 的连边 | `ops/rmsnorm_8x4_fp16_row8b`（行 8 B < 16 B）；fp32 单算子 RSQRT |

**判据只有一条：看 `EXIT=`**；层清单打印出来**不等于**成功（§2.6 与 `docs/OP-rmsnorm.md §2.6` 各踩过一次）。
看到第一类措辞时，先怀疑自己的 flatbuffer 构造，再怀疑编译器能力。

### 10.3 假设 / 预测（**写于跑板之前**，事后未改）

- **H1（主假设）**：`--heads 2` 的失败是构造器 bug（声明宽度 ≠ 数组行数），不是编译器边界。
  - 预测：修好后 h2 块图 `EXIT=0`、**单个** `MDLA_5_0` 执行步、无 bridge；层数 ≈27（单头 42 层减掉 RoPE 的 2×5 层，
    多出 RESHAPE×3 / BMM×2）。
  - 预测：h2 板端 **PASS**（atol/rtol = 1e-2 已在 `model.json` 里**事前登记**）。
- **H2**：14 头**真实形状**（H=896、HD=64）同样只需要"构造正确"就能一次编译过。
  - 预测：30 层左右、单执行步；`.dla` 与 E-112 的 7.2 MB decoder 层同量级（数 MB，不是数十 MB）。
  - 预测：板端 PASS。风险：(a) 7 MB 资产 push / 加载慢；(b) 多头图**没有 1/√HD 缩放**，
    score 幅值大（本构造 std≈18），fp16 指数上界可能不够 ⇒ **若出 NaN/Inf 就记失败，不调数据掩盖**。
- **H3**：D=896 的 RMSNorm 必须用 FC-tile（依据 `docs/OP-rmsnorm.md`「D=896 三变体」：A / A0 编不过）。
  - 预测：`--norm mean` 的 14 头块图在 ncc 上 **FAIL**（`Invalid RSqrt` / `Invalid ElementWiseAdd`）。
- **H4**：单头路径行为不变。
  - 预测：默认参数下改后的 `build_block2.py` 产出的 `model.tflite` 与 `golden/expected.bin`
    与改前**字节完全一致**（md5 相同）。

### 10.4 做法与结果（编译侧）

**做法**：`fc()` 内先 `w = check_fc_weight(out, wts, n, k)`，声明的张量形状取 `list(w.shape)`；
多头路径的权重按"展平后的每头布局"造：`Wq/Wk/Wv = [heads*HD, H]`、`Wo = [H, heads*HD]`；
新增 `--hidden / --head_dim / --ffn / --norm` 四个参数；多头 golden 由 numpy 独立重算
（`[T,H] → [heads,T,HD] → 打分 → softmax → PV → 拼回`），**不调用图构造代码**。

| op | 命令 | 层数 | 执行步 | bridge | `.dla` | EXIT |
|---|---|---|---|---|---|---|
| `ops/block_qwen25_multhead_h2_fp16` | `--heads 2` | 27 | **1**（`MDLA_5_0`） | 无 | 103,508 B | **0** |
| `ops/block_qwen25_multhead_h14_fp16` | `--heads 14 --hidden 896 --head_dim 64 --norm fctile` | 30 | **1**（`MDLA_5_0`） | 无 | 7,230,244 B | **0** |
| `ops/_block_qwen25_multhead_h14_meannorm_control` | 同上但 `--norm mean` | 27 | **1**（`MDLA_5_0`） | 无 | 7,156,280 B | **0**（见 10.4.2） |

`Graph[0] Step[0] / Target: MDLA_5_0`、`===Graph[0] has 1 Execution Step(s)===` 见各目录的 `model.compile.log`；
没有任何 `Bridging is disallowed` / `Cannot find an execution plan` 行。

**形状守卫自测**（`/usr/bin/python3 tools/build_block2.py --selftest`，不写任何文件）：

| case | 声明 `[n,k]` | 数组 | 期望 | 实际 |
|---|---|---|---|---|
| `q_ok` | [32,64] | (32,64) | 接受 | 接受 |
| `q_old_heads2_bug` | [64,64] | (32,64) | AssertionError | AssertionError（"声明 4096 个数，数组 2048 个数"） |
| `q_too_wide` | [32,128] | (32,64) | AssertionError | AssertionError |
| `q_transposed` | [32,64] | (64,32) | AssertionError | AssertionError |
| `q_rank1` | [1,64] | (64,) | AssertionError | AssertionError |

⇒ **5/5**。错形状在**构造期**就炸，不会再伪装成"编译器不支持"。

#### 10.4.1 单头路径行为不变（H4 成立，字节级）

| 产物 | 改前 md5 | 改后 md5 |
|---|---|---|
| `model.tflite`（默认参数） | `5d1b686e06901dbb4561eaef61987281` | `5d1b686e06901dbb4561eaef61987281` |
| `golden/expected.bin` | `3066ad33c56d5132a324440a51f9076f` | `3066ad33c56d5132a324440a51f9076f` |

改前那份也单独编过一次：`EXIT=0`、`.dla` 91,788 B。

#### 10.4.2 H3 被**证伪**（编译侧）：D=896 的 MEAN 在整图里其实是能编的

`--norm mean` 的 14 头块图（`ops/_block_qwen25_multhead_h14_meannorm_control`）**EXIT=0**、27 层、
单执行步、无 bridge —— 与预测相反。而 `docs/OP-rmsnorm.md` 的独立模型 `ops/rms896_d896_T8_vA0`
在同一个 ncc 上仍然 `EXIT=1`（`Invalid RSqrtLayer<2>`）。

两图的算子链几乎一样，差别在构造器：`tools/build_rms896.py` 里 `act("mean", …)` 之后又用 `const(…)`
追加了 axis 张量，**然后**才写 `idx["mean"] = len(tensors) - 1` —— 于是 `idx["mean"]` 指向的是后加的
int32 axis 张量，`RSQRT` 的输入其实是一个 int32 张量（A 变体的 ADD 同理指到 eps 张量）。
这正是 §8 在 `build_block2.py` 里已经定过的同一个根因。
⇒ **"D=896 不能用 MEAN" 极可能是那个脚本的 bug，不是宽度边界**。

- 本 ticket **不改** `tools/build_rms896.py` / `docs/OP-rmsnorm.md`（不在写权限内），已把这条报给 Lead；
- 出货的 h14 仍用 `--norm fctile`（`docs/OP-rmsnorm.md` 里唯一有**板端数值**证据的 D=896 变体），
  MEAN 变体作为 control 留证（目录以 `_` 开头，`run_all.sh` 会自动跳过）；
- **UNKNOWN**：MEAN 变体在 D=896 的**板端数值**未测（只证明了编译）。

### 10.5 板端

（跑完补记于本节末。）

### 10.6 适用范围

- `ncc-tflite 7.3.15` / `mdla5.0`，**厂商编译路径**（`.dla` 由 ncc 生成）；
- 块图 = 1 层 decoder：RMSNorm ×2 + （单头：RoPE + 动态权重 FC 打分 / 多头：RESHAPE + 3-D BMM +
  SoftMax + 3-D BMM）+ O 投影 + 残差 + SwiGLU；
- 多头路径**没有 RoPE、没有 GQA 的 KV 复制、没有 1/√HD 缩放**（与单头路径同一套约定）；
- 头数维度：2 头（H=64 / HD=32）与 14 头（H=896 / HD=64）；`heads*head_dim` 必须等于 hidden；
- h14 的 `.dla` 7.2 MB 由权重决定（`896×896` 的 4 个投影矩阵），不是图规模问题。

### 10.7 剩余疑问（明说）

1. 多头 attention 的**板端数值**见 10.5；没有 1/√HD 缩放时 fp16 量程是真实的边界；
2. GQA（14 Q / 2 KV）的 KV 显式复制（`SPLIT→CONCAT`）不在本块里 —— E-112 只做编译侧，ticket 04 负责；
3. `--norm fctile` 多 1 个 CONCAT 与一个 28×896 的常量（比 MEAN 多 3 层）；MEAN 变体层数更少，
   但 D=896 的板端数值未验（10.4.2）；
4. h14 的 FFN 用 `--ffn 128`（默认），**不是** Qwen2.5-0.5B 的 4864；FFN 宽度不影响头数/下沉结论，
   "整块真实规模" 要等 ticket 09/10；
5. 多头图里的 softmax / BMM 是原生层（E-112 已证），但**时序 / 性能未测**（本文无性能结论）。

