# OP-qkv —— Q / K / V 投影（含 bias）+ RoPE（T=8, fp16, 真实 Qwen2.5-0.5B 形状）

> 票 **03**（`impl/03-qkv`，worktree `wt-t03`，基线 `feat/qwen25-attention` @ `655109a`）。
> 三个 op 目录（q / k / v 形状不同，而 `e2e.sh`+`compare.py` 只支持单输入单输出）：
> `ops/qkv_q_bias_rope_t8_fp16/`、`ops/qkv_k_bias_rope_t8_fp16/`、`ops/qkv_v_bias_rope_t8_fp16/`。
> 一个 builder：`tools/build_qkv_bias_rope.py --which q|k|v`。
> 权重来源：票 02 的冻结夹具（`weights_digest=898fa31…18a4`，25 张量 sha256 校验）；RoPE 表本票自算。

---

## 0. TL;DR

（§4–§6 跑完后回填）

| 项 | 计划/判据 |
|---|---|
| 图 | `FC(W[N,896],bias[N]) → [8,N] → RESHAPE [8,H,64] → SPLIT(2) → 4×MUL(cos/sin 常量表) → SUB/ADD → CONCAT` |
| 形状 | q `[8,896]→[8,14,64]`；k/v `[8,896]→[8,2,64]`；fp16 in/out；输出 **rank 3** |
| 必须钉死 | ① bias 真进图（bias_dropped 反事实 ≫ 阈值）② k/v 宽度 = 128 且记录"误接 896"的失败形态 |
| RoPE | Qwen2 split-half，head_dim=64，θ=1e6，positions 0..7，sin/cos 为**常量表**（SIN/COS importer 拒绝），T 静态 |
| 输入口径 | 图输入 = **input_layernorm 之后的 hidden** `n1`（norm 不在图里；eps=1e-6；gamma 在 norm 里，**不**折进 FC 权重） |
| 判据 | `compare.txt` PASS + 噪声底实测 + 反事实表 + 编译无 bridge/单执行步/层数 |

---

## 1. 这一票要回答什么

票面（`.scratch/qwen25-attention/issues/03-qkv-projection-bias-rope.md`）要的四件事：

1. fp16 下 q/k/v 三个投影与**独立** numpy golden 对上，**bias 参与计算**；
2. k/v 输出宽度确认为 **128**，并记录"如果误接成 896 会怎样"；
3. RoPE 用**预算常量表**（sin/cos 不能作为图内算子）、head_dim=64、T=8，与 golden 对上；
4. 输出为 **3-D**（`[T,14,64]` / `[T,2,64]`），记录 rank 上限是 4；外加：无 bridge、单执行步、实测噪声底。

本票**不做**：attention 打分（票 04）、掩码/SoftMax（05）、PV+O（06）、RMSNorm 本身的板端验证
（`docs/OP-rmsnorm.md` 已有独立证据）。

---

## 2. 事前登记：假设与预测（写于任何板端运行之前，2026-11-13）

量级数字来自**只读夹具 + numpy** 的静态分析（设备无关），不是实验结果。它是登记的**预期**。

| # | 预测 | 若预测不成立意味着 |
|---|---|---|
| P1 | 三个图都 `EXIT=0`、**单** `ExecutionStep`、`Target: MDLA_5_0`、无 bridge、层数 **11**（FC 1 + RESHAPE 1 + Slice 2 + Mul 4 + Sub 1 + Add 1 + Concat 1） | 有算子落到 CPU（bridge）或逐元素层被拆成 Slice/op/Concat 块（rank 或行宽触发） |
| P2 | 板端 `max|dev−golden|` ≈ **1 ULP of max|out|**：q ≤ 1.6e-2（max|out|=25.41, ULP=1.56e-2）、k ≤ 6.3e-2（77.56, ULP=6.25e-2）、v ≤ 1.3e-4（0.128, ULP=1.22e-4）；`rel_l2` ≤ 1e-3 | 大于 1 ULP 说明不是"存量 fp16 舍入"，而是数学/布局错 |
| P3 | 板端输出与**逐 op fp16 模拟**（`fp16_sim`）逐位相同的比例 ≥ 90% | 若 dev 与 sim 差很多，MDLA 的 FC 累加/舍入口径与模拟不同（要改口径，不改判据） |
| P4 | **bias 反事实**：`bias_dropped` 与真值差 q 25.22 / k 77.81 / v 0.0428，分别是登记阈值的 **254× / 257× / 58×** ⇒ 板端若匹配真 bias golden，就排除了"bias 被静默丢掉" | 若板端同时接近"有 bias"和"无 bias"（差 250×，不可能），或与两者都差很多 ⇒ 图里 bias 接错 |
| P5 | **布局/旋转反事实**：`no_rotation` q 35.7 / k 108.2 / v 0.253；`head_major_reshape`（`[H,T,64]`，NOTES §10.2 那类）q 35.4 / k 99.2 / v 0.256；k/v 用 q_proj（"k 当成 896 宽"的前 128 维）k 82.7 / v 20.2 —— 全部 ≫ 阈值 | 任何一个接近阈值 ⇒ 该布局/旋转没生效 |
| P6 | **896 负对照**（编译期，无需板子）：(a) 896 宽 FC 接 `[8,2,64]` reshape（元素 7168 vs 1024）在 ncc **逐层校验期**报错；(b) 权重张量声明 `[896,896]` 但只写 128×896 个数，在**加载期**报 `Fail to process the given model`（NOTES §5 坑 2 / E-112 根因） | 两者都"能编过" ⇒ 这两类错误编译器不挡，只能靠 golden/形状检查挡 |
| P7 | **eps 口径不可判别**：eps=1e-6 vs 0 只让 q 输出动 1.56e-2 = **1 ULP**（阈值的 16%）；而"漏掉 gamma"会让 q 动 8.10（阈值的 82 倍）。即 gamma 必须记清、eps 在本算子里测不出来 | 若 eps 造成 >1 ULP 差异，说明 n1 的 rms 尺度接近 eps（本夹具 rms≈1.5e-2，离 1e-6 很远，不会发生） |

**容差登记（跑之前写进 `model.json`，规则见 NOTES §4/§10.1）**：`max|out|` 是 25.41 / 77.56，远大于 1，
§4 的"阈值 < 2e-2"在这两个 op 上等价于 7.9e-4 / 2.6e-4 **相对**误差，比 1 个 fp16 ULP 还紧 ⇒ 走 §10.1 例外：

* 下限由**设备无关**的 `fp16_sim`（numpy 逐 op 舍入模拟）实测：三者都是 **1.00 ULP**；
* 登记 `atol = 4·2⁻¹¹·max|out|`、`rtol = 4·2⁻¹¹`（k=4，阈值 ≈ 实测下限的 4 倍）；
* v 的 `max|out|=0.128 < 1`，§4 **原样适用**（atol/rtol 同样由实测下限推，远小于 2e-2）。

---

## 3. 做法

### 3.1 图（三个 op 只差 heads / 宽度 / 权重张量）

```
input [8,896] fp16
  → FULLY_CONNECTED(W[N,896] const, bias[N] const, keep_num_dims)   [8,N]
  → RESHAPE(NewShape=[8,H,64])                                      [8,H,64]     token-major
  → SPLIT(axis=2, num_splits=2)                                     [8,H,32]×2
  → MUL ×4（cos/sin 常量表，与激活同形状 [8,H,32]，**不用广播**）
  → SUB / ADD                                                       lo=a·cos−b·sin, hi=b·cos+a·sin
  → CONCAT(axis=2)                                                  [8,H,64]  = output
```

* q：`H=14, N=896`；k/v：`H=2, N=128`。
* 输入 `[8,896]`、输出 `[8,14,64]`/`[8,2,64]`：**rank 3**（`rank ≤ 4` 是 ncc 硬限制；
  rank-4 的 `[1,T,H,64]` 在 RoPE 上要 35 层，rank-3 只要 9–11 层，`docs/OP-rope.md` §5.3）。
* `keep_num_dims=True`、`weights_format=DEFAULT([N,K])`、`fused_activation=NONE`。

### 3.2 输入口径（票面要求记录"选了哪个"）

**选 B：图输入 = input_layernorm 之后的 hidden `n1`，RMSNorm 不在图里。**

```
n1 = fp16( embed.weight[TOKEN_IDS] / sqrt(mean(x²) + 1e-6) · layer0.input_layernorm.weight )
```

* `eps = 1e-6`（transformers `rms_norm_eps`），**不是**玩具块的 0；
* `gamma = layer0.input_layernorm.weight` **在 norm 里用掉，不折进 FC 权重**（折了的话 numpy golden
  也必须同步折，多一个静默错的机会）；
* `TOKEN_IDS = [9707, 11, 1879, 828, 11780, 13, 220, 16]` 固定，`golden/input.bin` 逐字节可复现；
* 为什么不在图里：RMSNorm 有自己的票与板端证据（`docs/OP-rmsnorm.md` 的 FC-tile+CONCAT 变体），
  塞进来只会让失配变成双因；为什么不用裸 embed（不 norm）：那样输入尺度是 0.015 的原始嵌入，
  与"真正喂给 q_proj 的东西"差一个量级。

`n1` 实测：`max|n1| = 1.0508`、std 0.0989 —— 与真实 Qwen2.5-0.5B（`layer0.input_layernorm`
增益 mean 0.032 / std 0.092）量级一致。

### 3.3 RoPE：常量表 + 约定 + 一个陷阱

* **Qwen2 split-half**：`out_lo = a·cos − b·sin`，`out_hi = b·cos + a·sin`（a=x[...,:32], b=x[...,32:]）。
  任务书里曾写错成 `b·sin + a·cos`，那不是旋转（不保 2-norm）——`docs/OP-rope.md` §2 的活教材。
  builder 里保留两道防线：golden 写盘前断言**每对 (a,b) 的 2-norm 不变**（`rotation_invariant_rel_err`
  写进 `model.json`），且 `--diag` 里 `no_rotation` 是登记反事实。
* **sin/cos 是常量表**：ncc importer 拒绝 `SIN`/`COS`（`docs/OP-rope.md` §0），所以表必须是数据；
  **代价是 T 必须静态**。本票按 prefill 的固定窗口做 const 表（positions 0..7）。
  decode 每步换表就得走"表当第二输入"的 dyn 路径（+2 个 SliceLayer，`docs/OP-rope.md` §4），
  **本票没做**（见 §9）。
* **表按 head 复制**成 `[8,H,32]`，而不是用 `[8,1,32]` 广播：rank-3 激活 × `[T,1,h]` 表在本仓库
  **没有板端证据**（只有 D=64 的 `[T,64]×[T,1]`，`docs/OP-rmsnorm.md`）。复制代价是 Static 7 KB（q）。
* **heads 是 token-major**：`[T,896] → [T,14,64]`。`[14,T,64]` 是登记反事实（diff 35.4）——
  NOTES §10.2 的事故类型（`[T,H]` 直接 reshape 成 `[heads,T,HD]`）在这里**不适用**：本算子不做
  head 与 token 的转置，只做最后一维的切分；能不能真对上由板端 `compare.txt` 说话，不由"编译过了"说话。

### 3.4 bias：为什么它是本票的核心

`config.json` 里**没有任何 bias 键**（q/k/v 的 bias 是 transformers 源码里硬编码的），而夹具实测
`q_proj.bias` std≈7.8、`k_proj.bias` std≈29.8（真实权重实测 7.85 / 32.2），**远大于权重本身**（0.067 / 0.081）。
所以 bias 是 graph 的第三个 FC 输入，且"漏掉它"这件事有一个量化上界很紧的判据：
`bias_dropped` 反事实与真值差 25.2 / 77.8，而登记阈值是 0.099 / 0.303 ⇒ 差 **250 倍**。

### 3.5 golden 与 fp16 下限

* `golden/expected.bin`：numpy float64，输入是**已存储的 fp16**（输入/权重/bias/表），**只在最后**舍入一次到 fp16。
  不调用 ncc / TFLite interpreter / 设备，也不与 flatbuffer 构造共享代码（R3）。
* `fp16_sim`（诊断用，不进 golden）：float32 FC + FC 输出按 fp16 存储 + RoPE 逐 op 舍入到 fp16 —— 模拟
  MDLA 的行为（`docs/OP-rope.md` §6 实测"逐 op fp16 模拟与设备逐位相同"）。它与 golden 的差就是
  **设备无关下限**，用来定阈值（`model.json.fp16_floor`）。

---

## 4. 编译器结果

（待填）

## 5. 板端结果 / 噪声底

（待填）

## 6. 反事实与负对照

（待填）

## 7. 票面 checklist 对照

（待填）

## 8. 适用范围 / 不适用

（待填）

## 9. 剩余疑问（UNKNOWN）

（待填）

## 10. 复现

（待填）

## 11. 交接

（待填）
