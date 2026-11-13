# OP-pv-o —— PV（3-D BMM）+ O 投影 + 残差（Qwen2.5-0.5B 真实形状 · T=8 · fp16）

范围：注意力输出那一半，一个可复现的单算子图：

\`\`\`
p [14,8,8] × v [14,8,64]  --BATCH_MATMUL(3-D, adj 0/0)-->  a3 [14,8,64]
  --TRANSPOSE perm=[1,0,2]--> [8,14,64] --RESHAPE--> a [8,896]
  --FULLY_CONNECTED(W=o_proj[896,896], bias=0)--> o [8,896]
  --ADD(x, o)--> out [8,896]                     # x 是残差输入
\`\`\`

- 分支 \`impl/06-pv-o\`（worktree \`wt-t06\`）· builder \`tools/build_pv_o_residual.py\` · 产物 \`ops/pv_o_residual_t8_fp16/\`
- 权重：票 02 的冻结夹具（\`weights/qwen25_frozen\`，\`weights_digest=898fa3195d627a4c…\`），**真实实测尺度**；
  取 \`layer0.attn.q_proj.*\` / \`k_proj.*\` / \`v_proj.*\` / \`o_proj.weight\` / \`input_layernorm.weight\` 与 \`embed.weight\`
- 编译器 \`ncc-tflite\` **7.3.15** · \`--arch=mdla5.0 --num-mdla=2 --l1-size-kb=3072 --opt=3 --disallow-bridge\` · 板 = Titan 1（mt6897 / Android 16）
- 板端全部经 \`tools/devlock.sh\` 串行；每次板端实验前后各一次 \`healthcheck.sh --quick\`（4 次全程 PASS）

---

## 0. 事前登记：假设与预测（写于任何编译/板端动作之前）

### 0.1 三个设计选择（先说清楚，避免结果被误读）

| # | 选择 | 理由 | 代价 / 风险 |
|---|---|---|---|
| D1 | **PV 走 3-D**（\`[H,T,T]×[H,T,HD]\`），不走 4-D \`[1,H,T,T]\` | E-112（\`docs/OP-gqa.md\`）实测 3-D 是 1 个 \`BatchMatMulLayer\`，4-D 被降成 5 层；层数预算按 3-D 记 | 3-D 的输出布局是 \`[H,T,HD]\`，必须先 \`TRANSPOSE perm=[1,0,2]\` 再 \`RESHAPE\` 成 \`[T,896]\`（NOTES §10.2 的教训：**直接 reshape 会编译通过但数学错**） |
| D2 | **p、v 打包成一个运行时输入** \`[14,8,72]\`，图内用两次 \`SLICE\` 拆成 p \`[14,8,8]\` / v \`[14,8,64]\`；第二个运行时输入是残差 x \`[8,896]\` | \`tools/e2e.sh\` 的缝只有 \`golden/input.bin\` + \`golden/input1.bin\`（NOTES §3、§8 不许改公共工具）。三个语义输入（p / v / x）只能压成两个文件。**不用**"把 x 做成常量"——常量 ADD 有被折进 FC bias 的风险，那样"残差加法"就没有层证据了 | 图上多 2 个 \`SLICE\` 层；"PV 一层"的判据要按层名数 \`BatchMatMulLayer\`，不能按总层数 |
| D3 | **p 用"真实权重 + 固定输入走一遍完整 attention"得到**（不是随便造的概率矩阵）：\`x0 = embed.weight[0:8]\` → RMSNorm(\`input_layernorm.weight\`, eps=1e-6) → q/k/v 投影（含 bias）→ RoPE(θ=1e6, split-half) → scores/√64 → 因果掩码 → softmax(\`axis=-1\`, float32)；v 按 GQA 分组（head h 用 kv head \`h//7\`）复制到 14 头 | 票据给的两条路里，这条路让 p / v / x 的**动态范围**来自真实权重尺度（\`q_proj.bias\` std≈7.85、\`k_proj.bias\` std≈29.8），而不是人手挑的数；同时它把"上游算子（票 03/04/05）在 CPU 上的语义"也钉住一份可复现数据 | p 只是**图输入数据**，其推导不进图；若它退化成近 one-hot（见 P1），单次对拍的"累加"判别力会变弱——已准备 alt-input 对照 |

### 0.2 假设 → 预测

| # | 假设 | 若成立，预期观察 |
|---|---|---|
| H1 | MDLA 把 3-D \`BATCH_MATMUL\` 映成一个原生 matmul 层（E-112 的结论在"头数=14、T=8、fp16、B 侧 64 宽"这组形状上仍成立） | exec plan 里 \`BatchMatMulLayer\` **恰好 1 个**，\`EXIT=0\`，\`Target: MDLA_5_0\`，无 bridge |
| H2 | 4-D 版 PV 被降成 5 层（切分/矩阵乘/切分/矩阵乘/拼接） | 4-D 对照模型的 exec plan 出现 **5 个层**（\`Slice\`/\`BatchMatMul\` 各 2 + \`Concat\` 1），与 3-D 的 1 层形成对照 |
| H3 | 整图（SLICE×2 → BMM → TRANSPOSE → RESHAPE → FC → ADD）全落 MDLA、单执行步 | \`Graph[0] Step[0]\` 只有一个 step；层清单含 2×Slice + BatchMatMul + Transpose + Reshape + FullyConnected + ElementWiseAdd（编译器可能把 TRANSPOSE/RESHAPE 之一折进相邻层，若层数少于此，按实际层名解释，**不硬凑**） |
| H4 | 板端数值 = "每层边界舍入到 fp16 + 层内 fp32 累加"的量级 | \`max_abs\` 与 builder 里 \`--fp16sim\` 算出的**层边界 fp16 下限**同量级（设备只多 1–2 ULP），而不是 \`2e-2\` 这种松阈值 |
| H5 | 硬件对同一 \`.dla\`、同一输入是确定性的 | 两次运行的 \`max|r1-r2| = 0\`（噪声底为 0），阈值退化为 fp16 末位口径 |
| H6 | 输入布局错（少一次 TRANSPOSE / SLICE 偏移错 / FC 权重转置）会大面积失配，而不是"看起来差不多" | 若 H1–H5 全部成立但数值 FAIL，第一嫌疑按 D1/D2 的顺序排（reshape 顺序 → slice 偏移 → FC 权重方向） |

### 0.3 三个数值预测（可在结果里被推翻）

| # | 预测 | 若被推翻意味着 |
|---|---|---|
| P1 | \`p\` **近 one-hot**：\`k_proj.bias\` std≈29.8 主导 scores，单头 scores 量级 ~10²，softmax 后 \`max(p) > 0.99\` | 说明我对 bias 量级的估计错了 → 要复查 q/k/v 的推导（阈值/缩放/RoPE 位置） |
| P2 | \`max|out|\` ≈ \`max|x|\` ≈ 0.05 量级（layer-0：x 是 embed 行 std≈0.0152，O 投影输出更小），即 \`2e-2\` 绝对阈值比输出本身还大 → **不适用 NOTES §4 的默认口径**，阈值必须由 fp16 模拟下限支撑（NOTES §10.1） | 若 \`max|out| >> 1\`，说明激活尺度与我的推导不符（例如 v 的量级被低估） |
| P3 | 板端 \`rel_l2\` 在 1e-3 量级或更小（fp16 末位） | 若 \`rel_l2 ~ 1e-1\`，几乎必然是布局错（§10.2 那类"编译通过但数学错"）而不是精度问题 |

---

## 1. 做法

### 1.1 图（7 层，全部在 MDLA_5_0）

| # | 层 | 作用 | 张量 |
|---|---|---|---|
| 1-2 | \`SLICE\` ×2 | 从打包输入拆出 p / v（D2 的搬运层，**不属于数学**） | \`in_pv [14,8,72]\` → \`p [14,8,8]\`、\`v [14,8,64]\` |
| 3 | \`BATCH_MATMUL\` | **PV**，3-D，\`adj_x=0 adj_y=0\` | \`p\`×\`v\` → \`a3 [14,8,64]\` |
| 4 | \`TRANSPOSE\` perm \`[1,0,2]\` | \`[H,T,HD]\` → \`[T,H,HD]\`（D1 / NOTES §10.2：少了这一步数学就错） | \`a3\` → \`at [8,14,64]\` |
| 5 | \`RESHAPE\` | 14 头拼回 896 宽（\`out[t, h*64+d]\`） | \`at\` → \`a [8,896]\` |
| 6 | \`FULLY_CONNECTED\` | **O 投影，无 bias**（W = 夹具 \`layer0.attn.o_proj.weight\` \`[896,896]\`；仍喂一个全零 \`[896]\` fp16 bias 张量，NOTES §5.1：FC 要第三个输入） | \`a\` → \`o [8,896]\` |
| 7 | \`ADD\` | **残差**（x 是第二个运行时输入，动态） | \`x\`+\`o\` → \`out [8,896]\` |

### 1.2 输入数据（D3）

\`\`\`
x0 = embed.weight[0:8]                      # [8,896] fp16，std 0.0150，|max| 0.0589
n1 = RMSNorm(x0, layer0.input_layernorm.weight, eps=1e-6)
q  = RoPE(n1·Wqᵀ + bq, θ=1e6)               # [8,896] -> [14,8,64]
k  = RoPE(n1·Wkᵀ + bk, θ=1e6)               # [8,128] -> [2,8,64] -> repeat 7x -> [14,8,64]
v  = n1·Wvᵀ + bv                            # [8,128] -> [2,8,64] -> repeat 7x -> [14,8,64]
scores = q·kᵀ/√64 + 因果掩码(-inf)  ->  p = softmax(float32)   # [14,8,8]
\`\`\`

p 的实测统计量（\`model.json:p_stats\`，全部 fp16 存储值）：

| 量 | 主输入（temperature 1.0） | alt 输入（temperature 0.02，合成） |
|---|---|---|
| \`max(p)\` | **1.000** | 1.000 |
| 元素 \`p>0.99\` 占比 | **10.4%** | 1.6% |
| 平均熵（bit，满熵 3） | **0.086** | 1.573 |
| 未掩码 \`|scores|\` 最大值 | **578.3** | 11.57 |
| 行和误差 | 9.4e-08 | 8.9e-08 |

→ **P1 成立**：真实权重（\`k_proj.bias\` std≈29.8、\`q_proj.bias\` std≈7.85）下 scores 到 10² 量级，
softmax 几乎是 one-hot。这正是"夹具复刻了真权重的 bias 量级"的直接后果（\`docs/OP-weights.md\` §4）。
为了让 **PV 的累加**也被真正压到（而不只是"选一行 V"），额外做了一个合成对照输入
（temperature 0.02，平均熵 1.57 bit，见 §3.3 的 \`compare_altp.txt\`）；它**不是**模型语义内的 p，
只作布局/累加判别用。

### 1.3 容差登记（**先登记，后跑**）

\`model.json\` 里 \`tolerance_revision = 1\`，\`atol = rtol = 4·2⁻¹¹·max|out| = 1.572e-04\`：

| 量 | 值 | 说明 |
|---|---|---|
| \`max|out| = max|golden|\` | 0.08051 | 其中 \`max|x| = 0.0589\`、\`max|o_proj| = 0.0720\`、\`max|attention(拼回 896 宽)| = 0.1367\` |
| 设备无关的 fp16 下限（\`--fp16sim\`：BMM/FC/ADD 每个层边界各舍入一次，层内 fp32） | **6.104e-05** | = \`max|out|\` 的 **1 个 ULP**（2⁻¹⁴） |
| 登记阈值 | **1.572e-04** | = 下限的 **2.58×**；相对口径 1.95e-3；绝对 1.6e-4 **≪ 2e-2**（NOTES §4 的上界） |

**为什么不直接抄 NOTES §4 的 2e-2**：本算子输出量级 0.08，2e-2 绝对值是输出本身的 25%，
任何布局错都能"通过"，判据会失效。因此按 NOTES §10.1 的例外规则，用**独立于设备的 fp16 模拟下限**
（numpy，与生成 flatbuffer 的代码零共享）支撑阈值。**这不是事后调阈值**：阈值在第一次板端运行之前
就写进 \`model.json\`，且第一次运行就 PASS（观测误差 6.10e-05 = 下限的 1.00×、阈值的 0.39×）。

### 1.4 两个运行时输入的打包（D2）

\`golden/input.bin\` = \`[14,8,72]\`（每个 \`(head,token)\` 行 = p 行 8 个值 + v 行 64 个值），
\`golden/input1.bin\` = \`x [8,896]\`。理由见 D2；代价是图上多 2 个 \`SLICE\`
（\`SliceLayer<0>{SLICE}\`、\`SliceLayer<1>{SLICE}\`）。**层数判据因此按层名数**：
\`BATCH_MATMUL\` 在整图里**只有 1 个**（\`BatchMatMulLayer<2>\`）。

---

## 2. 编译结果（\`model.compile.log\` 原文）

### 2.1 主图：7 层、单执行步、全 MDLA_5_0、EXIT=0

\`\`\`
===Graph[0] has 1 Execution Step(s)===
Graph[0] Step[0]
 ├ StepId: 0
 ├ Target: MDLA_5_0
 └ Subgraph:
   ├ SliceLayer<0>{SLICE}
   ├ SliceLayer<1>{SLICE}
   ├ BatchMatMulLayer<2>{BATCH_MATMUL}
   ├ TransposeLayer<3>{TRANSPOSE}
   ├ ReshapeLayer<4>{RESHAPE}
   ├ FullyConnectedLayer<5>{FULLY_CONNECTED}
   ├ ElementWiseAddLayer<6>{ADD}
   └ Output: OpResult (external)
DRAM Usage: [0]: MDLA 5.0  Input 29K  Output 14K  Temp 0  Static 1.5M  Code 11K  Total 1.6M
EXIT=0   SIZE=1630614
\`\`\`

→ **H1 / H3 成立**：\`BatchMatMulLayer\` 恰好 1 个；\`TRANSPOSE\` 与 \`RESHAPE\` **没有被折进** BMM/FC
（两个层都真出现），残差是独立的 \`ElementWiseAddLayer\`（**没有被折进 FC 的 bias**——这正是 D2 里
"x 必须动态"的理由被证实的部分）。\`--disallow-bridge\` 下无 bridge，单执行步。

### 2.2 对照：3-D = 1 层，4-D = 5 层（\`ops/pv_bmm{3,4}d_control_t8_fp16/\`）

| 对照 | 图 | \`EXIT\` | 层清单（原文） | \`.dla\` SIZE |
|---|---|---|---|---|
| 3-D | \`p[14,8,8] × v[14,8,64]\` | 0 | \`BatchMatMulLayer<0>{BATCH_MATMUL}\`（**1 层**） | 23394 |
| 4-D | \`p[1,14,8,8] × v[1,14,8,64]\` | 0 | \`SliceLayer<0>{BATCH_MATMUL}\`、\`BatchMatMulLayer<0>\`、\`SliceLayer<0>\`、\`BatchMatMulLayer<0>\`、\`ConcatLayer<0>\`（**5 层**） | 43978 |

→ **H2 成立**，与 E-112（\`docs/OP-gqa.md\`）逐项一致：4-D PV 被降成"切分/矩阵乘/切分/矩阵乘/拼接"，
且 DRAM \`Temp\` 从 0 涨到 14K（多了中间物化）。**不采用 4-D**：同样一次 PV，层数 1→5、临时缓冲 0→14K。

### 2.3 负对照：没有 \`TRANSPOSE\` 也能编译（\`ops/pv_o_badlayout_control_t8_fp16/\`）

| 图 | \`EXIT\` | 层数 | 板端 |
|---|---|---|---|
| \`BMM → RESHAPE[H,T,HD]→[T,896] → FC → ADD\`（**故意错**） | **0** | 6（少一个 Transpose） | **FAIL，rel_l2 = 0.923**（§3.4） |

→ 这就是 NOTES §10.2 说的"EXIT=0 + 单执行步 ≠ 数学正确"。它同时证明主图的 PASS **不是空的**：
判据真的能抓住"头布局错"这一类错误。

### 2.4 产物可复现性

- \`.tflite\`：同一 builder 连跑两次 **逐字节相同**（\`sha256=638f033272132c2710c5eb0b3bec2dab…\`）；golden 与 alt 输入也逐字节相同。
- \`.dla\`：同一个 \`.tflite\` 两次编译 **sha256 不同**（\`14e25e498f45f4a7…\` vs \`29d23db134635733…\`，SIZE 都是 1630614）
  —— 复验了仓库既有规则：**\`.dla\` 哈希不能当回归判据**。

---

## 3. 板端结果

### 3.1 主契约（\`compare.txt\`，\`ops/pv_o_residual_t8_fp16/\`）

\`\`\`
op        : PV_3D_BMM__O_PROJ__RESIDUAL {'T': 8, 'H': 14, 'KV': 2, 'HD': 64, 'D': 896, 'PV_rank': 3}  dtype=float16
expected  : 7168 vals   device: 7168 vals   logical shape=(8, 896)
max_abs   : 6.104e-05
rmse      : 6.739e-06
rel_l2    : 3.119e-04
nan/inf   : exp 0/0  dev 0/0
range     : exp [-0.08051,0.08014]  dev [-0.08051,0.08014]
tol       : atol=0.000157245 rtol=0.000157245
PASS
\`\`\`

- 相对口径：\`max_abs / max|exp| = 7.58e-04\`，\`rel_l2 = 3.12e-04\`（**P3 成立**：fp16 末位量级）。
- 绝对误差 6.10e-05 = \`max|out|\` 的 **1 个 ULP**（2⁻¹⁴），等于 §1.3 里设备无关的下限本身。
- 逐元素：7168 个里 **66.3% 与 golden 逐位相同**，169 个差 3.05e-05（1 个"大元素 ULP"），2 个差 6.10e-05。
- 端到端 \`max|out|\` 的 6.10e-05 与"FC 权重方向写反/拼头顺序写反"的误差量级（至少 1e-2）相差两个数量级，
  所以 PASS 说明 **BMM 布局、拼头顺序、FC 权重方向、残差位置**全部正确。

### 3.2 噪声底（NOTES §4 的测法）

同一 \`.dla\`、同一输入，连跑 **3** 次（每次 \`e2e.sh\`，板端 \`-c 3\`）：

| 运行 | \`out_*.bin\` sha256 | \`compare.txt\` 的 max_abs |
|---|---|---|
| run1 | \`67706d0f4231d770…\` | 6.104e-05 |
| run2 | \`67706d0f4231d770…\`（**逐字节相同**） | 6.104e-05 |
| run3 | \`67706d0f4231d770…\`（**逐字节相同**） | 6.104e-05 |

**噪声底 = max|r1−r2| = 0.0（三次输出逐字节相同）**。按 NOTES §4：噪声底为 0 时阈值退化为
fp16 末位口径（约 1e-3 相对）——本次登记值正好是 1.95e-3 相对（§1.3），并额外有"设备无关 fp16 模拟下限"
支撑，不是凭"噪声底 ×3–5"硬套（下限为 0 时该规则无定义）。**H5 成立**。

### 3.3 换输入：同一个 \`.dla\`，两次不同操作数

| 输入变体 | 变了什么 | \`compare\` 文件 | max_abs | rmse | rel_l2 | 结果 |
|---|---|---|---|---|---|---|
| 主输入 | — | \`compare.txt\` | 6.104e-05 | 6.739e-06 | 3.119e-04 | **PASS** |
| \`altp\`（temperature 0.02） | p 变平（熵 0.086→1.57 bit），v/x 不变 | \`compare_altp.txt\` | 6.104e-05 | 5.892e-06 | 3.129e-04 | **PASS** |
| \`altx\`（embed 行 8..15） | **p、v、x 三个操作数全变** | \`compare_altx.txt\` | 6.104e-05 | 6.382e-06 | 3.074e-04 | **PASS** |

- 三次的输出文件 sha256 互不相同（\`67706d0f…\` / \`6c6c14b1…\` / \`93b81f91…\`），且各自与**自己的** expected 对上
  → 三个运行时操作数都是**运行期读取**的，不是被编译期折掉的常量。
- \`altp\` 的作用：主输入的 p 近 one-hot（10.4% 元素 >0.99），PV 近似"选一行 V"；摊平后 8 个位置都参与累加，
  **PV 的累加路径**也被压到了（虽然它仍是合成输入，语义上不是模型内的 p）。
- \`altx\` 的作用：残差 x 与 p/v 一起换，证明 **ADD 的第二个操作数**是动态输入（它在 §2.1 里是独立层，
  不是被折进 FC bias 的常量）。

### 3.4 负对照的板端结果（\`ops/pv_o_badlayout_control_t8_fp16/compare_wronglayout.txt\`）

\`\`\`
max_abs   : 8.762e-02
rmse      : 1.995e-02
rel_l2    : 9.233e-01
range     : exp [-0.08051,0.08014]  dev [-0.07538,0.08704]
tol       : atol=0.000157245 rtol=0.000157245
FAIL
\`\`\`

→ 少了 \`TRANSPOSE\` 的图**编译 EXIT=0、单执行步、跑得起来**，但 \`rel_l2 = 0.92\`（几乎完全不相关）。
这正是 NOTES §10.2 记录的陷阱；也说明本 op 的 PASS 有判别力。

### 3.5 设备 vs "fp16 逐层舍入"模拟：7168 个元素里只有 **2** 个不同

| 位置 | device | fp16sim | golden(float64) |
|---|---|---|---|
| \`[2,758]\` | 0.014846802 | 0.014854431 | **0.014846802**（device 恰好等于精确值） |
| \`[6,106]\` | −0.045837402 | −0.045776367 | −0.045806885 |

即：设备行为 = "每个层边界舍入到 fp16、层内 fp32 累加"这个模型，**7166/7168 逐位吻合**
（包括 \`max_abs\` 两者都恰为 6.1035e-05）。这比"差 1–2 ULP"的预期更强，也因此把
"设备到底怎么累加"缩到了 2 个元素上（UNKNOWN，见 §7.2）。

### 3.6 设备状态

| 时点 | \`healthcheck.sh --quick\` |
|---|---|
| 主实验（healthcheck → altp → run1 → run2 → healthcheck） | PASS / PASS |
| 负对照实验 | PASS / PASS |
| altx 实验 | PASS / PASS |
| 最终主契约复跑 | PASS / PASS |

全程无 crash / 无 leak 标记，known-good op \`fc_1x64x64_int8\` 每次都 PASS → 本算子**没有污染后续任务**。

---

## 4. 判据与阈值审计（一次说清）

| 问题 | 回答 |
|---|---|
| 阈值哪来的？ | \`atol = rtol = 4·2⁻¹¹·max|out| = 1.572e-04\`（NOTES §10.1 的 k≈4 形式），由设备无关的 fp16 模拟下限 6.104e-05 支撑（2.58×） |
| 是先登记还是先跑？ | **先登记**：\`tolerance_revision=1\` 与理由写在 \`model.json\`，第一次板端运行之前就在文件里；第一次运行即 PASS（观测 6.10e-05 = 阈值的 0.39×），**没有事后放宽** |
| 噪声底多少？ | **0.0**（三次运行逐字节相同） |
| 观测误差 / 下限 / 阈值 | 6.104e-05 / 6.104e-05（1.00×）/ 1.572e-04（2.58×） |
| 相对口径 | \`max_abs/max|exp| = 7.58e-04\`，\`rel_l2 = 3.12e-04\` |
| 覆盖 2e-2 吗？ | 1.572e-04 **≪ 2e-2**（更严），且 2e-2 本身会把输出（0.08）的 25% 当噪声，已论证不可用 |

---

## 5. 与票面 checklist 对照

| 票面要求 | 状态 | 证据 |
|---|---|---|
| PV 走 3-D，编译后确认为**一个**原生层 | ✅ | §2.1 \`model.compile.log\` 里整图只有 \`BatchMatMulLayer<2>{BATCH_MATMUL}\` 一个；3-D 单独对照也是 1 层（§2.2） |
| 记录 4-D 会降成 5 层并说明不采用 | ✅ | §2.2：4-D 对照 5 层（Slice+BMM+Slice+BMM+Concat），\`.dla\` 43978 B vs 23394 B、DRAM Temp 14K vs 0 → 不采用 |
| O 投影**不带 bias**，与 golden 对上 | ✅ | §1.1 第 6 层；W=\`layer0.attn.o_proj.weight\`，bias 喂全零张量（NOTES §5.1 的第三个输入）；golden 不含 bias；PASS 的误差量级（1 ULP）排除"权重方向写反"（那会差 ~1e-2） |
| 残差加法与 golden 对上 | ✅ | \`ElementWiseAddLayer<6>{ADD}\` 独立成层（没被折进 FC）；x 是运行时输入，\`altx\` 换值后仍 PASS（§3.3） |
| 单执行步、无 bridge | ✅ | \`===Graph[0] has 1 Execution Step(s)===\`；\`--disallow-bridge\` 下 \`EXIT=0\` |
| 记录本算子的实测噪声底 | ✅ | §3.2：**0.0**（3 次运行逐字节相同） |

---

## 6. 适用范围 / 不适用

**适用（可被下游直接引用）**：

1. **PV 用 3-D BMM**：\`[H,T,T]×[H,T,HD]\`、\`adj 0/0\`、fp16、H=14、T=8、HD=64 —— 1 个原生层、板端 1 ULP 内对上。
2. **拼头必须 \`TRANSPOSE[1,0,2]\` 再 \`RESHAPE\`**：负对照实测 \`rel_l2=0.92\`（§3.4）。这条与 NOTES §10.2 互相印证，
   票 09（整块）可以直接照抄这一步。
3. **O 投影无 bias**：FC 第三个输入喂全零 \`[896]\`；\`o_proj\` 权重按 \`[N,K]=[896,896]\` 直接喂（设备算 \`a·Wᵀ\`）。
4. **残差是动态输入时 ADD 不会被折进 FC bias**（本图上 \`ElementWiseAddLayer\` 独立存在）。
5. **阈值口径**：当输出量级 ≪1 时，NOTE §4 的 2e-2 会失去判别力；用 \`4·2⁻¹¹·max|out|\` + fp16 模拟下限是可复用的做法。
6. **\`--disallow-bridge\` 下**：\`SLICE\`（rank-3、切片末维 8/64 fp16 = 16/128 B）、\`TRANSPOSE\`（rank-3 perm \`[1,0,2]\`）、
   \`RESHAPE\`、\`FC\`、\`ADD\` 在这组形状上都是原生层（本图 7 层全是）。

**不适用**：

- 性能结论：板端 703–1130 FPS 是"输入 8 KB、单步、启动开销主导"的小图，**不能**当吞吐/延迟结论。
- T≠8、H≠14、HD≠64、int8 路径、KV cache / 增量解码（本 spec 本就排除）。
- 只覆盖"p、v 已算好"的边界；p 本身来自 CPU 侧 numpy（票 04/05 的板端 SOP 不在本票）。

---

## 7. 剩余疑问（UNKNOWN，不圆）

1. **p 近 one-hot 是真实权重的性质，不是测试缺陷**：主输入平均熵 0.086 bit、10.4% 元素 >0.99。
   因此"单次主板端对拍"主要压的是**布局**（选对哪一行 V），**不是累加**。已用合成的 altp
   （temperature 0.02、熵 1.57 bit）补一次，但那是**语义外**的输入；若要把"真实 p 下的累加"也压住，
   需要票 03/04/05 的板端 SOP 输出作为 p 的来源（本票没有）。
2. **设备与 fp16 模拟的 2 个元素差异（§3.5）机制未知**：\`[2,758]\` 上 device 恰好等于 float64 精确值，
   \`[6,106]\` 上 device 比模拟值更接近精确值。候选解释：FC 的 896 项累加顺序/分块与 numpy 不同、
   或 MDLA 对部分通道用了更高的中间精度。要定死需要把 \`a\`、\`o\` 分别暴露成图输出的中间量探针（本票未做）。
3. **绝对误差由最大量级主导**：\`max_abs/ULP(该元素的 expected)\` 最大到 ~119（某些输出元素接近 0，
   而误差是 6.1e-05 的绝对量）。→ 票 09/10 若用纯 rtol 判据会在近零元素上误判，**必须以 atol 为主**。
   这一条是本文最值得下游抄走的一条。
4. **打包输入是 e2e 缝的产物**（D2）：如果 Lead 之后扩展 \`tools/e2e.sh\` 支持 ≥3 个运行时输入，
   本图应改成"p、v、x 三个独立输入"，层数会从 7 降到 5（少 2 个 \`SLICE\`），数值结论不受影响。
5. **T 只测了 8**；T=128（prefill 里程碑）未测，PV 的 B 侧 64 宽不变、A 侧 T×T 会变大，行宽规则仍满足
   （fp16 T=128 → 256 B/行），但**没有实测**。
6. **残差 x 只取过 fixture embed 行**（0..7 与 8..15，\`max|x|\`≈0.06）。更深的层 hidden 尺度会更大；
   若真权重档 \`max|out|\` 上到 O(1)，绝对阈值要按 \`4·2⁻¹¹·max|out|\` 重算（相对口径仍是 1e-3 量级）。
7. **性能未测**：见 §6。

---

## 8. 复现

\`\`\`bash
# 0) WSL 的 /usr/bin/python3（有 numpy + tflite schema）；夹具可指向已生成的那份
export QWEN25_WEIGHTS_DIR='/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen'

# 1) 生成 model.tflite + golden（同时写出 altp / altx 的对照输入）
/usr/bin/python3 tools/build_pv_o_residual.py --variant main --out ops/pv_o_residual_t8_fp16

# 2) 编译（锁外）：主图 + 两个层数对照 + 负对照
./tools/compile.sh ops/pv_o_residual_t8_fp16/model.tflite ops/pv_o_residual_t8_fp16/model.dla
/usr/bin/python3 tools/build_pv_o_residual.py --variant bmm3d --out ops/pv_bmm3d_control_t8_fp16
./tools/compile.sh ops/pv_bmm3d_control_t8_fp16/model.tflite ops/pv_bmm3d_control_t8_fp16/model.dla
/usr/bin/python3 tools/build_pv_o_residual.py --variant bmm4d --out ops/pv_bmm4d_control_t8_fp16
./tools/compile.sh ops/pv_bmm4d_control_t8_fp16/model.tflite ops/pv_bmm4d_control_t8_fp16/model.dla
/usr/bin/python3 tools/build_pv_o_residual.py --variant main-badlayout --out ops/pv_o_badlayout_control_t8_fp16
./tools/compile.sh ops/pv_o_badlayout_control_t8_fp16/model.tflite ops/pv_o_badlayout_control_t8_fp16/model.dla

# 3) 板端（持锁）：主契约 + 噪声底（连跑两次存两份）
export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"; export MSYS_NO_PATHCONV=1
./tools/devlock.sh run -- ./tools/healthcheck.sh --quick
./tools/devlock.sh run -- ./tools/e2e.sh ops/pv_o_residual_t8_fp16
cp ops/pv_o_residual_t8_fp16/out_device.bin ops/pv_o_residual_t8_fp16/out_r1.bin
./tools/devlock.sh run -- ./tools/e2e.sh ops/pv_o_residual_t8_fp16
cp ops/pv_o_residual_t8_fp16/out_device.bin ops/pv_o_residual_t8_fp16/out_r2.bin
./tools/devlock.sh run -- ./tools/healthcheck.sh --quick
# 噪声底：max|r1-r2|（本次 0.0）

# 4) 换输入对照：把 golden/{input,input1,expected}_{alt,altx}.bin 覆盖到
#    golden/{input,input1,expected}.bin 再跑一次 e2e（模型/.dla 不变），跑完恢复
# 5) 负对照：./tools/e2e.sh ops/pv_o_badlayout_control_t8_fp16（预期 FAIL，rel_l2≈0.92）
\`\`\`

---

## 9. 产物清单（入 git 的部分）

\`\`\`
ops/pv_o_residual_t8_fp16/            model.json  model.compile.log  compare.txt
                                      compare_run1.txt  compare_altp.txt  compare_altx.txt
                                      e2e.json
ops/pv_bmm3d_control_t8_fp16/         model.json  model.compile.log        # 3-D = 1 层
ops/pv_bmm4d_control_t8_fp16/         model.json  model.compile.log        # 4-D = 5 层
ops/pv_o_badlayout_control_t8_fp16/   model.json  model.compile.log  compare_wronglayout.txt
                                                                        # 负对照（FAIL 是判据）
tools/build_pv_o_residual.py
docs/OP-pv-o.md
\`\`\`

（\`*.bin\` / \`*.tflite\` / \`*.dla\` 按仓库约定不入 git，可由 builder + \`compile.sh\` 逐字节复现：
\`.tflite\` 与 golden 都已实测可复现，\`.dla\` 不可复现且不作判据。）
