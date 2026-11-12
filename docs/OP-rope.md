# E-ROPE —— RoPE（Qwen2 split-half）用 SPLIT / MUL / SUB / ADD / CONCAT 表达

> 范围：`feat/rope` worktree（`wt-rope`）。本文只记录**我自己的**实验（AGENTS.md R1–R10、`docs/AGENT-BRIEF.md`）。
> 前置知识来自 `docs/OP-shapeops.md`（SPLIT→SliceLayer、CONCAT→ConcatLayer）与 `docs/OP-elementwise.md`
> （MUL/SUB/ADD→ElementWise*Layer、`K%16` 粒度规则），本文不重复推导。

---

## 0. TL;DR

**结论三句话**：
1. **能跑**：`SIN`/`COS` 被 ncc importer 拒绝，但只要把 cos/sin 当**数据**喂进去，RoPE 可以纯用
   `SPLIT → MUL → SUB/ADD → CONCAT` 落在 MDLA_5_0 上，**10 个形状/分解全部编译 + 板端 + 数值通过**（fp16，atol 2e-3 事先登记）。
2. **编译器没有折叠任何东西**（除了"对常量张量做 SPLIT"这一步）：`SPLIT`→`SliceLayer`、`MUL`（**即使一个操作数是常量**）
   →`ElementWiseMulLayer`+Static DRAM、`SUB/ADD`→`ElementWiseSub/AddLayer`、`CONCAT`→`ConcatLayer`，
   全部在**同一个** `ExecutionStep` / `MDLA_5_0` 上。代价是**6–11 个真层**换 1 个理想融合算子（4-D 形状更差，**35 层**）。
3. **cos/sin 可以是运行时输入**（这是本题最关键的一条）：打包成一张 `[.., 2h]` 表当 input 喂，
   编译 `EXIT=0`、`Static=0`、板端结果与"表当常量"完全同级（1 ULP 内）。
   并用**随机噪声表**做对照证明该表被真的当成数据消费（换表则偏差 2.686，用对表则 9.8e-4）。

**约定修正（重要，先看 §2）**：任务书里写的 `b' = b*sin + a*cos` **不是旋转**（把单位向量转 0.7 rad 会得到 norm=1.08）。
Qwen2/HF 的真实形式是 **`out_hi = b*cos + a*sin`**。第一轮板端实验按任务书字面实现（自洽通过、但函数是错的），已全部作废重做。

| # | 变体 | shape | 表/分解 | 编译 | 层数 | 板端 | 数值（max_abs / 登记的 atol·rtol） |
|---|---|---|---|---|---|---|---|
| 1 | `rope_1x1x64_cosconst_fp16` | `[1,1,64]` | const2 / 4mul | ✓ EXIT=0 | **9** | ✓ 3 次 | 4.883e-04 ✓ (2e-3,1e-3) |
| 2 | `rope_1x8x64_cosconst_fp16` | `[1,8,64]` | const2 / 4mul | ✓ | **9** | ✓ | 9.766e-04 ✓ |
| 3 | `rope_1x8x2x64_cosconst_fp16` | `[1,8,2,64]` | const2 / 4mul | ✓ | **35** | ✓ | 9.766e-04 ✓ |
| 4 | `rope_1x8x64_dupconst_2mul_fp16` | `[1,8,64]` | const_dh / 2mul | ✓ | **6** | ✓ | 9.766e-04 ✓ |
| 5 | `rope_1x8x64_tableconst_fp16` | `[1,8,64]` | const1 / 4mul | ✓ | **9** | ✓ | 9.766e-04 ✓ |
| 6 | `rope_1x1x64_tabledyn_fp16` | `[1,1,64]` | **dyn1** / 4mul | ✓ | **11** | ✓ | 4.883e-04 ✓ |
| 7 | `rope_1x8x64_tabledyn_fp16` | `[1,8,64]` | **dyn1** / 4mul | ✓ | **11** | ✓ | 9.766e-04 ✓ |
| 8 | `rope_1x8x64_tabledyn_wild_fp16` | `[1,8,64]` | dyn1 / 4mul，**随机表** | ✓ | 11 | ✓ | 9.766e-04 ✓ |
| 9 | `rope_1x8x1x64_cosconst_fp16` | `[1,8,1,64]` | const2 / 4mul | ✓ | **35** | ✓ | 9.766e-04 ✓ |
| 10 | `rope_1x16x64_cosconst_fp16` | `[1,16,64]` | const2 / 4mul | ✓ | **9** | ✓ | 9.766e-04 ✓ |

附加的强结论：**设备输出与"逐 op fp16 舍入模拟"逐位相同 —— 5248/5248 个 code 零差异**（§6）。
即 MDLA 的 fp16 逐元素层就是 IEEE fp16 语义（每个 op 各自 round），与单次舍入 golden 相差 ≤1 ULP。

---

## 1. 证据绑定（R1）

| 项 | 值 |
|---|---|
| 仓库/分支 | `npu-ops` worktree `wt-rope`，分支 `feat/rope`，基线 `387bd71` |
| 编译器 | `ncc-tflite` **7.3.15**，`--arch=mdla5.0 --opt=3 --disallow-bridge --num-mdla=2 --l1-size-kb=3072`（`tools/compile.sh` 固定） |
| 设备 | adb `0123456789ABCDEF`，`ro.board.platform=mt6897`，Android 16，SELinux Enforcing，root |
| 板端执行器 | 厂商 `neuronrt`（`-m hw -a <dla> -i in [-i in1] -o out -c 3`） |
| 板端串行 | `NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" ./tools/devlock.sh run -- …`（R5） |
| 模型生成 | `tools/build_rope.py`（新增，本 worktree），seed 20261112；`model.tflite` + golden 重建**字节可复现**（§9 实测） |
| 生成/编译/板端 | 2026-11-12 23:37–23:47 (+08:00) |
| 设备状态 | 全部板端跑完后 `tools/healthcheck.sh` → **PASS**（known-good op + 重复性 + dmesg 干净） |

每个 `ops/rope_*/` 内含：`model.json`（形状/精度/量化/容差/约定/分解说明）、`model.compile.log`（cmd、ncc 版本、
`Target: MDLA_5_0`、层列表、DRAM 表、`EXIT=`、`SIZE=`、`SHA256=`）、`compare.txt`（`tools/compare.py` 原始输出）、`e2e.json`。
`.dla` 的 sha256 **只标识某一次产物**，不作回归依据（AGENTS §4）。

---

## 2. 约定修正：任务书的第二个式子不是旋转（**这轮唯一的"失败"，且它被独立参考抓住**）

### 2.1 事实

任务书给的是 `a' = a*cos - b*sin`、`b' = b*sin + a*cos`。两式**互相矛盾**：

* 旋转必须保持 2-norm。取 `(a,b)=(1,0)`、`θ=0.7 rad`：
  - 任务书字面：`(0.764842, 0.764842)` → **norm = 1.0816**（不是旋转）
  - 标准旋转 / HF：`(0.764842, 0.644218)` → norm = 1.0000 ✓
* HuggingFace Qwen2 的实现是 `q_embed = q*cos + rotate_half(q)*sin`，而
  `rotate_half(x) = cat((-x2, x1))`，展开正是：

  ```
  out_lo = a*cos - b*sin
  out_hi = b*cos + a*sin        <-- 注意：cos 配 in-half 的 b，sin 配 out-half 的 a
  ```

所以 **Qwen2 split-half 的正确形式是 `b*cos + a*sin`**，任务书里的 `b*sin + a*cos` 是笔误。

### 2.2 为什么会"假通过"（R3 的真实案例）

第一轮我按任务书字面实现（`MUL(a,cos)`、`MUL(b,sin)` 复用两次，只有 **2 个 MUL**），
golden 与图**共用同一个错公式**，于是 10 个模型全部 PASS（板端 max_abs 4.9e-4，看起来很漂亮）。
两边一起错 —— 正是 R3 要防的假通过。随后用 `transformers` 的 Qwen2 实现做约定审计（`--verify-hf`）才暴露：
偏差 1.9（不是舍入级别）。

### 2.3 现在的两道永久防线（已写进 `tools/build_rope.py`）

1. **生成期断言旋转不变量**：`build()` 里 `check_rotation_invariant()` 校验 golden 的每一对 `(a,b)` 的 2-norm
   与输入一致（fp16 存储允许 3e-3 相对误差；实测 3.9e-4–4.7e-4）。cos/sin 互换会立刻把它顶穿（1.08 ≫ 1.003）。
2. **`--verify-hf` 约定审计**（不是 golden，R3 保持）：
   ```
   rope_1x8x64_cosconst_fp16   ours(float64 tables) vs HF: 2.556e-07   ours(fp16 tables) vs HF: 4.976e-04
   ```
   第一列 ≪ 舍入 ⇒ **公式与 HF/Qwen2 一致**；第二列是用 fp16 存表的代价（≤6.15e-04，见 §8 Q4）。

### 2.4 修正后的代价

正确旋转需要**四个乘积**（`a*cos, b*sin, b*cos, a*sin`），"复用两个乘积"只在
那个错的（非旋转）式子下成立。因此主分解是 4 MUL（9 层）；此外另做了一个
"把 cos 复制成 `[cos|cos]`、把符号折进表里得到 `[-sin|sin]`"的 const-only 变体（6 层，见 §5.2）。

> 作废的第一轮产物从未提交为证据（只有修正后的 `model.json`/`model.compile.log` 进了 commit `9849fbc`），
> 板端 10 次运行在 23:38–23:44 之后全部重做。本节保留，供后续 Agent 不再踩（R4）。

---

## 3. 编译器保留了什么、折叠了什么（交付项 2）

### 3.1 逐 op 的落点（`model.compile.log` 原文，`rope_1x8x64_cosconst_fp16`）

```
===Graph[0] has 1 Execution Step(s)===
Graph[0] Step[0]
ExecutionStep
 ├ StepId: 0
 ├ Target: MDLA_5_0
 └ Subgraph:
   ├ SliceLayer<0>{SPLIT}
   ├ ElementWiseMulLayer<4>{MUL}
   ├ ElementWiseMulLayer<1>{MUL}
   ├ SliceLayer<0>{SPLIT}
   ├ ElementWiseMulLayer<3>{MUL}
   ├ ElementWiseAddLayer<6>{ADD}
   ├ ElementWiseMulLayer<2>{MUL}
   ├ ElementWiseSubLayer<5>{SUB}
   ├ ConcatLayer<7>{CONCATENATION}
   └ Output: OpResult (external)
```

| TFLite 算子 | 落点 | 是否折叠 |
|---|---|---|
| `SPLIT`（激活） | `SliceLayer`×2（**每个输出切片一层**） | 不折叠 |
| `MUL`（激活 × **常量**） | `ElementWiseMulLayer`，常量进 NVM（DRAM 表 `Static` 非 0） | **不折叠**成 scale/bias 层 |
| `MUL`（激活 × 激活） | `ElementWiseMulLayer` | 不折叠 |
| `SUB` / `ADD` | `ElementWiseSubLayer` / `ElementWiseAddLayer` | 不折叠（也没有融合成一层） |
| `CONCATENATION`（axis=last） | `ConcatLayer` | 不折叠 |
| **`SPLIT`（对常量张量）** | **没有任何层** | **折叠**（拆常量 = 编译期把它直接写进两块常量数据） |

最后一条是唯一被折叠的：`rope_1x8x64_tableconst_fp16`（打包常量表 + 图内 SPLIT）的 plan 里
**op 0（表 split）不出现**，只剩 op 1（激活 split）的 2 个 `SliceLayer`，所以层数与"两张半宽常量表"
的 `const2` 变体**完全一样（9 层）**，`Static` 也一样（1.0K = cos+sin 各 512 B）。原文：

```
### rope_1x8x64_tableconst_fp16            （op0 = SPLIT(const table)  → 无层）
 └ Subgraph:
   ├ SliceLayer<0>{SPLIT}                  <- op1 (activation split)
   ├ ElementWiseMulLayer<5>{MUL}
   ├ ElementWiseMulLayer<2>{MUL}
   ├ SliceLayer<0>{SPLIT}
   ├ ElementWiseMulLayer<4>{MUL}
   ├ ElementWiseAddLayer<7>{ADD}
   ├ ElementWiseMulLayer<3>{MUL}
   ├ ElementWiseSubLayer<6>{SUB}
   ├ ConcatLayer<8>{CONCATENATION}
```

### 3.2 与 shapeops 旧结论的关系

与 `docs/OP-shapeops.md` §2.4 一致：**没有任何模型出现 `IdentityLayer` 或"没有层"**（唯一的例外仍是
"可以当作别名/常量"的场合）。这里新增的观测是：**"常量操作数"不会让逐元素层消失**，
只有"对常量本身的 shape 变换"会消失。

---

## 4. cos/sin 作为**运行时输入**可行（交付项 3 —— 本题最关键的一条）

### 4.1 结论

**可以，不需要必须做成 const。** 而且这不是"编译过了"而已：

* 编译：`EXIT=0`，单 `ExecutionStep` @ `MDLA_5_0`，**`Static=0`**（模型里完全没有常量表），
  `Input` 从 1.0K 变 2.0K（激活 1K + 表 1K）。`slice=4`（两张表 split 都物化了，比 const 多 2 层）。
* 板端：`[1,1,64]` / `[1,8,64]` 两个 dyn 变体都真跑通（3 次重复），max_abs 4.883e-04 / 9.766e-04，
  与 const 变体同级（**同一个值**，因为表的内容相同）。
* **表被真的当作数据消费**（对照实验，§4.2）。

### 4.2 对照实验：随机噪声表（`rope_1x8x64_tabledyn_wild_fp16`）

如果"dyn 能跑"其实是因为编译器忽略了输入表、或把它当成某种可重算的东西，那么换一张
**随机噪声表**（与任何位置/角度无关）就会露馅。实测（`compare.txt` + 反事实 golden）：

| 模型（表怎么来的） | device vs golden(**实际喂进去的表**) | device vs golden(旋转表 p0=5) | device vs golden(随机表) |
|---|---|---|---|
| wild（喂随机表） | **9.766e-04** | 2.686e+00 | 9.766e-04 |
| dyn（喂旋转表） | **9.766e-04** | 9.766e-04 | 2.686e+00 |

"旋转表 golden"与"随机表 golden"本身相差 2.686。dev 永远贴着**它收到的那张表** ⇒ 运行时表被真的读入并按数据参与运算
（不可能被重算/常量折叠/忽略）。

### 4.3 工程含义

真实 decoder 里 cos/sin 取决于 position（且 KV cache 每步只增一个位置），所以：

* **dyn 路径**：模型只需要编译一次，表每次推理换一张（`golden/input1.bin` 这类张量），
  代价 = **+2 个 SliceLayer**（11 层 vs 9 层）、表从 NVM(Static) 移到 DRAM(Input)。
* **const 路径**：模型被钉死在 `[p0, p0+T)` 这段位置上（本实验 p0=5）；`T` 越大 `Static` 越大
  （`[1,1,64]` 时 128 B，`[1,8,64]` 时 1.0K，`[1,16,64]` 时 2.0K）。做 prefill 用固定窗口可行，
  做 decode 就必须每步换表 ⇒ 只能选 dyn。
* 注意 **`SPLIT` 常量表被折叠**（§3.1），所以 `tableconst` 与 `const2` 同价；也就是说
  "打包一张表"没有额外层开销，可以放心用。

---

## 5. 代价：多出来的层 vs 一个理想融合算子

### 5.1 层数（这就是本算子相对"一个融合 RoPE"的真实开销）

| 分解 | 层构成 | 层数 |
|---|---|---|
| 理想融合算子（不存在） | —— | **1** |
| `2mul`（const-only，§5.2） | Slice×2 + Mul×2 + Concat(swap) + Add | **6** |
| `4mul` + 两常量表（`const2`，3-D） | Slice×2 + Mul×4 + Sub + Add + Concat | **9** |
| `4mul` + 打包常量表（`const1`，3-D） | 同上（表的 split 被折叠） | **9** |
| `4mul` + **运行时表**（`dyn1`，3-D） | Slice×**4** + Mul×4 + Sub + Add + Concat | **11** |
| `4mul`，**4-D `[1,T,H,D]`** | Slice×16 + Mul×8 + Sub×2 + Add×2 + Concat×7 | **35** |

⇒ 在 3-D `[1, N, D]` 布局下，RoPE ≈ **6–11 个 MDLA 层**换 1 个融合算子；**4-D 头部布局是 35 层（≈4×）**。

### 5.2 6 层的那条路（const-only）

把符号折进表里：`out = x*[cos|cos] + swap(x)*[-sin|sin]`，`swap(x) = CONCAT(b, a)`：

```
### rope_1x8x64_dupconst_2mul_fp16
   ├ SliceLayer<0>{SPLIT}
   ├ SliceLayer<0>{SPLIT}
   ├ ConcatLayer<1>{CONCATENATION}      <- swap(b,a)
   ├ ElementWiseMulLayer<3>{MUL}        <- x * [cos|cos]
   ├ ElementWiseMulLayer<2>{MUL}        <- swap * [-sin|sin]
   ├ ElementWiseAddLayer<4>{ADD}
```

6 层、只需要 1 个 ADD（不再需要 SUB 和最后的 CONCAT）。**代价**：`cos` 被存了两份
（`Static` 1.0K → 2.0K）。仅当表是常量时可行；运行时表需要先做 `CONCAT(c,c)` 和变号，层数反而更多，
所以 dyn 路径只能用 9/11 层那套。

### 5.3 4-D 布局为什么炸开：单变量判别

| 对照 | shape | 元素数 | 层数 |
|---|---|---|---|
| rank-3，1 head，T=8 | `[1,8,64]` | 512 | **9** |
| **rank-4，1 head**，T=8 | `[1,8,1,64]` | 512 | **35** |
| rank-4，2 head | `[1,8,2,64]` | 1024 | **35** |
| rank-3，16 行 | `[1,16,64]` | 1024 | **9** |

`[1,8,1,64]` 与 `[1,8,64]` **元素数相同、head 数相同（1）**，唯一差别是 rank ⇒ 触发条件是 **rank 4**
（不是 head 数、不是行数）。展开方式：每个逐元素层被拆成 "Slice + 2×op + Concat" 的块，连 SPLIT 也被拆
（`SliceLayer<0>×4 + ConcatLayer<0>{SPLIT}`）。
**机制 = UNKNOWN**（AGENTS R10：不猜）。与 `docs/OP-MATRIX.md` §3 "C≥32 会被分解成 Slice+op+Concat"
是同一类现象，但这里触发它的是 rank 而不是通道数——两者是否同源未验证。
**可操作结论**：把 head 维折进前面的维（用 `[1, T*H, 64]` 并在外部做 head 的切片/偏移），能省 26 个层。

### 5.4 性能（只记录，不作结论 —— R7）

`neuronrt -c 3` 的 `Avg. FPS`（→ ms/inf）：1x1 const 509.7（1.96）、1x8 const 1039（0.96）、
`[1,8,2,64]` 876.9（1.14）、`dupconst_2mul` 759.1（1.32）、`tableconst` 776（1.29）、
1x1 dyn 560.9（1.78）、1x8 dyn 483.9（2.07）、**wild 1119（0.89）**、`[1,8,1,64]` 664.3（1.51）、
`[1,16,64]` 808.2（1.24）。

**这批数字不能用来比较层数**：`tabledyn` 与 `tabledyn_wild` 是**同一张图**（11 层、同一 `.dla` 结构），
FPS 却是 483.9 与 1119（2.3×差）。差异被启动/同步开销和板子状态淹没。
⇒ **层数是本报告唯一负责的代价指标，FPS 不是**；要谈性能需另做带对照、控温、多批次的实验。

---

## 6. 数值（五个结论分开，R7）

1. **编译**：10/10 `EXIT=0`，全部单 `ExecutionStep`、`Target: MDLA_5_0`、**无 bridge**（`--disallow-bridge` 恒开）。
2. **ioctl/提交**：`neuronrt` 10/10 正常返回，`-c 3` 无错误输出。
3. **数值**：10/10 `compare.txt` **PASS**，max_abs 4.883e-04（`[1,1,64]`）/ 9.766e-04（其余），
   登记的 `atol=2e-3, rtol=1e-3` 在跑之前就写进 `model.json`（commit `9849fbc` 早于板端证据 commit）。
4. **设备状态**：全部跑完 `tools/healthcheck.sh` **PASS**（known-good op + 重复性 + dmesg 无异常）。
5. **性能**：只记录（§5.4），不作结论。

**为什么 max_abs 是 4.9e-4 / 9.8e-4 而不是 0**：MDLA 的 fp16 逐元素层**每个 op 各自 round 到 fp16**。
把 golden 改成"逐 op fp16 舍入"再比：

| | 结果 |
|---|---|
| 设备 vs **逐 op fp16 模拟** | **5248 / 5248 个 code 逐位相同（0 差异，10 个模型全部）** |
| 设备 vs 登记的单次舍入 golden | ≤1 ULP（9.766e-04），68/512 个 code 差 1 个 fp16 code |

也就是说：**设备行为 = IEEE fp16 语义（逐 op 舍入）**，与"无限精度后存 fp16"的 golden 只差中间舍入。
这条也说明 `atol=2e-3` 的登记是**保守的**（实际只用掉 1 ULP）。

---

## 7. 失败清单（编译期 / 运行期）

**本算子在编译期与运行期都没有失败**：10 个模型全部 `EXIT=0` + 板端跑通 + 数值 PASS。
这本身是结论的一部分：3-D 的 `[1,N,64]` 与 4-D 的 `[1,T,H,64]`，行字节 128 B / 64 B，
**远高于 16 字节行规则**（`docs/OP-MATRIX.md` §2），所以本实验**没有**触碰那条边界（别把它当成"边界已验"）。

唯一真正的失败是**约定错误**（§2），且是离线被独立参考（HF）抓住的，不是设备报错。

失败知识（供后续 Agent）：
* 绝对不能只拿"golden vs device 一致"当通过——本实验第一轮就是两边同时错（R3 的活教材）；
  必须有一个**外部约定审计**（这里 `--verify-hf`）+ 一个**不依赖具体公式的不变量**（这里 2-norm 保持）。
* `SIN`/`COS` 在 ncc importer 阶段就被拒，不要试图用 `EXP`/`LOG` 拼三角函数，也不要指望 `PLE` 之类的查表算子。

---

## 8. 适用范围与剩余疑问（UNKNOWN）

**适用范围**：fp16；`head_dim=64`（`h=32`）；`T ∈ {1,8,16}`；head 数 1–2；`theta=1e6`；
Qwen2 **split-half** 约定；只有 `SPLIT/MUL/SUB/ADD/CONCAT`，`fused_activation=NONE`；
3-D `[1,N,D]` = 9 层（dyn 11 层），4-D `[1,T,H,D]` = 35 层；板端 3 次重复通过。

**未覆盖 / UNKNOWN**：
1. **其它约定**未测：interleaved / GPT-NeoX 风格（相邻两维成对旋转）、`partial rotary`（只转一部分维度）、
   `rope_scaling`（yarn/longrope 的 position 变换）—— 这些用同一套算子也能表达，但没有证据。
2. **int8 通路未测**：RoPE 的值域与量化 scale 的关系（每 token 动态 scale）没做；int8 表会掉精度。
   **fp32 直连不可用**（E-004）。
3. **更大的 T / batch / 多 head 联合**：`T=128/1024`、`batch>1`、`H=32`（真实 Qwen2-7B 的 attention 头数）；
   `[1, T*H, 64]` 折维后的 head 偏移（stride）如何在不加层的前提下表达，未测。
4. **fp16 表本身的精度代价**：用 fp16 存 cos/sin 会让每次旋转的 2-norm 漂 ~4e-4（实测 3.9e-4–4.7e-4），
   即旋转矩阵**非严格正交**。单层无所谓，但 32 层叠加后的累积效应**未测**（若是系统性同向，量级 ~1e-2）。
   缓解：表用 fp32 生成后 round（当前就是这么做的），或整层用 fp32/bf16 通路——都没验证。
5. **rank-4 分解的机制**未解释（§5.3），以及"折维到 3-D 省 26 层"是否在完整模型里仍然成立。
6. **与真实 attention 的衔接**：RoPE 后面接 Q·K（FC/1×1 conv）；`ConcatLayer` 出来的张量能否与
   FC 的输入布局直接对接（省一次搬运）未测。
7. 性能：见 §5.4，无结论。

**给 Lead 的建议（按价值）**：
1. 把 `--verify-hf` 这类"约定审计"提成公共做法：新算子只要是某个已知公开实现（HF/ONNX）的等价物，
   就顺手对一次，成本极低，能抓住本轮的这类错。
2. 若要走真实模型集成，先做 `[1, T*H, 64]` 折维 vs `[1,T,H,64]` 的单变量实验（26 层差异），
   这比调任何算子参数都值钱。

---

## 9. 复现

```bash
# 0) 环境：ncc-tflite 在 WSL，adb 在 Git Bash；设备共享，必须持锁
export MSYS_NO_PATHCONV=1
LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"

# 1) 生成（10 个变体）—— 也可只生成一个：… tools/build_rope.py rope_1x8x64_cosconst_fp16
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-rope" && \
  for s in $(/usr/bin/python3 tools/build_rope.py --list); do \
    /usr/bin/python3 tools/build_rope.py $s; done'

# 2) 约定审计（需要 torch + transformers；不参与 golden）
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-rope" && \
  /usr/bin/python3 tools/build_rope.py --verify-hf'

# 3) 编译（--disallow-bridge 已在 compile.sh 内固定）
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-rope" && \
  ./tools/compile.sh ops/rope_1x8x64_cosconst_fp16/model.tflite ops/rope_1x8x64_cosconst_fp16/model.dla'

# 4) 板端 + 对拍（dyn 变体自动用 golden/input1.bin 当第二个输入）
NPU_LOCK="$LOCK" ./tools/devlock.sh run -- ./tools/e2e.sh ops/rope_1x8x64_cosconst_fp16

# 5) 状态自检
NPU_LOCK="$LOCK" ./tools/devlock.sh run -- ./tools/healthcheck.sh
```

可复现性实测（2026-11-12 本 worktree）：重建 10 个变体后，`model.tflite`(10) + `golden/*.bin`(23)
共 **33 个文件 sha256 全部不变**（`.dla` 不是字节可复现的，见 AGENTS §4，只用数值回归）。

---

## 10. 交接（R8）

* 新增文件：`tools/build_rope.py`（生成器 + 约定审计 + 旋转不变量断言）、`ops/rope_*/` 10 套产物、
  `docs/OP-rope.md`（本文）。**公共工具、`probe/`、`PLAN.md`/`README.md`/`docs/EXPERIMENTS.md` 未改。**
* 新增用的公共能力：无（`e2e.sh` 已支持 `golden/input1.bin` 作为第二个运行时输入，dyn 变体直接复用，
  不需要改任何公共脚本）。
* 回归判据：数值（`compare.txt`，`atol=2e-3 / rtol=1e-3`，实测只用掉 1 ULP）；`model.json` 里的
  `rotation_invariant_rel_err` 可以作为"这还是不是旋转"的快速体检。
* 复跑：`./tools/e2e.sh ops/<name>` 即可（`model.tflite`/`.dla` 缺失时会自动重建重编）。
