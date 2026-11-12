# OP-shapeops —— RESHAPE / TRANSPOSE / CONCAT / SPLIT / PAD

范围：shape/layout 类算子（int8 为主，另加 2 个 fp16 边界档做规则区分）。
采集：2026-11-12 · ncc-tflite **7.3.15** · `--arch=mdla5.0 --num-mdla=2 --l1-size-kb=3072 --opt=3 --disallow-bridge` ·
板 = Titan 1（`mt6897`/`mt8792`，Android 16）· 全部板端任务持 `tools/devlock.sh` 锁串行执行。
实验编号：本文件用 **E-SHP-01…06**（`docs/EXPERIMENTS.md` 由 Lead 统一编号，未在此文件改动）。

---

## 0. 一页结论

| 算子 | 通过编译并上板 | 编译器发出的层 | 板端数值（逐位） | 可用性 |
|---|---|---|---|---|
| RESHAPE | ✅ 3 个形状 | `ReshapeLayer<0>{RESHAPE}` | 64/64、256/256、512/512 全等 | **可用**（载荷是否为真搬运仍 UNKNOWN，见 §4.1） |
| TRANSPOSE | ✅ 1 int8 + 1 fp16 | `TransposeLayer<0>{TRANSPOSE}` | 512/512 全等（int8 与 fp16 皆 0 误差） | **可用** |
| CONCAT | ✅ axis3 / axis1（C=16） | `ConcatLayer<0>{CONCATENATION}` | 1024/1024 全等（两档） | **可用**（C 需满足 §2） |
| SPLIT | ✅ slice0/slice1 各两档 | `SliceLayer<0>{SPLIT}` | 256/256、512/512、512/512 全等 | **可用**（注意 §2 的偏移规则） |
| PAD | ✅ | `PadLayer<0>{PAD}` | 960/960 全等（边界全 0，数据非 0） | **可用** |
| 链式 CONCAT→PAD | ✅ | `ConcatLayer<0>` + `PadLayer<1>` | 1920/1920 全等 | 两个层都**保留**，没有被折叠 |

三条主要结论：

1. **"纯数据搬运"的算子并不会被折叠成 `IdentityLayer on NVM`。** 五个算子在单算子图里都发出真层；
   两个 shape 算子串成一个图（CONCAT→PAD）时两层也都留在同一 `ExecutionStep` 里，**没有互相折叠**。
   （对比 `docs/EXPERIMENTS.md` 里"输入张量带 data buffer 被常量折叠成 IdentityLayer"的已知坑——这里输入都不带 buffer。）
2. **真正决定能不能用的不是算子，而是 DRAM 边界张量的"通道行字节数"**：每行 **< 16 B** 的张量
   既不能当外部输入也不能当外部输出，编译期直接判桥接失败（§2，10 条对照）。这条规则解释了本文件
   全部 6 个编译失败，且已被 fp16 档交叉验证为"字节"而非"通道数"。
3. 因此对真实 transformer block：**hidden 维是 16 的倍数时（128/256/768…），这五个算子都可以直接放进 NPU 图**；
   **hidden 维 < 16 或为 8 的倍数但不是 16 的倍数时（如 8、24），需要改布局或放弃 `--disallow-bridge`**。

---

## 1. 方法（先说清楚，避免误读结果）

### 1.1 为什么这些模型可以"逐位"验收

五个算子在 TFLite 语义里都是**纯数据搬运**（无算术、无 requant）。模型里刻意令
`input.scale = output.scale = 1/127`、`zero_point = 0`，于是：

- 设备侧不存在任何舍入/定点乘子，**契约是 `int8_tol = 0`（逐位相等）**，与 FC 的 ±1 requant 边界无关；
- 因此 `tools/compare.py` 报 `exact: YES` 才是真正的通过；任何一次位置错（转置写成 reshape、
  concat 两半互换、split 两半互换、pad 贴错边）都会变成大范围失配，而不是"看起来差不多"。

### 1.2 输入码值：种子化置换，小张量里每个位置码值唯一

`codes(n) = (permutation(0..n-1) mod 255) - 127`。对 n ≤ 255 的模型（64/256、64 元素等）**每个元素码值互不相同**，
所以任何"错位"都无法靠巧合对上。512 元素的转置档有码值碰撞，但线性模式的错位仍然是大面积失配（实测 512/512 全等）。

### 1.3 参考实现与被测实现独立（AGENTS.md R3）

golden 全部由 numpy 直接算：`x.reshape`、`np.transpose`、`np.concatenate`、`np.split`、`np.pad`；
与被测侧的 flatbuffers 建模代码没有任何共享的索引/packer 逻辑（`tools/build_shapeops.py` 只负责写模型与写输入）。
PAD 档另外令**输入码值永不为 0**，于是 0 值只可能来自 pad 常量——"pad 有没有贴对边"可直接判定。

### 1.4 单算子图 vs 链式图（回答"会不会被折叠"）

单算子图会**强制**编译器保留该层，因此它不能回答"放在两个 matmul 之间时还在不在"。
所以额外做了一个 `CONCAT → PAD` 的双层图（同样不含算术，仍然逐位可验），用来观察层是否被折叠。

---

## 2. 核心发现：DRAM 边界的"16 字节/通道行"规则（E-SHP-01…05 的共同解释）

### 2.1 假设（跑之前写下）

- H1：ncc 会把纯 reshape 折叠成免费别名，甚至发出 `IdentityLayer on NVM`；
- H2：这些算子都发出真层，能否编译只取决于算子本身是否被 MDLA 支持；
- H3：真正的约束在**张量布局**（DRAM 边界），而不是算子。

### 2.2 预测

若 H3 成立，则同一算子、同一 dtype 下，只把边界张量的最后一维从 8 改成 16，编译结果就会从失败变成功；
且 fp16（2 B/元素）下失败阈值应出现在 **C=4**（8 B）而 **C=8**（16 B）应通过。

### 2.3 做法

固定算子与形状，只改一个变量（最后一维 / dtype / 输出位置），逐个编译：

| # | 模型 | 边界张量（方向） | 每行字节 | 编译 | 原始错误 / 结果 |
|---|---|---|---|---|---|
| 1 | `reshape_1x4x8x8_to_1x8x4x8_int8` | 输入 C=8 | 8 | ✗ | `Bridging is disallowed but gap is detected between Input and MDLA_5_0` |
| 2 | `transpose_1x4x8x8_int8` | 输入 C=8 | 8 | ✗ | 同上（层名变成 `TransposeLayer`） |
| 3 | `concat_axis3_int8` | 两输入 C=8 | 8 | ✗ | 同上（层名变成 `ConcatLayer`） |
| 4 | `transpose_1x4x8x4_fp16` | 输入 C=4 (fp16) | 8 | ✗ | 同上 |
| 5 | `transpose_1x4x8x16_perm0312_int8` | **输出** C=8 | 8 | ✗ | `Bridging is disallowed but gap is detected between MDLA_5_0 and Output` |
| 6 | `split_axis3_part2_int8` | **输出**（第 2 片）C=8 | 8 | ✗ | 同上 |
| 7 | `reshape_1x4x8x16_to_1x8x4x16_int8` | 输入 C=16 | 16 | ✓ | `ReshapeLayer<0>`，DRAM 512/512/0/0/224 |
| 8 | `transpose_1x4x8x16_int8` | 输入 C=16 | 16 | ✓ | `TransposeLayer<0>` |
| 9 | `concat_axis3_c16_int8` | 两输入 C=16 | 16 | ✓ | `ConcatLayer<0>` |
| 10 | `transpose_1x4x8x8_fp16` | 输入 C=8 (fp16) | 16 | ✓ | `TransposeLayer<0>`（**fp16 8 通道 = 16 B，通过**） |

### 2.4 结果

**H3 成立**：dtype 从 int8 换成 fp16 时，失败阈值随"每行字节数"移动，而不是随通道数移动
（int8 C=8 失败 ↔ fp16 C=8 通过；fp16 C=4 回到 8 B 又失败）。
且规则在**输入侧和输出侧都成立**（同一条 16 B 门槛，分别报 `Input→MDLA` 与 `MDLA→Output` 的 gap）。

**H1 被否定**：没有任何模型出现 `IdentityLayer`；RESHAPE 两侧形状总元素相同（平坦无搬运），编译器仍然发
`ReshapeLayer<0>{RESHAPE}` 而不是折叠。
**H2 被部分否定**：算子支持不是判据——失败模型在 exec plan 里**同样打印了目标层**，只是桥接分析在打印之后才失败。

### 2.5 适用范围

- 该表只覆盖 int8 / fp16、4D NHWC、C ∈ {4,8,16}、`--disallow-bridge`；
- "16 B"是**归纳出的边界值**（8 B 全失败、16 B 全成功），没有测 1/2/9/12/15 B 这类非 2 的幂，
  因此不能写成"≥16 B 一律可以"；
- 该规则是**从编译行为反推**的，SDK 文档里没有对应表述；MDLA 内部为什么是 16 B 仍未解释（**UNKNOWN**）。

### 2.6 两个已知例外（必须写清楚，别当成"任意 8 通道都行"）

| 模型 | 边界张量 | 字节 | 结果 | 备注 |
|---|---|---|---|---|
| `split_axis3_part1_int8` | 输出 = 第 0 片 C=8 | 8 | **✓ 编译且逐位通过** | 同一个 8 通道形状作为"第 1 片"就失败（#6） |
| `reshape_1x4x64_to_1x4x8x8_int8` | 输出 C=8 | 8 | **✓ 编译且逐位通过** | 该 reshape 在平坦存储上是恒等映射，可零拷贝别名 |

两个例外都出现在"编译器可以把它当别名/左对齐子张量、不必物化"的场合，这与"需要物化才受 16 B 门槛约束"
的解释一致，但**该机制没有被独立证据证实**，见 §8 剩余疑问 Q1。

---

## 3. 逐算子记录

> 每个算子的格式：**假设 → 预测 → 做法 → 结果（编译 / 层 / 数值）→ 适用范围 → 剩余疑问**。
> 数值一栏的 `compare.txt` 均在 `ops/<name>/` 下，原始 exec plan 在 `model.compile.log`。

### E-SHP-01 RESHAPE

- **假设**：平坦恒等 reshape（元素数不变、只有维度切分变化）不需要数据搬运，编译器要么折叠成别名，要么发一个 0 成本的 `ReshapeLayer`；若发真层，则在"输入/输出张量布局对齐"的前提下数值必须逐位复现。
- **预测**：`[1,64]→[1,4,16]`、`[1,4,64]→[1,4,8,8]`、`[1,4,8,16]→[1,8,4,16]` 均可编译；设备输出与 `numpy.reshape` 逐位相等。反向预测：`[1,4,8,8]→[1,8,4,8]` 若失败，失败点必须落在桥接分析（而不是数值）。
- **做法**：`tools/build_shapeops.py reshape_*`（同时给 shape 张量与 `ReshapeOptions.new_shape`，与 converter 行为一致）→ `tools/compile.sh` → `tools/e2e.sh`。
- **结果**：
  - `reshape_1x64_to_1x4x16_int8`：编译 ✓，`ReshapeLayer<0>{RESHAPE}`，`Target: MDLA_5_0`，DRAM 64/64/0/0/224 B；板端 **64/64 逐位相等**，FPS 988。
  - `reshape_1x4x64_to_1x4x8x8_int8`：编译 ✓，`ReshapeLayer<0>`；板端 **256/256 逐位相等**（输出 C=8 也通过，见 §2.6），FPS 456。
  - `reshape_1x4x8x16_to_1x8x4x16_int8`：编译 ✓，`ReshapeLayer<0>`；板端 **512/512 逐位相等**（H/W 互换的"形态变化"也逐位对），FPS 1109。
  - `reshape_1x4x8x8_to_1x8x4x8_int8`：**编译失败**（输入 C=8，见 §2.3 #1）。
- **编译器发出的层**：`ReshapeLayer`（不是 Identity，也不是"没有层"）。
- **适用范围**：int8、rank 2→3 / 3→4 / 4→4、元素数守恒、输入末维 ≥16 字节；大的 H/W 尺寸与其它 perm 未测。
- **剩余疑问**：`ReshapeLayer` 到底搬不搬数据？平坦 reshape 的"正确"结果与零拷贝别名逐位相同，本实验无法区分（Q2）。

### E-SHP-02 TRANSPOSE

- **假设**：真重排必须物化，不能折叠；MDLA 有转置通路。竞争假设：MDLA 无转置，需要 CPU 桥接（`--disallow-bridge` 下应失败）。
- **预测**：若存在转置通路，`perm=(0,2,1,3)` 的 `[1,4,8,16]→[1,8,4,16]` 可编译，且板端输出与 `np.transpose` 逐位相等；若不存在，则报桥接 gap。
- **做法**：`transpose_1x4x8x16_int8`（int8）、`transpose_1x4x8x8_fp16`（fp16 交叉档）、两个反例档（`C=8` 输入、`C=8` 输出）。
- **结果**：
  - `transpose_1x4x8x16_int8`：编译 ✓，`TransposeLayer<0>{TRANSPOSE}`，DRAM 512/512/0/0/224；板端 **512/512 逐位相等**，FPS 924。
  - `transpose_1x4x8x8_fp16`：编译 ✓，`TransposeLayer<0>`；板端 **max_abs = 0.000e+00**（值恰为整数，fp16 精确表示），FPS 1824。
  - `transpose_1x4x8x8_int8` / `transpose_1x4x8x16_perm0312_int8`：编译失败，但**失败原因不是"没有转置能力"**，而是 §2 的 16 B 门槛（输入侧 / 输出侧）。
- **编译器发出的层**：`TransposeLayer`。
- **适用范围**：4D NHWC、perm 为轴交换、int8 与 fp16、末维 ≥16 B。
- **剩余疑问**：只测了 **交换 H/W**（perm 0,2,1,3）与 **把通道换到别的轴**（perm 0,3,1,2，失败）这两类；
  其它 perm（如 `0,1,3,2` 交换通道与 W）、5D 以上、以及"转置 + 通道 ≥16"的组合未测（Q3）。

### E-SHP-03 CONCAT（CONCATENATION）

- **假设**：channel 维拼接（transformer 里 q|k|v 合成一份）应落到 `ConcatLayer`；受限点与 §2 相同。
- **预测**：`C=16` 两输入沿 axis 3 拼接 → 编译成功，输出 `[1,4,8,32]` 与 `np.concatenate` 逐位相等；
  `C=8` 两输入 → 因外部输入每行 8 B 而编译失败（不是 concat 不支持）。
- **做法**：`concat_axis3_c16_int8`（axis=3）、`concat_axis1_c16_int8`（axis=1）、`concat_axis3_int8`（C=8 反例）。
  两输入由 `tools/e2e.sh` 的 `golden/input.bin` + `golden/input1.bin` 提供。
- **结果**：
  - `concat_axis3_c16_int8`：编译 ✓，`ConcatLayer<0>`，DRAM 1.0K/1.0K/0/0/448；板端 **1024/1024 逐位相等**，FPS 783。
  - `concat_axis1_c16_int8`：编译 ✓，`ConcatLayer<0>`；板端 **1024/1024 逐位相等**（非通道轴同样正确），FPS 483。
  - `concat_axis3_int8`（C=8）：**编译失败**，`Bridging is disallowed but gap is detected between Input and MDLA_5_0`。
- **编译器发出的层**：`ConcatLayer`（单算子图与链式图都保留，见 E-SHP-06）。
- **适用范围**：int8、2 输入、axis ∈ {1,3}、C=16；**3 输入及以上未测**（受 `tools/e2e.sh` 只支持 1–2 个输入的限制，Q4）。
- **剩余疑问**：`fused_activation_function` 只用了 NONE；不同 scale 的两输入（需要 requant）未测，按 FC 的经验会引入 ±1，需另立契约。

### E-SHP-04 SPLIT

- **假设**：`SPLIT`（axis + num_splits）会降成 `SliceLayer`；若输出切片能被直接物化，则两片都应可用。
- **预测**：`[1,4,8,16]` 沿 axis 3 二分（切片 C=8）时，第 0 片可用、第 1 片可能因"起始偏移落在 8 通道处"需要桥接；把输入加宽到 C=32（切片 C=16）应两片都可用。
- **做法**：每个切片单独建一个模型（graph output = 该切片，另一片保留为未使用的输出，因为 `e2e.sh` 只回读一个输出）。
- **结果**：
  - `split_axis3_part1_int8`（C=8 切片，第 0 片）：编译 ✓，`SliceLayer<0>{SPLIT}`，DRAM 512/256/0/0/224；板端 **256/256 逐位相等**，FPS 1329。
  - `split_axis3_part2_int8`（C=8 切片，第 1 片）：**编译失败**，`Bridging is disallowed but gap is detected between MDLA_5_0 and Output`。
  - `split_axis3_c16_part1_int8` / `_part2_int8`（输入 C=32，切片 C=16）：均编译 ✓、`SliceLayer<0>`，
    板端 **512/512 逐位相等**（两片分别验过），FPS 846 / 1226。
- **编译器发出的层**：`SliceLayer`（TFLite SPLIT 被降成 slice，而不是"多输出的 split 原语"）。
- **适用范围**：int8、axis=3、num_splits=2、切片末维 ≥16 B（或左对齐的第 0 片）。
- **剩余疑问**：为什么"第 0 片 8 通道"能物化而"第 1 片 8 通道"不能（§2.6 例外 1）——机制未证实（Q1）；
  `num_splits ≥ 3`、沿非通道轴切分、以及 SPLIT_V（不等宽切片）未测。

### E-SHP-05 PAD

- **假设**：零常量 pad 会落到 `PadLayer`，并且常量存在 NVM（DRAM 表的 `Static` 非 0）；数值上等于 `np.pad(..., constant_values=0)`。
- **预测**：`[1,4,8,16]` 在 H/W 各补 1 圈 → `[1,6,10,16]`，编译成功，板端 960/960 逐位相等；若 pad 贴错边（比如把 4D 当 NCHW 解释），则会出现大面积失配。
- **做法**：`pad_1x4x8x16_int8`，`paddings=[[0,0],[1,1],[1,1],[0,0]]`；输入码值永不为 0，让 0 值只能是 pad。
- **结果**：编译 ✓，`PadLayer<0>{PAD}`，DRAM **512/960/Temp 0/Static 224/Code 1.1K**（常量确实进 NVM），.dla 4703 B；
  板端 **960/960 逐位相等**，FPS 944。→ "4D paddings 与数据轴一一对应（NHWC 语义）"被证实，没有 NCHW 误解释。
- **编译器发出的层**：`PadLayer`（没有折叠，也没有折进相邻层——单算子图里无处可折，链式图见下）。
- **适用范围**：int8、常量 0、仅在 H/W 上补、NHWC；**非零常量 pad（PADV2）、负 padding（裁剪）、reflect 模式未测**。
- **剩余疑问**：pad 到"每行字节数变化"的场合（例如把 W 从 8 补到 9）对 §2 规则的影响未测（Q5）。

### E-SHP-06 链式 CONCAT → PAD（回答"层会不会被折叠"）

- **假设（竞争）**：H-a：两个 shape 算子在同一图里会被编译器合并/折叠（因为都是纯搬运）；
  H-b：两者都保留，只是共享一个 `ExecutionStep`。
- **预测**：若 H-a 成立，exec plan 只应出现 1 个层（或出现 `IdentityLayer`/无层）；若 H-b 成立，应出现 `ConcatLayer` + `PadLayer` 两个层，并且板端数值仍逐位正确。
- **做法**：`chain_concat_pad_int8`：两输入 `[1,4,8,16]` → CONCAT(axis3) → `[1,4,8,32]` → PAD(H/W +1) → `[1,6,10,32]`。无算术、无 requant，仍可按 `int8_tol=0` 验收。
- **结果**：**H-b 成立**。exec plan：
  ```
  Graph[0] Step[0]
   ├ Target: MDLA_5_0
   └ Subgraph:
     ├ ConcatLayer<0>{CONCATENATION}
     ├ PadLayer<1>{PAD}
     └ Output: OpResult (external)
  ```
  DRAM 1.0K/1.9K/0/**Static 448**/Code 1.3K = 4.6K；板端 **1920/1920 逐位相等**，FPS 974。
- **适用范围**：只有这一种双层组合；"FC→reshape→FC"这种跨 matmul 的组合未测（Q6）。
- **剩余疑问**：层不被折叠 ⇒ 每个 shape 算子都要付一次层的启动/带宽成本；**性能代价没有被单独测量**
  （本文件里的 FPS 都是 512–2000 元素级的小图，属于**启动开销主导**，不能当作吞吐结论——AGENTS.md R7：性能结论单独成立才写）。

---

## 4. 失败清单（编译期，原文照抄）

全部 6 个失败的模型都**保留在 `ops/` 下**（`model.tflite` + `model.json` + `model.compile.log`），没有数值结论。

### 4.0 可直接搬进能力表的失败原文（每个失败形状一行）

`--disallow-bridge` 下只观测到 **两种**失败句子（设备/工具链见页首；失败句都在 exec plan 打印**之后**出现）：

```
# ncc-tflite 7.3.15 · --arch=mdla5.0 --num-mdla=2 --l1-size-kb=3072 --opt=3 --disallow-bridge
IN_GAP    = "Bridging is disallowed but gap is detected between Input and MDLA_5_0"
OUT_GAP   = "Bridging is disallowed but gap is detected between MDLA_5_0 and Output"
FAIL_TAIL = "ERROR: Fail to compile ops/<model-dir>/model.tflite"   # 紧跟在 gap 句之后
```

| # | 算子（TFLite 语义） | 具体形状 / 属性 | 违反的边界张量 | 每行字节 | 失败句 | 证据目录 |
|---|---|---|---|---|---|---|
| F1 | RESHAPE | `[1,4,8,8]` → `[1,8,4,8]`，int8 | **外部输入** C=8 | 8 B | `IN_GAP` | `ops/reshape_1x4x8x8_to_1x8x4x8_int8/` |
| F2 | TRANSPOSE | `[1,4,8,8]`，perm `(0,2,1,3)`，int8 | **外部输入** C=8 | 8 B | `IN_GAP` | `ops/transpose_1x4x8x8_int8/` |
| F3 | CONCATENATION | axis=3，`2×[1,4,8,8]` → `[1,4,8,16]`，int8 | **两个外部输入** 均 C=8 | 8 B | `IN_GAP` | `ops/concat_axis3_int8/` |
| F4 | TRANSPOSE | `[1,4,8,4]`，perm `(0,2,1,3)`，**fp16** | **外部输入** C=4 | 8 B | `IN_GAP` | `ops/transpose_1x4x8x4_fp16/` |
| F5 | TRANSPOSE | `[1,4,8,16]`，perm `(0,3,1,2)` → 输出 `[1,16,4,8]`，int8 | **外部输出** C=8 | 8 B | `OUT_GAP` | `ops/transpose_1x4x8x16_perm0312_int8/` |
| F6 | SPLIT | axis=3，num_splits=2，输入 `[1,4,8,16]`，graph output = **第 1 片** | **外部输出** C=8 | 8 B | `OUT_GAP` | `ops/split_axis3_part2_int8/` |

配套的"应当通过"对照（写能力表时并排给出，说明这不是算子不支持）：

| 对照 | 形状 | 每行字节 | 结果 |
|---|---|---|---|
| F1' | RESHAPE `[1,4,8,16]`→`[1,8,4,16]`，int8 | 16 B | 编译 ✓ + 板端 512/512 逐位 |
| F2' | TRANSPOSE `[1,4,8,8]`，perm `(0,2,1,3)`，**fp16** | 16 B | 编译 ✓ + 板端 max_abs 0 |
| F3' | CONCATENATION axis=3，`2×[1,4,8,16]`，int8 | 16 B | 编译 ✓ + 板端 1024/1024 逐位 |
| F6' | SPLIT axis=3，输入 `[1,4,8,32]`，graph output = 第 1 片（C=16） | 16 B | 编译 ✓ + 板端 512/512 逐位 |

> 一句话进能力表：**shape/layout 五算子在 `--disallow-bridge` 下的唯一失败原因是"DRAM 边界张量每行 < 16 B"
> （`IN_GAP` = 外部输入不达标，`OUT_GAP` = 外部输出不达标），不是算子缺失；每行 ≥ 16 B 的同一算子全部编译通过且板端逐位正确。**
> 未验证：关掉 `--disallow-bridge` 时这些形状是否只是回退到 CPU 桥接（按策略我们不测、也不接受，记 UNKNOWN）。

| 模型 | 阶段 | 原始错误行 |
|---|---|---|
| `reshape_1x4x8x8_to_1x8x4x8_int8` | 编译期 | `Bridging is disallowed but gap is detected between Input and MDLA_5_0` / `ERROR: Fail to compile ops/reshape_1x4x8x8_to_1x8x4x8_int8/model.tflite` |
| `transpose_1x4x8x8_int8` | 编译期 | `Bridging is disallowed but gap is detected between Input and MDLA_5_0` |
| `concat_axis3_int8` | 编译期 | `Bridging is disallowed but gap is detected between Input and MDLA_5_0` |
| `transpose_1x4x8x16_perm0312_int8` | 编译期 | `Bridging is disallowed but gap is detected between MDLA_5_0 and Output` |
| `split_axis3_part2_int8` | 编译期 | `Bridging is disallowed but gap is detected between MDLA_5_0 and Output` |
| `transpose_1x4x8x4_fp16` | 编译期 | `Bridging is disallowed but gap is detected between Input and MDLA_5_0` |

**没有运行期失败**：12 个编译成功的模型全部在板上跑通（`neuronrt -m hw`，repeat=3），没有
`There is a runtime error during inference`。

⚠️ **读日志的坑**：失败的模型在 `model.compile.log` 里**也会打印 `Layer<Target: MDLA_5_0>` 的 exec plan**，
桥接检查发生在打印之后。看到层名不等于编译成功，必须看 `EXIT=` 与是否产出 `.dla`。

---

## 5. 板端证据汇总

| 模型 | 编译 | 层 | 形状 | 逐位一致 | FPS（小图，启动主导） | `.dla` sha256 前 16 |
|---|---|---|---|---|---|---|
| `reshape_1x64_to_1x4x16_int8` | EXIT=0 | ReshapeLayer | [1,64]→[1,4,16] | 64/64 | 988 | `b1d9f91714edfe11` |
| `reshape_1x4x64_to_1x4x8x8_int8` | EXIT=0 | ReshapeLayer | [1,4,64]→[1,4,8,8] | 256/256 | 456 | `1129e54a0123efe0` |
| `reshape_1x4x8x16_to_1x8x4x16_int8` | EXIT=0 | ReshapeLayer | [1,4,8,16]→[1,8,4,16] | 512/512 | 1109 | `ffb53818193fe51e` |
| `transpose_1x4x8x16_int8` | EXIT=0 | TransposeLayer | perm 0,2,1,3 | 512/512 | 924 | `078ae8ab0c0b78b7` |
| `transpose_1x4x8x8_fp16` | EXIT=0 | TransposeLayer | fp16 C=8 | 512/512（max_abs 0） | 1824 | `f3b79b002363173f` |
| `concat_axis3_c16_int8` | EXIT=0 | ConcatLayer | axis3 → C=32 | 1024/1024 | 783 | `ac336ebb3c75a9c1` |
| `concat_axis1_c16_int8` | EXIT=0 | ConcatLayer | axis1 → H=8 | 1024/1024 | 483 | `e8a82586daacb6c8` |
| `split_axis3_part1_int8` | EXIT=0 | SliceLayer | 第 0 片，C=8 | 256/256 | 1329 | `42d2f54a57c39b4e` |
| `split_axis3_c16_part1_int8` | EXIT=0 | SliceLayer | 第 0 片，C=16 | 512/512 | 846 | `39628a2f1d8c8f9a` |
| `split_axis3_c16_part2_int8` | EXIT=0 | SliceLayer | 第 1 片，C=16 | 512/512 | 1226 | `e8e35c6db7f29cc1` |
| `pad_1x4x8x16_int8` | EXIT=0 | PadLayer | [1,4,8,16]→[1,6,10,16] | 960/960 | 944 | `61ddb45ec6528eee` |
| `chain_concat_pad_int8` | EXIT=0 | Concat+Pad | →[1,6,10,32] | 1920/1920 | 974 | `09b750f8aa638fd0` |

设备状态：全部实验结束后跑 `tools/healthcheck.sh` → **PASS**（`device identity mt6897/16`、`/dev/apusys` OK、
known-good op `fc_1x64x64_int8` PASS、重复性 PASS、无 crash/leak 标记）。
即：这五个算子**没有污染后续任务**（AGENTS.md §3.4）。

---

## 6. 对"真实 transformer block"的适用性

- **能直接进 NPU 图**：`RESHAPE`、`TRANSPOSE`、`CONCAT(axis=3, C%16==0)`、`SPLIT`、`PAD` 都会成为真正的
  MDLA 层，并且数值逐位正确（整数路径）。所以"matmul 之间需要 shape 算子"这件事**不是桥接**。
- **前提**：进入/离开 NPU 的每个 DRAM 张量每行 ≥16 B（int8 即通道数 %16==0，或 16 的倍数关系按 dtype 折算）。
  hidden=128/256/768 这类常见配置天然满足；hidden=8/24 这类小模型会直接编译失败。
- **`PACK`/`UNPACK` 被 ncc 降成 Reshape+Concat 这件事**：本实验说明这两者**各自都会保留为独立层**
  （不会被折叠成一次搬运），且各自受 16 B 门槛约束，因此 unpack/pack 在 NPU 上会是**多一次层的开销**而不是免费别名。
  但这只是由"单算子/双层图不折叠"推出的**间接**结论，没有直接编译 pack/unpack 模型验证（Q7）。
- **成本量级**：本文件所有形状都是 64–1920 字节级，FPS 483–1824 完全由启动开销决定，
  **不能**用来推断大张量的 patch 带宽。性能结论必须另做（用真实尺寸 + 多次 repeat + FC 基准对比）。

---

## 7. 复现

```bash
# 在 worktree 内（WSL 里有 tflite schema 的 python）
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-shapeops" && \
  /usr/bin/python3 tools/build_shapeops.py --list'
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-shapeops" && \
  /usr/bin/python3 tools/build_shapeops.py concat_axis3_c16_int8 --out ops/concat_axis3_c16_int8 && \
  ./tools/compile.sh ops/concat_axis3_c16_int8/model.tflite ops/concat_axis3_c16_int8/model.dla'

# 板端（Git Bash，必须持锁）
export MSYS_NO_PATHCONV=1
NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" \
  ./tools/devlock.sh run -- ./tools/e2e.sh ops/concat_axis3_c16_int8
```

`tools/build_shapeops.py` 里每个 spec 上方都有注释说明"这个模型是为了区分哪两个假设"，改动前请保留（AGENTS.md R9）。

---

## 8. 剩余疑问（UNKNOWN / 待做）

- **Q1**：为什么 `SPLIT` 第 0 片（输出 8 通道 = 8 B/行）能物化，而第 1 片同样形状不能？候选解释是
  "左对齐子张量可被当作别名/部分 tile"；可判别的实验：让第 0 片输出**非左对齐**（例如 stride 不是
  连续通道），或对 16 通道输入做 `num_splits=4` 只看第 0 片——若仍通过，则"左对齐"是真正的判据。
- **Q2**：`ReshapeLayer` 是否真的搬运数据？平坦 reshape 的"正确"输出与零拷贝别名逐位相同，本实验无法区分。
  可判别的实验：在 reshape **前后**各接一个真实层（FC），比较两种情况下的 `.dla` 层数/命令区大小；
  或用 `probe/iolog2.c` 比较 ioctl 提交的 cmdbuf 长度。
- **Q3**：其它 perm（通道与 W 交换等）、perm 含 5D、以及"转置 + 通道 16"的组合是否都可用。
- **Q4**：**3 输入以上 CONCAT 未测**（q|k|v 的真实形态）——`tools/e2e.sh` 目前只喂 `input.bin` + `input1.bin`，
  且 `tools/compare.py` 只对拍单个 `output`。需要 Lead 扩展公共工具（多输入/多输出）。
- **Q5**：pad 之后每行字节数发生变化（如 W: 8→9）时，对 §2 的 16 B 规则有无影响。
- **Q6**：`FC → RESHAPE → FC`（真实跨 matmul 形态）是否仍保留 `ReshapeLayer`，以及它带来的额外延迟。
- **Q7**：直接编译 `PACK`/`UNPACK` 模型，验证"降成 Reshape+Concat"的说法与层数。
- **Q8**：§2 的 16 B 门槛只测了 C ∈ {4,8,16}；非 2 的幂（如 12、20）与 int16/int32 通路未测。
- **Q9**：两个 CONCAT 输入**不同 scale**（需要 requant）时的整数契约（预期 ±1，需单独登记 tol）。

---

## 9. 工具侧记录（不属于我的改动范围，交 Lead）

1. `tools/e2e.sh` 原先把 `WSLROOT` 写死成 `npu-ops` 检出路径，在任何 worktree 里跑都会让
   compare 步去错误目录执行（现象：`FileNotFoundError: ops/<name>/model.json`，而设备其实已经跑成功）。
   **Lead 已修**（改为按自身 checkout 推导），本文件所有板端记录都是修好后重跑的。
2. worktree 检出时 shell 脚本是 CRLF，WSL 里 `./tools/compile.sh` 报 `/usr/bin/env: 'bash\r'`。
   **Lead 已加 `.gitattributes`（`text eol=lf`）**。
3. `tools/compare.py` 的 dtype 取值表键是 `int8/int32/float32/float16`，`model.json` 里写 `fp16` 会
   `KeyError: 'fp16'`（我最初写 `fp16` 时踩到，已按工具约定改成 `float16`）。建议 Lead 在
   `AGENT-BRIEF.md` 里把这条写进"model.json 契约"，避免下一个 agent 再踩。
   **Lead 已修**：`compare.py` 现在同时接受别名 `fp16/fp32/i8/i32`；本文件的 `model.json` 仍统一写规范名
   （`int8` / `float16`），两种都能跑。
4. `tools/e2e.sh` 只支持 1 个输入 + 可选 `input1.bin`、只回读 1 个输出；`CONCAT` ≥3 输入、
   `SPLIT` 多输出只能拆成"每片一个模型"来做（本文件即如此，见 E-SHP-04）。若要做真实 q|k|v，
   需要工具升级。→ **Lead 记为待办**；本文件的证据就是"每片一个模型"的做法，验收时请按此口径看。
5. `.gitattributes` 只覆盖 `*.sh/*.py/*.c/*.h/*.md`；`model.json / model.compile.log / compare.txt`
   仍会触发 `LF will be replaced by CRLF` 警告（内容不受影响，但建议把 `*.json *.log *.txt` 一并加进
   `text eol=lf`，否则下次 checkout 会把这些证据文件变成 CRLF，diff 噪音大）。
   **Lead 已修**：`.gitattributes` 现覆盖 `*.json/*.log/*.txt/*.tsv`。
6. **`tools/healthcheck.sh` 会改写基线证据文件**：在全新 worktree 里 `ops/fc_1x64x64_int8/model.tflite`
   不存在（`.tflite` 被 gitignore），所以 healthcheck 会**重新生成并重新编译**基线模型，
   把 `model.compile.log`（`SIZE 7545 → 7549`、sha256 变化）和 `e2e.json` 时间戳写脏。
   多个 worktree 并行时这是**共享文件的写冲突源**。我这次已用 `git checkout -- ops/fc_1x64x64_int8/`
   还原，没有把它带进提交。建议 Lead 让 healthcheck 用独立的临时目录（或 `GOOD_OP` 指向 worktree 私有副本）。
   **Lead 已修**：healthcheck 现在跑完会 `git checkout -- ops/$GOOD_OP` 还原（`[restore] cleaned …`，
   可用 `HEALTH_RESTORE=0` 关闭）。

> 以上 1/2/3/4/5/6 都已在 master 修复并同步进本 worktree（`chore: sync shared tooling from master` 两个提交）；
> 本文件的板端记录是在 1、2 修好后重跑的，其余条目只影响工具体验、不影响任何数值结论。
