# E-108 — 逐元素二元算子：MUL / SUB / MAXIMUM / MINIMUM（int8，同形状无广播）

> 范围：`feat/elem` worktree（`wt-elem`）。本文只记录**我自己的**实验；ADD 本身（E-107）与公共工具由 Lead 维护。
> 方法约束见 `AGENTS.md`（R1–R10）与 `docs/AGENT-BRIEF.md`。

---

## 0. TL;DR

| op | shape | 编译（ncc-tflite 7.3.15） | 板端（neuronrt，MDLA_5_0） | 数值（int64 独立参考，int8_tol=0） |
|---|---|---|---|---|
| MUL | 1x64 | **PASS** `ElementWiseMulLayer` @ MDLA_5_0 | 跑通，3 次重复 | **PASS 0/64 不匹配** |
| SUB | 1x64 | **PASS** `ElementWiseSubLayer` @ MDLA_5_0 | 跑通 | **PASS 0/64** |
| MAXIMUM | 1x64 | **PASS** `ElementWiseMaxLayer` @ MDLA_5_0 | 跑通 | **PASS 0/64** |
| MINIMUM | 1x64 | **PASS** `ElementWiseMinLayer` @ MDLA_5_0 | 跑通 | **PASS 0/64** |
| MUL | 8x128 | **PASS** | 跑通 | **PASS 0/1024** |
| SUB | 8x128 | **PASS** | 跑通 | **PASS 0/1024** |
| MAXIMUM | 8x128 | **PASS** | 跑通 | **PASS 0/1024** |
| MINIMUM | 8x128 | **PASS** | 跑通 | **PASS 0/1024** |

* **四个算子全部编译成功并落到 MDLA_5_0，全部逐位通过**（8 个模型、共 4480 个输出编码 0 误差）。
* **没有任何一个"算子级"编译失败**：MUL/SUB/MAXIMUM/MINIMUM 都被 ncc 识别为原生 `ElementWise*Layer`。
* 真正的失败边界是**形状粒度**（E-108D）：`K` 必须是 16 的倍数（例外：`K=4` 且 `M` 是 4 的倍数），否则 ncc 要插一个非 DLA 的搬运层，被 `--disallow-bridge` 挡住：
  ```
  Bridging is disallowed but gap is detected between Input and MDLA_5_0
  ERROR: Fail to compile ops/.../model.tflite
  ```
  原始日志归档：`docs/OP-elementwise-compile-fail.log`。

---

## 1. 证据绑定（R1）

| 项 | 值 |
|---|---|
| 仓库/分支 | `npu-ops` worktree `wt-elem`，分支 `feat/elem`，基线 commit `bdd7668` |
| 编译器 | `ncc-tflite` **Version: 7.3.15**（NeuroPilot SDK basic 7.0.8 包内），`--arch=mdla5.0 --opt=3 --disallow-bridge --num-mdla=2 --l1-size-kb=3072` |
| 设备 | `adb` 序列号 `0123456789ABCDEF`，`ro.board.platform=mt6897`，Android 16，SELinux Enforcing，root |
| 板端执行器 | 厂商 `neuronrt`（`-m hw -a <dla> -i in -i in1 -o out -c 3`），**不是**自建提交器（见 §7） |
| 板端串行 | `NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" ./tools/devlock.sh run -- ...`（R5） |
| 模型生成 | 我自己的 `tools/build_binop.py`（新增；不改 `tools/build_ops.py`），seed 20261112 |
| 生成/编译日期 | 2026-11-12 20:57–20:59 (+08:00) |

每个 op 目录里 `model.compile.log` 含 cmd 行、ncc 版本、date、`Target: MDLA_5_0`、DRAM 表、`EXIT=0`、`SIZE=`、`SHA256=`；`compare.txt` 是 `tools/compare.py` 的原始输出；`e2e.json` 是 `tools/e2e.sh` 的机器可读收尾行。

`.dla` 指纹（sha256，取自各自 `model.compile.log`）：

| op | 1x64 | 8x128 |
|---|---|---|
| MUL | `0c6bf500c37b6d74…` (3071 B) | `e078eb818af23a18…` (3089 B) |
| SUB | `b85cb44a28711b0c…` | `cca0877bb986ff5e…` |
| MAXIMUM | `0fa025b2d936ea2c…` | `301bc6025aca59d7…` |
| MINIMUM | `eca7b2e54fea927b…` | `be4ea48388257715…` |

---

## 2. 公共方法（四个算子共用）

### 2.1 量化契约：为什么可以要求 `int8_tol = 0`

TFLite 的逐元素算子在 int8 上都要先算 `real_multiplier = sa*sb/so`（零点全为 0）：

* **SUB / MAXIMUM / MINIMUM**：取 `so = sa = sb = 1/127`。
  - SUB 的参考实现是 `(a-off) - (b-off)`，且第二个输入先乘 `input2_multiplier = sb/so = 1`；乘子退化为恒等，
    编码域运算就是 `clip(a-b)`（**饱和**，不回绕）；
  - MAX/MIN 的参考实现直接比较编码值（要求三个 scale 相同），编码域运算就是 `max(a,b)` / `min(a,b)`。
  三者都不产生非平凡的 requant 舍入 ⇒ 逐位可预测。
* **MUL**：取 **`so = sa*sb = 1/127²`**，于是 `real_multiplier = sa*sb/so = 1.0`，
  kernel 退化为**纯整数乘法 + 饱和**：`out = clip(a*b, -128, 127)`。
  这是唯一能让 int8 MUL 做到"逐位可预测"的量化配置；如果取 `so = 1/127`，就必须做乘 1/127 的定点
  requant，rounding 变成实现的内部细节（那样 `int8_tol` 只能写 ≥1，且参考实现与被测实现的舍入规则
  相同才可能假通过 —— 正是 R3 要避免的）。**所以本实验主动避开那条路**，见 §6 开放问题 2。

### 2.2 独立参考（R3）

`tools/build_binop.py` 先用 numpy 生成 `a`、`b`（int8），再在 **int64** 域算 `y`，最后 `clip(-128,127)→int8`
写成 `golden/expected.bin`。参考侧不经过任何 TFLite/ncc/MDLA 代码路径；设备输出只用于 `tools/compare.py`
对拍，从不参与构造参考。

输入码值域与注入的边界用例（写在 `model.json` 的 `quant_note` 里）：

| op | 码值域 | 注入的 (a,b) 对（扁平下标 0..7） | 目的 |
|---|---|---|---|
| MUL | [-12,12] | (12,12) (-12,12) (12,-12) (-11,12) (0,-12) (11,-11) (1,0) (-1,12) | 正/负饱和钳位、0、单位 |
| SUB | [-100,100] | (100,-100) (-100,100) (-100,-100) (100,100) (7,7) (-7,-7) (0,-100) (-100,0) | 两端饱和、相等、零 |
| MAXIMUM / MINIMUM | [-100,100] | 同上 | 相等（tie）、全量程选择 |

随机采样不一定命中饱和/tie，所以显式注入 —— "找边界而不是找样例"（AGENTS §3.3）。
实测钳位点：MUL 1x64 中 4/64、8x128 中 17/1024；SUB 1x64 中 12/64、8x128 中 155/1024；
MAX/MIN 按定义不产生钳位（只做选择）。

### 2.3 复现命令

```bash
# 生成 model.tflite + model.json + golden/{input.bin,input1.bin,expected.bin}
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-elem" && \
  /usr/bin/python3 tools/build_binop.py --op MUL --m 1 --k 64 --out ops/mul_1x64_int8'
# 编译（--disallow-bridge 已在 compile.sh 内固定）
wsl.exe -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-elem" && \
  ./tools/compile.sh ops/mul_1x64_int8/model.tflite ops/mul_1x64_int8/model.dla'
# 板端 + 对拍（两输入由 e2e.sh 的 input1.bin 支持）
export MSYS_NO_PATHCONV=1
NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" ./tools/devlock.sh run -- \
  ./tools/e2e.sh ops/mul_1x64_int8
```

---

## 3. MUL（int8，同形状）

**假设**
* H1（量化契约成立）：`so = sa*sb ⇒ real_multiplier ≡ 1`，MDLA 的 `ElementWiseMulLayer` 做整数乘法+饱和，
  设备输出应**逐位等于** `clip(a*b)`。
* H2（存在隐藏 requant）：即使 `real_multiplier = 1`，编译器/硬件仍走定点乘子路径（乘子被量化到 32 bit、
  带 round），会出现 ±1 的差 → `int8_tol` 必须 ≥1。
* H3（不支持/被桥接）：MUL 在 MDLA5 上没有原生层，ncc 需要 CPU 侧 fallback，`--disallow-bridge` 直接失败。

**预测（先写）**：H1 成立时 → 编译日志出现单个 `ElementWiseMulLayer`、`Target: MDLA_5_0`、`EXIT=0`；
板端 64/64 与参考完全一致（含 4 个饱和点）。H2 成立时 → 少数点差 ±1，`compare.py` 报 FAIL(tol=0)。
H3 成立时 → 编译期报 `Bridging is disallowed but gap is detected ...`。

**做法**：`ops/mul_1x64_int8`、`ops/mul_8x128_int8`，契约 `so=sa*sb=1/127²`，参考 = `clip(a*b)`。

**结果（五个结论分开，R7）**
1. 编译：**成功**，`ElementWiseMulLayer<0>{MUL}` @ `MDLA_5_0`，`EXIT=0`，DRAM 1x64 = 560 B（Input 128 / Output 64 / Code 368）；
2. ioctl/提交：`neuronrt` 正常返回，3 次重复推理无错误输出；
3. 数值：**逐位一致** —— 1x64 `exact: YES, mismatched=0/64`；8x128 `0/1024`，`max_abs_diff=0`；
4. 设备状态：本批 8 次推理后 `tools/healthcheck.sh` 全项 PASS（已知正常算子 + 重复性 + dmesg 无异常）；
5. 性能：仅记录，不作结论 —— 1x64 0.83 ms/inf（~1006 FPS），8x128 1.02 ms/inf（~835 FPS），3 次重复，无控温/无对照。

**结论**：H1 接受，H2/H3 否决（在本实验条件下）。

**适用范围**：int8、`zp=0`、`so=sa*sb`、2D `[M,K]`、`M∈{1,8}`、`K∈{64,128}`、无广播、无 fused activation。
**剩余疑问**：`so=1/127`（需要 requant）时的舍入行为未测；见 §6。

---

## 4. SUB / MAXIMUM / MINIMUM

三个算子共享同一套方法，各自的**竞争假设**不同：

### 4.1 SUB
* H1：饱和减法 `out=clip(a-b)`（TFLite int8 SUB 语义）。
* H2：int8 **环绕**减法（`a-b` 溢出回绕），则在注入的 ±200 附近会产生符号翻转的错值。
* **预测**：H1 → 12/64 个钳位点全部落在 ±127/-128；H2 → 这些点差 ≥1，且方向固定（回绕）。
* **结果**：`ElementWiseSubLayer` @ MDLA_5_0，`EXIT=0`；1x64 `mismatched=0/64`、8x128 `0/1024`（含 12/64、155/1024 个饱和点）。
  → **H1 接受，H2 否决**（MDLA 在编码域做了饱和，不是回绕）。
* `so = sa`，参考 `clip(a-b)` 整数精确。

### 4.2 MAXIMUM / MINIMUM
* H1：纯选择，`out = max(a,b)` / `min(a,b)`，逐位精确，且**不需要 requant**（`so=sa`）。
* H2：MDLA5 没有原生 max/min，ncc 用 `Relu`/比较+选择的组合或 CPU 侧算子实现 → `--disallow-bridge` 失败
  （或落成 `ElementWise*Layer` 但数值语义是"先 requant 再比较"）。
* **预测**：H1 → `ElementWiseMaxLayer` / `ElementWiseMinLayer` @ MDLA_5_0，`mismatched=0`；
  H2 → 编译失败，或相等/tie 点上出现意外值。
* **结果**：`ElementWiseMaxLayer<0>{MAXIMUM}`、`ElementWiseMinLayer<0>{MINIMUM}` @ MDLA_5_0，`EXIT=0`；
  1x64 与 8x128 均 `mismatched=0`（含注入的 tie 对与全量程 ±100 对）。→ **H1 接受，H2 否决**。

**适用范围**：同 §3；MAX/MIN 的 tie 语义只在"两值相等"时检验（相等时无法区分 upper/lower 语义）。
**剩余疑问**：MAX/MIN 在 `zp≠0`（非对称量化）时的 offset 处理未测；广播版（`[M,K]` vs `[K]`）未测。

---

## 5. 编译边界图（E-108D，"找边界而不是找样例"）

只做编译（不烧板）；用 MUL 作代表，另外三个算子在关键点复核。

### 5.1 M × K 网格（OK = `EXIT=0` 且落 MDLA_5_0；x = 编译失败）

```
M\K   1    2    4    8    16   24   32   48   64   128  136
1     x    x    x    x    OK   x    OK   OK   OK   OK   x
2     x    x    x    x    OK   x    OK   OK   OK   OK   x
3     x    x    x    x    OK   x    OK   OK   OK   OK   x
4     x    x    OK   x    OK   x    OK   OK   OK   OK   x
5     x    x    x    x    OK   x    OK   OK   OK   OK   x
6     x    x    x    x    OK   x    OK   OK   OK   OK   x
7     x    x    x    x    OK   x    OK   OK   OK   OK   x
8     x    x    OK   x    OK   x    OK   OK   OK   OK   x
12    x    x    OK   x    OK   x    OK   OK   OK   OK   x
16    x    x    OK   x    OK   x    OK   OK   OK   OK   x
```

补充点（MUL，编译）：`K=80 OK`、`K=96 OK`（M=1,2,7）、`K=144 OK`、`K=72 x`、`K=40 x`、
`K=4` 且 `M=20,24,32,100 → OK`、`K=4` 且 `M=9,10,11 → x`；
`K=8` 且 `M=2,4,32 → x`（total=16/32/256 仍失败）、`K=2` 且 `M=8,16 → x`、`K=1` 且 `M=16 → x`、`K=12` 且 `M=4,8 → x`。

### 5.2 经验规则（**操作规则，机制为 UNKNOWN**）

> `(M,K)` 可编译，当且仅当 **`K % 16 == 0`**，或者 **`K == 4 且 (M*K) % 16 == 0`**。
> 其余（`K ∈ {1,2,8,12,24,40,72,136,…}`）一律失败，**与 `M` 无关**。

被**否决**的竞争假设：
* "总字节数是 16 的倍数即可" —— 被 `8x8`（64 B）、`2x8`（16 B）、`4x12`（48 B）、`8x24`（192 B）、
  `8x136`（1088 B）反驳，全部失败；
* "`M` 必须满足某条件" —— 被 `M=1..16` 全列行为一致反驳（唯一例外是 `K=4` 时 `M%4`）；
* "`K` 是 4 的倍数即可" —— 被 `K=8`、`K=12`、`K=24` 反驳。

### 5.3 失败原文（逐字）

```
===Graph[0] has 1 Execution Step(s)===
Graph[0] Step[0]
ExecutionStep
 ├ StepId: 0
 ├ Target: MDLA_5_0
 └ Subgraph:
   ├ ElementWiseMulLayer<0>{MUL}
   └ Output: OpResult (external)

===End of Graph[0]===

Bridging is disallowed but gap is detected between Input and MDLA_5_0
ERROR: Fail to compile /tmp/fail/m1k8/model.tflite
EXIT=1
SIZE=0
```

完整原始日志（5 个代表用例：`1x4`、`1x8`、`8x8`、`8x24`、`1x136`）：**`docs/OP-elementwise-compile-fail.log`**。
注意：失败发生在**图切分阶段**（layer 已经识别成 `ElementWiseMulLayer`、target 已经算成 MDLA_5_0），
失败原因是"Input 与 MDLA_5_0 之间有 gap" —— 即 ncc 判定张量布局/粒度不匹配，需要在 DLA 之外插一个搬运层。
`--disallow-bridge` 让这个隐藏的 CPU 回退变成硬错误（AGENT-BRIEF §4：bridge = 没上 NPU = 失败）。

### 5.4 边界是否与算子有关

同点复核（编译）：`SUB/MAXIMUM/MINIMUM` 在 `1x8 → x`、`1x16 → OK`、`4x4 → OK`、`8x4 → OK`
与 MUL **完全一致** ⇒ 这条粒度约束是 **MDLA5 逐元素层的共性**，不是某个算子的特例。

---

## 6. 失败清单与"没有证明的事"

### 6.1 编译失败清单（本任务要求的四个算子）

**MUL / SUB / MAXIMUM / MINIMUM 在 1x64 与 8x128 上全部 `EXIT=0`，没有一个算子级编译失败。**
失败只出现在 §5 的**形状粒度**上（代表性错误行见 §5.3，原始日志 `docs/OP-elementwise-compile-fail.log`）。

### 6.2 运行期失败

无。8 次板端推理（4 op × 2 形状，每次 `-c 3`）全部正常返回，`compare.txt` 全 PASS。

### 6.3 明确**未**证明（UNKNOWN）

1. **机制**：为什么 `K=8, M=2`（total 16 B）失败、而 `K=4, M=4`（total 16 B）成功？**UNKNOWN**。
   候选解释（未验证，禁止当结论）：(a) MDLA 逐元素层以 16 B 向量为粒度，要求最后一个维度能整块覆盖；
   (b) ncc 只有在对齐成立时才把 `[M,K]` 视作可整体搬运的连续张量，否则插 gap 拷贝层。
   已经做过的可区分点是"固定 `total=16`"这一组：`(16,1) x`、`(8,2) x`、`(4,4) OK`、`(2,8) x`、`(1,16) OK`
   —— 同样是 16 B 的张量，只有"`K` 能被 16 整除"或"`K=4` 且 4 行拼满 16 B"两种分解可行，
   "总量对齐"假设因此被否决（数据见 §5.1）。
   **还没做**的区分实验：把同一个元素总数写成 3D/4D（如 `[1,4,4]`、`[1,1,4,4]`、`[1,4,1,4]`），
   若 4D 的 `H*W` 组合改变了可行性，说明约束作用在 ncc 的 NHWC 布局上而不是 2D 的 `[M,K]` 上。
2. **MUL 的非 1 乘子**：`so = 1/127`（`real_multiplier = 1/127`）时的舍入是否与 CPU 参考一致，未测。
   注意这与 `AGENT-BRIEF` 的"全量程 requant 允许 `int8_tol=1`"对应，属于另一个实验。
3. **零点非零**（`zp≠0`，非对称量化）未测。
4. **广播**（`[M,K]` op `[K]`、标量）未测，本任务的契约是"同形状无广播"。
5. **多算子链**（一个模型里 MUL→SUB→MAX 连续）与融合（`fused_activation=NONE` 之外）未测。
6. **性能**：只记录不结论（见 §3 第 5 条）。样本量小、无控温、无对照实现。
7. **板端回读路径**：本实验用的是厂商 `neuronrt`，即"厂商运行时"路径；"自建提交器"对 MUL/SUB/MAX/MIN 的
   覆盖是另一件事（§7），本文件不声称已完成。

---

## 7. 与自建提交器（probe/npuadd.c）的关系

`probe/npuadd.c` 的自建提交器（E-106B-3，L3.5）把一次合法 ADD 的 **ncc 命令流**（`cmd368/cmd32/cmd16/
subcmd/cmdin`，加三个旧 IOVA）存成资产，然后自己走 `HANDSHAKE → ALLOC → mmap → MEM MAP → 打补丁 →
CMD RUN → DMA_BUF_SYNC` 完成提交，不调用 `libneuron`/`neuronrt`。

它的"算子无关"部分（握手、分配、重定位、提交、同步）与"算子相关"部分（命令流资产 + 旧 IOVA + 元素数）
是混在一起的。把前者抽成 `npu_bin_binop(asset_dir, old_iova[3], nelem, a, b, out)` 是可行的，
因为四个算子的 `.dla` 都是**单层、同形状、两个输入一个输出**，命令流的布局（三处地址字、3 个 cmdbuf、
同一 cmdbin 结构）应当相同 —— 但"应当"不是证据：每个算子都需要**自己的一次抓取**（`probe/iolog2.c`
+ `probe/run_shim.sh`）来确定它自己的命令流资产与旧 IOVA。

本轮的交付边界：**没有**修改 `probe/`；若要推进，见 `probe/npu_binop.c`（新增文件，泛化版本；验证情况写在该文件头注释与本文 §7.1）。

### 7.1 泛化版状态

见 `probe/npu_binop.c` 头部注释（`npu_binop(op, a, b, out)` + 资产目录/IOVA 参数化；ADD 资产路径与
`npu_add64()` 行为等价，MUL/SUB/MAX/MIN 需要各自的抓取资产）。

---

## 8. 交接（R8）

* 新增产物：`tools/build_binop.py`（生成器）、`ops/{mul,sub,max,min}_{1x64,8x128}_int8/`、
  `docs/OP-elementwise.md`（本文）、`docs/OP-elementwise-compile-fail.log`（失败原文归档）、
  `probe/npu_binop.c`（泛化提交器，见 §7）。
* 公共工具**未改**。已向 Lead 报告过的两处共享工具问题（worktree 下 `WSLROOT` 写死、CRLF shebang）已由 Lead
  在 master 修复（`tools/e2e.sh` 自行推导 WSL 根、`.gitattributes` `text eol=lf`），本 worktree 已同步，
  之后所有板端命令都直接用 `./tools/e2e.sh`，无需任何 workaround。
* 下一步建议（按价值排序）：
  1. `(M,K)` 最小可行集合的机制实验（§6.3-1），并给 `tools/` 加一个"编译可行性扫描器"，把 §5 的网格变成可回归的测试；
  2. MUL requant（`so=1/127`）的舍入语义；
  3. 广播变体；
  4. 自建提交器对四个算子的覆盖（需要 per-op 命令流抓取）。
