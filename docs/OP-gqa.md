# E-112 — GQA 的多头编译边界：ncc 没有头数上限，`build_block2` 有张量声明 bug

> 占号说明：本文件占用 `E-112`，公共索引（`docs/EXPERIMENTS.md`、`PLAN.md`、`CAPABILITY.md`）未改，
> 沿用 `docs/OP-silu.md` 占 `E-110` 的做法。

## 1. 问题

`docs/OP-block.md` 原记：`--heads 2` 的 `RESHAPE → BATCH_MATMUL(adj_y) → SoftMax → BATCH_MATMUL → RESHAPE`
在 importer 阶段失败（`ERROR: Fail to process the given model`），结论是"多 head 未通"，
候选原因指向 `ReshapeOptions.NewShape` 的构造顺序或 BMM 的输入 rank。

Qwen2.5-0.5B 需要 **14 个 query head + 2 个 KV head**（GQA 7:1）。若"多 head 编不出来"成立，
整条路线要退到"14 次单头 unroll"或"2×7 头"的兜底设计——代价是每层 14 次 launch、24 层共 336 次，
整块"单步下沉"的优势全部消失。所以必须先判定这条边界是**编译器的**还是**我们自己的**。

## 2. 假设

- **H1**：ncc 对 `BATCH_MATMUL` 的头数有上限（头数 ≥14 编不过）。
- **H2**：失败与头数无关，是复合图里某个算子的构造问题。
- **H3**：失败源于 `tools/build_block2.py` 的 flatbuffer 张量声明，与编译器能力无关。

## 3. 做法

在仓库外的临时目录里，用 `tools/build_bmm.py` 的 flatbuffer 管线直接生成模型，再用
`tools/compile.sh`（`ncc-tflite 7.3.15 --arch=mdla5.0 --disallow-bridge`）逐个编译。
判据按错误措辞分阶段：**`the given model` = 加载/解析期**、**`the input model` = 逐层校验期**、
**`Cannot find an execution plan` / `Bridging is disallowed` = 后端期**。

## 4. 结果

| 变体 | rc | 层数 / 结论 |
|---|---|---|
| 3-D BMM 14 头 `[14,8,64]²` adj_y=1 | 0 | **1 个 `BatchMatMulLayer`** |
| 4-D `[1,14,8,64]²` adj_y=1 | 0 | 1 层 |
| PV 3-D 14 头 | 0 | **1 层** |
| PV 4-D 14 头 | 0 | 被降成 **5 层**（slice/bmm/concat） |
| GQA `SPLIT(2)→CONCAT(14)` | 0 | 12 层，1 执行步 |
| 完整 14 头 attention（q14/kv2, hd64, hidden896, T=8, fp16） | 0 | **16 层，1 执行步，2.1 MB** |
| 完整 decoder 层（RMSNorm + 14 头 attn + SwiGLU） | 0 | **27 层，1 执行步，7.2 MB** |
| 头数扫描 1,2,3,4,6,7,8,12,14,16,32,…,1024 | 全 0 | 始终**一个** `BatchMatMulLayer`，约 512 B 代码/头 |
| rank 5-D / 6-D BMM | 1 | `the rank of shape should be in range [0, 4]` |
| int8 14 头 head_dim=8 | 1 | `Bridging is disallowed`（16 B 行规则） |
| 仓库原版 `build_block2.py --heads 2` | 1 | `Fail to process the given model`（**加载期**） |
| 同上，仅把 q/k/v/o 权重改成 `(H,H)` | 0 | 通过 —— **H3 成立** |
| 修好后的块构造器（ticket 01）：`build_block2.py --heads 2` | 0 | **31 层，1 执行步，`.dla` 101 KB**；板端 **PASS**（1 ULP，见 `docs/OP-block.md §10.5`） |
| 同上 `--heads 14 --hidden 896 --head_dim 64 --norm fctile` | 0 | **33 层，1 执行步，`.dla` 7.06 MB**；板端 **PASS**（见 `docs/OP-block.md §10.5`） |

**H1 否定**：没有头数上限，扫到 1024 头仍是一个原生层。
**H2 部分成立**：复合图本身没问题，构造正确就编。
**H3 成立**：根因是 `fc()` 按 FC 的**目标宽度** `[n,k]` 声明权重张量，而 `heads>1` 时传入的
`Wq/Wk/Wv` 只有 `HD` 行。`heads=14` 时声明 `[896,896]`（802,816 个数）而实际数组是 `(64,896)`
（57,344 个数），flatbuffer 声明了一个大 14 倍的缓冲区，ncc 在**模型加载**阶段就退出——
从未走到逐层校验，所以打不出层清单。

## 5. 适用范围

- `ncc-tflite 7.3.15`，`--arch=mdla5.0 --num-mdla=2 --l1-size-kb=3072 --disallow-bridge`，**厂商编译路径**；
- fp16 全部通过；**rank ≤ 4** 是硬限制；int8 下每行 < 16 B 会触发 bridge；
- 图规模不是限制（14 个单头 BMM unroll = 15 层，rc=0）。

## 6. 剩余疑问（本实验未覆盖）

1. **上板数值未验**：本实验只做编译侧，`EXIT=0` 不等于数值正确。
2. PV 的 3-D（1 层）与 4-D（5 层）差异对板端时序/功耗的影响未测——**优先 3-D**是编译侧的结论，未过量测。
3. `SPLIT→CONCAT` 的 KV 显式复制在板上的数值正确性未验。
4. bias（Qwen 的 q/k/v 都有 bias）对 FC 层下沉与合并的影响未测。

## 7. 对后续的影响

- `docs/OP-block.md` 的"多 head WIP/未通过"归因**已订正**——不能按一条编译器的假限制去设计兜底方案；
- 兜底方案（2×7 头 23 层 / 14 次 unroll 15 层）相对于单体 1 层是**纯亏损**，不采用；
- GQA 的 KV 显式复制采用 `SPLIT→CONCAT`，代价 4 层、仍在同一执行步；
- **ticket 01 已把修复落进 `tools/build_block2.py`**（权重张量按数组真实形状声明 + `check_fc_weight()`
  形状守卫 + `--hidden/--head_dim/--ffn/--norm`），两个多头整块 op 编译与板端都通过；
- ticket 01 顺带发现**第二个坑**（本文只做编译侧，没看到）：`[T,H]` 直接 `RESHAPE` 成 `[heads,T,HD]`
  也能编、也能跑，但那是"按头分块重排 token"，板端 `rel_l2=0.143`；必须先 `[T,H]→[T,heads,HD]`
  再 `TRANSPOSE(perm=[1,0,2])`。⇒ **编译 EXIT=0 + 单执行步不等于数学正确**，多头必须补板端对拍
  （见 `docs/OP-block.md §10.5.1`）。
