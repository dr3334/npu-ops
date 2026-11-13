# OP-ncc-flags —— ncc 编译选项盘点与实测（2026-11-13）

> 背景：有人问「-O2/-O3、--gno=NDF/SMP、--opt-bw、--show-exec-plan、--show-memory-summary、--gen-debug-info 有没有用」。
> 结论先说：**`--show-exec-plan` 与 `--show-memory-summary` 一直在用；其余优化类开关本轮第一次实测，没有一个值得改**，
> 而 `--gen-debug-info` 是下一步该用的（它给逐算子信息，不是提速开关）。

## 1. 我们一直在用的（`tools/compile.sh` 固定）

```
ncc-tflite --arch=mdla5.0 --opt=3 --disallow-bridge --num-mdla=2 --l1-size-kb=3072 --show-exec-plan --show-memory-summary
```

即：`--opt=3`（等于 `-O3` 那一档，注意 ncc 的 **默认是 2**）、`--show-exec-plan`、`--show-memory-summary`，
外加本仓库的策略项 `--disallow-bridge`（bridge = 没上 NPU = 失败）、`--num-mdla=2`、`--l1-size-kb=3072`。

## 2. 从没用过、本轮才测的（ncc 7.3.15 全部接受）

`--gno=NDF`、`--gno=SMP`、`--gno=BMP`、`--opt-bw`、`--opt-accuracy`、`--opt-aggressive`、
`--gno-non-4d-tiling`、`--basic-tiling`、`--gen-debug-info`。

## 3. 编译层：全部通过，图结构不变

被测图：`ops/decoder_layer_t16_w8a16_native`（单层、T=16、8W16A、native int16 缝，tflite 15.3 MB）。
11 个变体**全部 EXIT=0、1 个 `Execution Step`、无 bridge**，`.dla` 体积 15,104,232 – 15,152,413 B（差 <0.4%）。
⇒ 这些开关**不改变图划分**，只影响算子实现选择，所以只能靠板端时间来判断。

## 4. 板端：单次对比会骗人，必须配对测

**第一轮（每个变体各跑一次，-c 20）**看起来很美：

| 变体 | ms/inf | 相对 baseline |
|---|---|---|
| `--opt=3`（现状） | 2.1451 | 1.00 |
| `--opt=2` | 1.9033 | **−11%** |
| `--opt-accuracy` | 1.8495 | **−14%** |
| `--opt-aggressive` | 1.6991 | **−21%** |
| `--gno=NDF` / `SMP` / `BMP` | 2.218 / 2.045 / 2.122 | +3% / −5% / −1% |
| `--opt-bw` / `NDF+--opt-bw` | 2.257 / 2.202 | +5% / +3% |
| `--gno-non-4d-tiling` | 2.338 | +9% |

**配对交替（AB / BA 交替，8 对，-c 50，算每对比值再取中位）把结论推翻了**：`--opt-aggressive` 的 8 个比值是
`1.041 / 0.910 / 0.851 / 0.919 / 1.028 / 1.012 / 0.971 / 0.863`，**中位 0.94、极差 0.85–1.04**。
同一份 `.dla` 的绝对时间在同一会话里能从 2.02 漂到 2.65 ms（±15%），**时钟/热态漂移比开关效应还大**。
⇒ 本轮**不采纳**任何优化开关：`--opt=3` 保持现状。

## 5. 两个开关会改数值（不是免费提速）

同一输入、同一驱动的输出 md5：`--opt=3` 与 `--opt=2` **相同**（`2a3de8f8…`）；
`--opt-accuracy`（`f907da91…`）与 `--opt-aggressive`（`980e6fb0…`）**都不同**。
⇒ 要采用它们，必须重跑一遍全部数值判据（实现门 + 分档质量期望），不能当作纯性能开关。

## 6. `--gno=NDF` / `--opt-bw` 为什么对我们没用（结构性原因）

这两个开关优化的是 **DRAM 访问**。而我们的 24 层 decode 塔是 **35.1 ms/token 读 16.7 MB 权重 ≈ 0.48 GB/s**，
远低于实测的 18.5 GB/s 带宽上限——**瓶颈不在 DRAM**（在 KV 的 host 拷贝 33.4 ms 与 T=8 padding）。
所以在当前形态下它们没有可省的东西；等 KV 常驻化之后再测才有意义。

## 7. 下一步该用的：`--gen-debug-info`

它不是提速开关，而是**逐算子性能信息**。当前 decode 的 68.5 ms/token 里，KV 宽度 33.4 ms、塔本体 35.1 ms，
但塔内哪一段最贵（attention 的 softmax？MLP 的三个 FC？padding 的浪费？）还没拆开。
下一步：用 `--gen-debug-info` 编译 decode 图，拿到逐算子时间，再决定是改结构（T=8 padding、KV 常驻）还是调开关。

## 8. 方法学（写进 NOTES）

**板端计时必须配对交替、报比值中位数，不报绝对 ms。** 同一 `.dla` 同一会话内可漂 ±15%，跨会话更糟；
单次 A/B 足以骗出「−21%」这种假结论。这也是为什么此前所有「x ms/inf」的数字都只作量级记录、不作性能结论。
