# 黑盒研究 02：三个可能改变策略的开关（先验）

日期：2026-11-14 · SDK 7.0.8 / `ncc-tflite` 7.3.15 · 板 Titan 1（MT6897）
票：[`02-three-switches-first.md`](../.scratch/compiler-blackbox/issues/02-three-switches-first.md)

## 0. 结论先放

| 开关 | 结论 | 对现有策略的影响 |
|---|---|---|
| **`--extract-static-data`** | ✅ **成立**：权重变成输入张量，喂回原始权重后输出与参考**逐字节相同** | 「权重一律运行时输入」从**自己改图**升级成**官方开关**；常量区落址那条悬案可以彻底绕开 |
| `.dlb` + `readdlb` | ✅ 能用：能打包、能读出结构（静态抽取开关、临时缓冲共享、DRAM temp 表） | 容器层有官方结构视图，不必逆向 |
| `--resize` / `--runtime-dynamic-shape` | ⏳ **未测完**（见 §4） | 若成立，「每形状一资产」可能作废 |

## 1. `--extract-static-data`：权重外置（重点）

```bash
./tools/compile.sh ops/fc_1x64x64_int8/model.tflite /tmp/bb02/fc_extract.dla \
  --extract-static-data /tmp/bb02/fc_static.bin
```

| 项 | 值 |
|---|---|
| 原 `.dla` | **7545 B**（权重编在里面） |
| 外置后 `.dla` | **3666 B**（−3879 B） |
| 侧文件 `fc_static.bin` | **262,178 B**（≠ 权重，见 §3） |
| 输入表（`neuronrt -d`） | input[0] = `[1,1,1,64]` 64 B；**input[1] = `[1,1,64,64]` 4096 B** |

把 `model.tflite` 里那个 4096 B 常数缓冲原样喂进去：

```bash
neuronrt -m hw -a f.dla -i in.bin -i w.bin -o out.bin -c 1
```

**输出与 `ops/fc_1x64x64_int8/out_device.bin` 逐字节相同**（sha `70d49412ab708da7`）。

-> 官方支持「权重当运行时输入」，而且**不需要我们改图**。这意味着：常量区那个 3/10 的落址问题，
在这条路上**根本不存在**（产物里没有常量区）。剩下的是我们提交器只有 2 个输入槽（真维度块需要 7~11 个）。

## 2. `.dlb` 与 `readdlb`

| 操作 | 结果 |
|---|---|
| `dla-packer -o fc.dlb model.dla` | 7899 B |
| `dla-packer -l 9 ...` | 5831 B |
| `readdlb fc.dlb` | 打印 DLA 列表 + 三个功能开关（Legacy Static Data Extraction / extracted-shared / DRAM Temp Buffer Sharing）+ DRAM 临时缓冲表 |

注意 `readdlb` 打印完以 **SIGABRT**（rc = −6）退出，输出完整 —— 不是我们调用错。

## 3. 未解（已记，不猜）

- `fc_static.bin` 262,178 B 是什么？`neuronrt` 不消费它，只要 4096 B 原始权重。
- 262178 ≈ 256 KiB + 34 B，像是「按 tile 展开/复制后的布局」，但没有证据。

## 4. `--resize` / `--runtime-dynamic-shape`：一个成立、一个不成立

**`--resize` ✅ 成立，但有语法坑**：必须给**全部输入**尺寸、逗号分隔、按输入顺序。少给一个就：

```text
ERROR: Please specify the dimensions of all inputs in order
ERROR: Number of inputs: 2
ERROR: Fail to run ResizeInputsPropagationPass
```

| 命令 | 结果 |
|---|---|
| `--resize 1x128,1x128` | ✅ 编译 3091 B；I/O 表 `[1,1,1,128]`；喂 128 B 输出与本仓库 128 资产的参考**逐字节相同** |
| `--resize 1x1x1x128,1x1x1x128` | ✅ 同上（4331 B） |
| `--resize 1x64,1x128`（两输入不同尺寸） | ❌ EXIT=1 |

**但它是「编译期换形状」，不是「一份产物吃多种尺寸」**：`--resize` 改的是编译时输入维，产物仍只认那个尺寸。

**`--runtime-dynamic-shape` ⚠️ 不成立（按这个用法）**：能编过（4091 B），但 `neuronrt -d` 的 I/O 表仍写死 `[1,1,1,64]`，
喂 128 B 直接 `The size of input[0] should be 64`。所以至少经 `neuronrt` 这条路，一份产物**不能**吃多种尺寸。

**对现有策略的影响**：`--resize` 只能把「为新形状重建 tflite」省成「同一个 tflite 加一个 flag」，
**「每形状一资产」依然成立** —— 自建库票 02/08 的结论不变。
