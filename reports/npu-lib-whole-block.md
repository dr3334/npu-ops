# 整块 = 单个算子：40 op 的 transformer block 一次提交跑通

日期：2026-11-14 · 板 Titan 1 / MT6897 · 目标 `ops/block_qwen25_1l_full_fp16`（T=8, H=64, 1 head, FFN=128, fp16，40 op / 63 张量）

## 0. 结果

| 项 | 值 |
|---|---|
| 提交形态 | **一次 `CMD RUN`，`num_subcmds = 2`**，5 个地址字全部重定位 |
| 我方 vs 厂商设备输出 | **逐字节相同**（512 个 fp16） |
| 我方 vs golden | `max_abs 9.766e-04`、`rmse 1.860e-04` —— 与厂商那条 `compare.txt` **完全一致** |
| 回归 | int8 矩阵 9/9 PASS；3 op 融合图（SwiGLU）仍逐字节一致 |

第一行输出就是 block 的整块结果（`[0.3147, 0.3987, 0.4885, -0.00818, …]`），两边一模一样。

## 1. 卡点回顾（上一轮的定位是对的）

整块图一次提交带 **2 个子命令**：

| 子命令 | cmdbuf | 里面的地址字 |
|---|---|---|
| subcmd0 | 32 / 16 / 6240 B | 2 处，都是输入 x（`0x50`、`0x340`） |
| subcmd1 | 32 / 16 / 5520 B | 3 处：常量区（`0x88`）、输入 x（`0x1060`）、输出（`0x1534`） |

之前只抓/只提交 subcmd0，所以输出地址根本不在资产里。

## 2. 做完的四件事

| # | 改动 | 文件 |
|---|---|---|
| 1 | shim 按 `num_subcmds` 枚举，逐个 dump `subcmd<i>` / `cmdbufs<i>` | [`probe/iolog2.c`](../probe/iolog2.c) + 新 [`probe/build_iolog2.sh`](../probe/build_iolog2.sh) |
| 2 | `mkasset` 从「断言正好 3 个 cmdbuf」改成按子命令分组，写 `subcmds.txt`（`<gi> <n> <size:dir>...`），单子命令仍走老文件名 | [`probe/mkasset.py`](../probe/mkasset.py) |
| 3 | `npu_run` 组装 N 个子命令、把重定位扫到**所有** cmdbuf | [`probe/npu_run.c`](../probe/npu_run.c) |
| 4 | `einfo_size` 进 plan（exec_infos 缓冲大小按抓取来的给） | 同上 |

## 3. 过程中撞的两个真 bug

1. **`patch_addrs` 会把计数清零**：逐块调用时只剩最后一块的命中数（`patched 3/5`）。改成逐块取回再加总。
2. **`exec_infos` 缓冲 76 B 不够**：CMA RUN 直接 `EINVAL`。这块图厂商用的是 **120 B**（`fd=13`，内容全 0）。
   现在由 plan 的 `einfo_size` 指定，缺省仍是 76（老资产不受影响）。

## 4. 必须说清的边界（这条很重要）

**常量区这次恰好落在抓取时的同一个 IOVA**（`0xcbaa0000 -> 0xcbaa0000`），所以：

- 这次成功**没有**证明「68 KB 常量区能在任意 IOVA 上重放」；
- [`docs/ISSUE-const-fc.md`](../docs/ISSUE-const-fc.md) 那个 3/11 的契约**仍然未解**；
- 下一步要专门做一次「先占位分配把常量区挤到别的 IOVA」的实验，才能判定整块路径对常量区的依赖有多硬。

其他边界：

- 一次提交 2 个子命令是我们**第一次**跑多子命令，别的图（24 层、T=128）子命令数可能更多，尚未扫；
- 整块仍含**常量权重**（`Static 64K`）；要换成动态权重（>2 个输入）还得扩 plan 的输入槽（见 [`reports/npu-lib-fusion.md`](npu-lib-fusion.md) §5）；
- 单次提交墙钟里含 `npu_run` 固定 200 ms 等待，**不是**算力时间，本报告不做性能结论。


## 6. ⚠️ 更正：整块重放**对常量区地址敏感**（2026-11-14 复测）

连跑 10 次同一份资产、同一份 plan，只记录常量区（B）实际落到的 IOVA 与输出 sha256：

| 常量区 IOVA | 次数 | 结果 |
|---|---|---|
| `0xcbaa0000`（= 抓取时那个） | 3 | ✅ 与参考输出逐字节相同 |
| `0xd36xxxxx`（别的地址） | 7 | ❌ 输出另一个稳定值（`6c4b3b1d…`），不是随机噪声 |

**3/10 正确** —— 这与 [`docs/ISSUE-const-fc.md`](../docs/ISSUE-const-fc.md) 记录的 3/11 是同一个现象：
常量区必须落在抓取时的 IOVA 上，命令流才对得上。

所以本文 §0 那次「一次提交跑通」是**地址巧合**，不是稳定能力。要让整块可靠，只有一条路：
**把权重从常量改成运行时输入**（`Static 28M → 0`），常量区的重放契约就不参与这条路径了。

## 7. 计时（能测到的与测不到的）

| 对象 | 数字 | 来源 |
|---|---|---|
| 玩具块（40 op，T=8）设备执行 | **1.039 ms/inf（696 FPS）** | 官方路径 `e2e.sh --repeat 5`（厂商 runtime 自己计时） |
| 真维度块（61 层，D=896，T=8）设备执行 | **2.3925 ms/inf（393 FPS）** | 官方路径 `e2e.sh --repeat 2` |
| 我们的提交路径 | **暂无可比数字** | `npu_run` 没有完成 fence：`ioctl` 之后是 `usleep(200 ms)` 硬等；同进程二次提交会被驱动拒绝（`CMD RUN: Invalid argument`） |

要给我们这条路出数，必须先做「提交完成等待」（fence 或可轮询的完成位），否则只能报 host 墙钟（0.73–0.85 s/次，其中含 200 ms 固定等待 + adb 推拉）。
## 5. 复现

```bash
# 抓取（会自动 dump 两个子命令）
# 资产: cmd0_0..cmd0_2.bin / cmd1_0..cmd1_2.bin / subcmd0.bin / subcmd1.bin / subcmds.txt / scratch.bin(常量区)
# plan: bytes 1024/69632/1024, n_inputs=2, extra_file=scratch.bin, expect_patches=5, einfo_size=120
# 板上: npu_run plan.txt
```
