# 算子融合：一张融合图在我们的提交器上一次提交跑完

日期：2026-11-14 · 板 Titan 1 / MT6897 · 目标图 `ops/gated_8x696_fp16`（`LOGISTIC + MUL + MUL`，fp16）

## 0. 结论

**融合图可以直接被我们的提交器重放，不需要 N 缓冲。** 一次提交算完 3 个算子，
输出与厂商设备输出**逐字节相同**。

| 项 | 值 |
|---|---|
| 图 | `LOGISTIC → MUL(g, sig) → MUL(m, u)`，`[8,696]` fp16 |
| 编译 | 3 层、**1 个执行步**、`.dla` 3757 B |
| DRAM | Input 21K / Output 10K / **Temp 0** / Static 0 → 中间张量全在 L1 |
| 我们的重放 | 地址字 **4 处**（gate 被引用两次：进 LOGISTIC 一次、进 MUL 一次） |
| 对比 | `our == vendor(e2e) == vendor(capture)`，**逐字节**（5568 个 fp16） |
| 对 golden | `max_abs 3.906e-03`（= 0.5~8 处 1 个 fp16 码）、`rmse 6.737e-04` —— 与厂商那条 compare 完全一致 |

## 1. 为什么这能成立（上一轮的疑问）

看编译日志的 `DRAM Usage`：融合图的 `Temp = 0`，中间张量全在 L1（本例 12K），
DRAM 上只有「图输入 + 图输出」。所以提交器的 A/B/C 三块缓冲模型**结构上就够**：

```
patch off=0x058 0x0c469800 -> 0x095ed400 (buf0)   # gate 的第一次引用
patch off=0x080 0x0c469800 -> 0x095ed400 (buf0)   # gate 的第二次引用（同一个值）
patch off=0x1b8 0x0c46b400 -> 0x0c473800 (buf1)   # up
patch off=0x284 0x0bfe2c00 -> 0x0bff2800 (buf2)   # output
RC=0 patches=4
```

## 2. 这一轮修掉的一个真问题：地址字个数不能靠「几个非 0 槽」推

之前 `expect_patches` = `iova.txt` 里非 0 的槽数（一元 2 / 二元 3）。融合图打破了这条：
**同一个缓冲可以被引用多次**（gate 进来两次），于是第一次跑就报 `patched 4/3`。

现在 `npu_lib` 直接**数命令流**（`_count_patches`）：读 `cmd<patch_idx>.bin`，数有多少个 32 位字等于某个
`iova>>4`。这是设备无关的、确定性的，而且正好等于 runner 会命中的次数。

## 3. 边界（不夸大）

- 这张图有 **2 个运行时输入**，正好落在 A/B 两个槽里；**超过 2 个输入**（例如整块权重全动态）还需要
  把 plan 从 `in_a/in_b` 扩成输入列表 —— 这是融合路线真正缺的一块基础设施。
- 权重：本例没有权重（SwiGLU 只有激活）。带权重的融合图如果权重是**常量**，会落进 `Static`，
  也就撞上 `docs/ISSUE-const-fc.md` 那个未解契约；所以带权重的融合必须走**动态权重**。
- 本例的 4 处地址字里有 2 处同值，说明「按值重定位」在融合图上依然成立（同值同改），但前提是
  **两个引用指向同一块缓冲** —— 若编译器给同一逻辑张量分配了两块缓冲，就得按位置（offset）重定位。

## 4. 复现

```bash
# 抓一次融合图的命令流（2 个输入，一次厂商 runtime 运行）
# 然后 plan: alloc 12288 x3 / bytes 11136 x3 / n_inputs=2 / expect_patches=4（数出来的）
# 板上: npu_run plan.txt
```

## 5. 下一步

1. `npu_lib` 加 `from_tflite(path, inputs=[...], output=...)`：把仓库现成的融合图
   （`build_silu.py` / `build_swiglu_ffn.py` / `build_decoder_layer.py` 的产物）直接当资产吃进来；
2. plan 支持 **N 个输入**（带权重的融合图必然 >2）；
3. 按段融合：RMSNorm 段 / attention 段 / SwiGLU 段各一张图，最后再试整块一张图。
