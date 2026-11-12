# S4 · 权重可换性（L2 → L3）实测

时间：2026-11-12 · 板：Titan 1 / MT6897(MT8792) · ncc-tflite 7.3.15 / mdla5.0。

## 1. 问题

`docs/05 §4 E2`：`.dla` 里的权重是否可换？如果换权重必须重编，就是 **L2（模板依赖）**；如果同 shape 换任意合法 A/B 都能跑，才算 **L3（独立固定形状算子）**。

## 2. 实验设计

两套模型，形状同为 `M=1,K=64,N=64`、int8：

| 模型 | 权重来源 | `.dla` 大小 | 说明 |
|---|---|---|---|
| `fc_1x64x64_int8` | 常量（TFLite constant buffer） | **7545 B** | 权重被编进 `.dla` |
| `fc_dynb_1x64x64_int8` | **运行期输入张量**（subgraph input #1） | **3085 B** | `.dla` 只含代码/布局，不带 4096 B 权重 |

`neuronrt -d` 打印的 IO 表证实：动态档有 **2 个输入**（`<1×1×1×64>` 激活 64 B，`<1×1×64×64>` 权重 4096 B）。

## 3. 结果

| 检查 | 结果 |
|---|---|
| 动态 B 编译（`--disallow-bridge`） | ✅ 通过，落 `MDLA_5_0` |
| W1 运行 vs W1 golden | 1/64 差 1 LSB（requant 边界，等同 S2 契约） |
| W2 运行 vs W2 golden | **逐位相等** |
| 交叉检查：W1 的 golden 对 W2 的运行 | 40/64 不同（最大差 255）→ **权重确实生效** |
| A/B/A 复现（W1 → W2 → W1） | W1 两次**逐位相同** → 无陈旧输出 |
| 新进程 | 每次 `neuronrt` 都是新进程，均通过 |

## 4. 结论

- **FULLY_CONNECTED 达到 L3**：一份 `.dla` + 运行期换权重 + 换输入 + 新进程，输出正确且可复现。
- 代价是权重每帧都要从主机传给设备（4096 B/次），是否值得取决于 batch/M；常量化（权重内嵌）在 decode 场景仍可能更快，**两条路都要保留**：
  - 静态档（内嵌权重）：`.dla` 大、每 token 无权重传输；
  - 动态档（运行期权重）：`.dla` 小、权重可换，但每次要传 B。
- 这也解释了 `docs/05 §7` 的卡点："B 不能换"在本板 **不成立**——至少 TFLite `FULLY_CONNECTED` 的 B 可以作为模型输入。

## 5. 产物

```
ops/fc_dynb_1x64x64_int8/{model.tflite, model.json, model.dla, model.compile.log,
                          golden/{input,weights,weights2,expected,expected2}.bin,
                          out_w1.bin, out_w2.bin, out_w1b.bin}
tools/build_dynb.py
```

## 6. 尚未验证

- 动态 B 在 `M>1`、`N=696/21704` 时是否同样可用（下一步）。
- 动态 B 的传输/打包开销（host→device 拷贝是否走 dma-buf，见 `probe/ioctl-catalog.md` 的 DMA_BUF_IOCTL_SYNC）。
