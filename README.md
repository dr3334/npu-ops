# npu-ops —— 单算子 → `.dla` → 板端实跑 → ioctl 探测

目标（对应 openmtk `docs/05`、`docs/07`）：把每个算子单独用 Neuron 工具链编成 `.dla`，在本板 **MT6897（Titan 1 / MT8792）** 上真实执行并与 CPU golden 对拍，最终固化成"换输入、换权重、新进程都能用"的算子组件；执行路径用 ioctl 层探测解释清楚，而不是只知道"命令返回 0"。

```
tools/build_ops.py    单算子 TFLite 生成器（当前：FULLY_CONNECTED；int8 / fp16 / fp32）
tools/compile.sh      ncc-tflite 封装：mdla5.0 + --disallow-bridge + 记录 exec-plan/sha256
tools/e2e.sh          build → compile → push → NPU 执行 → 回读 → 对拍（一条命令）
tools/compare.py      对拍：整数档逐位+容差、浮点档 max_abs/RMSE/relL2/NaN/范围
ops/<name>/           model.tflite · model.json · model.dla · model.compile.log
                      golden/{input,expected}.bin · out_device.bin · compare.txt · e2e.json
reports/              S0 环境基线、S2 FULLY_CONNECTED 结果、timings.txt
probe/                ioctl 探测：iolog.c(PRE/POST shim) · parse_io.py · ioctl-catalog.md
                      apusys.ko + mem_ioctl.dis(反汇编) · trace_ioctl/io.log/trace_kernel
```

## 快速开始

设备已连（`adb devices` 能看到 `0123456789ABCDEF`），SDK 在工作区同级 `neuropilot-sdk-basic-7.0.8-build20240807/`：

```bash
cd "/d/deepseek_ workspace/mtk/npu-ops"
MSYS_NO_PATHCONV=1 ./tools/e2e.sh ops/fc_1x64x64_int8          # 最小档
MSYS_NO_PATHCONV=1 ./tools/e2e.sh ops/fc_8x128x696_int8        # FFN 形状
MSYS_NO_PATHCONV=1 ./tools/e2e.sh ops/fc_1x128x21704_int8      # LM head 大 N
cat ops/fc_8x128x696_int8/compare.txt
```

首次跑会自动 `build_ops.py` 生成 tflite、`compile.sh` 编译、推送 `neuronrt`（已推过的跳过）、执行、回读、对拍。

## 当前状态（2026-11-12）

**编译 + 板端执行 + 对拍：9/9 通过**（`tools/run_all.sh`）

| 形状 (M,K,N) | dtype | 编译 | 板端执行 | 数值 |
|---|---|---|---|---|
| 1,64,64 | int8（small codes） | ✅ MDLA_5_0 无 bridge | 0.67 ms/inf | 逐位相等 |
| 1,64,64 / 1,128,128 | int8 全量程 | ✅ | 0.42 / 0.62 ms | 逐位相等 |
| 8,128,128 / 8,128,696 / 1,128,21704 | int8 全量程 | ✅ | 0.5–0.7 ms | max\|Δ\|=1（≤0.4%，requant 边界） |
| 1,64,64 / 8,128,128 | fp16 | ✅ | 0.5–0.6 ms | **max\|Δ\|=0（逐位）** |
| **1,64,64 动态权重**（`fc_dynb_*`） | int8 | ✅ `.dla` 仅 3085 B | 0.45–1.2 ms | **换 W1/W2 均正确、A/B/A 逐位复现 → L3** |
| 1,64,64 | fp32 | ❌ | – | fp32 IO 需要 bridge，拒绝（样本在 `rejected/fc_1x64x64_fp32/`）|

**ioctl 探测**（`probe/ioctl-catalog.md`）：一次 `.dla` 执行 = 1×未知探测 + 11×HANDSHAKE + 13×MEM + 2×CMD + 7×DMA_BUF_SYNC。
MEM 的 13 次调用**全部解码**（ALLOC/MAP/UNMAP + handle + IOVA 0x25cxxxxx）；HANDSHAKE 给出板级拓扑（dev_bitmask=0xd6、VLM 6 MiB）；
CMD 的 `RUN`/`DEL` 与 6.1 头布局一致，`priority=10` 与内核 trace 交叉验证。内核侧 1 cmd / 1 subcmd / 1 RV IPI，`executed_core_bmp=0x1`。
> 注意：早期"op 6–11 / ABI 已扩展"是**驱动清零 in-struct 后的 POST 误读**，已用 PRE/POST 双打修正。

详见 `reports/S0-env.md`、`reports/S2-fully_connected.md`、`reports/S4-weight-swap.md`、`probe/ioctl-catalog.md`。

## 契约（不写在代码注释里就会丢的规则）

1. **`--disallow-bridge` 永远开**：出现 bridge 就等于该算子没落到 NPU，编译判失败。
2. **整数档分两级**：`int8_tol=0` 表示要求逐位相等（`codes=small` 的累加证明档）；`int8_tol=1` 表示允许 requant 边界的 ±1 LSB，`compare.py` 会同时打印"实际是否逐位相等"和"偏离比例"，**不允许把 1 LSB 静默吃掉**。
3. **浮点档先登记容差再测**：fp16 默认 `atol=rtol=2e-2`（当前实测为 0，但阈值不因此收紧/放宽而不记录）。
4. **`.dla` 与其编译参数、sha256 一起归档**：`model.compile.log` 里记录完整命令行、ncc 版本、`EXIT`、`SIZE`、`SHA256`。
5. **两种权重形态都要保留**：常量档（权重编进 `.dla`，每 token 无权重传输）与动态档（权重作模型输入，可换但每次要传 B）。S4 已证明动态档成立（L3），不要只留一条路。
6. **探测必须 PRE/POST 双打**：驱动会把 in-struct 清零，只看调用后的 buffer 会把 out 字段当成调用者参数（本项目已踩过，见 `probe/ioctl-catalog.md §0`）。

## 已知限制

- `neuronrt` 依赖设备 `/vendor/lib64` 的 neuron runtime；固定版本对照实验时需要把我们 push 的 `mt6897/lib` 放到 `LD_LIBRARY_PATH` 前面。
- 板端耗时来自 `neuronrt -c N`，含进程/提交开销，不是定频性能结论。
- 只覆盖 FULLY_CONNECTED；`SIN/COS/EMBEDDING_LOOKUP/GATHER` 的既有否定结论见 `.tmp/findings/ops2_empirical.md`，不重复测。

## 缓冲地图（S5，见 `reports/S5-cmdbuf-capture.md`）

一次 `fc_1x64x64_int8` 推理实际用到的 dma-buf：

| 大小 | fd(=MEM handle) | IOVA | 角色 | 证据 |
|---|---|---|---|---|
| 4096 | 3 | 0x25c5f000 | 输入激活 | 前 64 B == `golden/input.bin` |
| 4096 | 0xb | 0x25c7b000 | **输出** | 推理后前 64 B == 板端输出 |
| 400 | 6 | — | 命令/寄存器描述区（静态） | 前后 0 字节变化 |
| 8192 | 7 | 0x25c78000 | 网络/参数区（静态） | 前后 0 字节变化 |
| 76 | 0xa | — | 小结构（未解码） | — |

工具：`probe/iolog2.c`（PRE/POST + mmap 跟踪 + 带信号保护的 dma-buf 快照）。
