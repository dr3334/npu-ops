# S2 · FULLY_CONNECTED（Matmul+Linear）逐档编译与板端对拍

时间：2026-11-12 · ncc-tflite 7.3.15 / mdla5.0 / MT6897（Titan 1, MT8792）· 全部 `--disallow-bridge`。

## 1. 结果总表

| op | 形状 (M,K,N) | dtype | 编译 | exec-plan | `.dla` 大小 | sha256(前16) | 板端 | 数值结论 |
|---|---|---|---|---|---|---|---|---|
| fc_1x64x64_int8_exact | 1,64,64 | int8（码 −1/0/1，零 bias，输出 scale=acc scale） | ✅ exit=0 | `FullyConnectedLayer` → `MDLA_5_0` | 7007 | a69657177245bf91 | 0.67 ms/inf | **逐位相等 64/64** |
| fc_1x64x64_int8 | 1,64,64 | int8 全量程 + requant | ✅ | `MDLA_5_0` | 7545 | 8529d9a4fe9460f8 | 0.42 ms/inf | 逐位相等 64/64 |
| fc_1x128x128_int8 | 1,128,128 | int8 | ✅ | `MDLA_5_0` | 20093 | 6996751d7c1399c6 | 0.62 ms/inf | 逐位相等 128/128 |
| fc_8x128x128_int8 | 8,128,128 | int8 | ✅ | `MDLA_5_0` | 20093 | 7a81fc5b0afeda77 | 0.70 ms/inf | max\|Δ\|=1，4/1024（0.39%） |
| fc_8x128x696_int8 | 8,128,696 | int8（FFN up/gate 形状） | ✅ | `MDLA_5_0` | 96093 | adba48a0c279f30d | 0.61 ms/inf | max\|Δ\|=1，7/5568（0.13%） |
| fc_1x128x21704_int8 | 1,128,21704 | int8（LM head 大 N） | ✅ | `MDLA_5_0` | 2869153 | 3daa499f6d6fb486 | 0.68 ms/inf | max\|Δ\|=1，35/21704（0.16%） |
| fc_1x64x64_fp16 | 1,64,64 | fp16 | ✅ | `MDLA_5_0` | 11641 | 9e21b28539b58663 | 0.61 ms/inf | **max\|Δ\|=0（逐位）** |
| fc_8x128x128_fp16 | 8,128,128 | fp16 | ✅ | `MDLA_5_0` | 36477 | 8dd469c3ff4903f8 | 0.51 ms/inf | **max\|Δ\|=0（逐位）** |
| fc_1x64x64_fp32 | 1,64,64 | fp32 | ❌ exit=1 | `MDLA_5_0` 有 FC 层，但 `Bridging is disallowed but gap is detected between Input and MDLA_5_0` | – | – | – | fp32 IO 需要 bridge（CPU 转换），本路径不接受 |

（板端时间为 `neuronrt -c 20` 的均值，含进程内提交/等待开销，仅作量级参考，不是性能结论。）

## 2. 关键发现

1. **int8 FULLY_CONNECTED 能完整落到 MDLA_5_0**，`--disallow-bridge` 通过 → 没有隐式 CPU 回退。这是「可直接用 NPU 组件」的第一块地基。
2. **精确性分两层**：
   - `codes=small`（码值 −1/0/1、零 bias、输出 scale = 累加 scale）时，设备输出与 INT64 golden **逐位相等**（64/64），证明 MAC + int32 累加 + 输出通路本身无误；
   - 全量程 + requant 时出现 **±1 LSB**，比例 0.13%–0.39%，**全部落在 requant 半边界**（实测 acc ≡ 15 (mod 32)，即 acc/32 的小数部分 0.46875 被设备向上取整）。
3. 该 ±1 不出自累加：把同一批 acc 用四种舍入规则（round-half-even / half-up / half-away / 定点移位）分别对拍，最好也只能降到 2/1024，说明 **MTK 的 requant 不是"精确浮点除法 + 标准舍入"**，而是定点乘子+移位链，边界行为需按其实现登记。
   → 处理办法：`model.json` 里登记 `int8_tol`（精确档 0，requant 档 1），`compare.py` 同时打印"逐位是否相等"和"是否满足容差"，**不把 1 LSB 当成 0**。
4. **fp16 路径更好**：两个形状设备输出与「FP32 累加后一次舍入到 fp16」的 golden **完全一致（max\|Δ\| = 0）**，说明 K=64/128 时设备内部就是 fp32 累加。
5. **fp32 输入不可直连**：MDLA 需要 int8/int16/fp16 IO，fp32 会被插入 bridge；在 `--disallow-bridge` 策略下直接失败（这也是真实模型的 I/O 必须显式量化/转 fp16 的原因）。
6. **M>1 可用**：`M=8`（按 tile 内 batch）编译与执行正常，输出按 `[8,N]` 排布，与 TFLite 语义一致（`keep_num_dims=true`）。
7. **大 N 可用**：`N=21704`（LM head 规模）编译通过并正确，`.dla` 2.87 MB（权重已内嵌进 `.dla`）。

## 3. 复现

```bash
cd "/d/deepseek_ workspace/mtk/npu-ops"
MSYS_NO_PATHCONV=1 ./tools/e2e.sh ops/fc_8x128x696_int8     # 编译→推送→NPU 执行→回读→对拍
MSYS_NO_PATHCONV=1 ./tools/e2e.sh ops/fc_1x64x64_int8_exact
cat ops/fc_8x128x696_int8/compare.txt
```

## 4. 下一步（S3 起）

1. **权重可换性（L2→L3）**：同 shape 下换 `B1→B2→B1` 与 `A1→A2→A1`，确认 `.dla` 里权重是否是"内嵌常量"（当前 `.dla` 大小随 N 变化，说明权重已编进去）；若不可换，就按 `docs/05` 如实标 L2，并转向 `NeuronModel_setOperandValue` 的运行时绑定路径。
2. **ioctl 探测**：`strace -f -e trace=ioctl,openat,mmap` + Frida hook，产出 `probe/ioctl-catalog.md`（fd / request / 结构体 / 方向 / 频次），把一次 `.dla` 执行解释到"buffer + 命令 + 完成"。
3. 算子补齐：`ADD/MUL/SOFTMAX/RESHAPE/TRANSPOSE` 按同一模板逐个编（复用 `.tmp/empirical2` 的既有结论，避免重测 `SIN/COS/GATHER/EMBEDDING_LOOKUP`）。
