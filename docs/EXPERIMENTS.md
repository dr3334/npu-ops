# EXPERIMENTS —— 实验记录（按 AGENTS.md §4 格式）

一条实验 = **假设 / 预测 / 做法 / 产物 / 结果 / 适用范围 / 剩余疑问**。
编号一次分配，永不重排；被否定的假设保留并标注。

设备基线（所有实验共用）：Titan 1 / `ro.board.platform=mt6897` / `ro.hardware=mt8792` / Android 16 (SDK 36) /
`ncc-tflite 7.3.15` / `--arch=mdla5.0 --num-mdla=2 --l1-size-kb=3072` / `--disallow-bridge` 恒开。

---

## E-001 工具链与板端可执行性

- **假设**：NeuroPilot 7.0.8 的 `ncc-tflite` 能在本机 WSL 编译，`neuronrt` 能在本板以 root 执行自定义 aarch64 二进制与 `.dla`。
- **预测**：`ncc-tflite --version` 返回 7.3.15；`neuronrt -m hw -a x.dla` 打印 IO 表并成功退出。
- **做法**：WSL 内跑 ncc；adb push `neuronrt` 到 `/data/local/tmp/npuops`；`-d` 打印 IO 映射。
- **产物**：`reports/S0-env.md`、`probe/trace_ioctl.txt`
- **结果**：成立。设备/工具链指纹、`/dev/apusys`、`/vendor/lib64/libneuron_*` 均确认。
- **适用范围**：本机 + 本板 + 该 SDK 版本；换板/换 SDK 需重测。
- **剩余疑问**：`nr=0` 的 ioctl（size 0x28，ENOTTY）用途未知。

---

## E-002 最小 FULLY_CONNECTED 能否落到 MDLA 并算对

- **假设**：`(M,K,N)=(1,64,64)` int8 FC 可在 `--disallow-bridge` 下编译到 `MDLA_5_0`，且板上输出的整数语义与独立 INT64 参考一致。
- **预测**：若累加/舍入语义正确，则用 `codes=small`（码 −1/0/1、零 bias、输出 scale=累加 scale）时设备输出与 INT64 参考**逐位相等**。
- **做法**：自建 TFLite（flatbuffers）→ ncc → push → `neuronrt -m hw` → 回读比对。
- **产物**：`ops/fc_1x64x64_int8_exact/`、`ops/fc_1x64x64_int8/`
- **结果**：成立（64/64 逐位）。`--disallow-bridge` 下 exec plan 只有 `FullyConnectedLayer → MDLA_5_0`。
- **适用范围**：M=1,K=64,N=64,int8；其它形状见 E-008。
- **剩余疑问**：无（此档为定义性实验）。

---

## E-003 全量程 int8 的 ±1 LSB 差异来自哪里

- **假设（竞争）**：
  - H1：设备累加器有误（int32 溢出/顺序）；
  - H2：requant 舍入规则不同（half-even vs half-up vs 定点乘子）；
  - H3：bias 或其 scale 在设备侧被重新量化。
- **预测**：若 H1 成立，差异应随 K 增大而增大且幅度 >1；若 H2 成立，差异只出现在 requant 半边界且恒为 ±1。
- **做法**：对同一批 INT64 acc 用 4 种舍入规则分别对拍；检查失配位置的 `acc mod 32` 分布。
- **产物**：`ops/fc_8x128x128_int8/compare.txt`、`ops/fc_1x128x21704_int8/compare.txt`
- **结果**：**H2 成立**。差异全部为 ±1、比例 0.13%–0.39%，且全部落在 `acc ≡ 15 (mod 32)` 的边界；H1/H3 被否定（small-codes 档逐位相等）。
- **适用范围**：本机 MTK requant 链路；不是"设备算错"。
- **剩余疑问**：MTK 定点乘子的确切位宽/舍入常数未知 → 只能登记 `int8_tol=1`，不能声称可逐位复现。

---

## E-004 fp16 与 fp32 的可行性边界

- **假设**：MDLA 原生支持 fp16 IO；fp32 IO 需要 bridge。
- **预测**：fp16 模型 `--disallow-bridge` 编译通过且数值与"FP32 累加后单次舍入"一致；fp32 模型报 `Bridging is disallowed but gap is detected`。
- **做法**：同形状分别构造 fp16 / fp32 tflite。
- **产物**：`ops/fc_1x64x64_fp16/`、`ops/fc_8x128x128_fp16/`、`rejected/fc_1x64x64_fp32/`
- **结果**：fp16 **max|Δ|=0（逐位）**；fp32 编译失败（bridge）。预测成立。
- **适用范围**：K≤128 的 fp16（fp32 累加）；更大 K 的累加顺序/精度需另测。
- **剩余疑问**：fp16 在 K>512 时是否仍逐位一致（设备累加精度）。

---

## E-005 权重是否可换（L2 → L3）

- **假设**：把权重改成模型第二输入后，一份 `.dla` 可服务不同权重，无需重编。
- **预测**：动态档 `.dla` 明显变小（不含权重）；W1/W2 各自输出正确；A→B→A 复现；新进程可跑。
- **做法**：`tools/build_dynb.py` 构造 2 输入模型，`neuronrt -i A -i W`。
- **产物**：`ops/fc_dynb_1x64x64_int8/`、`reports/S4-weight-swap.md`
- **结果**：成立。`.dla` 7545→3085 B；W2 逐位正确；A/B/A 逐位复现；交叉检查 40/64 不同证明权重生效。
- **适用范围**：M=1,K=64,N=64,int8。M>1 / 大 N 的动态 B 未测。
- **剩余疑问**：动态 B 的 host→device 传输成本；大 N 时是否仍可作输入。

---

## E-006 一次 `.dla` 执行的 ioctl 序列（含 PRE/POST 教训）

- **假设**：执行路径 = HANDSHAKE → MEM(alloc/map) → CMD(RUN/DEL)，参数布局与公开 6.1 `mdw_ioctl.h` 一致。
- **预测**：若一致，`op` 字段应落在 0..3（MEM）与 0..3（CMD），且 `priority` 能与内核 trace 对上。
- **做法**：strace（顺序/fd）+ LD_PRELOAD shim（payload）+ ftrace（内核侧）。
- **产物**：`probe/ioctl-catalog.md`、`probe/io.log`、`probe/trace_kernel.txt`、`probe/apusys.ko`
- **结果**：成立。11×HANDSHAKE / 13×MEM / 2×CMD / 7×DMA_BUF_SYNC；内核侧 1 cmd + 1 subcmd + 1 RV IPI，`executed_core_bmp=0x1`。
- **否定记录**：**第一版 shim 只在调用后 dump**，把驱动的 out 字段读成 `op=6..11`，据此得出的"ABI 已扩展"是**错误结论**；反汇编 `mdw_mem_ioctl` 见 `stp xzr,xzr,[x21]`（驱动清零 in-struct）后改为 PRE/POST 双打纠正。
- **适用范围**：本板该 BSP；MEM `flags` 位定义仍未解。
- **剩余疑问**：`nr=0`；CMD 里 `uid` 类"指针型 token"语义；MEM flags 位图。

---

## E-007 输入/输出 dma-buf 的字节级确认

- **假设**：MAP 后的 handle 就是 dma-buf fd，输入/输出可直接在主机侧快照。
- **预测**：命令提交前输入 buffer 前 64 B == 参考输入；推理完成、UNMAP 前输出 buffer 前 64 B == 板端输出。
- **做法**：`probe/iolog2.c`（mmap 跟踪 + 信号保护 memcpy；触发点 CMD_RUN 前 / MEM_UNMAP 前）。
- **产物**：`probe/buf/`、`reports/S5-cmdbuf-capture.md`
- **结果**：成立，两侧均**逐字节相等**。buffer 角色：输入 4 KiB(fd=3)、输出 4 KiB(fd=0xb)、命令 400 B(fd=6)、参数 8 KiB(fd=7)、小结构 76 B(fd=0xa)。
- **适用范围**：本模型；其它算子需重新确认哪块是命令区。
- **剩余疑问**：7545 B 的 map09（尺寸等于 `.dla`，内容不同）身份未确认；命令区里 0x025c3xxx 地址与 MAP 返回的 0x25c5f000 不同源。

---

## E-008 命令区里的形状字段（三形状差分）

- **假设**：同一算子族共享一份命令模板，形状只体现在少数几个字里。
- **预测**：若成立，命令区长度不随 K/N 变；差分后改动字数远小于 100，且能分成 K 依赖 / N 依赖两类。
- **做法**：64×64、64×128、128×128 三个形状各抓 400 B 命令区，逐字 diff（`probe/diff_cmd.py`）。
- **产物**：`reports/S6-cmd-fields.md`、`probe/buf/map10_*.bin`
- **结果**：成立。**25/100 字随形状变化**（12 N 依赖、10 K 依赖、2 地址、1 步长），其余 75 字与形状无关；`w82 = K/16`；形状以裸整数出现（0x40/0x80）。
- **适用范围**：M=1、int8、FC、K/N ∈ {64,128}。M>1 未测。
- **剩余疑问**：字段的寄存器名/位域未知（需 exec plan 对照）；0x025c3xxx 地址来源未知。
