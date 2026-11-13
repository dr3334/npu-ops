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

---

## E-101 基向量判布局（动态权重，K 方向是否按行主序）

- **假设（竞争，跑之前写定）**：
  - **H1** 行主序直接读 K：`out[r,:] == j_r`，对任意 j 成立；
  - **H2** K 被分块/置换：只有落在第一个 tile 内的 j 正确，跨 tile 的 j 映射到别的下标；
  - **H3** 权重被 repack 且偏移错：`out[r,:] == j_r + c`（对所有 j 同一偏移），或整行取到别的列。
- **做法**：动态权重模型（E-005），构造
  - `W[n,k] = k`（权重行自带 K 下标），`A[r] = e_{j_r}`（单位基向量），
  - 输出 scale = 输入×权重 scale（**倍率 1**，零 bias）→ 输出就是原始整数累加值，不经过 requant。
  - 一次运行 8 行不同 j；`tools/e101_basis.py` 生成、`e2e.sh` 执行、`compare.py` 逐位比对。
- **产物**：`ops/e101_basis_dynb/`（K=64，J=[0,1,15,16,31,32,47,63]）、`ops/e101b_basis_k128/`（K=128，J=[16,32,63,64,95,127,0,1]）
- **结果**：**H1 成立，H2/H3 被否定**。
  - K=64：512/512 逐位相等；
  - K=128（j 跨越 64 边界，到 127）：512/512 逐位相等。
  - 即：**K 方向按行主序读取，没有 tile 置换、没有 repack 偏移**，跨 64 边界也成立。
- **适用范围**：M=8、N=64、K∈{64,128}、int8、动态权重、零 bias、倍率 1 的 requant。更大的 K / N、M>8、常权重路径未测。
- **剩余疑问**：
  1. K>128（例如 696/1024）是否仍无置换；
  2. 常权重（编译期打包）路径是否与动态权重一致——若不一致，说明打包环节引入了重排；
  3. 该实验只证明"读的列下标正确"，尚未覆盖输出侧 N 方向的写回布局（需转置型探针）。

---

## E-102 权重绑定 + 地址重定位（命令区会随地址改写）

- **假设（跑前写定）**：
  - **H1 内容绑定**：`out[:, n]` 只由权重第 n 行决定；`W[n0,k0]=1`、A 全 1 ⇒ 只有第 `n0` 列等于 1，其余为 0，且换 `(n0,k0)` 只换列。
  - **H2 地址泄漏**：同一模型在不同 IOVA 下输出会错位/变化（重定位未做）。
- **做法**：
  1. 两个绑定探针：`W[5,3]=1` 与 `W[40,17]=1`（倍率 1、零 bias，输出即原始累加值）；
  2. 地址扰动：自己写 `probe/npu_hold`（**首次由我们自己的程序**对 `/dev/apusys` 做 HANDSHAKE + MEM ALLOC），先占 96 MiB 再跑同一模型，逼驱动换 IOVA；用 `probe/iova.py` 对比 MAP 返回值，用 `probe/iolog2.c` 快照命令区。
- **产物**：`ops/e102_col5/`、`ops/e102_col40/`、`probe/ioE102A.log`、`probe/ioE102C.log`、`probe/buf/e102_cmd_run{A,C}.bin`
- **结果**：
  - H1 成立：两个探针各 **512/512 逐位相等**（只有目标列亮，其余全 0）。
  - H2 **被否定**：96 MiB 占用后 MAP 的 IOVA 从 `0x25b93000/0x25b91000` 变为 `0x24fff000/0x24ffe000`，两次输出**逐字节相同**。
  - 附带强证据：命令区（368 B）在两种地址下**只差 5 个字节**，差异落在 3 处地址字上：
    | 偏移 | 无扰动 | 有扰动 |
    |---|---|---|
    | 0x07c | `0x025b8f00` | `0x024fff00` |
    | 0x09c | `0x025b9000` | `0x025b9400` |
    | 0x10c | `0x025b8e00` | `0x024ffe00` |
    ⇒ **命令区由 runtime 在每次提交前重定位**，这正是"同 shape 换资源仍正确"的机制。
- **附带成果**：`probe/npu_hold` 成功握手（`version=0x4 dev_bitmask=0xd6 mem_bitmask=0x36`）并拿到 `handle=0x4` —— 这是自主提交器 `apusctl` 的第一块砖。
- **适用范围**：M=8, K=64, N=64, int8, 动态权重；扰动为 96 MiB MAIN 分配。
- **剩余疑问**：
  1. 命令区里 `0x025b8f00/0x025b9000/0x025b8e00` 与 MAP 返回的 VA（`0x25b94000/0x25b93000/0x25b91000`）**并不同值**，说明它们指向运行时的其它子区（参数区/命令区自身），尚未与 buffer 角色对上；
  2. 本模型命令区 **368 B**，而 fc 系列是 400 B —— 哪个字段/配置造成 32 B 差异未知；
  3. 只测了"有/无 96 MiB 占用"，没有系统性扫描地址偏移量与对齐。

---

## E-106A 我们自己的 apusys 客户端：内存流水线（CMD 提交前的全部前置）

- **假设**：`HANDSHAKE → MEM ALLOC → MEM MAP → MEM UNMAP → MEM FREE` 这条内存流水线可以完全由我们自己的进程完成，不经过 `libneuron`。
- **预测**：四条 ioctl 全部返回 0；MAP 后拿到 IOVA；同时持有 4 个 buffer 时 IOVA 互不相同。
- **做法**：`probe/apusctl.c`（NDK 静态编译，aarch64），按 E-006 解出的 40/48 字节参数结构，用**字节级 put/get** 写字段，避免编译器对齐假设。
- **产物**：`probe/apusctl.c`、`probe/npu_hold.c`
- **结果**：成立。
  ```
  handshake: version=0x4 dev_bitmask=0xd6 mem_bitmask=0x36 meta_size=32
  buffer cmd    size=400   handle=0x4 iova=0x25b6d000
  buffer param  size=8192  handle=0x5 iova=0x25d14000
  buffer input  size=4096  handle=0x6 iova=0x25b6c000
  buffer output size=4096  handle=0x7 iova=0x25b82000
  ```
  早期版本曾因结构体多出 8 字节（56 B vs 48 B）报 `Bad address` —— 说明驱动会校验 `_IOC` 尺寸，**结构体布局必须与 ABI 精确一致**（已用 `_Static_assert` 固定 40/48）。
- **适用范围**：MT6897 本板；MAIN 内存类型、4096 对齐。
- **未完成**：`CMD RUN`。还缺 `mdw_cmd_in` / `mdw_subcmd_info` 的字段语义与 cmdbuf 内容重定位规则；
  按 AGENTS.md R10 **不在这里猜**，下一步用"在已抓到的合法提交上做单字段变异"（E-105B）来确定，而不是凭名字硬填。

---

## E-105B 在合法提交上做单字段变异（不猜偏移，直接看后果）

- **假设**：`mdw_cmd_in` 的字段可以靠"改一个 32-bit 字 → 观察三态"来归类，无需反汇编。
- **预测**：
  - 被驱动/运行时校验的字段 → 改后 `rc≠0` 且报错；
  - 只是 QoS 提示的字段 → 改后 `rc=0`、输出不变；
  - 参与寻址的字段 → 改后输出改变或崩溃。
- **做法**：给 `probe/iolog2.c` 加**补丁文件**机制（env 被 neuronrt 清空，故用 `/data/local/tmp/npuops/probe/patch.txt`，
  格式 `0xOFFSET 0xVALUE`），在 `CMD RUN` 真正下发前改写一个 4 字节字并记日志。基线：`md5=d7eb1d19…`，`rc=0`。
- **结果矩阵**（模型 `e102.dla`，M=8,K=64,N=64）：

  | 偏移 | 原始值 | 改成 | rc | 输出 |
  |---|---|---|---|---|
  | 0x20 | 0xa (10) | 0x14 (20) | 0 | 不变 |
  | 0x38 | 0x7d0 (2000) | 0x64 (100) | 0 | 不变 |
  | 0x3c | 0 | 1 | 0 | 不变 |
  | **0x24** | 0 | 1 | **1** | —（**被校验**） |
  | 0x28 | 0 | 1 | 0 | 不变 |
  | 0x40 | 0 | 1 | 0 | 不变 |
  | 0x44 | 0 | 1 | 0 | 不变 |
  | 0x60 | 0xffffffff (fence=-1) | 0 | 0 | 不变 |
  | **0x48** | 1 | 0 | **1** | `runtime error during inference`（**被校验**） |

- **结论**：
  1. **0x48 = num_subcmds**（由变异确认，不靠名字）：改成 0 直接运行期失败；
  2. **0x24 = hardlimit**：非零小值被拒，说明该字段有合法性约束（0 = 不限制）；
  3. `priority(0x20)`、`power_dtime/power_etime(0x38/0x3c)`、`softlimit(0x28)`、`app_type(0x40)`、`flags(0x44)`、`fence(0x60)` 在本路径下**不被校验、也不影响输出** —— 与 6.1 头文件的语义一致；
  4. 没有任何一次变异改变数值输出 ⇒ 这些字段都**不参与寻址**。
- **产物**：`probe/iolog2.c`（补丁机制）、`probe/io2.*.log` 中的 `PATCH` 行、本节表格。
- **适用范围**：M=8,K=64,N=64,int8,常权重路径（`e102.dla`）；只改了 4 字节对齐的字。
- **剩余疑问 / 下一步**：
  1. 指针字段（`subcmd_infos@0x50`、`adj_matrix@0x58`）尚未做变异 —— 计划改为"置 0 看是否 EFAULT"这种安全探测；
  2. `subcmd_info` 数组内部（`num_cmdbufs@+0x40`、`cmdbufs@+0x48`）需要同样手法确认；
  3. `0x68=9` 对应哪个字段未定（6.1 头里该偏移是 `exec_infos` 指针，但这里是 9，说明本 BSP 可能有重排）—— 标 UNKNOWN。

---

## E-105C 指针字段探测（0x50 / 0x58 / 0x68）

- **假设**：`subcmd_infos@0x50`、`adj_matrix@0x58` 是被解引用的指针；`0x68=9` 是标量。
- **预测**：置 0 或写入非法指针 → `rc≠0`；写回原值 → `rc=0`。
- **做法**：同 E-105B 的补丁机制，只改一个 4/8 字节字；对照组用"写回原值"。
- **结果**：

  | 偏移 | 改动 | rc | 判定 |
  |---|---|---|---|
  | 0x50 | 0 | 1 | **被解引用**（null 不可） |
  | 0x50 | 0xffffffffb4000010（非法指针） | 1 | 被解引用 |
  | 0x58 | 0 | 1 | **被解引用**（null 不可） |
  | 0x58 | 0xffffffffb4000007（非法指针） | 1 | 被解引用 |
  | 0x68 | 0 | 1 | **被校验** |
  | 0x68 | 9（原值） | 0 | 对照通过 |
  | 0x70 | 0（原值） | 0 | 对照通过 |
  | 0x10 | 1 | 0 | `id` 是 in/out 字段，可改 |

- **结论**：`0x50`、`0x58` 必须填**有效指针**；`0x68` 必须是合法非零标量；`0x70` 与 `id` 不敏感。
  与 6.1 头一致（`subcmd_infos@80=0x50`、`adj_matrix@88=0x58`、`fence@96=0x60`）。
- **剩余疑问**：`0x68=9` 的确切语义（6.1 头该处是 `exec_infos` 指针，但值是 9）→ 标 UNKNOWN。
- **对 W4 的意义**：要自己构造 `CMD RUN`，至少需要 **subcmd_infos 数组 + adj_matrix + 有效 0x68**，
  这三样都还没有构造器 → W4 的"完全自主提交"仍是待办，见 `PLAN.md`。

---

## E-103 形状能力边界（编译期 + 板端抽样）

- **假设**：存在可预测的合法性边界，且失败集中在对齐约束上。
- **预测**：若 K 需要 16 对齐，则 K∈{8,24,40,56,72} 编译失败，K∈{16,32,48,64,80,112,128…} 通过；N/M 无此约束。
- **做法**：`tools/scan_shapes.sh` 扫 31 个形状（K/N 各 12 档、M 7 档）→ `docs/CAPABILITY.tsv`；再补扫 12 个 K 值确认；最后抽 3 个形状做板端对拍。
- **结果**：**预测成立**。
  - K 必须是 **16 的倍数**（非 16 倍数在编译期报 `Bridging is disallowed but gap is detected`）；
  - N 从 8 到 512 全部编译通过；M 2/7/8/31/32/127/128 全部通过；
  - 板端抽样：`1,16,64` **逐位相等**；`1,64,8` **逐位相等**；`128,128,128` max|Δ|=1、24/16384（0.15%）只差 requant 边界。
- **产物**：`docs/CAPABILITY.md`、`docs/CAPABILITY.tsv`、`tools/scan_shapes.sh`、`ops/fc_1x16x64_int8/`、`ops/fc_1x64x8_int8/`、`ops/fc_128x128x128_int8/`
- **适用范围**：int8、FULLY_CONNECTED、单轴扫描（未扫组合）。
- **剩余疑问**：M 与 K 的组合是否引入新边界；K>512 未测。

---

## E-104 把"设备健康"变成回归门禁（而不是口头纪律）

- **假设**：只要套件前后都跑一次"已知正常程序"，就能把"自己算对但污染后续"的用例挡在门外。
- **预测**：正常套件下 pre/post 都 PASS；若某实验污染设备，post 会 FAIL 并使整轮退出码非 0。
- **做法**：`tools/run_all.sh` 增加 `[pre]`/`[post] healthcheck` 两处门禁（`HEALTHCHECK=0` 可跳过），
  门禁内部复用 `healthcheck.sh --quick`（设备存在性 + `/dev/apusys` + neuronrt + 一次已知正常算子）。
- **结果**：18 个算子全 PASS 且 `[post] healthcheck PASS`；两者任一失败会以 exit 20/21 终止整轮。
- **适用范围**：所有板端批量回归。
- **剩余疑问**：healthcheck 只做"能跑通且一致"，没有做 dmesg 严格断言（避免把历史告警当污染）。

---

## E-107 W5 的第一个"第二类算子"：ADD（同一条流水线复用）

- **假设**：ADD 可以完全复用 FC 的工具链（同一个 build/compile/e2e/compare），不需要新脚手架；
  只要 e2e 支持"两个常规运行时输入"即可。
- **预测**：`add_1x64_int8` 编译落到 `MDLA_5_0`、板端输出等于 `clip(a+b)` 逐位相等。
- **做法**：`tools/build_add.py` 用同一套 flatbuffers 助手生成 ADD（2 输入、int8、零 zp、同 scale）；
  `tools/e2e.sh` 扩展为"若 `golden/input1.bin` 存在则作为第二个 `-i`"。
- **产物**：`ops/add_1x64_int8/`（`.dla` 3071 B）、`tools/build_add.py`
- **结果**：成立 —— 板端 **64/64 逐位相等**，1.70 ms/inf（含进程开销）。
- **适用范围**：M=1,K=64,int8；未测 broadcast、未测 M>1。
- **剩余疑问**：broadcast（N≠M 形状）是否支持；int8 溢出语义（clip 还是 wrap）只在 clip 这一档验证过。

---

## E-106B（进行中）ADD 的完整提交结构已抓全，但"自建执行器"还差两块

- **目标**：用我们自己的函数库对一块自分配内存发一次 ADD，得到 `C = A + B`，不经过 `neuronrt`/`libneuron`。
- **本轮已完成（抓取面）**：把 shim 的两个盲点补掉后，**一次 ADD 提交的所有结构都拿到了**：
  - `mdw_cmd_in` 152 B：`op=0`、`priority=10`、`num_subcmds=1`、`subcmd_infos@0x50`、`adj_matrix@0x58`、`fence=-1`、`0x68=9`；
  - `mdw_subcmd_info` 80 B：`type@0=2`、`num_cmdbufs@64=3`、`cmdbufs@72` → 指针；
  - `mdw_subcmd_cmdbuf[3]`（每个 24 B）：`{handle=7,size=0x20,align=0x100,dir=1}`、`{8,0x10,0x100,1}`、`{2,0x170,0x100,1}`；
  - **三块 cmdbuf 的实际内容**：368 B 的是真正的命令流（含 `0xa5xxxxxx` 型命令字与 `0x0259…` 设备地址），32 B 与 16 B 两块是**全零占位**。
- **关键修正**：此前 mmap 捕获阈值是 `len>=64`，导致 32 B / 16 B 的命令缓冲被漏掉；改成 `len>=8` 后三块全部落盘。
- **仍然缺（未完成，不宣称实现）**：
  1. **地址翻译**：368 B 命令流里的地址字要映射到"我们自己的 buffer 的 IOVA"。已知提交前 runtime 会重写其中 3 处（E-102），但还需要把每个地址字与 A/B/C/命令缓冲的**新区**一一对上（旧 `handle→IOVA` 表在每次运行都不同，必须同一次抓取里配套记录）；
  2. **`adj_matrix`（0x58）**：驱动会解引用，但内容格式未定（置 0 会被拒，非法指针会被拒）；需要先做"指向全零矩阵"的实验；
  3. **完成等待**：`CMD RUN` 之后如何判断执行结束（fence 是 -1，可能要 poll 输出 buffer 或走 MEM SYNC）。
- **产物**：`probe/ioAdd*.log`、`probe/buf/add_map*.bin`（三块 cmdbuf + A/B/C）、`probe/iolog2.c`（`dump_subcmds` + 8 字节阈值）。
- **下一步（按成本排序）**：① 同一次抓取里把 `handle→IOVA` 表与命令流一起落盘 → 建地址翻译表；② `adj_matrix` 用全零 buffer 试提交（干净失败也被接受）；③ 若 ② 通过，先跑"数据不变、只换地址"的复现，再换输入值。

---

## E-106B 结果（三小步：-1 完成、-2 完成、-3 卡在 EINVAL）

### B-1 ✅ 完整抓取（含 handle→IOVA 表）
一次 ADD 运行里同时拿到：`cmd_in`(152 B)、`subcmd_info`(80 B)、`cmdbuf[3]` 描述符、
三块 cmdbuf 内容（368 B 真有内容 / 32 B / 16 B 全零）、以及 `handle→IOVA`：
`4→0x25969000(A)`、`10→0x25968000(B)`、`11→0x2596c000(C)`。

**关键发现**：命令流里的地址是 **`iova >> 4`**（0x25969000 → 字面 `0x02596900`），
这就是第一次打补丁"0 命中"的原因；按 `>>4` 后三处地址字全部命中（0x054=B、0x074=A、0x118=C）。

### B-2 ✅ `adj_matrix` 可以是全零（用原始 runtime 做替换验证）
在**合法提交**上把 `cmd_in@0x58` 换成我们自己的 256 B 全零缓冲（shim `adjzero` 模式）：
`rc=0`，输出 `md5=6360b7b3d32c94a46585a3f6a54f50ff`，**与 golden 完全一致**。
⇒ 该字段对 `num_subcmds=1` 不要求特定内容，全零可用。这是自建提交少掉的一个未知。

### B-3 ❌ 我们自己的提交仍被拒：`CMD RUN -> EINVAL(22)`
`probe/apusctl_run.c` 已经能：自己握手、自己 ALLOC/mmap/MAP、写内容、按 `>>4` 打三处地址补丁、
组装 `cmdbuf[3]`/`subcmd_info`/`cmd_in` 并调用 `ioctl(fd, 'A',34, &cmd)`。
已排除：`uid`（0 / 1 / 0x1000 / 0x10000 / 指针值 全 EINVAL）、地址补丁（命中 3 处）、
`adj_matrix`（全零已证明可用）。
**尚未试**（下一步）：① 复制 runtime 的完整握手序列（BASIC×2 + DEV×5 + MEM×4，我们只做了 1 次 BASIC）；
② cmdbuf 是否需要 MEM MAP 后才可作为命令源；③ `subcmd_info` 其余字段（boost/affinity 等）是否需要与 device type 匹配。

---

## ✅ E-106B-3 成功：我们自己的进程让 NPU 完成了加法

**结果（最终）**

```
addin/out.bin md5 = 4cc17bf2a68a4bd5cf16e516d2450c06
期望      exp2.bin md5 = 4cc17bf2a68a4bd5cf16e516d2450c06      ← 逐字节相同
A=[10,20,30,40,50,60,70,80]×8   B=[1..8]×8   →   C=[11,22,33,44,55,66,77,88]×8
两次运行输出逐字节相同；healthcheck PASS
```

路径：`probe/apusctl_run.c`（NDK 静态 aarch64）——**全程自己的 ioctl，没有任何 `libneuron`/`neuronrt` 调用**：
`HANDSHAKE(BASIC×2 + DEV×5 + MEM×4) → ALLOC → mmap → MEM MAP → 写内容 → 复制命令流并重定位地址
→ 组装 cmdbuf[3]/subcmd_info/cmd_in → CMD RUN → 等待 + DMA_BUF_SYNC → 读回`。

**最后三个卡点是怎么解的（都有驱动日志作证）**

1. **`exec_infos`（`cmd_in@0x68`）是 MEM handle，不是用户指针**
   先错传用户指针 → 内核打印
   `mdw_mem_get get dma_buf handle(4963376) fail` / `mdw_cmd_create get exec info fail`。
   改成"分配一块 76 B 缓冲、把它的 **handle** 填进 0x68"后立刻 `rc=0`。
   （这也解释了抓取里 `0x68=9`——那正是 runtime 的 76 B 缓冲 handle。）
2. **命令流里的地址是 `iova >> 4`**（`0x25969000` 存成 `0x02596900`），三处地址字：`0x054`=B、`0x074`=A、`0x118`=C。
3. **完成后必须等 + 同步缓存**：`rc=0` 只代表提交成功；加 `usleep(200 ms)` 再对输出缓冲做
   `DMA_BUF_IOCTL_SYNC(START|READ)` / `(END|READ)`，读回才是正确数据（否则读到预填的 0xAA）。

**另外两条已经验证的结论**
- `adj_matrix`（0x58）可以指向一块**全零**缓冲（E-106B-2，用原始 runtime 替换验证，输出与 golden 一致）；
- cmdbuf 描述符的**顺序有语义**：必须 `[0]=32 B(dir1) [1]=16 B(dir2) [2]=368 B(dir1)`；
- 驱动不支持 `MEM op=1 (FREE)`（内核稳定打印 `not support memory op(1)`），所以收尾不做 FREE。

**等级变化**：行为与独立性同时达成 → **L3.5（自有提交器 + 模板命令）**。仍不是 L4：
命令流仍是 ncc 编译产物，我们只做重定位，没有自己生成 MDLA 命令字。

**封装成函数库**：`probe/npuadd.h` + `probe/npuadd.c` 提供
`int npu_add64(const int8_t *a, const int8_t *b, int8_t *out);`
（固定 64 元素，内部完成握手/分配/重定位/提交/等待/同步，不依赖厂商 runtime）。
`probe/npuadd_demo.c` 实测：

```
$ ./probe/npuadd_demo
handshake sequence: 2 basic + 5 dev + 4 mem done
patch off=0x054 ... / 0x074 ... / 0x118 ...
npu_add64 rc=0
C[0..7] = 11 22 33 44 55 66 77 88        ← A=[10..80]×8 + B=[1..8]×8
```

---

# W6 并行批次索引（2026-11-12）

四个 worktree 分支合并进 master，各自的分内实验保留在自己的文档里；这里是编号映射与一句话结论。

| 分支 | 文档 | 本地编号 | 一句话结论 |
|---|---|---|---|
| `feat/shapeops` | `docs/OP-shapeops.md` | E-SHP-01…06 | RESHAPE/TRANSPOSE/CONCAT/SPLIT/PAD 全部落 MDLA 且逐位相等；失败只发生在 DRAM 边界行宽 <16 B |
| `feat/actreduce` | `docs/OP-actreduce.md` | E-ACT-01…09 | RELU/LOGISTIC/TANH/MEAN/SUM 逐位；**SOFTMAX 是真实 NPU 算子（fp16+int8，beta 生效）** |
| `feat/elem` | `docs/OP-elementwise.md` | E-ELEM-01…05 | MUL/SUB/MAX/MIN 逐位（4352 codes 零误差）；`K==4` 例外机制 UNKNOWN |
| `feat/convpool` | `docs/OP-convpool.md` | CP-01…05 | CONV/DW/POOL 逐位；**反推出真实 int8 requant 规则并用未见乘子验证**；C=8 失败机制 UNKNOWN |

合并后回归：`PASS=75 XFAIL=16 FAIL=0`（`tools/run_all.sh`，前后 healthcheck 均 PASS）。
汇总表见 `docs/OP-MATRIX.md`。

## 由本批次修正的既有结论

1. **E-003 的 ±1 LSB 得到解释**：不是"MTK 的 requant 无法复现"，而是 gemmlowp 的**两次舍入**
   （`round_half_up(acc·M0/2^31)` 再 `RoundingDivideByPOT(shift)`）。按该规则写 golden 即可逐位。
2. **`.dla` 不可字节复现**（同 tflite 三次编译、大小相同、sha256 三次不同）；回归只比数值。
3. **`--disallow-bridge` 的语义边界**被推到台面：convpool 发现 C=4/8 的失败其实由 **eDPA relayout**
   （MDLA 自己的 DMA，仍是 NPU）填充，被我们的严格策略拒绝。是否允许它，是一个**政策选择**，待定。
---

## qwen25-attention 批次（2026-11-13，spec: `.scratch/qwen25-attention/spec.md`）

背景：仓库已有 178 个单算子目录与一个玩具形状（单头 / head_dim=32）的整块。这一批把目标推到**真实形状**：
hidden 896、14 query head / 2 KV head、head_dim 64、vocab 151936、T=8。全部 fp16、全部 `--disallow-bridge`、
权重用「随机生成后冻结」的夹具（尺度=真实 Qwen2.5-0.5B-Instuct 逐张量实测，见 `docs/OP-weights.md`）。

| 票 | 文档（本批次的实验记录） | 一句话结论 |
|---|---|---|
| 01 | `docs/OP-block.md` §10 / `docs/OP-gqa.md`（E-112） | 多 head「编不过」是 builder 的权重张量声明 bug，不是 ncc 的限制；修好后 2 头 31 层 / 14 头 33 层，**各 1 个执行步**，板端 PASS |
| 02 | `docs/OP-weights.md` | 25 张量冻结夹具：同 seed 两次生成 **25/25 sha256 相同**；含 q/k/v 的 bias（真实权重里 bias std 7.85/32.2，比权重还大） |
| 03 | `docs/OP-qkv.md` | QKV 投影+bias+RoPE，11 层/1 步；bias 反事实差 254–257×、误接 896 宽在**加载期**失败、无 RoPE 差 343–360× |
| 04 | `docs/OP-gqa-scores.md` | 2→14 KV 复制（SPLIT+CONCAT）5 层、仍在 1 步；3-D BMM 出 `[14,8,8]` **逐位一致**；2×7 头=13 层、14 次 unroll=33 层（不采用） |
| 05 | `docs/OP-masked-softmax.md` | 掩码+原生 SoftMax 3 层/1 步；被掩码位 **392/392 逐位 0**；掩码 −4096 vs −8192 两份 dump 逐位相同（已饱和）；**EXP 前提被推翻** |
| 06 | `docs/OP-pv-o.md` | PV 3-D = **1 个原生层**，4-D 被降成 5 层；O 投影无 bias；残差独立成层；7 层/1 步 |
| 07 | `docs/OP-swiglu-ffn.md` | SwiGLU 6 层/1 步/Temp 0；SiLU **不融合**（+2 层、同一步）；三个大张量各 8.72 MB |
| 08 | `docs/OP-lm-head.md` | tied LM head：**FC 的 N ≤ 65535**（单体 151936 被编译期拒绝）→ 4×FC+CONCAT，5 层/1 步，板端 99.77% 逐位 |
| 09 | `docs/OP-decoder-layer.md` | 1 层自包含整块（`[8,896]` 进 / 出）——见该文档 |
| 10 | `docs/OP-two-layer.md` | 2 层串联 + 输出投影，logits 向量 atol/rtol 对拍——见该文档 |

### 由本批次新增的方法论（值得复用）

1. **每条「编译侧已通」都必须配一次板端对拍**。本批次三个负对照（h2 的 reshape、GQA 的 wrong_reshape、PV 的 badlayout）
   都是 `EXIT=0` + 单执行步，但板端 `rel_l2` 0.14–1.23。编译通过只证明「能下沉」，不证明「算得对」。
2. **没有正对照的「某算子不支持」不算结论**。EXP 探针第一版把 opcode 表偏移当成 opcode 索引，连 SOFTMAX 都报
   `Fail to process the given model`——先跑一个已知可用的正对照，才能把「构造错」与「真不支持」分开（E-112 同类错误）。
3. **阈值必须由独立于设备的下限支撑**：本批次噪声底实测全为 0（硬件确定性），于是阈值改按 fp16 末位口径
   `4·2⁻¹¹·max|out|` 登记，并用「把 golden 逐层按 fp16 舍入」的模拟给出下限；登记一律在跑之前（`tolerance_revision`）。
