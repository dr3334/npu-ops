# OWN-SUBMITTER —— 不经过厂商 runtime 的算子覆盖

> **库化版本**：把下面这套东西做成可调用库（`Graph`/`DType`，定义 -> 编译 -> 运行 -> 两道验收，
> plan 驱动、换算子/换形状不重编 C）见 **[docs/OWN-SUBMITTER-LIB.md](OWN-SUBMITTER-LIB.md)**
> 与 `tools/npu_lib/`。本文记录的是提交器本身逐项实测的历史与细节。

目标（用户口径）：**"跑通"= 用我们自己的库在 NPU 上算出结果并对拍**，不是"用 `.dla` + 厂商 runtime 跑通"。
本文件只记录**已经用自建提交器实测通过**的算子；其余算子的"能跑"是厂商路径，另见 `docs/OP-MATRIX.md`。

## 1. 实测覆盖（2026-11-12）

| 算子 | 入口 | 元素数 | 结果 |
|---|---|---|---|
| ADD | `npu_add64()` / `npu_binop(NPU_OP_ADD, …)` | 64×int8 | rc=0，**0/64 失配** |
| MUL | `npu_binop(NPU_OP_MUL, …)` | 64×int8 | rc=0，**0/64**（含 5 个 clip 到 127） |
| SUB | `npu_binop(NPU_OP_SUB, …)` | 64×int8 | rc=0，**0/64**（含负值） |
| MAXIMUM | `npu_binop(NPU_OP_MAXIMUM, …)` | 64×int8 | rc=0，**0/64** |
| MINIMUM | `npu_binop(NPU_OP_MINIMUM, …)` | 64×int8 | rc=0，**0/64** |

设备实测输出（`probe/npu_binop_demo`，A=[1..8]×8、B=[10..80]×8）：

```
ADD      rc=0  mismatched=0/64  C[0..7]=11 22 33 44 55 66 77 88
MUL      rc=0  mismatched=0/64  C[0..7]=10 40 90 127 127 127 127 127
SUB      rc=0  mismatched=0/64  C[0..7]=-9 -18 -27 -36 -45 -54 -63 -72
MAXIMUM  rc=0  mismatched=0/64  C[0..7]=10 20 30 40 50 60 70 80
MINIMUM  rc=0  mismatched=0/64  C[0..7]=1 2 3 4 5 6 7 8
RESULT: ALL PASS
```

**全程没有 `libneuron` / `neuronrt`**：握手（BASIC×2 + DEV×5 + MEM×4）→ 自己 ALLOC/mmap/MEM MAP →
写内容 → 命令流地址重定位（`iova >> 4`）→ 组装 `cmdbuf[3]`/`subcmd_info`/`cmd_in` → `CMD RUN` →
等待 + `DMA_BUF_SYNC` → 读回。每次运行都是新的进程、新的 IOVA。

## 2. 资产（asset）是什么

一个算子 = 一份 **asset 目录**（设备 `/data/local/tmp/npuops/assets/<op>/`）：

| 文件 | 来源 | 说明 |
|---|---|---|
| `cmd368.bin` / `cmd32.bin` / `cmd16.bin` | ncc 编译产物的命令流 | 顺序语义固定：`[32B dir1][16B dir2][368B dir1]` |
| `cmdin.bin` (152 B) | 抓取到的 `mdw_cmd_in` 模板 | 指针字段在提交时替换成我们的 |
| `subcmd.bin` (80 B) | 抓取到的 `mdw_subcmd_info` 模板 | `num_cmdbufs` / `cmdbufs` 指针在提交时替换 |
| `iova.txt` | 抓取那次运行的 `A B C` 设备地址 | 与命令流**配对**：重定位的匹配键 |

关键点：`iova.txt` 与命令流必须来自**同一次**抓取。换了资产就要一起换（代码不用改，运行期读取）。

## 3. 新增一个算子要做什么

```bash
# 1) 生成并编译该算子的 .dla（厂商编译器仍用于"产生一份合法提交"）
./tools/compile.sh ops/<op>/model.tflite ops/<op>/model.dla
# 2) 抓一次资产（厂商 runtime 只用来记录这一次提交；之后不再用它）
adb push ops/<op>/model.dla /data/local/tmp/npuops/probe/<op>.dla
adb push ops/<op>/golden/input.bin  /data/local/tmp/npuops/probe/<op>_a.bin
adb push ops/<op>/golden/input1.bin /data/local/tmp/npuops/probe/<op>_b.bin
./probe/mkasset.sh <op> /data/local/tmp/npuops/probe/<op>.dla \
        /data/local/tmp/npuops/probe/<op>_a.bin /data/local/tmp/npuops/probe/<op>_b.bin
# 3) 在 probe/npu_binop.c 里注册一条 plan（asset_dir + n_inputs + 期望的重定位处数）
# 4) 自建提交器跑一遍，和独立 golden 对拍
```

## 4. 尚未覆盖 / 下一步

1. **一元算子**（RELU / LOGISTIC / TANH / SOFTMAX）：`npu_binop` 已加 `n_inputs` 字段（1=一元），
   抓取脚本也已支持单输入（`./probe/mkasset.sh <op> <dla> <a.bin> "-"`），**只差一次实测**；
2. **FULLY_CONNECTED / LINEAR**：需要"动态权重"那套（权重是第二个运行时输入），命令流里会有更多地址字，
   且权重缓冲要更大（8192/20480 B）；`npu_binop` 的 patch 计数从 3 变成 N，需要泛化；
3. 目前只有 `bytes = 64` 的形状；扩展长度要重新抓资产（命令流里含长度字段，E-008 标出过 25 个形状相关字）。

## 5. 一元算子（2026-11-12 更新）

`npu_binop` 现在支持 `n_inputs = 1`（一元：A → C），并已注册 RELU：

| 算子 | 状态 | 证据 |
|---|---|---|
| RELU | ✅ **自建提交器跑通** | `rc=0`，`0/64` 失配；输入含负值时输出正确截断为 0 |
| LOGISTIC | ⚠️ 提交成功但数值未通过 | `rc=0`，但与现有 golden 比 64/64 失配（`dev=0` vs `exp=127`）→ 怀疑 golden 的量化尺度/存储类型与 int8 传输不一致，**未作为结论** |
| TANH | ⚠️ 同上 | 同上 |

一元抓取的两个技术点（都已落到工具里）：
1. **命令流大小不固定**：一元算子是 336 B（不是 368 B）。`mkasset.py` 改为从
   `mdw_subcmd_cmdbuf` 描述符（handle/size/dir）定位缓冲，不再硬编码尺寸；
2. **重定位要跳过空槽**：一元的 B 槽 IOVA 是 0，`old>>4 == 0` 会匹配满屏零字
   （第一次跑出 "patched 34/2"）。修复后 2 处精确命中（RELU：0x050=A、0x0f4=C）。

### 关于 SOFTMAX 的说明（回答"是不是该用函数近似"）

已有的板端证据（E-ACT/OP-actreduce）：**SOFTMAX 在 MTK 上是原生层 `SoftMaxLayer<0>`**，
int8 `[1,8,16]` 逐位相等、fp16 差 1 个 code、`beta` 生效——**不是"NPU 不支持、只能 CPU 近似"**。
函数近似（LUT/多项式）在 MTK 上的价值是**性能选项**，不是可用性前提：
先用原生层保证正确性，再用 `npu_add64/npu_binop` 这套自有提交器做 A/B 对比是否值得替换。
（RK 那边用 DPU LUT，是因为它的 softmax 走 DPU；MTK 有独立 SoftMax 层，策略不同。）

## 6. 🔑 矩阵乘级算子：FULLY_CONNECTED（动态权重）

这是"自己算 transformer"最关键的一步：现在**权重也是运行时输入**的 FC 已经能由自建提交器跑起来。

| 项 | 值 |
|---|---|
| 形状 | `A[1,64] int8` × `W[64,64] int8` → `C[1,64]`（W 是运行时输入，不是常量） |
| 资产 | `probe/assets/fcdyn`（3 个 cmdbuf：32/16/368 B；iova.txt 三个地址） |
| 提交 | `CMD RUN rc=0`，`probe/npu_fc_demo` |
| 重定位 | 3 处命中：`0x078`=B(权重)、`0x098`=A(激活)、`0x110`=C |
| 数值 | **1/64 元素差 1 个 code**（`first@46 dev=-75 exp=-74`），两次运行完全一致 |
| 判定 | 与厂商路径 FC 的 `int8_tol=1` 契约一致（requant 舍入边界，见 `docs/OP-convpool.md` 的反推规则） |

新增机制：plan 里加了 `bytes_b`（第二个输入的字节数——权重 4096，激活只有 64），
其余（握手/分配/重定位/提交/等待/同步）与二元算子完全复用。

**至此自建提交器覆盖 7 个算子 + 1 个矩阵乘级算子**：ADD / MUL / SUB / MAXIMUM / MINIMUM / RELU /
（以及）FULLY_CONNECTED(动态权重)。仍未覆盖：常量权重的 FC、多输入 >2、非 64 的形状。

## 7. 🔑 自建提交器拼出"注意力核心"：FC → FC → BMM（17 次连续提交，0 失配）

`probe/npu_attn_demo`（同一进程内连续提交，**无 libneuron/neuronrt**）：

```
stage1 BMM(golden a,b)   rc=0  mismatched=0/64      ← BMM 资产本身与 golden 逐字节一致
stage2 FC->FC->BMM       rc=0  mismatched=0/64      ← 8 次 FC(Wq) + 8 次 FC(Wk) + 1 次 BMM
RESULT: PASS
```

- **q = x·Wqᵀ、k = x·Wkᵀ**：逐 token 走 `FULLY_CONNECTED(动态权重)`（每次提交 1 个 token）；
- **s = q·kᵀ**：走新抓的 `BATCH_MATMUL (T=8, D=64, int8)` 资产，**一次提交算完 8×8**；
- 参考值用**我们实际产生的 int8 q/k** 在 int64 下精确乘加后夹到 int8 —— 这样验证的是"串联链路"，
  不把 FC 的 requant ±1 code 混进结论（FC 单算子的 1/64 差 1 code 已单独记录）。

**这一步的意义**：矩阵乘级算子（投影、QKᵀ）已经能由我们自己的提交器链起来，
构成 attention 的 q/k/scores 部分。**还缺**：softmax 与 P·V 两段（需要各自的资产），
以及把它们和 v 的投影接起来。

覆盖清单更新：ADD / MUL / SUB / MAXIMUM / MINIMUM / RELU / FC(动态权重) / **BATCH_MATMUL(QKᵀ)**。

## 8. Attention 链路（T=16, int8）：三段通、一段卡

`probe/npu_attn16_demo`：同进程内
`q,k,v = x·Wᵀ`（48 次 FC 动态权重）→ `s = q·kᵀ`（BMM T=16）→ `p = softmax(s)` → `o = p·v`（BMM PV T=16）。

| 段 | 结果 |
|---|---|
| q/k/v（48 次提交） | `rc=0` |
| s = q·kᵀ | **0/256 失配**（与用设备实际 int8 q/k 的 int64 参考完全一致） |
| p = softmax(s) | ❌ 输出全 0（`rc=0` 但缓冲未被写） |
| o = p·v | 受上游影响 1008/1024 失配 |

**softmax 段单独复现**（`probe/npu_sm_check`，输入用抓取时的 golden、对照厂商 runtime 的同一输出）：
`rc=0`，但设备侧输出全 0 —— 说明**这个算子的资产不完整**：抓取时 runtime 分配了 **3 个 4 KiB 缓冲**
（fd7/fd4/fd11），而命令流里只出现 2 个地址（A 与输出）。也就是说 softmax 很可能依赖那个
**不参与重定位的 scratch/LUT 缓冲**（内容/存在性都有影响），而我们只重放了 3 个 cmdbuf + 2 个地址。

下一步（具体）：在 `mkasset.py` 里把"额外的 4 KiB 缓冲"也纳入资产（分配 + 映射 + 抓取内容 +
按需放入命令流），或者对比 eDPA 反汇编确认它是否读取该缓冲。**在此之前不声称 attention 全链路打通。**

已经可以确定的是：**投影 + QKᵀ（矩阵乘级）已经能由自建提交器链式完成并逐字节正确**。

> **⚠️ 本节结论已于 2026-11-14 被推翻**：softmax 现在在我们的提交器上**逐字节跑对**（0/128，与厂商设备输出相同）。
> 根因最可能是 `mkasset.py` 在一元分支里把「输出」和「LUT」两个 4 KiB 缓冲的角色认反了（旧资产 `softmax16` / `softmax16b`
> 的 `iova.txt` 里 B/C 两槽互换就是当时的痕迹），而 fd 复用让 `.dla` 代码缓冲在 dump 里看起来也像输出。
> 现在判据已加：候选必须能在 MAP 表里解析出 IOVA。证据与复现见 [`reports/npu-lib-softmax.md`](../reports/npu-lib-softmax.md)。
> 下面这段保留为失败知识的原始记录。
### 8.1 softmax 的第三块缓冲：找到了，但还不够

用抓取日志 + 命令流逐字比对，确认 softmax 的 `.dla` 引用了 **3 个数据缓冲**：

| 缓冲 | 大小 | 命令流里的地址字 | 内容 |
|---|---|---|---|
| 输入 (fd4) | 4 KiB（用 256 B） | `0x050` | 我们的 int8 输入 |
| **LUT/scratch (fd7)** | 4 KiB（用 96 B） | `0x448` | 运行前就有内容（pre==post，96 个非零字节）→ 是**常量表** |
| 输出 (fd11) | 4 KiB（用 256 B） | `0x128` | 运行前全 0，运行后有 256 B |

已做的改动：
- `mkasset.py`：一元算子自动把第三个 4 KiB 缓冲也纳入资产（`scratch.bin`），并把它的 IOVA 写进 `iova.txt` 第二位；
- `npu_binop`：新增 `extra_file` 字段——如果资产里有 `scratch.bin`，就把它加载进第二个缓冲并从命令流里重定位（现在 3 处地址全部命中：0x050 / 0x128 / 0x448）。

**但输出仍然是全 0**。也就是说：地址齐了、LUT 内容也照搬了，kernel 仍然没写输出。
剩下的候选（下一轮按序排除）：
1. LUT 的**分配方式**（vendor 可能是只读映射 / 特定对齐），我们用的是普通 ALLOC+MAP；
2. `.dla` 里对这块缓冲除了地址还有**长度/偏移字段**（1184 B 的命令流比其它算子大得多）；
3. 该模型需要 **2 个执行步**（第一个准备 LUT），而我们的资产只抓了 3 个 cmdbuf。

**当前结论不变**：投影 + QKᵀ 已经逐字节正确；softmax 还没通，PV 因此也未验证。

### 8.2 三个缓冲的角色**已确证**（用厂商输出反查），但 kernel 仍写 0

最新一次抓取的逐字核对：

| 缓冲 | IOVA | 命令流地址字 | 厂商 post 内容 |
|---|---|---|---|
| 输入 (fd4) | `0x25a06000` | `0x050` | 非零 255 B（我们的输入） |
| **LUT (fd7)** | `0x25939000` | `0x448` | 96 B 非零，pre==post（常量表） |
| 输出 (fd11) | `0x259cf000` | `0x128` | 运行前 0，运行后 256 B，**与 vendor_out 完全一致** |

我们的资产把三块都映射了、三处地址都重定位了（日志 3 行 patch），内核日志显示任务正常执行
（`apu_ipi_send` + handler 返回，无 error）。但设备输出缓冲仍然**被写成全 0**
（用 0x5A 预填后读回全 0 → 说明 kernel 确实写了，只是写出来是 0）。

⇒ 不是"没跑"，是"算出来就是 0"。剩下的候选：
1. **输入量化域**：模型输入 scale=1/16, zp=0；我们喂的是 QKᵀ 的 int8 码。这个应该没问题（vendor 用同一份字节），
   除非 LUT 决定了输入的解读方式而我们的 LUT 内容/位置不对；
2. **LUT 内容必须在运行前被 vendor 侧再写一次**（我们抓的是 map 时刻的快照）；
3. `.dla` 里还有别的字段指向那块缓冲（1184 B 命令流远大于其它算子）。

下一步建议：用 `diff_cmd.py` 比较 "vendor 成功那次" 与 "我们这次" 的**命令流字节**（我们不改原缓冲，
另存一份 patch 后的），看除了 3 个地址字之外还有哪里有差异。

**结论保持**：投影 + QKᵀ 逐字节正确；softmax 未通（原因已缩小到 LUT/命令流字段，不再是"缺缓冲"）。

## 9. ✅ Attention 全链路打通（自建提交器，T=16, int8, 51 次提交）

```
stage q/k/v (48 submissions) rc=0
stage s = q*k^T              rc=0
stage p = softmax(s)         rc=0
stage o = p*v                rc=0
mismatch: s=0/256  p=0/256  o=0/1024
```

**没有 libneuron / neuronrt**：48 次 `FULLY_CONNECTED(动态权重)`（逐 token 投影 q/k/v）
+ `BATCH_MATMUL(QKᵀ)` + `SOFTMAX` + `BATCH_MATMUL(PV)`，全部由我们自己的 ioctl 提交。

### softmax 那两个坑（都是靠"字节级 diff"定的，不是猜）

1. **命令流本身没问题**：把"我们 patch 后的 cmd2"和"厂商抓取的 cmd2"逐字节比，
   1184 字节里**只有 3 处地址字不同**；两次独立厂商抓取之间也只差这 3 处。
   ⇒ 命令流是确定性的，问题一定在缓冲。
2. **拿错了 LUT**：`mkasset.py` 找输出缓冲时用了 `dirname(outdir)`（资产目录的上一层），
   于是 vendor 匹配失败，退化成 `other[0]` —— 结果把 **LUT 缓冲当成了输出、把输出当成了 LUT**，
   等于给 kernel 喂了一张全零的 LUT，所以它"算出来就是 0"。
   修好路径后 `scratch.bin` 从全 0 变成 **96 个非零字节**，softmax 立刻 `0/256`。

### 还有一处必须自己做：softmax→PV 的 requant

softmax 模型输出域是 `(1/256, zp=-128)`，而 PV 模型的 A 输入域是 `(1/127, zp=0)`。
TFLite 图里这步由 ncc 插入；手搓链路里必须自己转：
`code_pv = round(((code_sm+128)/256)*127)`。加上这一步，`o` 就是 `0/1024`。

### 结论

**"自己的库在 NPU 上算 attention" 已经成立**（q/k/v 投影 → QKᵀ → softmax → PV，逐字节正确）。
仍未覆盖：常量权重 FC（当前走动态权重）、KV cache、multi-head、以及把这些接成一个完整 block。

## 10. 常量权重 FC：资产抓到了，但输出饱和（未通过）

目标：省掉"每次运行都把权重当输入传"的开销（真实推理里权重是常量）。

抓取结果（`probe/assets/fc_const`）：

| 缓冲 | 大小 | 命令流地址字 | 说明 |
|---|---|---|---|
| 常量权重 | **8192 B** | `0x05c` | 内容 4255 个非零字节，前段像 int16 打包权重；**pre == post**（运行前后不变） |
| 输入 | 4096（用 64 B） | `0x0c0` | |
| 输出 | 4096（用 64 B） | `0x138` | |

已做：
- `mkasset.py` 新增**通用规则**：命令流里出现、且既不是 A 也不是 C 的缓冲 = 常量/scratch 缓冲（大小不限，这里是 8192）；
- `iova_of()` 改成 `(fd, 文件大小)` 双键匹配 —— 抓取目录里 **`.dla` 代码缓冲与输入缓冲同 fd**，只按 fd 匹配会拿错；
- `npu_binop`：第二个缓冲按 `bytes_b` 分配（8192），`extra_file` 从资产加载常量内容；
- 新增 `NPU_OP_FC_CONST` 计划 + `probe/npu_fcc_demo`。

**结果**：`rc=0`，三处地址全部命中，但数值 `43/64` 失配且大量饱和到 ±127。
说明提交本身没问题（同命令流、同缓冲），而是**常量缓冲的语义没对上**（可能是 DLA 期望 loader 写入的
布局/校验，或需要只读映射）。

**下一步候选**（按序）：① 对比 `_post` 之后的缓冲与 DLA 静态段（`map09`）里是否也有一份同样的权重；
② 试着把常量缓冲分配成"映射后不写"（只读）；③ 用 `dump_tflite` + `.dla` 反汇编确认常量段的偏移。

### 10.1 常量权重 FC 的实测：**能对，但不稳定**

同一份资产（`probe/assets/fc_const`）连续跑 11 次：

| 配置 | 结果 |
|---|---|
| 常数缓冲 align=0x100、ALLOC 的 offset40 留 0 | **3/5 次 `0/64`，2/5 次 `43/64`**（`first@0 dev=127 exp=-128`） |
| 常数缓冲 align=0x10（照抄厂商抓取） | 6/6 次 **`43/64`** |
| ALLOC offset40 = 5（照抄厂商抓取） | 6/6 次 **`43/64`** |

对照组：**动态权重 FC 同一套代码连跑 5 次全部一致**（`1/64`，requant 边界），所以问题特定于
"8192 B 常量缓冲"这条路径，不是通用提交逻辑。

失败的数值特征很刺眼：`dev=+127` vs `exp=-128`（同一个位置出现正负极值），
说明**常量缓冲里那 4352 字节的权重没有被 DLA 完整读到**（读到的是陈旧/零数据时会饱和到一侧）。

已排除：命令流（3 处地址全命中、与厂商逐字节一致）、缓冲内容（与厂商 pre/post 完全一致）、
8 KB 对齐、ALLOC 参数、命令缓冲 flush。

剩余候选（下一轮）：
1. 常数缓冲可能需要 **8 KiB 的连续 IOMMU 映射**（我们的 ALLOC 只保证 size 内连续，DLA 可能按更大粒度取址）；
2. 抓取时该缓冲的 **MAP 参数**（flags=2）与我们一致，但 runtime 可能对常量区用 **只读映射**；
3. `.dla` 里除 3 个地址字外还有**常量长度/偏移**字段（400 B 命令流）。

**结论**：常量权重 FC 的资产与提交链路已经打通（`rc=0`、可复现地算出结果），但**数值正确性未达成**，
不并入"已支持"清单。

## 11. ✅ 大尺寸 FC 也走通了（256 KiB 与 1 MiB 权重）

| 形状 | 权重 | 自建提交器结果 |
|---|---|---|
| 1×512×512 动态权重 | 256 KiB | **0/512**（连跑 3 次一致） |
| 1×1024×1024 动态权重 | 1 MiB | **0/1024**（连跑 3 次一致） |

两例都落在 §MAPPING 记录的**高段地址**（`0xcbbc0000` / `0xcba00000`），说明"按值重定位、不假设地址范围"这条规则是对的。

映射协议、每个算子需要几块缓冲、地址字偏移、缓存同步要求，全部写进了 **[docs/MAPPING.md](MAPPING.md)**。

### 10.2 常量权重 FC 的后续排查（本轮新增，仍未通过）

在 §10.1 基础上又做了三组对照，结论是**更接近根因但还没解决**：

| 实验 | 结果 | 说明 |
|---|---|---|
| 常量缓冲 **8192 vs 4096** 分配（`NPU_B_4096`） | 两者都 `43/64` | 分配大小不是主因 |
| 命令缓冲 **flush vs 不 flush**（`NPU_CMDFLUSH=0`） | 两者都 `43/64` | 之前"3/5 次正确"与 flush 无关 |
| **交换常量/输出两处地址**（`NPU_SWAP_BC`） | `64/64`（输出全 0） | 反过来证明**角色标注是对的**：`0x05c`=常量、`0x0c0`=输入、`0x138`=输出 |

另外验证了设备端资产与本地**逐字节一致**（iova/scratch/cmd2/cmdin/subcmd 五个文件 md5 全同），
所以不是"推错文件"。

失败特征始终是 `dev=+127 vs exp=-128`（同一位置正负极值），即**累加器读到了错误符号/错误数据**，
而不是精度问题。剩下最可疑的一条：**常量区可能要求由 kernel/loader 在提交时填充**，
我们"预先把内容拷贝进去"这条路径在语义上不等价（厂商 runtime 也可能是每 run 重新写的）。

下一轮做法（明确）：抓住"vendor 成功那次"与"我们这次"的 **`.dla` 代码段 + 常量段**在内存里的差异，
或者用 `NPU_VERBOSE=1` 把两块缓冲在 CMD RUN 前的内容 dump 下来与抓取逐字节比对。

### 10.3 常量权重 FC：找到**决定性相关性**（仍未通过）

用 `NPU_VERBOSE=1` 打印三块缓冲的实际 IOVA，再和控制结果对照，规律非常干净：

| 我们的常量缓冲地址 | 结果 |
|---|---|
| **`0x25318000`**（= 抓取时的地址） | **`0/64` 正确** |
| `0x25342000` / `0x2531e000` / `0x25340000` 等其它地址 | `43/64`（`dev=127 vs exp=-128`） |

10 次里 4 次正确，**全部是"恰好分配到抓取地址"的那几次**。

还做过的对照（都排除）：
- 分配大小 8192 vs 4096 —— 无关；
- 命令缓冲 flush —— 无关；
- **写完内容再 MAP**（厂商抓取里 MAP 前内容就已就位）—— 无关，仍 4/10；
- 角色交换（常量/输出）—— `64/64` 全 0，**反证角色标注正确**；
- 打印我们 patch 后的 cmd2：`0x05c` 处确实是我们的 `B>>4`（`0x2531e00`），**重定位本身生效**。

⇒ 结论：**命令流里那处地址确实是常量指针，但 DLA 实际取数时还依赖抓取时的那个地址**。
最可能的解释是 `.dla` 的代码段里也内嵌了一份常量地址（由 loader 在加载时写入代码缓冲），
而我们只重放了 cmdbuf、从未加载代码段。这与"地址对上就全对、对不上就符号错乱"完全吻合。

**下一步（唯一有意义的方向）**：把 `.dla` 的代码段也纳入资产（加载到我们自己分配的缓冲、并把代码里的
常量地址一并重定位），而不是只重放 3 个 cmdbuf。
