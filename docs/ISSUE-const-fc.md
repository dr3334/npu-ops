# 求助分析：常量权重 FULLY_CONNECTED 在自建提交器上"地址对上才对"

> 本文是一份**自包含的技术档案**，供外部分析者在不接触设备的情况下推理。
> 所有数字都来自实测抓取；文件路径相对 `npu-ops/` 仓库根。

## 0. 一句话问题

我们自己写了一个 NPU 提交器（不用厂商 `libneuron`/`neuronrt`），它能正确跑 8 个算子（含动态权重 FC、
BATCH_MATMUL、SOFTMAX）。**唯独"常量权重 FULLY_CONNECTED"**：命令跑成功（`rc=0`），
但**只有当我们把常量缓冲恰好分配在"抓取时那个 IOVA"上时结果才对**，分配在别的地址就错。

## 1. 平台与工具链

| 项 | 值 |
|---|---|
| SoC | MediaTek MT6897（HEADWOLF Titan 1），Android 16 / arm64，`adb root` |
| 驱动 | `/dev/apusys`，magic `'A'=0x41`：nr 32 HANDSHAKE(40 B)、33 MEM(48 B)、34 CMD(152 B) |
| 编译器 | ncc-tflite 7.3.15，`--arch=mdla5.0 --num-mdla=2 --l1-size-kb=3072 --disallow-bridge` |
| 目标 | MDLA_5_0，单 execution step |

## 2. 模型

`ops/fc_1x64x64_int8/`：TFLite FULLY_CONNECTED，**权重是常量**（编进 `.dla`），int8，
M=1 K=64 N=64，输入 int8[1,64]，输出 int8[1,64]。`.dla` = **7545 B**，1 个 `FullyConnectedLayer<0>`。

对照（同一套提交器、能跑对）：`ops/fc_dynb_1x64x64_int8/`，权重是**运行时输入**（第二输入），int8。

## 3. 厂商 runtime 的抓取（一次合法运行）

工具：`probe/iolog2.c`（PRE/POST 双打 ioctl + mmap 缓冲快照），`probe/mem_map_flow.py` 解析。
资产目录：`probe/assets/fc_const/`（含 `cap.log`、`buf/`、`cmd0/1/2.bin`、`cmdin.bin`、`subcmd.bin`、`iova.txt`）。

### 3.1 MEM 调用序列（全量，按时间顺序）

```
 ALLOC  size=  400 align=0x100  type=0 -> handle=6
 ALLOC  size= 8192 align=0x1000 type=0 -> handle=7
 MAP    handle=7  size= 8192 flags=2 -> iova=0x25318000     ← 常量区
 ALLOC  size=   32 align=0x100  type=0 -> handle=8
 ALLOC  size=   16 align=0x100  type=0 -> handle=9
 ALLOC  size=   76 align=0x0    type=0 -> handle=10
 ALLOC  size= 4096 align=0x100  type=0 -> handle=4
 MAP    handle=4  size= 4096 flags=2 -> iova=0x258e0000     ← 输入
 ALLOC  size= 4096 align=0x100  type=0 -> handle=11
 MAP    handle=11 size= 4096 flags=2 -> iova=0x25733000     ← 输出
```

注意：常量区**先分配、先 MAP**，且 `align=0x1000`（其它是 `0x100`）。

### 3.2 命令流（`cmd2.bin`，400 B）

三个地址字（`iova>>4` 的 u32）：

| 偏移 | 值 | 指向 |
|---|---|---|
| `0x05c` | `0x02531800` = 0x25318000>>4 | 常量区 |
| `0x0c0` | `0x0258e000` = 0x258e0000>>4 | 输入 |
| `0x138` | `0x02573300` = 0x25733000>>4 | 输出 |

另有两个小 cmdbuf：`cmd0.bin` 32 B、`cmd1.bin` 16 B（内容全 0，未见地址）。
**两次独立厂商抓取对比**：1184/400 字节里**只有这 3 处地址字不同**（其余逐字节相同）→ 命令流是确定性的。

### 3.3 缓冲内容

| 缓冲 | 大小 | 内容 |
|---|---|---|
| 常量区 | 8192 | **4255 个非零字节**，集中在 `[0, 4352)`；前 16 字节 `f5 12 00 00 8f 1d 00 00 be 15 00 00 90 e4 ff ff`（像 int16 序列）；**pre == post**（运行前后不变） |
| 输入 | 4096（用 64 B） | 64 B 输入（与 golden input 逐字节相同） |
| 输出 | 4096（用 64 B） | pre 全 0 → post 有价值，**与厂商输出/`golden/expected.bin` 逐字节相同** |

### 3.4 用 `probe/scan_blob.py` 扫过常量区

在 `scratch.bin`（8192 B）里搜索 `0x25318000` 的 u64 / u32 / `>>4` / `>>12` 各种编码：**0 命中**。
⇒ 常量区里**没有内嵌自身地址**。

## 4. 我们的重放实现（`probe/npu_binop.c`）

```
open("/dev/apusys")
HANDSHAKE: BASIC×2 -> DEV(1,2,4,6,7) -> MEM(1,2,4,5)
ALLOC+mmap+MAP: cmdbuf[3]（32/16/400）、A(4096)、B(常量 8192)、C(4096)、einfo(76)
写 A/B 内容；DMA_BUF_SYNC(START|WRITE / END|WRITE)
从资产读 cmdin.bin(152 B) 模板：清 0x10/0x18，num_subcmds=1，指向 subcmd、adj(64 B 全零)、fence=-1，
   并把 einfo 的 **MEM handle** 填进 `cmd_in@0x68`
从资产读 subcmd.bin(80 B) 模板：num_cmdbufs=3，cmdbufs 指向我们构造的 3×24 B 描述符
重定位：在 cmd2 里按值搜索 old_iova[k]>>4 → 替换成 newv[k]>>4（k=0/1/2 = A/B/C）
ioctl(CMD RUN) -> rc=0；usleep(200ms)；DMA_BUF_SYNC(START|READ / END|READ) 后读 C
```

**已知细节**：`cmd_in@0x68` 必须是 76 B 缓冲的 **MEM handle**（不是指针）；不这样做会得到
`mdw_mem_get get dma_buf handle(...) fail`。命令流地址 = `iova>>4`。驱动不支持 `MEM op=1(FREE)`。

## 5. 现象（这是要分析的核心）

`probe/npu_fcc_demo` 连跑 10 次：**4 次 `0/64` 正确，6 次 `43/64` 错误**。
用 `NPU_VERBOSE=1` 打印每次分配的 IOVA 后，规律非常干净：

| 我们的常量缓冲 IOVA | 结果 |
|---|---|
| **`0x25318000`**（= 抓取时那个地址） | `0/64` ✅ |
| `0x25342000` | `43/64` ❌ |
| `0x2531e000` | `43/64` ❌ |
| `0x25340000` | `64/64`（全 0，配合下面"保留地址"实验） |

错误时的特征：`first@0 dev=+127 exp=-128` —— **同一位置出现正负极值**，像累加器读到了错误符号的数据，
而不是精度误差。`rc` 永远是 0，内核日志无 error（`apu_ipi_send` / handler 正常返回）。

## 6. 已经排除的（都有实测数据）

| 假设 | 实验 | 结果 |
|---|---|---|
| 分配大小不对 | 常量缓冲分配 8192 vs 4096 | 都是 `43/64` |
| 命令缓冲没 flush | `NPU_NO_CMDFLUSH=1` | 都是 `43/64` |
| 写入顺序（应"写完再 MAP"） | 改成 alloc→写→再 MAP | 仍 4/10 |
| 角色标注错（常量/输出互换） | `NPU_SWAP_BC=1` 交换两处地址 | `64/64` 全 0 → **反证角色正确** |
| 重定位没生效 | `NPU_DUMP_DIR` 导出 patch 后的 cmd2 | `0x05c` 处确实是我们的 `B>>4` |
| 推错资产 | 设备端/本地 5 个文件 md5 比对 | 完全一致 |
| 常量区里内嵌地址 | `scan_blob.py` 扫 4 种编码 | 0 命中 |
| 保留抓取地址不重定位（`NPU_KEEP_CONST=1`） | 常量地址仍是 0x25318000 | 当我们的 B 恰好在 0x25318000 时 `0/64`；B 在别处时 `64/64` 全 0 |

最后一行很关键：**不重定位时行为与"命令流地址被 DLA 直接使用"完全一致**（地址上没有数据 → 全 0）；
**重定位到我们的地址后**却不是全 0，而是 `43/64` 的错乱值 —— 说明 DLA **确实读到了我们缓冲里的数据**，
只是解读方式/位置和厂商那次不同。

## 7. 候选假设（按可信度排序）

### H1（当前最可信）：常量区是"代码+常量"的混合块，DLA 按**基址+固定偏移**取代码与常量
- 支持：8192 B 比 64×64 权重（4096 B）大一倍；非零区 4352 B；`pre==post`；`type/align` 与其它缓冲不同（`align=0x1000`）。
- 预测：若把我们的常量缓冲**内容**按不同偏移摆放（例如整体后移 0x1000），结果会变成另一种错误模式；
  或者真正的常量在 `+0x1000` 偏移处（我们看到的非零区在 `[0,0x1100)`，其余为 0）。
- 判别实验：把 `scratch.bin` 的后 4096 B 与前 4096 B 互换后再跑，看错误是否变成"部分正确"。

### H2：常量区必须在 MAP 之前由内核/loader 写入，用户态写的内容会被覆盖或不被采信
- 支持：厂商流程里 MAP 时内容已就位；`align=0x1000` 暗示特殊分配路径。
- 反驳线索：我们"写→MAP"仍 4/10（但那次实验里 DLA 也可能读到我们的数据）。
- 判别实验：MAP 之后**不改内容**，只改地址（分配到不同 IOVA）→ 已经做过，失败。

### H3：DLA 用"基址低 13 位"做索引（即要求常量区与代码区同页对齐/同域）
- 支持：只有恰好落在 0x25318000（`align=0x1000` 且页内偏移 0x8000）时才正确。
- 判别实验：故意分配一个**同样页内偏移、不同基址**的缓冲（如 `0x…b8000`），看是否也正确。
  如果"页内偏移一致就正确"，则说明 DLA 只用了低 13 位，高位移位/取模有 bug（可能性低但可测）。

### H4：常量区地址被写进了**驱动内部的会话表**（第一次 MAP 的地址被记住），后续 DLA 用那张表
- 支持：跨进程仍只在"那个地址"上正确，像是全局缓存了 0x25318000 这个映射。
- 判别实验：在**同一进程**里先后映射两块常量缓冲（不同地址），看第二块是否也能工作；
  或者重启 APU（power off/on）后第一次就跑，看是否"第一个分配的地址"总是对的。

## 8. 需要外部分析者回答的问题

1. 从 `apusys.ko` 的 `mdw_mem_*` / `mdw_cmd_*` 看，DLA 提交时**常量/代码区**是怎么被引用的？
   命令流里的那个地址字（我们认定的 0x05c）到底是"数据缓冲"还是"代码缓冲"？
2. MTK 的 `.dla` 里，**常量（weights）与代码**是同一段还是两段？loader 是否会在加载时做重定位？
3. 是否有一个"提交前必须由驱动写入某块缓冲"的步骤（我们在用户态看不到）？
4. 为什么只有"抓取时的那个 IOVA"能让结果正确——驱动里是否有按 IOVA 缓存的映射表/TLB？

## 9. 可用的原始产物清单

| 路径 | 内容 |
|---|---|
| `probe/assets/fc_const/cap.log` | 完整 ioctl 日志（PRE/POST、MAPS、MMAP、BUFDUMP） |
| `probe/assets/fc_const/buf/` | 每个 mmap 缓冲的 pre/post 快照（含 7545 B 的 `.dla` 代码段 `map09_fd4_len1d79.bin`） |
| `probe/assets/fc_const/{cmd0,cmd1,cmd2,cmdin,subcmd}.bin` | 重放用的命令模板 |
| `probe/assets/fc_const/{iova.txt,scratch.bin,vendor_out.bin}` | 抓取时的三个地址、常量区内容、厂商输出 |
| `ops/fc_1x64x64_int8/` | 模型 + golden（input/expected）+ compile log |
| `ops/fc_dynb_1x64x64_int8/` | 对照：动态权重版本（我们的提交器**能跑对**） |
| `probe/npu_binop.c` | 我们的提交器实现（含本次所有诊断开关） |
| `docs/MAPPING.md` | 已确认的映射协议（MEM op 0/1/2、两段地址、每算子缓冲表） |

## 10. 复现命令

```bash
# 抓一次（厂商 runtime 只用来记录一次合法提交）
./probe/mkasset_u.sh fc_const <device.dla> <a.bin> ops/fc_1x64x64_int8/golden/input.bin
# 解码映射时序 / 资产摘要
/usr/bin/python3 probe/mem_map_flow.py probe/assets/fc_const/cap.log
/usr/bin/python3 probe/asset_report.py
# 用我们自己的提交器跑（设备上）
LD_LIBRARY_PATH=/data/local/tmp/npuops/lib ./probe/npu_fcc_demo in.bin expected.bin
# 诊断开关
NPU_VERBOSE=1   打印三块缓冲 IOVA
NPU_KEEP_CONST=1 常量地址不重定位
NPU_SWAP_BC=1    交换常量/输出地址
NPU_B_4096=1     常量缓冲按 4 KiB 分配
NPU_DUMP_DIR=... 导出重定位后的命令流
```
