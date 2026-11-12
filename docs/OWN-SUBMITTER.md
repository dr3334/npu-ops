# OWN-SUBMITTER —— 不经过厂商 runtime 的算子覆盖

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
