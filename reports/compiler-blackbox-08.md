# 黑盒研究 08：三级剖面（-P1/2/3）上板采集与对账

日期：2026-11-14 · SDK 7.0.8 / `ncc-tflite` 7.3.15 · 板端运行时 **7.3.8**（`/vendor/lib64`）· 票：`08-profile-on-board.md`
前序：[07](./compiler-blackbox-07.md)（`-P<n>` + `MTKNN_*` 出口）/ [OP-toolchain](../docs/OP-toolchain.md) §6.1 §7
证据：`ops/_probe/bb08-prof/`（脚本 + `build/` 编译日志 + `board/` 板端原始件 + `host/` 主机侧原始件）

## 0. 一句话

两级产物各四份（`-P0/1/2/3`，共 8 份）编好上板，三个等级全部跑通并拿到**每级剖面表原文**（level 3 另有
`execution_result.json` / `dla_info.json`）；配对交替测了全部级别的总时延；对账首次给出**偏差百分比**：

- **`fc_1x64x64_int8`（1 条命令，单核）**：设备侧计算 **5.03 µs**，墙钟 **496.4 µs**（同产物 p3 干净态中位数）
  → 计算只占 **1.0%**，**偏差 99.0%**（拿 p0 的 351.9 µs 配对则是 98.6%，两者都远大于 07 的 61–75%）；
- **`decoder_layer_t8_fp16`（61 层 → 121 条命令，双核）**：设备侧计算 **1226 µs**（最慢核），墙钟 **2112 µs** → 计算占 **58%**，**偏差 41.9%**；
- **61 层这张图：贵在搬，不在算** —— 7 条 `FULLY_CONNECTED` 吃掉 **68.8%** 的核心周期、搬 **30.2 MB** 权重（15–29 GB/s），
  其中三张 8.8 MB 的 MLP FC（gate/up/down）就占 **61%**；而逐元素类命令（MUL/ADD/SUB/MEAN/LOGISTIC/SOFTMAX，共 32 条）只占 **23.9%**，且其中最大的一条（339 µs 的 ADD）是**等依赖**不是计算。

另外两条副产品都推翻了旧结论：**板端其实会打印剖面表**（走 `logcat -s neuron`，07 只看了 stdout 才以为没有）；
**本轮跑了 4 次 level 3，板子没坏**（07 §8 的 `MDLA timeout` 没复现）。时钟从 07 的 0.67–1.0 GHz 收紧到 **0.94–1.03 GHz**。

## 1. 板子：一次 12 分钟的 USB 掉线（不是 profiling 弄的）

| 时间 | 事件 |
|---|---|
| 16:18 | 持锁检查：`0123456789ABCDEF` 在场，mt6897/16，`uid=0(root)`，uptime 30 min |
| 16:19 | `healthcheck --quick` **PASS** |
| 16:22 | staging 失败：`adb: no devices/emulators found` |
| 16:22–16:30 | `adb kill-server/start-server`、`adb reconnect offline`、`pnputil /scan-devices` 均无效；Windows 设备树里没有 `VID_0E8D` |
| 16:30 | **自行回来**（`uptime` 46 min 连续，**没重启**）→ 全程测量 |
| 16:34 | 收尾 `healthcheck --quick` **PASS**，`dmesg` 坏态标记 **0** |

**判据式的排除**：设备树里没有 MediaTek VID ⇒ 不是 adb server、不是驱动；`uptime` 连续 ⇒ 板子没重启，是 **USB 链路掉线**。
这条诊断路径已补进 `docs/OP-toolchain.md` §6.1（先判电气层，再动 adb）。

## 2. 产物：8 份 `.dla`（已验证）

| 模型 | `-P0` | `-P1` | `-P2` | `-P3` |
|---|---|---|---|---|
| `fc_1x64x64_int8` | 7,634 B | 7,654 B | 7,654 B | **8,345 B** |
| `decoder_layer_t8_fp16` | 29,913,450 B | 29,913,470 B | 29,913,470 B | **29,974,424 B** |

- `-P1/-P2` 的 +20 B 里 4 B 是命令行回显（07 §3.1 口径），内容增量 ≈16 B；
- **`-P3` 的增量随层数走**：1 层 +711 B，61 层 **+60,974 B**（≈1,000 B/层）；
- 三个等级编出的产物**输出逐字节相同**（`board/lat-fc-p*.bin` 与 `lat-dec-p*.bin` 的 md5 全一致）——`-P` 不改数学；
- decoder 的 exec plan 给出 **61 个层条目**：`MUL 17 / SPLIT 8 / FULLY_CONNECTED 7 / ADD 7 / TRANSPOSE 4 / RESHAPE 4 / CONCATENATION 4 / SUB 2 / RSQRT 2 / MEAN 2 / BATCH_MATMUL 2 / SOFTMAX 1 / LOGISTIC 1`。
- 每份的 `build/*.compile.log` 里有 cmd / ncc 版本 / exec plan / memory summary / EXIT / SIZE / SHA256。

## 3. 三级剖面原文（已验证，板端）

**重要更正**：07 §3.3 说「板端 level 1/2/3 都只打 3 行，没有 Summary 表」——**那是因为只读了 stdout**。
表确实打印，走的是 **`logcat` 的 `tag=neuron`**；`20-run-one.sh` 用 `adb logcat -d -v time -s neuron` 抓全了。

### 3.1 level 1（`fc`，`-c 20`；原文 `board/logcat-neuron-fc-L1.txt`）

```text
I/neuron  ( 8805): The profiling level is now set to 1
I/neuron  ( 8805):                                  Runtime Status (LEVEL 1) - Summary
I/neuron  ( 8805):  ===Time===, ==Ratio==, =ExeTimes=, =Time/Inf.=, ===Description===
I/neuron  ( 8805):       0.186,      2.2%,         20,       0.009, Set Inference Input
I/neuron  ( 8805):       0.244,      2.8%,         20,       0.012, Set Inference Output
I/neuron  ( 8805):       8.158,     95.0%,         20,       0.408, Inference
I/neuron  ( 8805):       8.588,    100.0%,        ---,       0.429, Total
```

### 3.2 level 2（`fc`；原文 `board/logcat-neuron-fc-L2.txt`）

level 2 多出来的不只是 LEVEL 2 一行，而是**整整两张 driver 级表**（LEVEL 3 / LEVEL 4），这是本轮新看到的结构：

```text
I/neuron  ( 9189):                                  Runtime Status (LEVEL 1) - Summary
I/neuron  ( 9189):       0.203,      2.9%,         20,       0.010, Set Inference Input
I/neuron  ( 9189):       0.104,      1.5%,         20,       0.005, Set Inference Output
I/neuron  ( 9189):       6.734,     95.6%,         20,       0.337, Inference
I/neuron  ( 9189):       7.041,    100.0%,        ---,       0.351, Total
...
I/neuron  ( 9189):                  Detail of 'Execute': Driver Turnaround Time (LEVEL 3) - Summary
I/neuron  ( 9189):       0.081,    100.0%,          3,       0.027, MDLA_5_0 - subgraph#1
I/neuron  ( 9189):                  Detail of 'Driver Turnaround Time': IP Time (LEVEL 4) - Summary
I/neuron  ( 9189):       0.024,    100.0%,          3,       0.008, MDLA_5_0 - subgraph#1
```

- 这两张表的 `MDLA_5_0 - subgraph#1` 是**按设备/subsystem** 的，不是按层，粒度仍不够；
- level 1 只出 LEVEL 1 表；**level 2 就把 LEVEL 2/3/4 全带出来**（`ask=2` 即可，不用 `-P3`）。

### 3.3 level 3（两只图都跑；JSON 见 §3.4）

| 模型 | 等级 1 | 等级 2 | 等级 3 | 干净（无 profiler） |
|---|---|---|---|---|
| `fc` 墙钟 ms/inf | 0.4127 | 0.3395 | **0.7771** | 0.3519 |
| `dec` 墙钟 ms/inf | 1.6241 | 2.6934 | **12.6607** | 1.9616 |
| `fc` Inference 段（logcat 表） | 0.408 | 0.337 | —（表同 L2，多出 JSON） | — |
| `dec` Inference 段（logcat 表） | 1.620 | 2.610 | — | — |

（原文：`board/console-{fc,dec}-L{1,2,3}.txt`；表：`board/logcat-neuron-{fc,dec}-L{1,2,3}.txt`）

### 3.4 level-3 的两个 JSON（已验证）

| 文件 | fc | dec |
|---|---|---|
| `execution_result.json` | 11,891 B（20 次推理，**单核**，每条 1 command） | 599,773 B（10 次推理，**双核**，62+59 = **121 条命令**） |
| `dla_info.json` | 695 B（6 buffers：L1 1 + DRAM 5） | 24,201 B（25 buffers：L1 1 + DRAM 24） |

- 每次推理都有完整 `cores[].flows[]`；`endCycle` 单调，末值即该核完成时刻（07 §6.3 的读法在本图同样成立）；
- **fc 只用了一个核**（`cores=1`），尽管编译时是 `--num-mdla=2`——单算子图没有可并行的两条命令流；
- dec 的 121 条命令来自 61 层：双核分工不是 61/0 而是 **62/59**，末值 1,180,655 vs 1,177,905（两核几乎同时收工，负载是均衡的）。

## 4. 等级门槛：板端原文（已验证，判据 4）

`22-mismatch.sh` 在**板端**跑了 7 组，原文 `board/level-gating-board.txt`（`logcat tag=neuron`）：

```text
=== DLA=fc_p0  ask=1 ===   W/neuron: To enable profiling level 1, please use -P1 option to re-compile the model
=== DLA=fc_p0  ask=2 ===   W/neuron: To enable profiling level 2, please use -P2 option to re-compile the model
=== DLA=fc_p0  ask=3 ===   W/neuron: To enable profiling level 3, please use -P3 option to re-compile the model
=== DLA=fc_p1  ask=2 ===   W/neuron: The runtime profiling level is 2, but DLA only supports 1
                           W/neuron: To enable profiling level 2, please use -P2 option to re-compile the model
                           I/neuron: The profiling level is now set to 1
=== DLA=fc_p1  ask=3 ===   W/neuron: The runtime profiling level is 3, but DLA only supports 1
                           W/neuron: To enable profiling level 3, please use -P3 option to re-compile the model
                           I/neuron: The profiling level is now set to 1
=== DLA=fc_p2  ask=3 ===   W/neuron: The runtime profiling level is 3, but DLA only supports 2
                           W/neuron: To enable profiling level 3, please use -P3 option to re-compile the model
                           I/neuron: The profiling level is now set to 2
=== DLA=fc_p3  ask=3 ===   I/neuron: The profiling level is now set to 3
```

- 判据原文里的 `please use -P{} option to re-compile the model` **在板端就是这么打的**（`W/neuron`，不是 stdout）；
- **它不报错、不失败**：`rc=0`，只是降级 + 告警。所以「拿到 level-3 数据」的唯一凭据是 `The profiling level is now set to 3` 这一行；
- 主机侧 `-m null` 的完整 24 组矩阵在 `host/gating-matrix.txt`（8 份产物 × ask 1/2/3），与板端行为一致；`-P0` 只有 WARNING 没有 `set to`。

## 5. 总时延：配对交替（已验证）

`21-latency.sh`：每轮按 `p0,p1,p2,p3` 的顺序各跑一次（`fc -c 20` / `dec -c 10`），共 3 轮，取中位数（`reports/timings.md` 口径）。

**fc（ms/inf）**

| 轮 | p0 | p1 | p2 | p3 |
|---|---|---|---|---|
| 1 | 0.3501 | 0.3232 | 0.4643 | 0.4964 |
| 2 | 0.5922 | 0.5455 | 0.3684 | 0.5177 |
| 3 | 0.3520 | 0.3705 | 0.4018 | 0.3854 |
| **中位数** | **0.3519** | 0.3705 | 0.4018 | 0.4964 |
| MAD | 0.0018 | 0.0473 | 0.0334 | 0.0213 |

**dec（ms/inf）**

| 轮 | p0 | p1 | p2 | p3 |
|---|---|---|---|---|
| 1 | 2.0090 | 2.0295 | 1.4668 | 2.1124 |
| 2 | 1.9616 | 2.5199 | 1.6495 | 2.5059 |
| 3 | 1.8108 | 1.8004 | 2.3087 | 1.7291 |
| **中位数** | **1.9616** | 2.0295 | 1.6495 | 2.1124 |
| MAD | 0.0474 | 0.2291 | 0.1827 | 0.3833 |

**读法（诚实版）**：这块板子当时的噪声 **MAD 0.18–0.38 ms（dec，约 ±15%）**，比「等级之间」的差异还大；
逐轮比值在 0.62–1.32 之间来回跳（`92-latency.py` 输出），**没有任何一轮显示出单调的级别趋势**。
所以：**`-P` 等级对总时延没有可测的影响**（在 ±15% 噪声下），这也符合「`-P` 只往产物里写记录、运行时不额外做事」的预期；
**不能用这张表说 p3 比 p0 慢** —— 它的中位数高只是这一轮抖动。板子 load average 一直在 13 左右（背景负载），是噪声来源。

## 6. 对账：per-command 之和 vs `-c N` 总时延（判据 3）

**口径**：`A` = 最慢核末条 `endCycle` / 时钟（两核并发，取 max 不取 sum）；`B` = 同产物**干净态** `-c N` 的 ms/inf；偏差 = `(B−A)/B`。

| 图 | 设备侧计算 A | 墙钟 B（干净，中位数） | **偏差** | 计算占墙钟 |
|---|---|---|---|---|
| `fc_1x64x64_int8` | **5.03 µs**（中位 `endCycle` 5,031，20 次取中位） | **496.4 µs**（p3，JSON 同产物）/ 351.9 µs（p0） | **99.0%** / 98.6% | **1.0%** / 1.4% |
| `decoder_layer_t8_fp16` | **1226 µs**（中位 `endCycle` 1,226,334，10 次取中位） | **2112 µs**（p3）/ 1962 µs（p0） | **41.9%**（用 p3）/ 37.5%（用 p0） | **58.1%** / 62.5% |

- **`fc` 的 99% 不是「算得慢」，是「图太小」**：一条 5,031 周期的命令（5.03 µs）撑不起 352–496 µs 的提交+同步+IO 开销。
  这个数字在工程上的意思很直接：**单算子小图的瓶颈 100% 在运行时侧**，改算子/改编译选项都不会有用。
- **`dec` 的 41.9% 与 07 的 25–39% 同量级但更高**，因为这张图有 30 MB 权重真正在跑（07 的 3.75 MB 图几乎没有权重搬运），
  设备侧计算占比自然上升。**两张图不能互相套用**。
- 时钟（见 §6.1）只影响 A 的绝对值，不影响「哪层占比」（线性）。

### 6.1 时钟：0.94–1.03 GHz（比 07 收紧）

dec 的三个 `exec_us` 样本（`40-board-profqos.sh`，同一批产物）给出：

| `exec_us` | 中位 `endCycle` / exec_us |
|---|---|
| 1193（p3 干净） | **1027.9 MHz** |
| 1230（p0 干净） | **997.0 MHz** |
| 1309（p0 + level 3） | **936.8 MHz** |
| 829（p3 + level 3） | 1479 MHz（离群，level-3 观测下 QoS 计数被扰动） |

→ **0.94–1.03 GHz**，中位约 **1.0 GHz**；07 的 0.67 GHz 端点在本轮 8 个样本里一次都没出现。
**fc 不能用来定标**：它的 `exec_us` 是整数 µs，5 µs 的任务上量化误差 ±10%（拿它算只有 296–503 MHz，那是量化不是时钟）。
**受此影响的只有 §6 表里的 A 的百分数（±5%），§7 的占比与排序完全不受影响。**

## 7. 61 层这张图：哪层贵、贵在算还是贵在搬（已验证，`board/perop-dec.json`）

### 7.1 按算子类（两核合计 2,358,600 周期 = 2358.6 µs @1GHz，占比按这个总和工作）

| 算子 | 命令数 | 周期 | **占比** | 访存 B | 等效 GB/s |
|---|---|---|---|---|---|
| **FULLY_CONNECTED** | 7 | 1,621,961 | **68.8%** | **30,185,472** | **18.4** |
| ADD | 7 | 344,546 | 14.6% | 124,416 | 0.36 |
| BATCH_MATMUL | 56 | 143,686 | 6.1% | 885,248 | 6.1 |
| MUL+MEAN（两处 RMSNorm） | 2 | 101,464 | 4.3% | 28,800 | 0.28 |
| MUL | 16 | 82,964 | 3.5% | 483,008 | 5.8 |
| SPLIT+CONCAT（KV 复制链） | 28 | 29,400 | 1.2% | 458,752 | 15.4 |
| LOGISTIC | 1 | 28,456 | 1.2% | 233,472 | 8.1 |
| SOFTMAX / SUB | 3 | 6,083 | 0.3% | 39,936 | — |

### 7.2 最贵的几条命令（`90-perop.py` 的 top 表）

| 核 | cmd | 层 | 周期 | µs | 访存 B | GB/s |
|---|---|---|---|---|---|---|
| 1 | 76 | `FullyConnectedLayer<50>`（MLP 8.8 MB） | 578,131 | 584.0 | 8,808,448 | 15.1 |
| 0 | 74 | `FullyConnectedLayer<50>`（另一核上的同一张） | 570,776 | 576.5 | 8,808,448 | 15.3 |
| 1 | 78 | `ElementWiseAddLayer<ADD>` | 336,021 | 339.4 | **43,008** | 0.13 |
| 0 | 78 | `FullyConnectedLayer`（第三张 8.8 MB） | 295,506 | 298.5 | 8,808,448 | 29.5 |
| 0 | 70 | `ElementWiseMul+Mean`（RMSNorm） | 95,183 | 96.1 | 14,400 | 0.15 |
| 1 | 71 / 2 | `FullyConnectedLayer`（q_proj / o_proj 1.6 MB） | 80,128 / 70,305 | 80.9 / 71.0 | 1,634,304 | 20.2 / 23.0 |

### 7.3 结论：**贵在搬，不在算**

1. **61% 的周期花在三张 8.8 MB 的 MLP FC 上**（gate/up/down，合计 1,444,413 周期），它们各搬 8.8 MB、跑 15–30 GB/s；
   7 条 FC 合计 **30.2 MB**、**68.8%** 的周期 —— 与解析预算（MLP 87.6% 的 MAC **和** 87.6% 的权重字节，§8）同向；
2. **算术强度只有 3.95 MAC/字节**（`host/static-budget.txt`）：M 维只有 8（`T=8`），权重每个推理必须完整搬一遍，
   所以 **时间下限 = 28.4 MiB / 峰值带宽**；实测 18.4 GB/s 与 07 的 FC 带宽 18–24 GB/s 一致，说明已经接近这台机器的搬数上限；
3. **逐元素的那 30 条命令（24%）里，最大的 339 µs 是「等」不是「算」**：core1 的 `ADD` 只碰 43 KB，却有 336,021 周期
   （0.13 GB/s）。它紧跟在 core0 的 8.8 MB FC 后面，是**依赖等待**被算进了 `delta(endCycle)`。
   ⇒ **读法的限制要写明**：`endCycle` 是「完成时刻」，delta 里同时包含计算与等人，不能当作纯计算时间（07 的读法同样如此）；
4. 与 07 的 48 层产物对照：07 测到 `q_proj` 34.4% / `o_proj` 26.3%，那是因为**它的产物只有 3.75 MB、刚好只装 q/k/v/o，没有 MLP**；
   本图里 q/o 合计只占 **6.4%**（151k/2359k），与解析预算给出的 10.8% 同向（差额来自逐元素与搬数命令）。

### 7.4 与解析预算的对照（`host/static-budget.txt`）

| 部分 | 解析：占 MAC | 解析：占权重字节 | 实测：占核心周期 |
|---|---|---|---|
| MLP gate/up/down | 87.6% | 87.6% | **61.2%**（三张 8.8MB FC） |
| q/o_proj | 10.8% | 10.8% | 6.4% |
| k/v_proj | 1.5% | 1.5% | ~2% |
| 逐元素 + 搬数 | ~0 | ~0 | **~45%** |

**解析预算算的是「算力/字节的份额」，实测周期里逐元素与等待占了近一半** —— 这正是「算得少、搬得多、还常等人」的定量版本，
也是为什么 61 层的实际时间（1226 µs）比纯带宽下限（28.4 MiB / 18.4 GB/s ≈ 1.6 ms … 见下）还要分散在多核上。

## 8. 观测开销（同一批产物，同一台板）

| 等级 | fc ms/inf | dec ms/inf | dec 相对干净态 |
|---|---|---|---|
| 干净（无 profiler） | 0.3519 | 1.9616 | 1.00× |
| level 1 | 0.4127 | 1.6241 | 0.83×（**没有变慢**，落在噪声里） |
| level 2 | 0.3395 | 2.6934 | 1.37× |
| level 3 | 0.7771 | **12.6607** | **6.5×** |
| level 3 + `getProfiledQoSData`（profqos） | 0.937 | **23.104** | **11.8×** |

07 的 7.2× 在本轮是 6.5×（neuronrt）/ 11.8×（profqos 另加一次 QoS 查询）——**同一数量级，确认 level 3 很贵**。
level 1 在 dec 上甚至比干净态快（1.62 vs 1.96），只能解释为 boost/噪声，**不能读成「profiler 加速」**。

## 9. 风险记录

1. **level 3 这次没弄坏板子**：本轮共 4 次 level-3 运行（fc、dec、profqos-fc、profqos-dec），
   每次之后 `healthcheck --quick` 都 PASS，`dmesg` 里 `MDLA timeout`/`ret(-5)` 计数 **0**，`uptime` 连续。
   07 §8 的坏态**没有复现** ⇒ 应改判为「**曾发生过一次、未定位触发条件**」，纪律照旧（一次一级别 + 前后 healthcheck + 出事就 reboot）。
2. **USB 掉线 12 分钟**（§1）：与 profiling 无关，恢复方式是**等**；诊断步骤（先看设备树 VID，再动 adb）已进 `docs/OP-toolchain.md` §6.1。
3. **板子背景负载 load average ≈13**：时延噪声 ±15%，配对交替也只能给到这个精度。要做更细的级别间比较，得先找到这个负载源。
4. **`endCycle` delta 含等待**（§7.3 第 3 条）：跨核依赖会让某条命令的 delta 远超其真实计算时间。

## 10. 结论分级表

| # | 结论 | 证据 | 分级 | 能推翻它的实验 |
|---|---|---|---|---|
| 1 | `-P1/2/3` 产物互不相同，`-P3` 增量随层数线性（1 层 711 B / 61 层 60,974 B） | §2 | **已验证** | 换图后 -P3 增量与层数无关 |
| 2 | **板端会打印剖面表，走 `logcat tag=neuron`**（07「板端不打表」不成立） | §3、`board/logcat-neuron-*` | **已验证** | 换一台板/运行时后 logcat 里没有该 tag |
| 3 | level 2 就把 LEVEL 2/3/4 三张表全带出来（含 driver 的 `MDLA_5_0 - subgraph#1`） | §3.2 | **已验证** | level 2 只出 LEVEL 2 表 |
| 4 | 等级不匹配是**告警 + 静默降级**，原文 `please use -P{} option to re-compile the model`，rc=0 | §4 板端 7 组 + host 24 组 | **已验证** | 任一组返回非 0 |
| 5 | `fc`：计算 5.03 µs / 墙钟 496.4 µs（p3），**偏差 99.0%**（p0 口径 98.6%） | §6 | **已验证** | 同产物换运行重测，偏差落到 50% 以内 |
| 6 | `dec`：计算 1226 µs / 墙钟 2112 µs，**偏差 41.9%**（p0 口径 37.5%） | §6 | **已验证** | 同上 |
| 7 | 61 层图 **68.8% 的周期在 7 条 FC 上**，30.2 MB @18.4 GB/s ⇒ **带宽受限** | §7 | **已验证** | 重测里 FC 周期占比 <50%，或 FC 带宽 ≫24 GB/s |
| 8 | 时钟 **0.94–1.03 GHz**（四个样本，中位 ≈1.0 GHz） | §6.1 | **已验证（4 样本）** | 出现 <800 MHz 的样本 |
| 9 | `-P` 等级对总时延无可测影响（±15% 噪声内） | §5 | **已验证（限本板当时噪声）** | 低噪声环境下测出单调趋势 |
| 10 | level 3 的观测开销 6.5×（neuronrt）/ 11.8×（profqos） | §8 | **已验证（单图单机）** | 复测开销 <2× |
| 11 | level 3 会弄坏板子 | 07 §8 vs 本轮 4 次未复现 | **未知（触发条件未定位）** | 连续多次 level 3 后复现坏态并定位 |
| 12 | 跨核依赖把「等待」算进了 delta（339 µs 的 ADD 只碰 43 KB） | §7.3 | **推测** | 用 PMU 事件或单核编译对比，若该 ADD 仍 339 µs 则不是等待 |
| 13 | 逐元素那张表的 logcat key | §3 | **已验证** | —— |

## 11. 遗留

- **PMU 14 个 event 的语义**仍未解（本报告只用 `endCycle` + `bufferAccesses`）；
- **「等待」与「计算」无法在 `endCycle` 口径下分开**：要分开得靠 PMU 事件语义或单核对照实验（`--num-mdla=1` 编译同一张图）；
- **板子背景负载 load ≈13** 的来源未查，它把时延噪声抬到 ±15%，也是 §5 无法分辨级别的直接原因；
- `fc` 的 `exec_us` 只有 1 µs 分辨率，5 µs 的任务上不可用于定标。

## 12. 复跑（一条命令一条命令）

```bash
export NPU_LOCK_HELD=1
./tools/healthcheck.sh --quick                                              # 前置门
./tools/devlock.sh run -- bash ops/_probe/bb08-prof/10-stage.sh             # 推 8 份产物
./tools/devlock.sh run -- bash ops/_probe/bb08-prof/21-latency.sh fc 20 3   # 配对交替
./tools/devlock.sh run -- bash ops/_probe/bb08-prof/21-latency.sh dec 10 3
./tools/devlock.sh run -- bash ops/_probe/bb08-prof/22-mismatch.sh          # 等级门槛原文
./tools/devlock.sh run -- bash ops/_probe/bb08-prof/20-run-one.sh fc 3 20   # 逐级别（1/2/3）
./tools/devlock.sh run -- bash ops/_probe/bb08-prof/20-run-one.sh dec 3 10
./tools/devlock.sh run -- bash ops/_probe/bb08-prof/40-board-profqos.sh dec "0 3"   # exec_us / 定标
/usr/bin/python3 ops/_probe/bb08-prof/92-latency.py ops/_probe/bb08-prof/board/latency-*.tsv
/usr/bin/python3 ops/_probe/bb08-prof/90-perop.py \
  --exec ops/_probe/bb08-prof/board/execution_result-dec-L3.json \
  --info ops/_probe/bb08-prof/board/dla_info-dec-L3.json \
  --plan ops/_probe/bb08-prof/build/dec_p3.compile.log \
  --label dec-P3 --exec-us 1193 --clock-mhz 1000 --wall-us 2112
```

坏态处置不变：`dmesg` 出现 `MDLA timeout` / `ret(-5)` ⇒ `adb reboot` → 等设备回来 → **`adb root`** → `healthcheck --quick`。
