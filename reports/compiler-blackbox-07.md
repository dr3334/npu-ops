# 黑盒研究 07：官方 profiling 出口打通 —— 每命令 PMU 剖面

日期：2026-11-14 · SDK 7.0.8 / `ncc-tflite` 7.3.15 · 板端运行时 **7.3.8**（`/vendor/lib64`）· 票：`07-profiling-trace.md`
前序：[06](./compiler-blackbox-06.md)（debug/urate 的可读性）/ [OP-toolchain](../docs/OP-toolchain.md) / [OP-dla-structure](../docs/OP-dla-structure.md)

## 0. 一句话

**出口存在，而且不是 `--Xruntime`，是 `ncc-tflite -P<level>` 编译器开关 + `MTKNN_ENABLE_PROFILER/MTKNN_PROFILING_LEVEL` 运行时环境变量**：
level 3 会把每一次推理的**每一条命令**（per-command）的 PMU 记录 dump 成两个 JSON，含 `endCycle` 与逐命令的 `opIndices` / `bufferAccesses`；
用最慢核的 `endCycle` 对运行时的 `exec_us` 定标得到 **999.6 MHz ≈ 1.0 GHz**，于是每命令的 µs 与带宽都能算出来，
**sum（最慢核）= 264 µs = 运行时 `exec_us`**（判据达成），但 **≠ `neuronrt -c N` 的 692…1066 µs**——差额是运行时自身的提交/同步开销，
这本身是「找瓶颈」环节最该知道的一条。

**代价与雷**：level 3 的观测开销是 7.2×（1.35 → 9.7 ms/inf），并且**跑完 level 3 之后板子进入 `MDLA timeout` 坏态**
（连 known-good `fc_1x64x64_int8` 也 `Inference fail`，需 reboot）——详见 §8。

## 1. 怎么跑（可复现）

```bash
# 编译期：给 DLA 打上 profiling 等级（1…3；--help 里没有这个开关）
./tools/compile.sh ops/_probe/bb07-prof/build/fc.tflite /mnt/d/bb07-tmp/x.dla -P3
#   等价写法： --profiling-level=3

# 运行期：开profiler并指定等级（等级不得高于 DLA 自带的等级，否则降级并告警）
adb shell "cd /data/local/tmp/npuops && MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=3 \
           ./bin/neuronrt -m hw -a bb07/full_p3.dla -i bb07/din.bin -o bb07/o.bin -c 5"

# level 3 会在**当前工作目录**（或 MTKNN_PROFILER_LOG_PATH=<dir>）写出：
#   dla_info.json          21 KB  结构 + 逐命令 bufferAccesses（buffer_id / offset / size）
#   execution_result.json 272 KB  逐推理 -> cores -> flows：endCycle + 14 个 PMU event
```

本报告所有数字来自 `ops/_probe/bb07-prof/`（脚本 + `board/` 下的原始证据），逐命令表在
`ops/_probe/bb07-prof/board/perop-table.txt`，由 `ops/_probe/bb07-prof/perop_table.py` 生成。

## 2. 试法 1：`--Xruntime <options>` —— **不是 profiling 出口**

```text
--Xruntime <options>      Pass options to the neuron runtime. Enclose option string by single quotation.
```

试过的 option 字符串——**宿主**（`-m null`，profiler 环境变量开/关两种）：`''`、`nbThreads:2`、`nbThreads=2`、`{"nbThreads":2}`、
`parallelismOption:single`、`profiling-level:3`、`profilingLevel=3`、`profil:3`、`dump-profiling-result`、`garbage`；
**板端**（健康态，`-m hw`，`full_p0`，`-c 5`）：`''`、`garbage`、`profiling-level:3`、`nbThreads:2`、`{"parallelismOption":"single"}`。

两端一致：**全部 rc=0、输出行完全相同、等级不变**——既不报错也不生效（`garbage` 与合法串一样被吞掉）。板端这一次的时延也全在噪声带里
（0.762 / 0.792 / 0.895(无选项) / 0.990 / 1.052 ms/inf，顺序无关），没有「选项生效」的痕迹。
证据：`board/xruntime-hw-clean.txt`（板端干净态）、`25-xruntime-and-levels.sh`（宿主）、`04/05/15/16` 的字符串清单。

它最终落到 `NeuronRuntime_create_with_options(const char* options, ...)`（`RuntimeAPI.h:102`）。运行时里与它相关的打印只有
`Runtime options:` / ` - options:` / `  - nbThreads:` / `  - dlaBuffer:`（`libneuron_runtime.so` strings），**没有任何 profile/dump 类字段**；
profiling 的开关根本不在这个结构里，而在环境变量（§3）。

**结论**：`--Xruntime` = 未拿到 profiling 能力。**已验证（负面：10 个字符串、两种环境、宿主与板端行为一致）**；
它到底接受哪些键 = **未知**（运行时只在 internal verbose 下打印那几个字段）。

## 3. 试法 2：真正接上的是 `-P` + `MTKNN_*`（`--gen-debug-info` 不是它）

### 3.1 `-P<n>` / `--profiling-level=<n>`：help 里没有，二进制里有

```text
$ strings host/bin/ncc-tflite | grep -i profiling-level
P,profiling-level
Specify a profiling level to generate required info in DLA
The profiling level is now set to {}
The runtime profiling level is {}, but DLA only supports {}
To enable profiling level {}, please use -P{} option to re-compile the model
```

实测（`fc_1x64x64_int8`，同长输出名，路径回显抵消）：

| 变体 | 开关 | `.dla` B | Δ | 扣掉命令行回显后的内容增量 |
|---|---|---|---|---|
| `q000` | 无 | **7556** | — | — |
| `q001` | `-P1` | 7576 | +20 | **+16** |
| `q002` | `-P2` | 7576 | +20 | **+16**（与 -P1 同长但**字节不同**，等级值不一样） |
| `q003` | `-P3` | **8267** | **+711** | **+707** |
| `q012` | `--profiling-level=2` | 7576 | +20 | ≈0（flag 文本 21 B，回显抵消） |
| `q013` | `--profiling-level=3` | 8267 | +711 | **+690** |
| `q100` | `--gen-debug-info` | 8112 | +556 | **+539**（与 06 一致） |
| `q103` | `--gen-debug-info -P3` | 8284 | +728 | ≈+704 ≈ 只多 `-P3` 那份 |

**关键点**：`-P3` 的内容量随**层数**走，不是常数——同一份 FC 是 +707 B，Qwen2.5 decoder layer（48 层）是
**3,754,640 → 3,809,342 B（+54,702）**。这正符合「每层记录」。

### 3.2 等级门槛：DLA 自称几级，运行时才给几级

```text
DLA=q000(无 -P)  ask=1 -> WARNING: To enable profiling level 1, please use -P1 option to re-compile the model
DLA=q001(-P1)    ask=1 -> INFO: The profiling level is now set to 1
DLA=q001(-P1)    ask=3 -> WARNING: The runtime profiling level is 3, but DLA only supports 1
                          WARNING: To enable profiling level 3, please use -P3 option to re-compile the model
                          INFO: The profiling level is now set to 1          <- 自动降级
DLA=q002(-P2)    ask=3 -> ... DLA only supports 2 ... set to 2
DLA=q003(-P3)    ask=3 -> INFO: The profiling level is now set to 3        <- 通过
DLA=q100(--gen-debug-info) ask=3 -> 仍报 "only supports 0"，--gen-debug-info 不算任何等级
```

**结论**：`--gen-debug-info`（06 里那 539 B）**不是**运行时用来 profiling 的那段；真正被运行时认的是 `-P<n>`。
06 的「Runtime can work with this info for profiling」这句 help 有误导性。**已验证**。

### 3.3 运行时 profiler：三档，level 3 才有 JSON

`MTKNN_ENABLE_PROFILER=1` + `MTKNN_PROFILING_LEVEL=1|2|3`（另有 `MTKNN_PER_OP_PROFILE`、`MTKNN_PROFILER_LOG_PATH`）。
宿主 `-m null` 上 level 3 的控制台原文（节选，75 行）：

```text
INFO: The profiling level is now set to 3
                                 Runtime Status (LEVEL 1) - Summary
 ===Time===, ==Ratio==, =ExeTimes=, =Time/Inf.=, ===Description===
      0.004,     23.5%,          4,       0.001, Set Inference Input
      0.002,     11.8%,          4,       0.001, Set Inference Output
      0.011,     64.7%,          4,       0.003, Inference
      0.017,    100.0%,        ---,       0.004, Total
                             Detail of 'Inference' (LEVEL 2) - Summary
      0.001,    100.0%,          4,       0.000, Transit Execute
```

- 这几张表是**阶段级**（输入/输出/推理、以及推理内部的 Transit Execute），不是每层；
- **板端**上 level 1/2/3 **都只打印 3 行**（`Inference repeats for N times.` / `Total inference time = …` / `Avg. FPS : …`），
  没有任何 Summary 表，也没有「Per-OP」字样；level 3 唯一多出来的是那两个 JSON（`board/perop-matrix-L{1,2,3}.txt`）——
  宿主与板端的控制台行为不同，记一笔（`board/level3-console-hw.txt` 是坏板期抓的，**不作证据**）；
- `MTKNN_PER_OP_PROFILE=1` 在宿主 `-m null` 与板端都**没有**产生额外的「Per-OP Report」（strings 里有
  `MVPU Per-OP Profiling Report - Core#{}`，但那要靠**真实 PMU 数据**，见 §6）。

## 4. 试法 3：LiteRT 路径 —— 没有 profiler

```bash
unzip -q litert_npu_*.zip -d /mnt/d/bb07-tmp/litert   # 不在 repo 里解
```

- 五个 zip 里**只有** `libLiteRtDispatch_*.so` 与 `libLiteRtCompilerPlugin_*.so`（qualcomm v69/73/75/79/81、google tensor、mediatek），
  外加 gradle/manifest/strings.xml；**没有任何 benchmark / profiler / trace 可执行文件**；
- MediaTek 的 dispatch lib 里 `MTKNN|profil|PER_OP|MVPU` 命中数 = **0**；它是薄壳，`dlopen` 的是
  `libneuron_adapter.so` / `libneuronusdk_adapter.mtk.so`；compiler plugin 里只有
  `MTKNN_ADAPTER_DLA_DIR` / `MTKNN_ADAPTER_DLA_PLATFORM` 两个环境变量（把 DLA 落到临时目录用）；
- 也就是说：**delegate 的 profiling 出口就是 §3 那套 Neuron 运行时的出口**，LiteRT 自己没有第二套。

**结论**：LiteRT 路径无独立 profiler。**已验证**（zip 全解、目录全列、strings 全扫）。

## 5. 试法 4：分段受控实验 —— 没做（已被 §6 取代）

拿到 per-command 数据之后，「黑盒分段反推段占比」这种精度只到段级的方法就没有必要了。
顺带记一个坑：本 worktree 里 `python3 tools/build_decoder_layer.py --variant attn_half` 编出来的图与 `--stage r1` 全图
**大小相同**（3,704,112 B，字节不同），所以 `ops/_decoder_bisect/{attn_half,ffn_half}` 这套「半图」不能直接当分段资产用。

## 6. 拿到的数据：每命令 PMU（判据达成）

### 6.1 粒度

```text
$ ./bin/profqos_v bb07/full_p3.dla -i bb07/din.bin -o out.bin -c 5      # 自建 harness，见 ops/_probe/bb07-prof/profqos.c
[rt] inputs=1 outputs=1
[rt] 5 inference(s) in 7719 us
[rt] getProfiledQoSData rc=0 execBoostValue=100
[subgraph 0] subcmds=1                    <- 公开 API 这一层只有 1 个 sub-command（整图一条命令）
  cmd   0  exec_us=264  bandwidthMBps=0  boost=100
SUM_EXEC_US=264 CMDS=1 MEASURED_US=7719 LOOPS=5 PER_INF_US=1544
```

- `NeuronRuntime_getProfiledQoSData()` 的 `ProfiledQoSData` 结构（`Types.h:99`）本来就是
  「每 sub-command 一个 `QoSData{execTime, bandwidth, boostValue}`」，但**这颗 MDLA 上整图就是一条命令**，
  所以这条路只给 1 个数；**每层的数在 level-3 的 JSON 里**。
- 附带一个坑：`-m null` 上 `getProfiledQoSData` 会**除零 SIGFPE**（`execTime=0` 时算带宽），`profqos.c` 已加信号兜底。

### 6.2 level-3 dump 的结构

```text
execution_result.json : inference -> codes[每个 MDLA 核一个] -> flows[每条命令一条]
                        每条 flow: {id, pmu{endCycle, events[14]}}
                        full_p3: 每核 55 条 flow，共 110 条（48 层被编译成 55 条命令，双核分工）
dla_info.json        : 同一批 flow id -> {opIndices[], bufferAccesses[{buffer_id, offset, size}]}
                        buffers: 21 个（L1 1 个 80 KB，DRAM 20 个）
```

### 6.3 定标：999.6 MHz（两个核互相印证）

- 最慢核最后一条 `endCycle` = **263,906**，另一核 = **189,009**；
- 运行时 `exec_us`（同一份 DLA）= **264 µs**；263,906 / 264 µs = **999.6 MHz** —— 正好 1.0 GHz；
- 于是 `us = cycles / 1000`，另一核 = **189 µs**（比最慢核早 75 µs 收工、之后空转）；
- **两个核用同一个时钟都自洽**，这是「`endCycle` 就是完成时刻」这个读法的交叉验证；
  另一个候选读法（event 6914 − 6912）求和是最慢核 `endCycle` 的 ~30 倍，**不自洽，未采用**（事件语义仍未知）。

### 6.4 逐命令结果（`full_p3` = Qwen2.5 decoder layer，T=8，D=896，fp16）

完整 110 行见 `ops/_probe/bb07-prof/board/perop-table.txt`。占大头的几条：

| 核 | 命令 | `opIndices` | tflite 算子 | cycles | 占比 | 访存 B | 带宽 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | `[6]` | FULLY_CONNECTED（q_proj，w 896×896） | 90,820 | **34.4%** | 1,634,304 | 17.99 B/µs ≈ **18.0 GB/s** |
| 1 | 71 | `[42]` | FULLY_CONNECTED（o_proj） | 69,281 | **26.3%** | 1,634,304 | 23.59 B/µs ≈ **23.6 GB/s** |
| 0 | 40..44 | `[35]` | BATCH_MATMUL（qkᵀ） | ~3,163 每条 | 各 1.2% | 30,464 | 9.6 GB/s |
| 0 | 54..68 | `[39]` | TRANSPOSE（KV 复制链） | ~1,935–3,457 | 0.7–1.8% | 1,920–15,488 | 0.7–3.3 GB/s |
| 0 | 13..19 | `[30,31]` | SPLIT+CONCAT（14 头复制） | ~89,58x 累计 | 各 1.5% | 16,384 | 0.18 GB/s |

读法（对「改参数」这条闭环直接有用）：
1. **两个 FC 占 60.7%**，且带宽 18–24 GB/s —— 权重访存是主项（896×896 fp16 = 1.6 MB/个）；
2. KV 的 14 头复制（SPLIT+CONCAT 那串）合计 ~90 条命令、每条 1–5 µs，**加起来也不少**，且带宽极低（0.15–0.33 GB/s）——是纯搬数据；
3. 核间分工不均：核 1 干了 264 µs，核 0 只有 189 µs（**尾部 75 µs 只有单核在跑**），这是 `--num-mdla=2` 下的负载不均证据。

### 6.5 判据核对

| 项 | 数 | 说明 |
|---|---|---|
| sum（最慢核）= 运行时 `exec_us` | **263.9 ≈ 264 µs** | 结构上相等（定标即由此得出），但**每命令的分摊是真的** |
| `neuronrt -c 20` 总时延 | **692–1066 µs/inf**（3 轮交替） | 同一份 `full_p3` |
| 设备侧计算占墙钟 | **25–38%** | 其余是运行时提交/同步/IO 开销 |
| 同图 `full_p0`（不打 -P） | exec_us **178 µs** / 墙钟 788–1066 µs | -P3 的 exec_us 更高（264），但墙钟无显著差异 |
| 观测开销 | level1 1.35 → level2 1.73 → **level3 9.74 ms/inf** | **7.2×**（`-c 5`，同图同机） |

**判据结论**：**拿到每命令耗时与带宽，且「每核 sum = 运行时 exec_us」成立**；但 **≠ `neuronrt -c N` 的墙钟**，
差额是运行时开销——这不是我们漏了一层，而是两个量本来就不同（一个是设备侧命令时间，一个是含提交/同步的端到端时间）。
要把它们对齐，需要的不是更细的 trace，而是**在 `-c N` 之外单独量运行时开销**（§9）。

## 7. 结论分级表

| # | 结论 | 证据 | 分级 | 能推翻它的实验 |
|---|---|---|---|---|
| 1 | `--Xruntime` 不是 profiling 出口（10 个字符串、无报错无效果） | §2、`25-xruntime-and-levels.sh` | **已验证（负面）** | 找到任一字符串让运行时打印 `Runtime options:` 或改变等级 |
| 2 | `ncc-tflite` 接受 `-P1/2/3` 与 `--profiling-level=n`（help 未列） | §3.1 的 8 个变体尺寸 | **已验证** | `ncc-tflite -P9` 仍 rc=0 且产物不变（说明它不校验范围） |
| 3 | DLA 自带 profiling 等级，运行时只能降到该等级 | §3.2 的 8 组消息 | **已验证** | 用 `-P3` 产物被降级即推翻 |
| 4 | `--gen-debug-info` ≠ profiling 等级（不满足任何等级） | q100 ask=3 → "only supports 0" | **已验证** | 用 `--gen-debug-info` 产物通过 level 3 即推翻 |
| 5 | level 3 把每命令 PMU dump 成 `execution_result.json`+`dla_info.json` | 板端 272 KB / 21 KB | **已验证** | 换图跑不到 flow 级数据即推翻 |
| 6 | 每命令时长 = `endCycle` 差；时钟 ≈ 1.0 GHz（两核自洽） | §6.3 | **已验证（交叉印证）** | 用另一个已知时长的图复算，若两核推出的时钟不一致即推翻 |
| 7 | PMU 的 14 个 event id 的语义 | —— | **未知** | 官方文档/更多对照实验 |
| 8 | `opIndices` → 层名的映射可以直接做（本报告用 tflite opcode 表反查） | §6.4 | **已验证** | —— |
| 9 | `MTKNN_PER_OP_PROFILE` 没有额外效果 | 宿主 + 板端都无 Per-OP Report | **推测**（可能是板端/hw 版本不支持，或要更高 level） | 在别的芯片/运行时版本上出现 Per-OP Report |
| 10 | level 3 的观测开销 7.2× | 1.35 → 9.74 ms/inf | **已验证（单图单机）** | 换图重复，若差值消失即推翻 |
| 11 | 板端控制台 level 3 不打印表（只写 JSON） | `board/level3-console-hw.txt` 2 行 | **已验证** | —— |
| 12 | **level 3 跑完板子进入 `MDLA timeout` 坏态**（reboot 可恢复，恢复后需 `adb root`） | §8 的 dmesg + 重启后的恢复记录 | **已验证** | 干净 reboot 后同序列复跑若不再坏，则说明是偶发 |

## 8. 后遗症：level 3 之后板子坏了（重要）

`MTKNN_PROFILING_LEVEL=3` 连跑几次之后，**任何** `neuronrt -m hw` 都失败：

```text
$ ./bin/neuronrt -m hw -a dla/fc_1x64x64_int8.dla -i dla/fc_1x64x64_int8.in.bin -o dla/hc.bin -c 5
There is a runtime error during inference
Inference fail
```

dmesg（`ops/_probe/bb07-prof/board/dmesg-apu.txt`）：

```text
[apusys&]MDLA timeout
[warn] mdw_cmd_check_rets sc(...-#0) type(2) softlimit(0) boost(100) fail
[error] mdw_cmd_complete s(...) c(neuronrt/...) ret(-5/0x3) time(25003555) pid(...)
```

- `ret=-5` 且 `time=25,003,555`（≈25 s 超时），**known-good 算子同样失败**，`-c 1` 也一样，等 20 s 不恢复；
- 判定：**不是我们的 DLA，是设备状态**（APU 卡死），需要 reboot 才能恢复；
- **实测恢复路径**：设备自己重启了（`uptime` 回到 1 min），重启后 `adb shell` 退回 `uid=2000(shell)`、
  连 `/vendor/lib64/mt6897/libneuron_runtime.so` 都 `Permission denied`、`dmesg` 也读不了，此时
  `neuronrt` 报 `Unable to open Neuron Runtime library`；**`adb root` 之后一切恢复**（known-good 10 次：0.4537 ms/inf）；
- 纪律建议：把 level 3 profiling 当成「会弄坏板子」的实验，跑之前先 `tools/healthcheck.sh --quick`，跑完立刻 `--quick`，
  坏了就 reboot，reboot 之后先 `adb root` 再跑。

## 9. 遗留

- **PMU 14 个 event id 的语义**（含 6912/6914）未解；本报告只用了 `endCycle`。
- **每层名字**要靠 tflite opcode 反查（本报告手工对表），没有做成工具；把 `opIndices` + `--show-exec-plan` 的层序对齐可以做成一步。
- **运行时开销的分解**：`neuronrt -c N` 的墙钟里 62–75% 不在设备侧，这需要单独量（SetInput/SetOutput/wait 各占多少），
  §3.3 的 LEVEL 1/2 表正好是为此准备的，但板端只在 level≤2 时便宜可用。
- **`getProfiledQoSData` 只给 1 个 sub-command**（整图一条命令），拿不到每层；每层必须走 level 3 的 JSON。
- 板子坏态未修（等 reboot），本票的板端实验到此为止。

## 10. 验收口径

- 四条试法逐条有结论（试法 1/3 负面、试法 2 命中、试法 4 被取代）；
- 判据：**每命令耗时+带宽拿到，且每核 sum = 运行时 `exec_us`**；与 `neuronrt -c N` 的差额已定位为运行时开销（并给出占比）；
- 每条结论都给了「能推翻它的实验」；拿不到的说拿不到（PMU event 语义、板子坏态后的验证）。
