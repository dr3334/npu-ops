# OP-kv1 —— KV cache 上设备：v1 API 不行，v2 API 行（但每 token 一次 24 MiB 的 CPU 拷）

> 分支 `impl/kv-resident`，worktree `wt-kv1`；前驱 docs/OP-q06-prefill-decode.md（票 06）、
> docs/OP-vendor-runtime.md、docs/adr/0006-execution-form-shared-weights.md（本票给它加了一条订正）。
> 设备 mt6897 / Android 16（0123456789ABCDEF），板端全部持 `/d/deepseek_ workspace/mtk/.npu-board.lock`，
> 每个板端块前后各一次 `healthcheck --quick`（全部 PASS）。公共工具与 `.scratch/` 一行未改；
> 新增 `tools/kv1_runner.c`、`tools/kv2_runner.c`、`tools/build_kv1.sh`、`tools/e2e_kv1.sh`。

---

## 0. TL;DR

**目标达成，但要走 RuntimeV2，而且它每 token 会把 24 MiB 的 KV 在 CPU 上重拷一次。**

| 问题 | 答案 | 证据 |
|---|---|---|
| KV 能不能常驻设备、每 token 只写新行 | **能**——用 `NeuronRuntimeV2_run` 每请求送 `IOBuffer{fd,offset}` | §6：8 个 token 每步 hidden/k/v **全部逐字节等于** CLI 路径 |
| 那 RuntimeAPI v1（`setInput`）行不行 | **不行**：绑一次用户 dma-buf 之后，图输入就"死"了，写、重绑都无效 | §5：毒化 0/20000 输出 sha256 相同；ping-pong 换 fd 无效；预置缓存的控制组逐字节一致 |
| 每 token host 写多少 | **12,288 B**（K+V 各 48×128 B）vs CLI 形态 **25,640,088 B**（≈2085×） | §7.4 |
| 有没有额外拷贝 | **有**：v2 每请求把 24 MiB 从用户 buffer 拷进运行时 staging，**CPU 12.95 ms/invoke** | §7.2 `/proc/self/stat` |
| 每 token 的时间（配对交替） | CLI 每 invoke **59.1 ms**（斜率法）/ v1 **49.4 ms**（但数值不成立）/ **v2 62.9 ms** | §7.1 |
| 端到端每 token | CLI 形态 **≈1.2–1.9 s**（每 token 一个进程 + adb 推 25.6 MB）；v2 常驻 **62.9 ms** | §7.3 |
| 稳定性 | v2 400 次：median **62.84 ms**、p10 58.4 / p90 67.0；输出 400 次 == 1 次 | §7.5 |

一句话：**"KV 常驻设备"在 RuntimeV2 上是成立的（数值 8/8 步逐字节成立），代价是运行时每 token 把
24 MiB 再拷一遍（CPU 13 ms），换来的是把"每 token 一个进程 + 25.6 MB 搬运（1.2–1.9 s）"变成
"一个进程、每 token 12 KB 写入（62.9 ms）"。**

---

## 1. 两条 API 路线（带行号）

| 事实 | 位置 |
|---|---|
| v1：用户提供输入 buffer `setInput(runtime, handle, buffer, length, BufferAttribute)` | `RuntimeAPI.h:144` |
| v1：带偏移的变体 `setOffsetedInput(..., size_t offset)` | `RuntimeAPI.h:159` |
| `BufferAttribute` 只有 `int ionFd`；`-1` = 非 ION；注释："may benefit from this information to **eliminate unnecessary data copy**" | `Types.h:201-209` |
| **v2：每次请求都重新给 buffer** —— `IOBuffer{buffer, length, int fd, int offset}`，随 `SyncInferenceRequest` 传给 `NeuronRuntimeV2_run` | `RuntimeV2.h:23-37`、`64-72`、`174` |
| 没有"取回运行时内部输入 buffer 指针"的 API（只有 size / paddedSize / rank / dims） | `RuntimeAPI.h:252-424` |

设备侧：`/dev/dma_heap/mtk_mm-uncached` 存在（正是厂商那份 239 MiB 权重 dma-buf 的 exporter，
OP-vendor-runtime §6）；`/vendor/lib64/libneuron_runtime.so` 自报 **7.3.8**，装载 ncc 7.3.15 编的 `.dla` 正常，
并导出了 `NeuronRuntimeV2_create/run/release`（`strings` 计数 32）。

---

## 2. 做法：两个最小 aarch64 runner

都是 NDK r26d 编、只依赖 libc/libdl、运行时 `dlopen` 设备上的 `libneuron_runtime.so`（不钉版本）。

- **`tools/kv1_runner.c`（v1 路线）**：`loadNetworkFromFile` → 每个输入一块常驻 buffer（KV 用
  `DMA_HEAP_IOCTL_ALLOC` 从 `mtk_mm-uncached` 分配 + `mmap`）→ `setInput` **绑一次** →
  循环 invoke，每 token 只把新行（48×128 B/张）写进 dma-buf。
- **`tools/kv2_runner.c`（v2 路线）**：`NeuronRuntimeV2_create` → 同样两块常驻 dma-buf →
  **每次 `run` 都构造 `IOBuffer{ptr,len,fd,offset}` 数组** → 循环 invoke，每 token 同样只写新行。

两者的 IO 表相同（9 输入 3 输出，**padded == logical，连 `suppressInputConversion` 都不需要**）：

```
[in 6] size=12582912 padded=12582912 rank=4   <-- KV(k)      [out 1] size=49152 <-- k_out [48,8,64]
[in 7] size=12582912 padded=12582912 rank=4   <-- KV(v)      [out 2] size=49152 <-- v_out
[in 8] size=16674816 padded=16674816 rank=0   <-- 外部权重 bin
```

---

## 3. 8 步对拍怎么搭的（可复现）

`tools/e2e_kv1.sh --steps 8`：**同一份 prefill 输出**做种子，两条路径跑同一条 8 步序列，
**每步都比、比 3 张张量**：

```
phase 0  prefill 一次（CLI，未改）        -> 拉 out_hidden/out_k/out_v
phase 1  build_q06.py --seq-init         -> 两条路径同种子、同 step0 输入；clone 状态
phase 2  BASE  8 步：每步 push 8 个输入文件 + neuronrt -c 1（step0 用 -c 20）
phase 3  KVRES 8 步：一个进程、KV 在 dma-buf、每步只写新行
phase 4  逐字节 cmp（8 步 × 3 张量）
phase 5  两条路径各自与同一套 numpy golden 对拍
```

**可复现性**：本次重建 tflite/.dla/bin 打出来的 BASE 8 步与票 06 §5.2 的表**逐位相同**
（370/84.56、135/27.76、143/36.90、98/21.68、144/34.31、67/17.93、152/36.43、249/59.76），
并补齐了票 06 的 UNKNOWN 8：**decode 各步 rel_l2 0.00251–0.01280（门 0.05，PASS）**。

---

## 4. 结论一：v2 路线把目标做成了（8 步逐步逐字节）

`tools/kv2_runner` 用同一套 `--steps` 序列跑，与 CLI 基线逐字节比：

```
[v2] create(q06/decode.dla, threads=1) rc=0
[v2 in 6] dma-buf fd=14 len=12582912
[v2 in 7] dma-buf fd=15 len=12582912
step 00: hidden=SAME k=SAME v=SAME
step 01: hidden=SAME k=SAME v=SAME
...
step 07: hidden=SAME k=SAME v=SAME
```

**24/24 张量逐字节相同**（不是"cos 很像"，是 `cmp` 过）。也就是说：
KV 缓冲**一次分配**、进程内**反复使用**，host 每 token 只写 48×128 B/张，设备确实读到了新行。

---

## 5. 结论二：同一件事在 v1 的 `setInput` 上做不到（三个实验）

v1 runner 跑同一条序列：**step 0 三条张量全同，step 1 起全不同**。三个实验定位到原因。

### 5.1 毒化实验

把新行写成常量 **int16 20000**（而不是图输出的那一行），跑 2 步，再把 KV 的 dma-buf **读回来**：

| 检查 | 结果 |
|---|---|
| 读回 cache 与种子差多少字节 | **正好 6,144 B** = 48×64×2 = 恰好一行/层，位置 `p*C*128 + 16*128` |
| 种子 row16 | `0 0 0 0` |
| 写后 row16 | `20000 20000 20000 20000` |
| **step1 hidden（写 0 vs 写 20000）** | **sha256 相同** `4e748e43bc1b13af…` |

=> 写确实落到该落的地方、且只落那里；**下一次 invoke 没读它**。

### 5.2 换一块 dma-buf 重新 setInput（ping-pong）

再分配一对 dma-buf（fd=30/31），每次写完把整块 cache memcpy 进去，用**新 ptr+fd** 重新 `setInput`：

```
[pingpong] alt buffer for in6 fd=30
[pingpong] alt buffer for in7 fd=31
step-1 hidden: pingpong vs plain = IDENTICAL
```

=> 绑一次之后，后续 `setInput` 对这个 handle 等于空操作。

### 5.3 控制实验：框架本身是对的

把 step 0 的输出行**预先写进种子文件**，起新进程只跑 1 次，与 CLI 的 step 1 比：

```
SAME: runner==CLI when the cache is pre-seeded (framework correct)
```

=> 行映射、mask、输入文件、比较链全对，**唯一不成立的是"进程内更新"**。

### 5.4 另外三个解释都排掉了

| 解释 | 测法 | 结果 |
|---|---|---|
| 需要 `DMA_BUF_IOCTL_SYNC` | `--sync 1` 每次写前后各一次 ioctl | step1 起仍 DIFF |
| 需要 `suppressInputConversion` | `--noconv`（输入输出都关转换） | 仍 DIFF（`--noconv --sync 1` 也试了） |
| 我们写错行/越界 | §5.1 的"差 6144 B、位置正确" | 排除 |

一句话：**v1 的"用户持有 buffer"只是"绑一次时的快照"，不是可更新的句柄。**

---

## 6. 两个 API 的 8 步对拍总表

| step | pos | BASE 码级max | BASE |Δcos| | v1 路线 | v2 路线 |
|---|---|---|---|---|---|
| 0 | 16 | 370 | 8.13e-5 | 逐字节 SAME | **逐字节 SAME** |
| 1 | 17 | 135 | 8.81e-6 | DIFF（码级 39,573 / |Δcos| 1.07） | **逐字节 SAME** |
| 2 | 18 | 143 | 1.34e-5 | DIFF（48,174 / 0.995） | **逐字节 SAME** |
| 3 | 19 | 98 | 5.19e-6 | DIFF（41,265 / 1.033） | **逐字节 SAME** |
| 4 | 20 | 144 | 1.45e-5 | DIFF（34,634 / 0.996） | **逐字节 SAME** |
| 5 | 21 | 67 | 3.13e-6 | DIFF（39,577 / 0.984） | **逐字节 SAME** |
| 6 | 22 | 152 | 1.47e-5 | DIFF（37,603 / 1.064） | **逐字节 SAME** |
| 7 | 23 | 249 | 3.77e-5 | DIFF（41,704 / 1.005） | **逐字节 SAME** |

BASE 侧 8 步 |Δcos| ≤ 8.2e-5（门 0.003）、码级 ≤370（门 1200）、rel_l2 ≤0.0128（门 0.05），
与票 06 一档；v2 侧与 BASE **逐字节相同**，所以这些门它自动满足（它不是在"接近"，它就是同一串字节）。

---

## 7. 性能：三个臂的配对交替

方法照 NOTES §11.5：同一会话、配对交替、报比值中位数。臂：
**A** = `neuronrt` CLI；**V1** = kv1_runner（dma-buf KV，但数值不成立）；**V2** = kv2_runner（dma-buf KV）。

### 7.1 每 invoke（同一会话，N=120，A 用 -c20/-c120 的斜率扣掉进程启动）

| 臂 | 每 invoke | 算法 |
|---|---|---|
| A CLI | **59.1 ms** | (7331−1421)/100 与 (7315−1408)/100，启动 ≈232 ms |
| V1 | **49.4 ms**（median） | 内部单调时钟，n=120 |
| V2 | **62.9 ms**（median） | 内部单调时钟，n=120；n=400 时 62.84 |
| V2（KV 在匿名内存） | 65.2 ms | 同一 runner 只换 `--kv-heap host` |

**比值**：V2/A = **1.07**（v2 每 invoke 比 CLI 慢 7%）、V2/V1 = **1.27**、V2/V2-anon = **0.96**。
读法：v2 的"每请求重送 buffer"不是免费的（§7.2 给出这 13 ms 是什么）。

### 7.2 额外拷贝：v2 每 token 在 CPU 上重拷 24 MiB

`/proc/self/stat` 的 utime+stime 差值（n=400）：

| 臂 | wall/invoke | **CPU/invoke** |
|---|---|---|
| V2（KV 在 dma-buf） | 62.9 ms | **12.95 ms** |
| V2（KV 在匿名内存） | 64.8 ms | 15.18 ms |

12.95 ms 的 CPU 时间 ≈ 25.2 MB / 1.95 GB/s ⇒ **运行时每个请求把 24 MiB 的 KV 从用户 buffer 拷进
自己的 staging**（v1 没有这份拷贝：49.4 ms 与"零 CPU 平移"一致）。这正是 V2 比 V1 慢 13.5 ms 的来源，
也解释了为什么"dma-buf 版比匿名版快 2.2 ms"（少了一次中转）。

所以对任务卡里那三个问题要照实回答：
- （a）invoke 时间：V2 **62.9 ms**（CLI 同会话 59.1、V1 49.4）；
- （b）host 写入量：**12,288 B/token**（K+V 各 6,144）vs CLI 形态 25,640,088 B；
- （c）**有额外拷贝**：运行时每 token 拷 24 MiB（CPU 12.95 ms），不是零拷贝；
- （d）与票 06 的 68.5 ms/token：**同会话配对 A/V2 = 1.07**（V2 慢 7%），见 §7.3 的口径说明。

### 7.3 与票 06 的 68.5 ms/token 怎么比（口径要小心）

票 06 的 68.5 是 `neuronrt -c 20` 的**进程内摊薄值**，它**不含**"每 token 一个新进程 + adb 推 25.6 MB"。
本次同会话 A 臂测得 59.1 ms/invoke（−12%，在 NOTES §11.5 说的 ±15% 漂移内），**不作性能结论**。

真正的"每 token 端到端"（这才是 decode 循环里发生的）：

| 形态 | 每 token | 依据 |
|---|---|---|
| 票 06 / CLI 形态 | **≈1.2–1.9 s**：进程启动 ~0.23 s + 装载/读 25.6 MB + adb push 0.88–1.46 s + run 0.25 s | 本票实测 push 0.88–1.46 s（§7.4）+ §7.1 的 232 ms 启动 |
| V2 常驻 | **62.9 ms** | §7.5 |

**≈19–30× 的端到端改善，且 host→device 搬运量降 2085×**。这是本票的实际交付。

### 7.4 每 token 的 host 写入量

| 形态 | 每 token host 写入 | 备注 |
|---|---|---|
| CLI（票 06） | **25,640,088 B**（8 个输入文件全推） | 实测 0.88–1.46 s/步 |
| V2 常驻 | **12,288 B**（K+V 各 48×128 B，写进 dma-buf） | 无 adb、无进程重启 |
| 另：小输入 | 493,264 B（x/mask/cos/sin 六个文件） | 两种形态都要，与 KV 无关 |

### 7.5 稳定性（V2，400 次）

```
[time] n=400 mean=62.905 median=62.836 p10=58.363 p90=67.009 min=54.449 max=81.369
[cpu]  wall_total=25162 ms cpu_total=5180.0 ms cpu_per_invoke=12.950 ms
```

- 400 次 invoke 的输出与 1 次 invoke 的 **sha256 相同**（确定性）；
- `rchar` 在循环里**一个字节不涨**、`read_bytes = 0`（v1 runner 的 `/proc/self/io` 实测：
  400 次只涨 101 B，那 101 B 是读 `/proc/self/io` 自己）；
- 极差 54.4–81.4 ms（±21%），比 v1 的 44.0–56.7 宽——多出来的抖动就是那 13 ms 的 CPU 拷贝在抢。

---

## 8. 每个 token 到底发生了什么（一句话版）

```
v1（不行）：host 写 dma-buf 第 m 行(12 KB) -> invoke -> 设备读的是"绑定时的那份快照" -> 结果错
v2（可以）：host 写 dma-buf 第 m 行(12 KB) -> 构造 IOBuffer{fd} -> V2_run
             -> 运行时 CPU memcpy 24 MiB 到 staging(12.95 ms) -> 设备读 staging -> 结果对
CLI（可以但慢）：host 推 25.6 MB -> 新进程 -> 装载 232 ms -> 设备 -> 结果对
```

---

## 9. 下一步（按性价比）

1. **把 V2 的 13 ms 拷贝拿掉**：V2 若能像 v1 那样"导入一次 fd、之后只重送描述符"，就同时拿到
   "数值正确"和"零拷贝"。线索：V2 的 `create_with_options` 的 options 字符串未探索；
   `MemImportWithFlag` 在厂商 trace 里出现 202 次（OP-vendor-runtime §8），说明厂商路径是导入一次。
   建议下一票先做**V2 options 扫描**（`suppressInputConversion` 之类）+ 用本票的 8 步 harness 判成败。
2. **把 T=8 的 7 行 padding 去掉**（票 06 的硬边界：T=1 编不过）。这是 35 ms 塔本体里 7/8 的浪费，
   比 13 ms 的拷贝更值钱。
3. 真要零拷贝 + in-place，还是得"图自己持有 KV"（厂商 `libmtk_llm` 的形态）；TFLite 函数式图表达不了，
   要么换执行形态，要么走 PLAN.md 里已有的自建提交器方向。

---

## 10. UNKNOWN（不圆）

1. **V2 的拷贝发生在哪一层**（runtime 的 descriptor 处理 / apusys 驱动 / 还是每个请求重新 import）：
   只测到"CPU 12.95 ms、24 MiB 量级"，没有系统调用或驱动级证据。
2. **v1 为什么"latch"**：没有分离"setInput 时快照"与"首次 invoke 时快照"（本票所有实验都是
   bind → invoke → 再写）。对结论无影响（两种都不支持更新）。
3. **V2 的 options 字符串格式未知**：`create_with_options` 存在（`RuntimeV2.h:107`）但没试过任何取值。
4. **V2 与 v1 的数值可复现性**：8 步逐字节相同是"同一张 .dla、同一会话"下的结论；跨会话/跨重编
   （.dla 不可复现）没有验证。
5. **窗口滚动（m ≥ C）没跑**：8 步离 2048 还很远。
6. **V2 的 400 次抖动（±21%）**：怀疑与那 13 ms CPU 拷贝抢占有关，未取证。

---

## 11. 复现

```
export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"

# 产物（与票 06 同一条命令）：tflite -> dla -> extract-shared 出 16,674,816 B 的 bin
/usr/bin/python3 tools/build_q06.py --mode decode --t 8 --cache 2048 --layers 24 \
  --tab ops/q06_tab.json --out ops/q06_decode_t8_c2048_24l
./tools/compile.sh ops/q06_decode_t8_c2048_24l/model.tflite ops/q06_decode_t8_c2048_24l/model.dla

./tools/build_kv1.sh                       # out/kv1/kv1_runner 和 kv2_runner

./tools/devlock.sh run -- ./tools/healthcheck.sh --quick
./tools/devlock.sh run -- ./tools/e2e_kv1.sh --steps 8      # v1 路线：step1 起 DIFF（本票的证据）
./tools/devlock.sh run -- ./tools/healthcheck.sh --quick
# v2：同一套 --stepdir/--outdir 契约
#   ./bin/kv2_runner -a q06/decode.dla <9 个 -i> -o <3 个 -o> --steps 8 --m0 16 \
#       --stepdir kv1/plan --outdir kv1/out8_v2
```

runner 的开关（证据工具，不是产品接口）：

| 开关 | 作用 |
|---|---|
| `--kv-heap host` | KV 用匿名内存（`ionFd=-1`/`fd=-1`） |
| `--sync 1` | 每次 host 写前后各一次 `DMA_BUF_IOCTL_SYNC` |
| `--noconv` | `EnvOptions.suppressInputConversion/OutputConversion = true`（仅 v1） |
| `--rebind` / `--pingpong` | 写完用同一 ptr / 换另一块 buffer 重新 `setInput`（仅 v1） |
| `--poison N` | 新行改写常量 N（毒化实验，仅 v1） |
| `--dumpcache F` | 结束时把两块 KV buffer 写盘（验证"写落在哪"，仅 v1） |
| `--steps S --stepdir DIR --outdir DIR --m0 M` | 多 token 常驻序列（DIR/stepNN/*.bin） |
| `-c N` | 同一输入连跑 N 次（计时用） |

证据产物：`ops/kv1_evidence/`（`seq_base.jsonl`、`seq_kvres.jsonl`、`api-latch-verdict.txt`、
`run400.json`、`e2e.log`、两轮 `round*.txt`）。
