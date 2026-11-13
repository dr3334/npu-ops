# 厂商 runtime 的真实形态（tok/s / 内存 / 加载阶段）——板上实测

日期：2026-11-13 · 板 Titan 1 / MT6897 / Android 16 · 全部板端操作持 `/d/deepseek_ workspace/mtk/.npu-board.lock`
对象：设备上现成的厂商资产 `/data/local/tmp/llm_sdk/`（`main` runner + `config_qwen2.5_0.5b_mt6897_2048c_benchmark.yaml`）
票：`.scratch/quant-runtime/issues/01-measure-vendor-runtime.md`

**一句话**：厂商这条路在 2048 上下文下跑 **57.4–58.2 tok/s（17.2–17.4 ms/token）**，其中 NPU 自身
**14.3–15.1 ms/token**；250.7 MB 的共享权重是**一块 239.125 MiB 的 DMA-BUF**（inode 64129），被同一个
进程**映射 3 次（别名，Rss=0）**，**在初始化期一次装好，decode 期间一个字节都不重读**。进程 PSS 稳态
**346.5–354.8 MB**（匿名堆 322.8–330.6 + 文件 23.5–24.1，见 §5 的三次运行差异），
另有 **564.2 MiB DMA-BUF**（dumpsys 记为 Graphics）。

---

## 0. 一句话之外的六条（都用后面 §3–§7 的证据支撑）

| 问题 | 实测答案 |
|---|---|
| decode tok/s | **57.4–58.2**（2048c，7 次运行）；256c/512c 是 **64.0–64.3** |
| 与矩阵哪一档对得上 | NPU 时间落在我们 **4W16A**（11.6–14.5 ms）**区间外沿**，与 **4W8A（5.94 ms）差 2.1–2.6 倍** → 与 4W8A 不相容（详见 §4 的告诫） |
| 250.7 MB 权重是否一次性映射 | 是。**一块 dma-buf**，大小 239.125 MiB = 250,737,664 B（= 文件字节数），**3 个别名映射** |
| decode 期间 DMA-BUF 是否稳定 | 稳定。apusys 85.65 MB 从 +1.0 s 到最后一次采样（+4.1 s）**逐字节相同** |
| 加载发生在哪个阶段 | **初始化**。`/proc/pid/io` 的 rchar/syscr 从 +0.83 s 起**完全冻结**，`read_bytes=0` |
| 进程 PSS / DMA-BUF | PSS **346.5–354.8 MB**（smaps_rollup，三次运行）；DMA-BUF 唯一 **564.18 MiB / 117 块**；dumpsys TOTAL PSS **932 MB**（两者相加） |

---

## 1. 纪律与还原性（先交代，免得后面数字没根）

- **只读资产，一个字节没改**。跑前把 `prompt*.txt` 与 4 个 config 拷到
  `/data/local/tmp/llm_sdk_backup_q01/`（含 `SHA256SUMS.txt`）；收工后对**现存文件**重算 sha256 与备份
  逐行 `diff`：**`RESTORE_VERIFIED_IDENTICAL`**。没有改 config、没有改 prompt、没有改 `.dla`。
- 板端健康：实跑前后各一次 `./tools/devlock.sh run -- ./tools/healthcheck.sh --quick`，两次都
  **`HEALTHCHECK: PASS`**（含 `fc_1x64x64_int8` 已知-good 算子）。收工后 `/proc/meminfo` 与
  `dmabuf_dump` 回到基线（apusys dma-buf = 0）。
- 主检出 `npu-ops` 只读；本票全部改动只在 worktree `wt-q01`，产物（日志/采样/atrace 解包）放在仓库外
  `.qwen25-impl/probe-vendor-q01/@。

---

## 2. runner 的形态（只读探测）

`main` 是 MTK `llm_sdk` 的 demo runner，源码路径字符串带 `jni/main/main.cpp`。

- **CLI**：`./main <config.yaml>`。无参 → `ERROR: No yaml config file provided. [jni/main/main.cpp:378]`；
  `--help` → `Unrecognized argument: --help`（然后仍缺 config）。已认的参数只有 6 个：
  `--input-file` / `--one-prompt-per-line` / `--benchmark-quiet` / `--preload-shared-weights` /
  `--read-tokens` / `--preformatter`。
- **必须设 `LD_LIBRARY_PATH`**，否则 `CANNOT LINK EXECUTABLE "./main": library "libtokenizer.so" not found`。
- 一次运行的阶段与自报数字（下面每一行都是 runner 自己打印的）：

```
Done model init. (Time taken: 0.268s)          <- 装两张 .dla + 250.7 MB 共享权重 + 272.3 MB 词嵌入
Done analyzing prompt in 0.0882s (1451 tok/s)  <- prefill，128 token（promptTokenBatchSize=128）
Hot swapping to 1t model... (0.0162s)          <- prefill 图 -> decode 图
Response [Max Length = 200]:                   <- 固定生成 200 token（stopToken: 999999 只影响是否提前停）
  Decode Runtime: 57.958 tok/s
  Decode Host overhead: 0.0427 ms/token        <- 宿主侧开销，可忽略
```

- **深挖 API**：`strings` 里有 `mtk_llm_shared_weights_preload(SharedWeightsHandle&, LlmRuntimeOptions&)`、
  `mtk_llm_shared_weights_release_handle`、`mtk_llm_swap_model(void*, size_t, size_t)`，以及日志串
  `Preload shared weights took: `。→「共享权重」是 SDK **一等公民的 API**，不是文档外的技巧；
  `--preload-shared-weights` 就是显式触发 preload 的开关（本票没测该开关）。

---

## 3. tok/s（7 次实跑，都是同一份 benchmark config）

全部 `stopToken: 999999`、生成 200 token、同一 prompt（runner 内置的那段 PPC/Intel 文本，128 token）。

| # | init (s) | prefill (s) | prefill tok/s | swap (s) | **Decode Runtime** | Generative Mode | ms/token |
|---|---|---|---|---|---|---|---|
| A（`--benchmark-quiet`，无采样器） | 0.4486 | 0.08966 | 1427.5 | 0.01880 | **57.509** | 57.505 | 17.39 |
| B1 | 0.2682 | 0.08821 | 1451.0 | 0.01622 | **57.958** | 57.815 | 17.25 |
| B2 | 0.2741 | 0.08804 | 1453.8 | 0.01554 | **57.561** | 57.415 | 17.37 |
| C | 0.3002 | 0.08717 | 1468.4 | 0.01618 | **57.820** | 57.628 | 17.30 |
| E | 0.2614 | 0.08711 | 1469.4 | 0.01529 | **58.161** | 57.977 | 17.19 |
| F | 0.2682 | 0.08899 | 1438.3 | 0.01657 | **57.938** | 57.761 | 17.26 |
| G | 0.2623 | 0.08674 | 1475.7 | 0.01466 | **57.601** | 57.437 | 17.36 |

**换算**：decode **17.19–17.40 ms/token**（均值 17.31，标准差 0.08 → 板很稳）；
prefill **128 token / 0.087 s = 0.68 ms/token（1427–1476 tok/s）**。

缓存长度的影响（同一批资产，不同 config）：

| config | cacheSize | prefill | prefill tok/s | decode tok/s | ms/token |
|---|---|---|---|---|---|
| `..._256c.yaml` | 256 | 0.0665 s | 1925.3 | **64.04** | 15.62 |
| `..._512c.yaml` | 512 | 0.0688 s | 1861.8 | **64.30** | 15.55 |
| `..._2048c_benchmark.yaml` / `..._mt6897.yaml` | 2048 | 0.0884 s | 1447.2 | **57.46** | 17.40 |

→ 256c 与 512c **几乎相同**（差 0.4%），2048c 比它们慢 **10%**。原因未查（见 §10 UNKNOWN-3）。

**关于外部流传的「56 tok/s」**：那是别人报的数，**不是我们测的**。我们在同一块板、同一批资产上测到的
是 **57.4–58.2 tok/s**（`..._mt6897.yaml` 即非 benchmark 配置：57.46）。两者差 2–3%，量级一致；
但**出处无法溯源**，本票只认上表的数字。

**首 token 延迟**：runner 的字符串里有 `First output token `，但这几次运行**没有打印**（UNKNOWN-4）。
可由自报阶段合成：`init 0.26–0.45 s + prefill 0.087 s + swap 0.016 s ≈ 0.35–0.37 s`
（进程启动另计，见 §7 的存活时间）。

---

## 4. 与我们量化矩阵的对齐（**只比对，不下结论说厂商用了什么**）

我们自己的矩阵（`skills/mtk-llm-pipeline/references/02-quantization-matrix.md:23-28`）是
**LFM2.5-230M**、单 DLA decode、converter 7.16.0 / ncc 7.3.15：

| 我们的档 | decode（单 DLA） |
|---|---|
| 4W16A | 11.6–14.5 ms |
| 4W8A | **5.94 ms** |
| 8W8A | 8.09 ms |
| W8A16F | 15.08 ms |

厂商（Qwen2.5-0.5B，24 层，2048c）：**17.2–17.4 ms/token 端到端**；从 atrace 拆出的 **NPU 自身 14.3–15.1 ms**
（§8），256c/512c 的 NPU 自身是 **12.56–13.23 ms**。

对齐结论（措辞刻意保守）：

- 与 **4W8A（5.94 ms）不相容**：即使拿最有利的 512c NPU 时间（12.56 ms）比，也是 4W8A 的 **2.1 倍**；
  2048c 是 **2.4–2.6 倍**。要么档位不是 4W8A，要么矩阵那一格的模型/形状完全不同——两者其一必真。
- 与 **4W16A（11.6–14.5 ms）相容**：512c 的 NPU 时间 **12.56–13.23 ms 正落在该区间内**；2048c 的
  14.3–15.1 ms 只比上沿高 4%。
- **反证也要写**：矩阵是 230M 模型，本机是 0.5B（层数 24、隐藏 896）。**参数量差 2 倍，时间却落在
  同一区间**，说明矩阵那几格与本机不是同一把尺子——所以「对得上 4W16A」只能算**相容**，不能算**证明**。
- 旁证（只是文件名，不是内部读证）：两张 `.dla` 的文件名都带
  `Qwen2.5-0.5B-Instruct_asym4W_sym16A_Overall_hessian_24layer_*_shared.dla`，即厂商自己的档位命名 =
  非对称 4-bit 权重 + 对称 16-bit 激活；`embedding_..._int16.bin` 也是 int16。
- **本票不主张「厂商一定用了 W4/W8」**。凡是要写进 ADR/文档的，请只引「17.2–17.4 ms/token、
  NPU 14.3–15.1 ms、与 4W8A 不相容、与 4W16A 相容、文件名叫 asym4W_sym16A」。

---

## 5. 三个时点的内存（同一进程，50 ms 细采样）

设备侧采样器（`sampler_v2.sh`）每 ~50 ms 记一行 `/proc/meminfo` + `/proc/pid/smaps_rollup`，每 4 行记
`dmabuf_dump -b` 与 `/proc/pid/io`；t=0 取「`pidof main` 第一次非空」的采样点。

| t (ms) | 阶段 | Pss (MB) | Pss_Anon | Pss_File | MemFree (kB) |
|---|---|---|---|---|---|
| 0 | 进程刚起 | 26.7 | 5.9 | 20.0 | 623,768 |
| +164 | **装权重的瞬间** | 195.3 | 58.1 | **137.1** | 605,112 |
| +306 | init 结束（自报 0.261 s） | 304.2 | 280.8 | 23.3 | 260,016 |
| +636 | 权重/工作缓冲装完 | 342.2 | 318.6 | 23.5 | 206,468 |
| +758 | prefill 完 / 换图 | 345.9 | 322.2 | 23.5 | 203,648 |
| **+876 → +4123** | **decode 200 token 全程** | **346.5** | 322.8 | 23.6 | 187,724 → 322,672 |
| +4322 | 退出中 | 74.8 | 51.7 | 23.0 | 636,784 |
| 收工 | — | — | — | — | 746,164 |

**decode 期间 Pss 是常数**：连续 26 个采样点（+876 → +4123 ms）Pss **恒为 346.5 MB**（标准差 < 0.02 MB）、
Pss_Anon 恒为 322,828 kB、Pss_File 在 23.5–23.6 MB 之间抖 1–2 个页、Pss_Shmem 117–120 kB。
**没有逐 token 的增长**（KV cache 是静态预留，cacheSize 2048）。

**三次运行之间的 8 MB 差异（未解释）**：run E（本表）稳态 346.5 MB / Anon 322.8；run C 与 run F 的同一进程
稳态是 **354.8 MB / Anon 330.6**。三次曲线都是「装完即平」，只是平台高度差 8.3 MB，原因没查到（UNKNOWN-8）。
所以引用时请写 **347–355 MB**，不要写单一值。

**MemFree 的绝对值不要当预算用**：本机常态 MemFree ~250 MB、MemAvailable 6.1 GB（NOTES.md §6 已订正过
这个指标误用）。跑的时候 MemFree 最低掉到 **187.7 MB**，MemAvailable 从 7.75 GB 掉到 6.60 GB
（−1.15 GB，含共享权重 + 词嵌入 + 页缓存被挤）。注意 MemFree 在 +2773 ms 回到 338 MB 后仍缓降——
那是后台进程在还页，不是模型在释放。

**PSS 的构成**（`smaps_mid.txt` 逐 VMA 归属，decode 中）：

| 类 | Pss (MB) | 说明 |
|---|---|---|
| `[anon:scudo:*]` | **318.4** | 其中**单块 259.7 MB** = 272,269,312 B（= `embedding_..._int16.bin` 的文件大小） |
| 文件（.so 等） | ~24 | `libneuron_runtime.so` 1.8、`libsentencepiece` 1.2、`libre2` 1.3 … |
| `/dmabuf:*` VMA | **0.0** | 见 §6：这些映射 `Rss=0`，Pss 不计 |

**词嵌入是堆里的拷贝，不是文件映射**：`272,269,312 B / 1048576 = 259.66 MiB`，与那块
scudo:secondary 的 259.7 MB 逐字节对上。decode 中 `/proc/pid/maps` 里**找不到**任何 `embedding_` 或
`.dla` 的文件映射。

**dumpsys meminfo（decode 中）**：

```
Native Heap   324680 kB   .so mmap  28039 kB   EGL mtrack  577724 kB
TOTAL PSS     932292 kB   TOTAL RSS 952388 kB  Swap 0
```

→ **两个口径都要报**：`smaps_rollup` 的 PSS 347–355 MB **不含** dma-buf；Android 的 `dumpsys` 把 dma-buf
算进 Graphics/EGL mtrack（577.7 MB）→ TOTAL PSS **932 MB**。以后写「常驻」必须说清用哪个口径。

---

## 6. 250.7 MB 共享权重的真实形态（本票最关键的一节）

**它是 dma-buf，不是堆、不是文件映射；一块 buffer，映射三次。**

decode 中 `/proc/<pid>/maps` 里三条**互不相同地址**的映射，**inode 全是 64129**：

```
6fa5dc2000-6fb4ce2000 rw-s 00000000 00:09 64129  /dmabuf:soc:mtk-apu-mem-data
6fc24e2000-6fd1402000 rw-s 00000000 00:09 64129  /dmabuf:soc:mtk-apu-mem-data
6fe3de1000-6ff2d01000 rw-s 00000000 00:09 64129  /dmabuf:soc:mtk-apu-mem-data
```

- 每条长度 = **239.125 MiB = 250,737,664 B**，与 `shared_weights_2048c_0.bin` 的字节数**完全相同**。
- 三条 inode 相同 → **同一块 buffer 的 3 个别名**（0.5B 的两张图 + 1）；
- 每条的 smaps 是 `Rss: 0 kB / Pss: 0 kB / Shared_*: 0` → **CPU 从来没碰过这些页**，NPU 直接吃 buffer
  句柄，进程这边只是把 VMA 挂上去（`VmFlags: rd wr sh mr mw me ms pf io de dd`）。这解释了为什么
  smaps 的 PSS 不含权重，而 dumpsys 的 Graphics 含它。

**整份 dma-buf 账**（decode 中，按 inode 去重：117 块 / 564.18 MiB）：

| 唯一大小 | 块数 | 读法 |
|---|---|---|
| **239.12 MiB** | 1 | 共享权重（inode 64129，映射 3 次） |
| 49.46 MiB | 1 | `/dmabuf:APUCB0:<pid>:MDLA`（NPU 命令缓冲） |
| 37.09 MiB | 1 | `/dmabuf:soc:mtk-apu-mem-data` |
| 15.32 MiB | 1 | `/dmabuf:APUBUF0:<pid>:MDLA` |
| **4.50 MiB** | **48** | 每块映射 3 次；48 = 24 层 × 2 张图（自然解读，未证实） |
| 其余小块 | 65 | 0.5–1.5 MiB 级 |

地址空间合计 1553 MiB（大小块都被多次别名映射），**去重后 564.18 MiB**，与 dumpsys 的
`EGL mtrack 577,724 kB` 同量级。

**时点**：从 `dmabuf_dump -b` 的时间序列看，`mtk_mm-uncached`（这批 `/dmabuf:soc:mtk-apu-mem-data`
的 exporter）在 **+0.60 s 一次性 +431 MiB**；`apusys` exporter 在 **+0.88 s 到 77.68 MiB、+1.43 s 到
81.68 MiB 后不再变**。→ 权重与工作缓冲都在 **init 期（0.26–0.9 s）**装好。

---

## 7. 加载发生在初始化，decode 期间零重载（三条独立证据）

1. **`/proc/pid/io`**（每 ~0.2 s 一次）：`rchar` 4,434,420 → **4,696,728 B 之后完全冻结**（+0.83 s 起），
   `syscr` 1384 恒定，`read_bytes = 0`，`wchar` 单调增长（那是往 stdout 打 token）。
   → decode 期间**没有任何文件读取**，连一次 4 KB 的 read 都没有。
2. **`dmabuf_dump -b` 的 apusys 总量**：81.68 MiB（85,651,475 B）从 +1.0 s 到 +4.1 s **逐字节相同**
   （7 个时点）。
3. **Pss / Pss_Anon / Pss_File 全程常数**（§5），**没有** 239 MB 级的周期性抖动。

反过来，init 期的加载是看得见的：`Pss_File` 在 +0.16 s 冲到 **137 MB**、`rchar` 在 0.2–0.8 s 之间从
0.5 MB 涨到 4.4 MB、`mtk_mm-uncached` +431 MiB。**注意 `rchar` 只有 4.4 MB，而资产是 547 MB**：
→ 权重/嵌入**不是走 `read(2)` 进来的**，是 mmap 后按页取（`read_bytes=0` 也排除了块层直读）。
具体的拷贝路径（mmap+memcpy → dma-buf？）没拿到系统调用级证据（UNKNOWN-2）。

**进程时间线**（采样器实测）：pid 可见到 pid 消失 **≈ 4.49 s**；runner 自报 init 0.26 + prefill 0.087 +
swap 0.016 + decode 3.46 = **3.82 s**；余下 ~0.6 s 是 SDK `dlopen`（`libsentencepiece` 17 MB、
`libneuronusdk_adapter` 13.6 MB、`libre2` 11 MB …）与退出。

---

## 8. 厂商自己的 atrace（解包成功，是上板形态的旁证）

设备上两个 `.atrace` 不是二进制 ftrace ring buffer，而是 **`TRACE:` + zlib 压的文本 ftrace**
（`qwen25_coarse.atrace` 363 KB → 5.1 MB 文本 / 38,783 行，`tracer: nop`）。把它们解出来、按 pid
栈配对 `tracing_mark_write` 的 B/E，得到厂商（自家环境、2026-07-30 录的）每次 NPU 推理的耗时：

| 量 | `qwen25_coarse`（512c 档） | `qwen25_2048_coarse`（2048c 档） |
|---|---|---|
| NPU 推理次数 | 17 | 17 |
| prefill（最大那次） | **66.0 ms** | **84.7 ms** |
| decode 每步 `APUSys Real Execution` | 12.56–14.18 ms（中位 12.80） | **14.25–15.10 ms（中位 14.70）** |
| 相邻两次触发间隔（含宿主） | 14.6–14.8 ms | 16.6–16.7 ms |
| 其它小段 | `ExecuteEnqueue` 0.32–1.78、`BindImpl` 0.10、`FlushInputs` 0.12、`MemImportWithFlag` 0.078（×202） | 同量级 |

**交叉验证**：今日实测 2048c = 17.40 ms/token，atrace 的触发间隔 16.6–16.7 ms（差 4–5%）；512c 实测
15.55 ms，atrace 14.6–14.8 ms（差 5%）。→ trace 里的东西**和今天的运行是同一个形态**，只是 trace 用的
那个 `libmtk_llm.so`（07-27 构建）比今天这块板上的状态稍快一点（UNKNOWN-1）。**注意**：trace 里
没有「每 token 一次 250 MB 传输」的迹象——每步就是一个 14–15 ms 的 `APUSys Real Execution`，加一个
0.4 ms 级的小段。

---

## 9. 对我们自己的形态意味着什么（对照 ADR 0005 / 0006）

- **「每层一个 `.dla`」的代价不是 145 ms/层，而是「进程级装载」这件事本身**：厂商这边整网
  （24 层 × 2 张图 + 250.7 MB 权重 + 272.3 MB 嵌入）**0.26 s 装完**，换图只要 **15–19 ms**
  （`Hot swapping`，prefill↔decode 各一次；反向换回 128t 只要 2.9 ms）。这个数字可以直接当
  票 05/06 的验收参照：**「两张图 + 共享权重」的装载上限就在 0.3 s 量级**。
- **共享权重的形态现在是可复述的**：一块 dma-buf，按文件字节数分配，多图多重映射，CPU 不碰页。
  这是票 03（ncc 外部/共享权重机制）要复现的目标形态；`dmabuf_dump -b` + `maps` 的 inode 去重
  就是它的验收手段（本票脚本可直接复用）。
- **KV cache 没有单独可见**：dma-buf 里没有与 cacheSize 成正比的明显块（256c/512c/2048c 的 decode
  ms 只差 10%）。KV 大概在 4.5 MiB 那 48 块里，**未证实**。
- **tok/s 的可比性**：以后我们报 decode 数字，必须同时报「NPU 自身」与「端到端（含宿主+采样）」。
  本例二者差 **2.0–2.3 ms/token**（14.3–15.1 → 17.2–17.4），是 12–15% 的口径差。

---

## 10. UNKNOWN（不圆场）

1. **512c 的 NPU 时间差 5%**：atrace（07-30 录）12.56–13.23 ms vs 今日实测折算 15.55 ms 端到端。
   trace 的构建与今日 `libmtk_llm.so` 未必同一版；未做版本比对。
2. **权重/嵌入的拷贝路径**：`read_bytes=0`、`rchar=4.4 MB`、init 期 `Pss_File` 峰 137 MB → 说明
   不是 `read(2)`；但「mmap → memcpy 进 dma-buf」还是「mmap 后由驱动直取」没有系统调用级证据
   （没跑 strace/perfetto 的 syscall 追踪）。
3. **256c 与 512c 速率几乎相同**（64.04 vs 64.30 tok/s，差 0.4%）：2048c 才慢 10%。KV 寻址为何在 512
   以内不进关键路径，未查。
4. **首 token 延迟没有直接打印**：字符串 `First output token ` 存在但未触发（可能与
   `--benchmark-quiet` 或某个门控有关）。只能用 init+prefill+swap 合成 0.35–0.37 s。
5. **3 次映射分别给谁**：inode 64129 被映射 3 次，只能确认「同一块 buffer」，无法从 `maps` 区分哪一
   条给 prefill 图、哪一条给 decode 图、第三条是做什么的。
6. **48 块 4.50 MiB = 24×2 的解读**未证实（数量吻合，但没有逐块的身份证据）。
7. **`--preload-shared-weights` 与 `--read-tokens` 两个开关没测**：本票只用默认路径跑。
8. **两次运行的稳态 PSS 差 8.3 MB**（run E 346.5 vs run C/F 354.8，Anon 322.8 vs 330.6）：三次都
   「装完即平」，没有逐 token 增长；但平台高度差了整整一个 8 MB 块，没找到原因（没做同进程多样性
   对照，也没有 heap 级 dump）。引用时写区间，不要写单值。

**票面附录**：`.qwen25-impl/probe-w4a8/RESULTS.md` **不存在**（该目录只有 `build_int4_*.py`、
`cmp_int4.py`、`int4probe/` 等产物，没有 RESULTS.md）。所以本票没有任何结论引自「w4a8 探针结果」，
只用 `probe-precision/RESULTS.md` 与 `probe-int16/RESULTS.md`（两份都在，已读）。

---

## 11. 复现

```
export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
cd '/d/deepseek_ workspace/mtk/npu-ops'
./tools/devlock.sh run -- ./tools/healthcheck.sh --quick          # 前后各一次

# 1) 只读探测（CLI 形态）
./tools/devlock.sh run -- bash '.../probe-vendor-q01/probe_usage.sh'
# 2) 短跑一次拿 tok/s（最干净，无采样器干扰）
./tools/devlock.sh run -- bash '.../probe-vendor-q01/run_a_short.sh'
# 3) 设备侧细采样跑（PSS / dma-buf / io 时间序列）
./tools/devlock.sh run -- bash '.../probe-vendor-q01/run_e_measure.sh'
# 4) decode 中抓 maps/smaps/dumpsys（主机侧对齐）
./tools/devlock.sh run -- bash '.../probe-vendor-q01/run_f_measure.sh'
# 5) dma-buf 时间序列
./tools/devlock.sh run -- bash '.../probe-vendor-q01/run_g_measure.sh'
# 6) 缓存长度对照（256c/512c/2048c）
./tools/devlock.sh run -- bash '.../probe-vendor-q01/run_h_cachecompare.sh'

# atrace 解包 + 时长统计（WSL）
wsl.exe -e bash -lc '/usr/bin/python3 .../probe-vendor-q01/decompress_atrace.py'
wsl.exe -e bash -lc '/usr/bin/python3 .../probe-vendor-q01/durations.py'
# dma-buf 账 / smaps 归属
wsl.exe -e bash -lc '/usr/bin/python3 .../probe-vendor-q01/analyze_inodes.py'
wsl.exe -e bash -lc '/usr/bin/python3 .../probe-vendor-q01/analyze_smaps2.py'
```

（`.../probe-vendor-q01` = `/d/deepseek_ workspace/mtk/.qwen25-impl/probe-vendor-q01`）

**产物**（仓库外）：`main.strings.txt`（runner 的 944 条字符串）、
`qwen25{,_2048}_coarse.{atrace,txt}`、`samples.csv`（50 ms 曲线）、`aux_v2.txt`（dma-buf/io 序列）、
`maps_mid.txt` / `smaps_mid.txt` / `fd_mid.txt` / `dumpsys_mid.txt`、`dmabuf_series.txt`
（19 个时点的全量 `dmabuf_dump -b`）、各次 `run*.log`。
设备侧：`/data/local/tmp/llm_sdk_backup_q01/`（备份 + `SHA256SUMS.txt`）、`/data/local/tmp/q01/`
（脚本与采样）。
