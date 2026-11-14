# OP-24layer —— 24 层单图 + 外部共享权重（prefill），8W16A + native 缝

> 票：.scratch/quant-runtime/issues/05-24layer-single-dla-shared-weights.md；spec .scratch/quant-runtime/spec.md
> 上位证据：docs/OP-shared-weights.md（票 03 机制）、docs/OP-tier-matrix.md（票 04f 档位矩阵）、
> docs/OP-w4a16-layer.md（04d native 缝）、docs/OP-vendor-runtime.md（厂商形态与内存口径）。
> ADR：docs/adr/0005-quantization-tier-target.md **第六次订正**（实现门 + 按档质量期望）。
> 分支 impl/q05-24layer，worktree wt-q05；编译器 ncc-tflite 7.3.15
> （--arch=mdla5.0 --opt=3 --disallow-bridge --num-mdla=2 --l1-size-kb=3072）；
> 设备 mt6897 / Android 16（0123456789ABCDEF），板端全部持 /d/deepseek_ workspace/mtk/.npu-board.lock，
> 每个板端块前后各一次 healthcheck --quick（**全部 PASS**）。
> 权重夹具 npu-ops/weights/qwen25_frozen（只读；**24 层用的是同一套层 0 权重重复 24 次 = 权重复用，不是真实 24 层**）。
> 公共工具（e2e/e2e_i16/compare/compare_i16/compile/devlock/healthcheck/build_ops/quant_contract/build_w4a8）**一行未改**。

---

## 0. TL;DR

**24 层能装进一个图、能编、能跑、权重能外置成一份 bin，全部成立。三张图（T=8 / T=16 / T=128）
都是 EXIT=0、1440 层、单执行步、无 bridge、板端实现门 ｜Δcos(device, golden)｜ ≤ 4.7e-5。**

| 量 | T=8 | T=16 | T=128 |
|---|---|---|---|
| 原生层数 / 执行步 / EXIT | 1440 / **1** / 0 | 1440 / **1** / 0 | 1440 / **1** / 0 |
| bridge | 无 | 无 | 无 |
| .dla（抽取前） | 18,302,350 B | 18,284,064 B | 29,351,954 B |
| Static / Code / Total | 15M / 897K / 16M | 15M / 823K / 16M | 26M / 823K / 31M |
| model_shared.dla（抽取后） | 3,317,256 B | **1,489,594 B** | 1,489,644 B |
| 外部 bin（本次配对） | 16,878,592 B（与 T=16 共用一份） | 同左 | 27,039,744 B（与 T=16 配对） |
| 板端 prefill 一次 | **17.47 / 18.21 ms** | **22.82 / 24.14 ms** | **94.55 ms** |
| ms/token | 2.18–2.28 | 1.43–1.51 | **0.74** |
| device vs 量化 golden cosine | 0.9999877 | 0.9999852 | 0.9999727 |
| ｜Δcos(device, golden)｜（门 0.003） | **1.19e-5 PASS** | **4.68e-5 PASS** | **7.54e-6 PASS** |
| rel_l2(device, golden)（门 0.05） | 4.97e-3 PASS | 5.44e-3 PASS | 7.39e-3 PASS |
| 码级 device vs golden（max / mean） | 267 / 30.5 | 354 / 32.5 | 512 / 37.4 |
| device vs 独立 fp64 链 | 0.9967964 | 0.9961469 | 0.9636675 |
| 量化 golden vs fp64 链 | 0.9967845 | 0.9961937 | 0.9636600 |
| 噪声底（3 次独立装载） | **0 字节** | **0 字节** | **0 字节** |
| 质量期望（int8 权重 ≥0.9998） | FAIL | FAIL | FAIL |

四条要点：

1. **单图 24 层没有编译器上限的迹象**：1440 层仍是 **1 个执行步**（1440 = 24 × 60，60 正是票 04f 单层
   8W16A native 的层数），无 bridge，编译 2–3 分钟。**没有触发降级到分块**。
2. **权重外置成立且比预期更省**：ncc 把 24 份**字节相同**的静态权重合并成一份（Static 15M，控制实验
   见 §3.2），所以 .dla 只有 18 MB；extract-shared 之后图只剩 **1.49 MB**、bin **16.88 MB**。
   **真实 24 层不同权重应当是 ~366 MB**（2 层控制实验实测 2× → §3.2）。
3. **实现门三张全过**（｜Δcos｜ ≤ 4.7e-5，比 0.003 的门小两个数量级），**质量期望三张全不过**
   （0.9962 / 0.9968 / 0.9637）——这不是实现错，是 24 层串联累积 + T=128 的 per-tensor 尺度被拉坏（§8）。
4. **外部 bin 是 dma-buf**：进程 PSS 21.6 MB（不含权重），一块 16.10 MiB 的 APUBUF0 = bin 的页对齐大小；
   400 次 Run 期间 PSS 恒定、rchar 冻结、read_bytes=0（§6）。

---

## 1. 票面要回答什么 / 做法

票面三件事：**①24 层一个图**（复用 04 的 8W16A 构件）、**②权重外部化**（票 03 的 extract-shared 机制）、
**③prefill 上板**（T=8 / T=16，T=128 时间允许就试）。

档位按 Lead 的指令锁死，**没有为了过而改档**（builder 里 lock_tier() 会直接拒绝其它组合）：

| 项 | 取值 |
|---|---|
| 权重 | int8，per-output-channel 对称（s[n] = max abs(W[n,:]) / 127） |
| 激活 | int16 per-tensor（s = max abs(v) / 32767） |
| bias | int32 per-channel |
| 掩码 SOFTMAX | **native**（int16 加法 + int16 SOFTMAX，无 fp16 缝）→ 每层 60 层 |
| T | 8 / 16 / 128（都试） |

---

## 2. 图怎么造的：tools/build_24layer.py（与 build_layer 只差两处）

层本体**没有重写**：tools/build_24layer.py 逐调用照抄 tools/build_w4a8.py::build_layer(upto="out") 的算子序列
（qkv + RoPE 单 scale、kv14 CONCAT、BMM-QK、native 缝 4 层、BMM-PV、o_proj、残差、第二 RMSNorm、SwiGLU、down、残差），
只改了两处：

1. **张量名前缀**（L00_ … L23_），24 份同名算子不会在同一个 flatbuffer 里撞名；
2. **层 i 的输入张量就是层 i−1 的输出张量**——层间没有图输入/输出，所以整串是一个执行步。

### 2.1 N=1 回归（证明抄对了）

| 项 | 官方 tools/build_w4a8.py layer --w-bits 8 --a-bits 16 --seam native | 本票 --layers 1 |
|---|---|---|
| model.tflite | 15,335,424 B | 15,335,872 B（只多 448 B = 前缀名） |
| golden/input.bin | 逐字节相同 | 同 |
| golden 的 cosine vs fp64 | 0.99984464 | 0.99984454（同到 1e-7） |

**码级不完全相同**（max 151 码、mean 30.9、逐位 169/14336），原因是一处**有意的建模差异**：官方用
fp64 链的中继值挑 per-op 输出 scale；本票用**该层量化后的输入**（n1 的反量化值）挑，这才能让第 2..24 层
有正确的参考。两边与 fp64 的 cosine 差 1e-7，说明差异只在量化格点选取，不在数学。

### 2.2 事前登记（第一次板端运行之前写进 model.json，未事后调整）

| 项 | 登记值 | 依据 |
|---|---|---|
| int8_tol（码） | **1200** | √24 × 票 04f 单层 device-vs-golden max（247 码）= 1210 → 取 1200。这是**传播上界**，不是把单层契约放宽 |
| ｜Δcos(device,golden)｜ | **0.003** | ADR 0005 第六次订正（实现门） |
| rel_l2(device,golden) | **0.05** | 单层 ~1e-2 的 √24 传播（√24×1e-2 = 4.9e-2） |
| 质量期望 | **0.9998** | ADR 第六次订正（int8 权重）；24 层累积照实报，最终门在票 06 的 logits |

---

## 3. 编译：1440 层、单执行步、无 bridge

### 3.1 三张图

| 图 | T | 层数 | 执行步 | bridge | Input/Output/Temp | Static | Code | .dla |
|---|---|---|---|---|---|---|---|---|
| ops/q05_prefill_t8_24l | 8 | 1440 | 1 | 无 | 14K/14K/14K | 15M | 897K | 18,302,350 |
| ops/q05_prefill_t16_24l | 16 | 1440 | 1 | 无 | 28K/28K/0 | 15M | 823K | 18,284,064 |
| ops/q05_prefill_t128_24l | 128 | 1440 | 1 | 无 | 224K/224K/3.8M | 26M | 823K | 29,351,954 |

- 1440 = **24 × 60**，60 与票 04f 单层 8W16A native 的层数一致 → 每层拓扑没有被堆叠改动。
- model.compile.log 里 Graph[0] has 1 Execution Step(s)、TargetExecutionOrder 只有 MDLA_5_0、全程没有 bridge 字样。
- T=8 **能编**：票 04 的「T=8 的 P·V 被 16 B 行规则挡住」是 **int8 激活**下的现象；int16 下
  p 是 [14,8,8] × 2 B = **16 B 行**，正好过线。这是本票的第一条新边界。
- T=128 的 Static 从 15M 涨到 26M，多出来的 ~11 MB 是**每层一份的 causal mask 常量**（[14,128,128] int16
  = 458,752 B × 24）——见 §4.3 的建议。

### 3.2 ncc 会把字节相同的静态张量合并（0.5 小时的控制实验，避免把「省内存」误读成「我们的功劳」）

24 层用的是**同一套夹具权重重复 24 次**，于是 24 份权重张量字节完全相同：

```text
ops/q05_dedupe_same       (2 层，权重字节相同)  -> Static 14M  .dla 15,243,260 B
ops/q05_dedupe_distinct   (2 层，W*(1+0.01*i))   -> Static 28M  .dla 30,168,544 B
```
**结论**：ncc 对**内容相同**的静态张量做了合并（2 层 = 1 份；24 层 = 15M 而不是 360M），
字节不同就老老实实各存一份（2 层 = 2×14M）。所以：

- 本票的 Static 15M / bin 16.88 MB 是**权重复用**的产物，**不是**真实 24 层的大小；
- **真实 24 层（24 套不同权重）按控制实验线性外推 ≈ 24 × 15.24 MB ≈ 366 MB 的 .dla/Static**，
  与 ADR 第六次订正里「24 层 ≈336 MiB」的量级一致。

---

## 4. extract-shared：两张图 + 一份 bin

### 4.1 命令与结果（两个配对，各自出 bin）

```bash
# 配对 A（T=16 与 T=8，两张都是 prefill）
extract-shared -i ops/q05_prefill_t16_24l/model.dla,ops/q05_prefill_t8_24l/model.dla \
               -o ops/q05_prefill_t16_24l/model_shared.dla,ops/q05_prefill_t8_24l/model_shared.dla \
               -s ops/q05_shared.bin
# 配对 B（T=128 与 T=16）
extract-shared -i ops/q05_prefill_t128_24l/model.dla,ops/q05_prefill_t16_24l/model.dla \
               -o ops/q05_prefill_t128_24l/model_shared.dla,ops/q05_prefill_t16_24l/model_shared_t128pair.dla \
               -s ops/q05_shared_t128.bin
```
| 配对 | bin | 第一张图 reuse / nonreuse | 第二张图 reuse / nonreuse |
|---|---|---|---|
| A（T=16 第一） | **16,878,592 B**（414 个静态张量） | 16,761,856 / **0** | 14,979,264 / 1,637,152（372 个） |
| B（T=128 第一） | **27,039,744 B**（421 个） | 27,829,696 / **0** | 14,955,072 / 1,806,784（396 个） |

- **-e diff 的语义是「从第一张图抽」**，所以第一张图的静态数据全部进 bin（nonreuse 0），第二张图只复用
  字节相同的那部分，其余**留在自己 .dla 里**（票 03 §4.6 的安全降级，不报错）。
- 两张图**权重字节一致**（同一份量化产物）→ 权重全部命中 reuse，这是前提，票 03 已警告过。
- **T=8 的 1.64 MB nonreuse** 是它的 bias / RoPE 表 / mask 常量（这些值依赖 T 与激活尺度）；
  这也**反向验证**了它确实落在设备上：T=8 运行时的 dma-buf 里多出正好 1.57 MiB 的一块（§6）。

### 4.2 抽取后的图

| 文件 | 字节 | 说明 |
|---|---|---|
| ops/q05_prefill_t16_24l/model_shared.dla | **1,489,594** | 只剩 code + 非静态部分，权重全在 bin |
| ops/q05_prefill_t8_24l/model_shared.dla | 3,317,256 | 含 1.64 MB 未共享静态 |
| ops/q05_prefill_t128_24l/model_shared.dla | 1,489,644 | 与 T=16 同档 |

厂商 prefill 图是 3,238,205 B（int4、24 个不同层、lm_head 在图内）——**同一个数量级**。

### 4.3 bin 里不只有权重：把 mask / RoPE 提成输入（对票 06 的直接建议）

配对 B 的 bin 是 27.04 MB，比配对 A 的 16.88 MB 多 10.2 MB —— 多的就是 **T=128 的每层 causal mask 常量**
（24 × 458,752 B ≈ 11 MB）。它们进 bin 是因为 -e diff 把第一张图的静态全抽走，而不是因为它们被共享。

⇒ 真实 24 层图应该把 **causal mask、RoPE cos/sin、位置相关常量做成图输入**（每层共享一张），
这样 (a) bin 只剩权重、(b) bin 与 T 无关 → prefill 图与 decode 图才可能**真正共用一份 bin**。
厂商的 prefill 图正是这么做的：它的 IO 表里 mask 是**输入**（557,056 B），不是静态数据（docs/OP-shared-weights.md §4.2）。
本票没有改图（票面只要求「图 + bin 能跑」），这条留作票 06 的输入。

---

## 5. 板端：延迟、cosine / rel_l2、噪声底

三张图各自「独立装载 → Run，neuronrt -m hw，喂 [-i 激活 -i bin]」，持 devlock，前后 healthcheck PASS。
对照三列：**device vs 量化 golden**（实现门）、**device vs 独立 fp64 链**（质量）、量化 golden vs fp64（量化本身的代价）。

| 量 | T=8（4,096 值） | T=16（14,336 值） | T=128（114,688 值） |
|---|---|---|---|
| prefill 一次（两次会话） | 17.47 / 18.21 ms | 22.82 / 24.14 ms | 94.55 ms |
| ms/token | 2.18–2.28 | 1.43–1.51 | **0.74** |
| device vs golden cosine | 0.9999877 | 0.9999852 | 0.9999727 |
| ｜Δcos｜ | 1.19e-5 | 4.68e-5 | 7.54e-6 |
| rel_l2(device, golden) | 4.97e-3 | 5.44e-3 | 7.39e-3 |
| 码级 max / mean / p99 | 267 / 30.5 / 171 | 354 / 32.5 / 152 | 512 / 37.4 / 170 |
| 逐位相同 | 102/7168 | 172/14336 | 1295/114688 |
| device vs fp64 链 | 0.9967964 | 0.9961469 | 0.9636675 |
| golden vs fp64 链 | 0.9967845 | 0.9961937 | 0.9636600 |
| 1 码（真实量） | 2.453e-4 | 2.691e-4 | 3.249e-4 |

- **实现门全过**：三张 ｜Δcos｜ 4.7e-5 / 1.2e-5 / 7.5e-6，全部远小于 0.003；rel_l2 也远小于 0.05；
  码级 354 / 267 / 512 全部小于登记的 1200（票 04f 单层 247 码，24 层只涨到 354，说明设备舍入在链上**没有放大**）。
- **噪声底 = 0 字节**：每张图 **3 次独立装载**（每次进程重启）输出 sha256 完全相同；
  另有一次 **400 次 Run 的同一次装载**，输出与单次运行**逐字节相同**（§7 三联表）。
  ⇒ 本票所有码级差异都是模型差异，不是抖动；ADR 的 0.003 门在有 24 层串联时仍有余量。
- **延迟读法**：24 层单图 22.8–24.1 ms（T=16）= **0.95–1.0 ms/层**；票 04f 的**单层图**是 2.16 ms/层。
  同一份权重、同一档，差在「一次装载 + 一个执行步」vs「一次装载 + 一次 Run 的固定开销」。
  这条对票 06 有意义：**每层一个 .dla 的开销主要不在算子上**。
- 时间没有时钟/热态控制（同 04c/04e/04f），同一天两次会话能差 1.3 ms（T=16: 22.82 → 24.14），
  **只当档位/形态的粗比较**。

---

## 6. 常驻内存实测（PSS / DMA-BUF）

方法照 docs/OP-vendor-runtime.md §5–6：neuronrt -c 400（**一次装载、400 次 Run**），每 0.5 s 采
/proc/pid/smaps_rollup + /proc/pid/maps 的 /dmabuf: 行 + dmabuf_dump -b + /proc/pid/io。
脚本 tools/mem_q05.sh（设备侧 tools/mem_q05_device.sh），原始采样在 ops/<name>/mem/。

| 量 | T=16 | T=8 |
|---|---|---|
| PSS 稳态（14 个采样点） | **21.61–21.64 MB** | 21.61–21.63 MB |
| 其中 Pss_Anon / File / Shmem | 19.29 / 2.25 / 0.10 MB | 19.22 / 2.28 / 0.12 MB |
| Rss | 25.37 MB | 25.36 MB |
| dma-buf 去重合计 | **16.98 MiB / 10 块** | 18.62 MiB / 12 块 |
| 其中最大一块 | **16.10 MiB** = APUBUF0 = bin 的页对齐大小（16,878,592 → 16,879,616） | 同 16.10 MiB |
| 第二块 | — | **1.57 MiB** = 该图未共享的静态（extract-shared 报 1,637,152 B） |
| rchar（装载后冻结） | 17,169,689 B，**不变** | 17,155,353 B，**不变** |
| read_bytes | **0** | **0** |
| MemAvailable 变化 | 7,543,684 → 7,532,604 kB（−11.1 MB） | 7,529,872 → 7,528,988 kB（−0.9 MB） |

三条读法：

1. **外部 bin 在 NPU 的 dma-buf 里，进程 PSS 不含它**：一块 16.10 MiB 的 APUBUF0 正好是 bin 的页对齐大小，
   PSS 只有 21.6 MB。这与厂商「250.7 MB 权重是一块 dma-buf、Rss=0、PSS 不含」是**同一形态**（票 01 §6）。
2. **一次装载、多次 Run 是稳定的**：400 次 Run 期间 PSS 只抖 30 kB（噪声级），rchar 从装载结束起**一个字节不涨**，
   read_bytes 恒 0 → 运行期不重读图/权重。
3. **不能与厂商的 346–355 MB 横向比**：他们没有把 16-bit 词嵌入从堆里摘出去（259.7 MB 的 scudo 块），
   我们这张图没有嵌入、没有 KV、没有 lm_head。可比的是**形态**（外部权重走 dma-buf、PSS 不含权重），不是绝对值。

---

## 7. 共享权重的完整性三联（票 03 警告：设备**不校验** bin 内容）

| 物 | sha256 |
|---|---|
| bin（T=16 / T=8 配对，ops/q05_shared.bin） | 6b58dd0bf6112123bf26b8e98d383624cebf9ef44f498a01740955debe1ab371 |
| bin（T=128 / T=16 配对，ops/q05_shared_t128.bin） | c1854263c8bbef3dce63fd8ab17a0de5a0e0b5a3f14fc9725f1e5fa54b2f5acf |
| 图 ops/q05_prefill_t16_24l/model_shared.dla | 968bf97006387813041f3f2201f12e3dff330877889b196736237ef61b429c03 |
| 图 ops/q05_prefill_t8_24l/model_shared.dla | e8144c9146071732029e6c46b4cb85bd2748faedff1520e5a903fc785dad61b4 |
| 图 ops/q05_prefill_t128_24l/model_shared.dla | faf702ea185ea06a0d49aa56bb0dade8e0b027e24b860245765cff786193299e |
| 输出 T=16（out_r1/r2/r3 与 400-Run 的 out_mem.bin **全同**） | 69b8ba5a66cbf1b8c6af694543fd37d59269f6b31c6fb2982a9e7f4783d85421 |
| 输出 T=8（r1/r2/r3 与 out_mem.bin 全同） | 97fcf057cef6c8b2135674818af607c4c0b7eba0304f8bf270ba4ffef00a0575 |
| 输出 T=128（r1/r2/r3 全同） | 3638774c959cd097ba2dacb6a0b4403cc936313f9fc1ab7199cc3301fa42f046 |

（每张图目录里的 e2e.json 记了 bin / dla / out 三个 sha256，tools/e2e_q05.sh 自动写。）
**两张图共用的那份 bin 与 T=128 配对的 bin 不同**（内容不同：后者含 T=128 的每层 mask），
说明 **bin 必须与图配对使用**，而设备不会替你检查——配错就是静默算错（票 03 §4.5 实测过全零 bin）。

---

## 8. 判据对照（ADR 0005 第六次订正）

### 8.1 实现门（硬，全部 PASS）

| 门 | 登记值 | T=8 | T=16 | T=128 |
|---|---|---|---|---|
| ｜Δcos(device, 量化 golden)｜ | ≤ 0.003 | 1.19e-5 **PASS** | 4.68e-5 **PASS** | 7.54e-6 **PASS** |
| rel_l2(device, golden) | ≤ 0.05 | 4.97e-3 **PASS** | 5.44e-3 **PASS** | 7.39e-3 **PASS** |
| 码级 max | ≤ 1200 | 267 **PASS** | 354 **PASS** | 512 **PASS** |

⇒ **图搭对了、设备忠实**。24 层串联没有把设备舍入放大（码级 max 354 对单层 247，只涨 1.4×）。

### 8.2 质量期望（按档，三张全不过 —— 照实报）

| 量 | 单层（票 04f / 本票 T=16 单层回归） | 24 层 T=16 | 24 层 T=8 | 24 层 T=128 |
|---|---|---|---|---|
| device vs fp64 cosine | 0.9998598（04f）/ 0.9998446 | 0.9961469 | 0.9967964 | 0.9636675 |
| 质量期望（int8 权重） | ≥0.9998 | **FAIL** | **FAIL** | **FAIL** |

**为什么 24 层一定掉**（本票不信口径、给了拆分）：

- 单层 1−cos 在 T=16 是 **1.55e-4**（本票 N=1 回归 0.99984454）；
- 2 层是 **2.8e-4**（0.9997203），24 层是 **3.8e-3**（0.9961937）—— 与「每层加一点、按层线性累积」吻合；
- ADR 第六次订正已经把 0.9998 定义为**单层/算子级**的期望，并把**最终质量门放到整网 logits（票 06）**。
  本票据此不改任何阈值，**把 0.9962 / 0.9968 作为「24 层串联的实测值」记录**。

### 8.3 顺带一条对档位有意义的新数据：T=128 的 per-tensor int16 被拉坏

单层 device-free 的 golden vs fp64：

| 单层，T=16 | 单层，T=128 |
|---|---|
| 0.9998445（1−cos 1.6e-4） | **0.9965765（1−cos 3.4e-3）** |

**同样的图、同样的权重，只把 T 从 16 换成 128，单层就掉了 22 倍。** 机制是 per-tensor（整张激活一个 scale）：
128 行 × 896 列的 max 比 16 行更容易被 outlier 顶高，典型值落在更粗的格点上；再叠加 24 层累积 → 0.9637。
⇒ 若票 06 的 prefill 要跑 128 token，**per-tensor 激活尺度需要按行（或按 head）细化，或把 T 分块**；
本票没有 bisect 到具体阶段（见 UNKNOWN）。

---

## 9. 与厂商形态的差距（数量级对齐，不做性能结论）

| 物 | 厂商（04 层 W4，24 个不同层） | 本票（int8，同一套夹具权重 ×24） |
|---|---|---|
| prefill 图 | 3,238,205 B | **1,489,594 B**（T=16） |
| 权重 bin | 250,737,664 B（4-bit 打包 + 分组 scale） | **16,878,592 B**（int8，**一份**权重） |
| 空载 PSS | 346–355 MB（含 259.7 MB 词嵌入堆拷贝） | 21.6 MB（无嵌入、无 KV、无 lm_head） |
| prefill 时间 | 128 token / 87 ms（0.68 ms/token） | 128 token / 94.55 ms（**0.74 ms/token**） |

- **ms/token 与厂商同档**（0.74 vs 0.68），而我们用的是 int8 权重（他们是 4-bit）——**这条不宜当性能结论**：
  权重来源不同（夹具 vs 真权重）、没有 lm_head、没有 KV、T 相同但图内容不同。
- **bin 体积的真实差距已经量出来了**：我们 16.9 MB 是**一份**权重；真实 24 层不同权重按 2 层控制实验外推
  ≈ **366 MB**（int8），而厂商 4-bit 只要 250.7 MB —— 差的正是「8-bit vs 4-bit」（ADR 第六次订正写的 2× 代价）。

---

## 10. UNKNOWN（不圆）

1. **T=128 单层为什么从 0.99984 掉到 0.99658**：只做了「同图换 T」的对照，没有逐段（n1 / q3 / scores / probs /
   attn / mlp_y）bisect 到具体阶段。猜测是 per-tensor 激活尺度 + 每层 mask 常量，未证。
2. **ncc 去重的机制**（内容哈希？还是张量字节比对？）：只用 2 层控制实验证明了「字节不同就不合并」，
   没有从工具侧确认判据，也没有试「部分相同」的中间情形。
3. **24 层链上设备舍入的传播规律**：实测码级 max 从单层 247 只涨到 354、T=128 到 512，**没有**按 √24 涨满；
   但只有 3 个点，不足以给出模型。
4. **延迟没有时钟/热态控制**（同 04c/04e/04f），两次会话差 1.3 ms；0.95 ms/层 只是形态对比，不是性能结论。
5. **只做了 prefill**：decode（T=1）+ KV cache 不在本票；两张图共用一个 bin 的**真实**用例（prefill+decode）
   本票没有跑，只证明了「两张 prefill 图共用一份 bin」成立。

---

## 11. 复现

```bash
export QWEN25_WEIGHTS_DIR="/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen"
export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
WT="/mnt/d/deepseek_ workspace/mtk/wt-q05"

# 1) 三张图（编译放锁外；24 层各 ~30 s build + 2-3 min ncc）
/usr/bin/python3 tools/build_24layer.py --layers 24 --t 16  --out ops/q05_prefill_t16_24l
/usr/bin/python3 tools/build_24layer.py --layers 24 --t 8   --out ops/q05_prefill_t8_24l
/usr/bin/python3 tools/build_24layer.py --layers 24 --t 128 --out ops/q05_prefill_t128_24l
./tools/compile.sh ops/q05_prefill_t16_24l/model.tflite ops/q05_prefill_t16_24l/model.dla
# N=1 回归（对官方单层）
/usr/bin/python3 tools/build_w4a8.py layer --w-bits 8 --a-bits 16 --seam native --out /tmp/ref_l1
/usr/bin/python3 tools/build_24layer.py --layers 1 --t 16 --out /tmp/my_l1
# 去重控制
/usr/bin/python3 tools/build_24layer.py --layers 2 --t 16 --out ops/q05_dedupe_same
/usr/bin/python3 tools/build_24layer.py --layers 2 --t 16 --distinct 0.01 --out ops/q05_dedupe_distinct

# 2) 抽共享权重
export LD_LIBRARY_PATH="/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk/host/lib"
.../host/bin/extract-shared -i ops/q05_prefill_t16_24l/model.dla,ops/q05_prefill_t8_24l/model.dla \
    -o ops/q05_prefill_t16_24l/model_shared.dla,ops/q05_prefill_t8_24l/model_shared.dla \
    -s ops/q05_shared.bin

# 3) 板端（一个锁会话；每块前后 healthcheck）
./tools/devlock.sh run -- ./tools/healthcheck.sh --quick
./tools/devlock.sh run -- ./tools/e2e_q05.sh ops/q05_prefill_t16_24l ops/q05_prefill_t8_24l
./tools/devlock.sh run -- ./tools/repeat_q05.sh ops/q05_prefill_t16_24l 3
./tools/devlock.sh run -- ./tools/mem_q05.sh ops/q05_prefill_t16_24l 400
./tools/devlock.sh run -- ./tools/healthcheck.sh --quick
```
**可复现性**：同参数重建 model.tflite / golden/*.bin 逐字节稳定；.dla 不可复现（仓库既有约定），
所以本票的绑定靠 **bin / dla / 输出 三个 sha256**（§7），不靠 .dla 的哈希做回归判据。

---

## 12. 与票面 checklist 对照

| 票面要求 | 状态 | 证据 |
|---|---|---|
| 24 层整块图（复用 04 的构件），权重外部化 | ✅ | 三张图 1440 层 / 1 执行步 / EXIT=0；extract-shared 出 1.49 MB 图 + 16.88 MB bin |
| 编译：层数、执行步、无 bridge、Static、外部权重字节 | ✅ | §3.1 + §4.1 |
| 上板：T=8 与 T=128（+T=16）prefill，与 CPU 链对拍（码级 + cosine） | ✅ | §5（三张，全部持锁，前后 healthcheck PASS） |
| 常驻内存（PSS / DMA-BUF）+「一次装载、多次 Run」稳定性 | ✅ | §6（400 次 Run，PSS 抖 30 kB，rchar 冻结）+ §5 噪声底 0 |
| 若 03 说「不能外部化」则退到 24 层单图内嵌 | — | 03 说能，且本次实测成立，**没有走降级** |
| 若 24 层一个图编不过则降级为分块 | — | **不需要**：1440 层一次编过（1 执行步），**降级链未触发** |
| 不以改档位换通过 | ✅ | builder 的 lock_tier() 只接受 8W16A + native |
