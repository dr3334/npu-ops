# 执行形态抄厂商：两张图 + 外部共享权重 + KV，而不是每层一个 `.dla`

**Status**: accepted

板上现成的厂商资产给出了他们的答案：`24layer_128t2048c_0_shared.dla`（3.2 MB，prefill）、`24layer_1t2048c_0_shared.dla`（21.6 MB，decode）、`shared_weights_2048c_0.bin`（250.7 MB，24 层 W4 权重）、`embedding_..._int16.bin`（272.3 MB，INT16 词嵌入）。权重外部化、两张图各自承载一种执行形态、`SwapModel` 只在 prefill↔decode 边界发生。

对照我们自己的做法：`每层一个 .dla` 是票 09 为了「误差可定位到层」选的**诊断形态**，代价是每层一次 145 ms 的进程级装载（24 层 ≈ 3.5 s/token）。

决定：**部署形态抄厂商**（一个 prefill 图 + 一个 decode 图 + 外部共享权重 + KV cache），`每层一个 .dla` 只保留为诊断手段，不作为部署形态。第一块砖是「最小 runtime 探针」：用厂商 SDK 的 RuntimeAPI 自己装载两张图 + 共享权重、重复 `Run()`、量 PSS 与 DMA-BUF，证明「执行计划复用」能被我们自己的进程握住。

**Considered Options**
- 继续每层一个 `.dla`，靠进程池或 Loader 优化：装载成本是进程级的，优化空间有限；且与厂商资产形态不同构，无法对照。
- 直接把整网交给厂商 runtime：那就没有「自建执行路径」这件事了（ADR 0002 的目标作废）。

**Consequences**
- 需要先确认 ncc 的外部/共享权重机制我们能不能用——这是当前最大 UNKNOWN。
- KV cache、decode 图、词嵌入装载都是新工作，之前的 spec 明确把它们放在范围外。
---

## 实测补充（2026-11-13，票 q05 之后）

24 层单图 + 外部共享权重已经在板上跑通，四条会影响后续做法的实测：

1. **1440 层（24×60）仍能装进一个 `.dla`、单执行步、无 bridge**；三档 T=8/16/128 都成立。板端 prefill：T=8 **17.5 ms**、T=16 **22.8 ms**、T=128 **94.6 ms**（0.74 ms/token，与厂商 0.68 同档但不可横比）。
2. **`ncc` 会合并字节相同的静态张量**：2 层控制实验的 Static 是 14M vs 28M，因此 24 层图的 Static 15M 是「**权重复用**」造成的假象（夹具只有 layer0/layer1 两套）。**真实 24 层权重约 366 MB**，做内存预算时必须按这个数，不能看 Static。
3. **每层 causal mask 随 T² 进 Static**（T=128 让 bin 多出 ~11 MB）⇒ 要把 **mask / RoPE 表提成运行时输入**（厂商的 prefill 图正是这么做的），否则共享 bin 与 T 绑定，prefill 与 decode 也不可能共用一份。这是票 06 的第一条设计约束。
4. **int16 下 T=8 编得过**（`[14,8,8]×2B` 正好 16 B 行，int8 时被行规则挡住）；但**单层 T=128 的质量掉到 0.99658**（per-tensor int16 被 128 行里的 outlier 拉坏）⇒ 128 token 要按行/按 head 细化激活尺度，或分块，否则它会成为 24 层累积误差的主要来源。

另外，常驻形态已经与厂商同构：PSS **21.6 MB**（Anon 19.3 / File 2.25）、dma-buf **16.98 MiB / 10 块**（最大一块 16.10 MiB = bin 页对齐，CPU 从不触页）、`rchar` 冻结、`read_bytes=0`；**400 次 `Run()` 与单次运行逐字节相同**。
### 实测闭环（2026-11-13，票 q06 之后）

「两张图 + 一份外部权重」这条形态已经在我们自己的栈上完整跑通：

- **共用一份 bin**：prefill(T=16，1346 算子) 与 decode(T=8/C=2048，1396 算子) 都单执行步、无 bridge；`extract-shared` 出 bin **16,674,816 B**；prefill 的 nonreuse 张量 **0 个**、decode 只有 **2 个/8,224 B**（mask/RoPE 提成输入后，票 05 里 1.64 MB 的 nonreuse 基本消失）。三张不同 T 的图抽出的 bin **体积完全相同** ⇒ bin 与 T 无关。
- **T=1 编不过**（prefill 图报 SOFTMAX `Act1 shift exceeds range of MDLA command field`；decode 图报 Input↔MDLA gap，两者在 T=8 消失）⇒ decode 走 **T=8 只用第 0 行**，其余 7 行同喂该 token、mask 放行槽位 `C+r`；代价是 **7/8 的 query 行算力浪费**（多算不多存）。这条要记成「T=1 在本栈不可编」的硬边界，除非 16 B 行规则或 SOFTMAX command field 有别的绕法。
- **KV cache 只能 host 持有**：K/V 各一张 `[48, C, 64]` int16（48 = 24 层 × 2 KV head）作为图输入，图每步只输出新行 `[48, T, 64]`，host 写第 m 行后整块回传。体积 **24.0 MiB**（12.0 MiB × 2；此前按 12.6 MB 只算了 K）。per-token 拷贝 25.2 MB 上行 + 0.2 MB 下行（adb 实测 0.73–0.97 s / 0.26–0.35 s）——**这一项是当前 decode 延迟的大头**，`RuntimeAPI` 常驻可降到 O(256 B/层)，是下一轮的第一块砖。
- **延迟分解**（C=2048）：decode **68.5 ms/token** = KV 宽度 **33.4 ms** + 塔本体 35.1 ms + **权重带宽 ≈0**（bin 常驻 dma-buf，200 次 `Run()` 只读一次，`rchar` 冻结 42.59 MB）。对照厂商 17.2 ms/token：差距主要在 KV 的 host 拷贝与 T=8 padding，不在权重。
- **逐 token 正确性**（8 步，每步都记）：`|Δcos(device, 量化 golden)| ≤ 8.2e-5`（门 0.003）、码级 max ≤370（门 1200）、噪声底 0 字节；vs 独立 fp64 链 hidden/logits cosine **0.9727–0.9985**、**argmax 6/8**。prefill 的 `|Δcos|` 2.24e-5、cos 0.996148。

结论：本 ADR 的形态判断被证实——**「两张图 + 一份 bin」可行、KV 是 host 侧状态、权重可以真正常驻**；剩余差距是**工程**（KV 常驻、T=1 的绕法、padding 成本），不是形态问题。

## 订正（2026-11-13，票 kv1 之后）：KV 常驻要换 RuntimeV2，且每 token 多一次 24 MiB 拷贝

上一节结尾断言「用 RuntimeAPI 把缓存指针常驻，per-token 拷贝可以从 25 MB 降到 O(256 B/层)」。
半对半错，位置很具体（docs/OP-kv1-resident-kv.md）：

- **v1 的 `NeuronRuntime_setInput(handle, ptr, len, {ionFd})` 不行**：用户 dma-buf 能绑上（rc=0），
  但图输入在第一次推理后就不再被读——host 就地写新行（读回 buffer 证明正好落下 6,144 B）、
  甚至另分配一块 dma-buf 重新绑，下一次 invoke 的输出**都逐字节不变**。控制实验（把同一份 cache
  预先写进种子文件）逐字节复现 CLI，所以不是行映射写错，是"绑一次 = 快照"。
- **v2 的 `NeuronRuntimeV2_run(runtime, SyncInferenceRequest)` 行**：每请求重送
  `IOBuffer{ptr,len,fd,offset}`，8 个 token 的 hidden/k/v **24/24 张量逐字节等于** CLI 路径。
  KV 缓冲因此可以一次分配、进程内反复用，host 每 token 只写 12,288 B（CLI 形态是 25.6 MB）。
- **但 v2 每请求把 24 MiB 拷进自己的 staging**：`/proc/self/stat` 的 CPU 差 12.95 ms/invoke
  （≈25.2 MB @1.9 GB/s）。所以 v2 每 invoke 62.9 ms，比"绑一次就不动"的 v1（49.4 ms）慢，
  比同会话 CLI（59.1 ms）也慢 7%。

结论口径：**"KV 常驻设备"已经做到（架构上不再是 25.6 MB/token 搬运），剩下的是运行时那 13 ms 的
拷贝与 T=8 的 7 行 padding**——两者都不在图的语义里，属运行时/编译侧优化。上一段"下一轮第一块砖"
的说法改成："这一块砖已经搬开，露出的是运行时 descriptor 处理与 T=1 编译这两块。"
### KV 归属与上下文长度：两条实测结论（2026-11-13，票 kv1 / kv-maxlen）

1. **KV 可以常驻设备，但只走 `RuntimeV2`**：v1 的 `setInput(handle, ptr, len, {ionFd})` 是**绑一次的快照**——绑上后 host 就地写新行、甚至换另一块 dma-buf 重绑，下一步输出都逐字节不变（三组实验钉死）。v2 的每请求 `IOBuffer{fd, offset}` 才行：KV 两块 12.58 MB dma-buf 一次分配、进程内反复用，**每 token host 只写 12,288 B**（vs 每 token 25.6 MB，2085×），8 个 token × 3 张量与 CLI 路径 **24/24 逐字节相同**。
   代价：运行时每请求把 24 MiB 拷进自己的 staging，多 **12.95 ms**（v2/CLI 每 invoke = 1.07）。**但端到端每 token 从 1.2–1.9 s（起进程 + adb push）降到 62.9 ms ⇒ 19–30×** —— 收益在「不再每 token 起进程」，不在单次 invoke。
2. **24 层 decode 图的 KV 上限 = C 8192**（今天）：编译 67 s(2048) / 84 s(4096) / **878 s(8192)**、ncc RSS 6.4→9.0 GB；C=16384 已解析出 1 个执行步与 1532 层清单，但**后端内存规划 55 分钟不返回**（RSS 9.05 GB，被 SIGTERM）。同 C 的单层只要 13 s ⇒ 墙是「层数 × 缓存宽度」的编译期内存规划，不是图输入上限、不是设备内存、不是执行步、不是 16 B 行规则。
   理论天花板由编译器点名：`L = C + t ≤ 65535` ⇒ C ≤ 65527；但 65527 又撞 `gap between Input and MDLA_5_0`，单层 57344 可编、65527 不可 ⇒ 真实硬边界在 (57344, 65527)，24 层形态下不可达。
   KV 宽度占每 token 时间 **59%(2048) / 74%(4096) / 88%(8192)**，且 4096→8192 超线性（×2.69；ncc Code 924K→1.6M，执行计划换了）⇒ **长上下文的首要成本是 KV 带宽，不是权重**。

**长上下文（≥16K）在本栈只有三条未实测的路**：拆图（每层或每 chunk 一张图，用重装换编译可行性）、降 `--opt`（换编译时间）、窗口滚动（C 固定、越界丢最老的 KV）。这也是「prefill 固定 32768」目前做不到的原因：24 层单图在 C=8192 就撞编译墙。
