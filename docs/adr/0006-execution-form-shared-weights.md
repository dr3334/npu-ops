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
