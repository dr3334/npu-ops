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
