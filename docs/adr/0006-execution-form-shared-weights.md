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
