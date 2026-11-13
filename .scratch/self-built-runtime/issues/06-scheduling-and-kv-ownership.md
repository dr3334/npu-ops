# 06: 多图调度与 KV 归属（自建库自己管）

**Type:** grilling
**Status:** open
**Blocked by:** 01, 05

## Question

自建库跑 1 层（后续 24 层）时谁负责：①按层/按图切换执行；②权重与激活缓冲的分配与复用；③KV cache 的持有与每步写入。

**已知**：厂商侧答案是「RuntimeV2 每请求 IOBuffer」（docs/adr/0006）；自建库若自己持有 KV，就要把新行写进常驻 dma-buf 并让命令流引用它 —— 即 docs/MAPPING.md 的缓冲绑定层。

**要拍**：本图（1 层）先不做 KV；24 层那张图必须在这里定形态。
