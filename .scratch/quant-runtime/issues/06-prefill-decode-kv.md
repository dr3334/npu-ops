# 06: prefill/decode 双图 + KV cache

**What to build:** 补上 decode 路径：T=1 的图 + 跨 token 的 KV cache（cacheSize 2048，int16），并验证跨 token 状态正确。

**Blocked by:** 05

**Status:** ready-for-agent

- [ ] KV cache 的存放形态（设备内存张量 / 外部 bin / 图内常量）选型并记录
- [ ] decode 图：T=1、读取并更新 KV、单执行步
- [ ] 正确性：连续 N 个 token 的 hidden/logits 与 CPU 参考对拍（每步都要，不能只比最后一步）
- [ ] 记录每个 token 的延迟分解（权重带宽 / KV 读写 / 装载是否为零）
- [ ] 明确「这一档能说明什么、不能说明什么」（不能说明采样/长上下文正确）