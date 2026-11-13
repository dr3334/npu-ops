# 04: GQA 打分 + KV 显式复制

**What to build:** 分组查询注意力的打分那一半。真实模型是 14 个 query head 对 2 个 KV head，编译器**不会**替我们广播——2 个 KV head 必须在图内显式复制成 14 份，否则形状对不上。复制要用已证的形状算子完成，并且不能让数据回到 host。

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] 2 个 KV head 在图内复制成 14 份（用 SPLIT + CONCATENATE），数据不回到 host
- [ ] 3-D BMM（adj_y）产出 `[14,8,8]` 的 scores，与 golden 对上
- [ ] 记录复制带来的层数开销与执行步数（预期仍是单执行步）
- [ ] 明确记录**不采用**"2×7 头"与"14 次单头 unroll"两种兜底，并写出它们的层数代价
- [ ] 记录 rank 上限是 4：5-D/6-D 会明确报错，不要试
- [ ] 记录本算子的实测噪声底
