# 04: 验收口径落定：同源等价 + CPU golden 两道

**Type:** grilling
**Status:** open
**Blocked by:** None

## Question

把两道验收写成规则——①同源等价：同一命令流同一输入，自建库输出与厂商 runtime 输出逐字节相同（不同 ⇒ 提交层问题）；②数值正确：自建库输出 vs CPU golden（int8 逐位 / fp16 容差）。①不得替代②，②不得替代①。

**要拍**：术语（CONTEXT.md 新增「同源等价」）、写进 docs/AGENT-BRIEF.md 的位置、docs/OWN-SUBMITTER.md 覆盖表是否加一列「厂商输出一致性」。
