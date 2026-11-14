# 计划先记在本地 `.scratch/`，不记进公开的 GitHub Issues

**Status**: accepted

`origin` 是公开仓库 `dr3334/npu-ops`，而本阶段的代码与提交只留在本地（分支 `feat/qwen25-attention`，不推远端）。spec 与实现票如果直接发到 GitHub Issues，等于把尚未完成的路线图即时公开——代码不推、计划却推，是自相矛盾的口径。所以 spec 与 issue 一律落在 `.scratch/<feature>/`，轨道迁移等整块 PASS、由人决定公开时机后再做一次。

**Considered Options**
- 直接发 GitHub Issues：零摩擦，但与"不推远端"冲突
- spec 落 `docs/`、票落 `.scratch/`：两套约定并存，读的人要多记一条规则

**Consequences**
- `/to-tickets` 产出的阻塞边是文本行 `Blocked by: NN`，而不是 GitHub 的原生 dependency；前沿要靠扫文件算，不能靠 UI
- 迁移到 GitHub 时是一次性搬运，票号需要重编
- 仓库设为 public 这件事本身没有改变，随时可以推翻本决定——代价是已经把计划公开了，收不回来
