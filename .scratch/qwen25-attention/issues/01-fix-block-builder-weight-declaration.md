# 01: 修块构造器的权重张量声明

**What to build:** 让 `heads>1` 的块实验重新能进图。当前块构造器按 FC 的**目标宽度**声明权重张量，而 `heads>1` 时传进去的权重数组行数不足，flatbuffer 于是声明了一个远大于实际数组的缓冲区。编译器在**模型加载**阶段就退出，且这条失败被误读成"编译器不支持多头"，进而挡住了所有多头实验。修好之后，`heads>1`（含 14 头）的块图应当能编到单个执行步。

**Blocked by:** None (can start immediately)

**Status:** done

- [ ] `heads=2` 与 `heads=14` 的块图均编译通过（EXIT=0、无 bridge、单执行步）
- [ ] 权重数组尺寸与声明形状不一致时**立刻**报错，而不是交给编译器去报一句含混的话
- [ ] 证据订正落地：不再把多 head 描述成编译器不支持，并指向新的实验记录
- [ ] 单头路径行为不变，既有单头块实验仍然 PASS
- [ ] 在实验记录里写清楚"加载期 / 逐层校验期 / 后端期"三种错误措辞的区分，供后续排障引用
## Comments

- 2026-11-13 Lead: 合并到 feat/qwen25-attention（merge 2220a2f）。h2=31 层 / h14=33 层，均单执行步无 bridge，板端 PASS（h14 契约 3e-2，见 model.json tolerance_revision）。额外发现：多头必须 CONCAT-free 的 RESHAPE+TRANSPOSE 重排（见 docs/OP-block.md §10）。

