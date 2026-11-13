# 03: 摸清 ncc 的外部/共享权重机制

**What to build:** 回答「我们能不能自己造一份『图 + 外部权重 bin』」。这是第 05 张票的前提，也是当前最大 UNKNOWN。

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] 查 ncc-tflite 的编译开关（`--help` 全量）与 SDK 文档：有没有 external/shared weight、权重外置、多图共享常量的选项
- [ ] 解剖厂商 `.dla`：`24layer_1t2048c_0_shared.dla` 里是否记录了权重 bin 的路径/偏移/校验；`strings`、头部字段、与 `shared_weights_2048c_0.bin` 的对应关系
- [ ] 查 `shared_weights_*.bin` 的内部布局（头部/张量目录/是否 nibble 打包 + 分组 scale）
- [ ] 用一个最小图 + 手工外置权重做编译探针（能编就先看是否生成外部引用）
- [ ] 结论写成「能/不能/需要什么工具」，并明确第 05 张票是否成立
- [ ] 记录到 `docs/OP-shared-weights.md`（假设/预测/做法/结果/适用范围/剩余疑问）