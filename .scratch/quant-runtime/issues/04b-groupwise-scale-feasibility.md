# 04b: 分组 scale（group-wise）能否在本栈表达

**What to build:** 回答「4-bit 的 14–16% 每矩阵误差能不能靠分组 scale 压下去」。这是票 05 选档的前提。

**Blocked by:** 04

**Status:** done

- [ ] 查 TFLite schema 与 ncc：per-axis（quantized_dimension）之外还有没有分组量化的表达方式；厂商 MTKEXT 算子的 qparams 可否经 `--import_qparams_config_file` 导入我们的图
- [ ] 试两种降误差的可行形态：① 把 K 切成 G 段、每段一次 FC 再求和（代价 G−1 层/矩阵）；② 把 N 切块、每块独立 per-channel（对 K 方向的动态范围没用）
- [ ] 用厂商 `shared_weights_2048c_0.bin` 的体积口径（0.5075 B/param ≈ 4bit + 1.5% scale）反推他们的分组大小，作为目标误差的参照
- [ ] 给出「能/不能/代价多少倍层数」的结论，并附最小复现（编译 + 板端各一例）

## Comments
- 完成（结论：不可行）：分组 scale 属性级不可表达；K 切段可表达但代价 2G−1 层/矩阵且救不了 4-bit；厂商分组是 256、增益在 hessian/AdaRound。
