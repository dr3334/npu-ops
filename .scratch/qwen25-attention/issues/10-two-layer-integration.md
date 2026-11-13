# 10: 2 层 transformer 集成对拍

**What to build:** 本 spec 的终点证据。两层 decoder 串联 + 输出投影，从 `[T,896]` 一路算到词表 logits，全程与同一份冻结随机权重的 CPU 参考对拍。两层足以暴露所有跨层问题（残差跨层累积、层与层之间的契约、每层权重的重新装载），继续加到 24 层不再提供新的信息，只是 12 倍的机械重复。

**Blocked by:** 08, 09

**Status:** done

- [ ] 两层各是一个独立的图；第 1 层的板端输出**直接**作为第 2 层的输入，不经 CPU 修正
- [ ] 两层的 hidden 各自落盘并与 golden 比对，误差可定位到具体层
- [ ] 经输出投影得到 logits，logits 向量 atol/rtol 对拍通过
- [ ] 记录跨层重新装载权重的方式与耗时
- [ ] **top-1 仍不启用**；本票不含真实权重
- [ ] 在实验记录里写明"这一档证据能说明什么、不能说明什么"（结构对 ≠ 真实 prefill 可用）

## Comments

- 2026-11-13 Lead: 合并到 feat/qwen25-attention（merge cf0f9ad）。三图（l0/l1/logits）各自 EXIT=0/1 执行步/无 bridge；板端三个 compare 全 PASS（h1 9.50 ULP、h2 6.50 ULP、logits 7.50 ULP，噪声底 0）。交接方式=device out_device.bin 字节覆盖下一层 golden/input.bin（cmp 相同，无 CPU 修正）。传播误差 vs 本层误差已分开：h2 总 6.348e-3 → 本层 3.418e-3；logits 总 3.662e-3 → 本层 2.441e-4。反事实（换错权重 24.9×、跳过第 0 层 25.3–30.3×、分块顺序反 37.9×）证明判据不空转。
