# 06: prefill/decode 双图 + KV cache

**What to build:** 补上 decode 路径：prefill 与 decode **共用一份外部权重 bin**，KV cache 跨 token 保留，连续 N 个 token 的输出与 CPU 链对拍。

**Blocked by:** 05

**Status:** ready-for-agent

票 05 给的三条设计约束（必须遵守）：
1. **mask / RoPE 表提成运行时输入**——否则 mask 随 T² 进 Static，bin 与 T 绑定，prefill 与 decode 无法共用一份 bin（厂商的 prefill 图就是把 mask 当输入）；
2. **每层权重复用是夹具的假象**：`Static 15M` 不代表真实占用，真实 24 层 ≈366 MB；内存预算按 366 MB + int16 embedding 260 MB 算；
3. **T=128 的 per-tensor int16 会被 outlier 拉坏**（单层 0.99658）——decode 用 T=1、prefill 用 T=8/16 时不受影响，若要做 128 需先按行/按 head 细化尺度。

- [ ] prefill 图与 decode 图 **共用一份 bin**（`extract-shared` 要求两张图权重字节一致；mask/RoPE 作输入后 T 不再进 bin）
- [ ] decode 图：T=1（若 16 B 行规则挡住则 padding 到 T=8/16 并只用第 0 行，记录取舍）
- [ ] KV cache：24 层 × 2 KV head × 64 × 2048 × int16 ≈ **12.6 MB**；决定它在图上如何进出（TFLite 是函数式，写入只能是输出 → host 拷回），记录 per-token 的拷贝成本
- [ ] 正确性：连续 N（≥8）个 token 的 hidden/logits 与 CPU 参考逐步对拍，**每一步都记**，不只比最后一步
- [ ] 判据：实现门（device == 量化 golden + |Δcos| ≤0.003）；最终质量门在 logits（cosine / token 命中率），并写清「这一档能说明什么、不能说明什么」
- [ ] 记录每个 token 的延迟分解：权重带宽（共享 bin 常驻，应无重载）+ KV 读写 + host 拷贝