# 04e: 8W8A 用真实权重在设备上复核（档位决定）

**What to build:** 夹具给出的「int8 激活天花板 0.96」可能是夹具性质：真实 Qwen2.5-0.5B 权重离线跑 8W8A 单层是 0.9999157。这一票用**真实权重**在**设备上**复核同一形状，决定主档到底是 8W8A 还是 4W16A。

**Blocked by:** 04b, 04c

**Status:** ready-for-agent

- [ ] 从 `D:\ai\Qwen2.5-0.5B-Instruct`（bf16 safetensors）取 **layer0** 的 q/k/v/o/gate/up/down 权重，按 per-channel max-abs 量化到 int8（记录 scale 生成规则）
- [ ] 输入用真实 token 的 embed 行（与票 08/09 同法）或真实 prompt 的 hidden，记录选择
- [ ] 设备上跑 8W8A 单层（T=16 seam），与独立 fp64 链（同一份真实权重、同一输入）比 cosine / rel_l2；分段给 `scores → probs → out`
- [ ] 同一 harness 再跑夹具权重一版，把「夹具 vs 真权重」的差异写成一张对照表（这是本条最重要的产出）
- [ ] 结论：8W8A 在真实权重下是否 ≥0.9999（或差多少）；若成立则主档锁 8W8A，否则与 04d 的 4W16A 结果二选一
- [ ] 记录：真实权重的 int8 量化误差（每矩阵）、`.dla` 体积、板端单层时间