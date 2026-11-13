# 04f: 真权重下的档位矩阵，决定主档

**What to build:** 用同一 harness 在**真权重**上把三个还没测的组合补齐：4W8A、4W16A、8W16A，与已测的 8W8A（0.8801）一起构成完整的二维矩阵，据此锁主档。

**Blocked by:** 04d, 04e

**Status:** done

已知（真权重、设备）：8W8A = **0.8801**（probs 0.7677，int8 激活是瓶颈）；夹具下 4W16A = **0.9510**（int16 把 probs 修到 0.9999993，天花板改成 int4 权重）。

- [ ] `--w-bits {4,8} --a-bits {8,16} --weights real --input real`，四个组合里补 4W8A / 4W16A / 8W16A 三格（8W8A 已有，沿用同判据复核一次）
- [ ] 每格给：整层 device vs fp64 的 cosine、分段表（`scores`/`probs`/`out`）、码级误差、`|Δcos(device, golden)|`
- [ ] 预期与判据：若 **8W16A ≥ 0.9999**（权重侧 0.9999157、激活 int16 已把 probs 修好），则主档锁 8W16A；若 4W16A 在真权重下也 ≥0.999 则作为「更省内存」的备选
- [ ] 记录每格的 `.dla` 体积 / Static / 板端单层时间（8W16A 的 .dla ≈ 15 MB、4W16A ≈ 7.6 MB）
- [ ] 结论写进 `docs/OP-tier-matrix.md` 与 ADR 0005 的第五次订正
- [ ] 若某格编不过（例如 int16 激活的某个 seam），记录报错阶段与替代方案（native seam / fp16 缝）

## Comments
- 完成：真权重四格矩阵，8W16A+native 0.9998598 唯一近标档；缺口 84% 来自 int8 权重、器件定点链路 4.59e-5；主档锁 8W16A。
