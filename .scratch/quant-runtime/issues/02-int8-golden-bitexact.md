# 02: 把 int8 golden 修到逐位（两次舍入 + M0 取整）

**What to build:** 让 `int8_tol=1` 从「规则未知」的兜底变成「残留 1 码」的例外：golden 按量化契约实现，离线重放已有板端 dump 验证 0 失配。

**Blocked by:** None (can start immediately)

**Status:** done

- [ ] `tools/build_ops.py`（FC）与 `tools/build_bmm.py` 的 golden 改成两次舍入：`round_half_up(acc·M0/2^31)` → `RoundingDivideByPOT(shift)`，M0 用整乘子（如 1/32 → 2³⁰），不从 float32 scale 反算
- [ ] 离线重放：用已有 int8 op 的板端输出（没有就重跑一次）验证 0 失配；对照「历史 float64 单次舍入」差多少
- [ ] 受影响的 `ops/*/model.json` 里 `int8_tol` 从 1 收到 0（或写明为何仍需 1）；`compare.txt` 重跑刷新
- [ ] 顺手订正 `docs/OP-MATRIX.md` §3 与 `docs/OP-convpool.md` 里「conv golden 仍用旧 rational 规则」的残留在案项（div3 的 167/512）
- [ ] 记录：哪些形状/乘子仍需 ±1（per-channel scale、非对称 zp、int16 累加都还没验证）

## Comments
- 完成：quant_contract（两次舍入 + 整乘子），145,296 值 0 失配（旧规则 127），17 个 op 收为 0 码；公共工具改动为票面授权例外。
