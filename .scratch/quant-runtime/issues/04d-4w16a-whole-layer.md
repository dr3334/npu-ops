# 04d: 4W16A 整层（质量决定性实验）

**What to build:** 用票 04c 的 harness，把激活从 int8 换成 **int16**（权重保持 int4 per-channel），量整层 cosine。这是「4W16A 能不能越过 0.9999」的唯一决定性证据，也是厂商出厂格式的复刻。

**Blocked by:** 04c

**Status:** done

- [ ] 沿用 `tools/build_w4a8.py`（已在 04c 参数化成 `--w-bits 4|8`），加 `--a-bits 8|16`：激活与输出 int16、bias 仍 int32 per-channel、op version 取 ncc 接受的那几档（int16 激活实测 {1,3,4,5,6,7,9} 可编，canonical v13 被拒）
- [ ] 组件级先过：FC / SwiGLU / BMM-QK / BMM-PV 的 4W16A 版（板端 device vs 量化 golden；vs fp64 cosine）
- [ ] 整层 4W16A（T=16 seam）：层数/执行步/Static/噪声底/码级误差/cosine
- [ ] 关键对照：`scores → probs` 段的 cosine 是否从 0.96 抬到 ≥0.999；整层是否 ≥0.9999
- [ ] 记录 int16 激活的代价：`.dla` 体积、Static、板端单层时间（与 4W8A 1.44 ms / 8W8A 2.17 ms 对比）
- [ ] 若 int16 激活在某个 seam 上编不过（例如 CONCAT/ADD 的 dtype 组合），记录该 seam 并说明用 fp16 替的具体位置与层数代价

## Comments
- 完成：int16 激活把 probs 从 0.96 抬到 0.9999993，整层夹具 0.9510 = int4 权重天花板；native 缝 59 层、不需要 fp16 缝。
