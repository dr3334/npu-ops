# bb04-fc —— 差分探查自动报告

生成：/usr/bin/python3 tools/bbdiff.py --matrix tools/bbdiff-matrix-blackbox04.json（WSL /usr/bin/python3）
时间：2026-11-14T13:55:27+0800 · 每变体保留 5 次同类别编译（另跑 4 次做类别校验）· 编译入口 tools/compile.sh
噪声口径：判定用 **per-variant**；对照列同时给出 union（03/04 旧口径）

## 变体

| 变体 | tflite B | 权重 B | .dla B（保留各次） | rc | 尝试/保留/丢弃 | 自身噪声 B | 稳定字节 B |
|---|---|---|---|---|---|---|---|
| fc_a | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 115 | 7520 |
| fc_b_n128 | 9456 | 8192 | 11987 / 11987 / 11987 / 11987 / 11987 | 0/0/0/0/0 | 9/5/4 | 116 | 11871 |
| fc_c_k128 | 9200 | 8192 | 11731 / 11731 / 11731 / 11731 / 11731 | 0/0/0/0/0 | 9/5/4 | 60 | 11671 |
| fc_d_w_cal | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 110 | 7520 |
| fc_d_w_swap | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 115 | 7520 |
| fc_e_out16 | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 60 | 7575 |
| fc_e_abs2 | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 115 | 7520 |
| fc_e_out8 | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 110 | 7525 |
| fc_e_fp16 | 8896 | 8192 | 11731 / 11731 / 11731 / 11731 / 11731 | 0/0/0/0/0 | 9/5/4 | 115 | 11612 |
| fc_f_gno_ndf | 5104 | 4096 | 7645 / 7645 / 7645 / 7645 / 7645 | 0/0/0/0/0 | 9/5/4 | 105 | 7530 |
| fc_f_gno_smp | 5104 | 4096 | 7645 / 7645 / 7645 / 7645 / 7645 | 0/0/0/0/0 | 9/5/4 | 60 | 7585 |
| fc_f_gno_bmp | 5104 | 4096 | 7645 / 7645 / 7645 / 7645 / 7645 | 0/0/0/0/0 | 9/5/4 | 115 | 7530 |
| fc_f_gno_all | 5104 | 4096 | 7653 / 7653 / 7653 / 7653 / 7653 | 0/0/0/0/0 | 9/5/4 | 39 | 7614 |
| fc_g_meta | 5104 | 4096 | 7820 / 7820 / 7820 / 7820 / 7820 | 0/0/0/0/0 | 9/5/4 | 60 | 7760 |
| fc_h_resize | 5104 | 4096 | 7633 / 7633 / 7633 / 7633 / 7633 | 0/0/0/0/0 | 9/5/4 | 117 | 7512 |

### .dlb 打包（容器层，不进 .dla 字节差分）

| 变体 | .dla B | level0 | level9 |
|---|---|---|---|
| fc_a | 7635 | 8003 | 5858 |

## 已知噪声区（两次编译不稳定）

并集口径 **217 B** / 92 段（占最大产物 1.8%）；由每变体前 4 次编译两两比对取并集（第 5 次留作 holdout）。判定已改用**每变体各自标定**：跨变体差分只豁免**参与这次差分的两个变体**自己的噪声，逐段见 noise.txt。

每变体噪声 B：{"fc_a": 115, "fc_b_n128": 116, "fc_c_k128": 60, "fc_d_w_cal": 110, "fc_d_w_swap": 115, "fc_e_out16": 60, "fc_e_abs2": 115, "fc_e_out8": 110, "fc_e_fp16": 115, "fc_f_gno_ndf": 105, "fc_f_gno_smp": 60, "fc_f_gno_bmp": 115, "fc_f_gno_all": 39, "fc_g_meta": 60, "fc_h_resize": 117}

| zone | 字节 | 来源 |
|---|---|---|
| weight | 4096 | fc_a vs fc_d_w_cal（离噪） |

## 判定

「结果」按 per-variant 口径；「union 旧口径」列是同一份产物在旧口径下的判定（改口径前后对照）。

| check | kind | 结果 | union 旧口径 | 关键数字 |
|---|---|---|---|---|
| case-b-n-changed | diff_outside_noise | **PASS** | PASS | {"diff_bytes": 10308, "outside_noise": 10158, "inside_noise": 150, "noise_share": 0.0146} |
| case-b-size-grows | size_delta | **PASS** | - | {"delta": 4352, "ratio": 0.531} |
| case-c-k-changed | diff_outside_noise | **PASS** | PASS | {"diff_bytes": 9971, "outside_noise": 9885, "inside_noise": 86, "noise_share": 0.0086} |
| case-c-size-grows | size_delta | **PASS** | - | {"delta": 4096, "ratio": 0.5} |
| case-d-weights-confined | diff_within_zone | **PASS** | PASS | {"zone_bytes": 4096, "diff_bytes": 4136, "outside_noise": 4080, "inside_noise": 56, "escaped_zone": 0} |
| case-e1-out-scale-changed | diff_outside_noise | **FAIL** | FAIL | {"diff_bytes": 36, "outside_noise": 1, "inside_noise": 35, "noise_share": 0.9722} |
| case-e1-quant-enters-artifact | diff_outside_noise | **FAIL** | FAIL | {"diff_bytes": 36, "outside_noise": 1, "inside_noise": 35, "noise_share": 0.9722} |
| case-e2-abs-scale-changed | diff_outside_noise | **FAIL** | FAIL | {"diff_bytes": 101, "outside_noise": 0, "inside_noise": 101, "noise_share": 1.0} |
| case-e1b-out8-changed | diff_outside_noise | **FAIL** | FAIL | {"diff_bytes": 106, "outside_noise": 1, "inside_noise": 105, "noise_share": 0.9906} |
| case-e3-fp16-differs | diff_outside_noise | **PASS** | PASS | {"diff_bytes": 10040, "outside_noise": 9954, "inside_noise": 86, "noise_share": 0.0086} |
| case-repeat-b | repeatable | **PASS** | PASS | {"diff_bytes": 110, "outside_noise": 0, "inside_noise": 110, "size_delta": 0} |
| case-repeat-c | repeatable | **PASS** | PASS | {"diff_bytes": 37, "outside_noise": 0, "inside_noise": 37, "size_delta": 0} |
| case-repeat-e1b | repeatable | **PASS** | PASS | {"diff_bytes": 35, "outside_noise": 0, "inside_noise": 35, "size_delta": 0} |
| case-repeat-e1 | repeatable | **PASS** | PASS | {"diff_bytes": 38, "outside_noise": 0, "inside_noise": 38, "size_delta": 0} |
| case-repeat-e2 | repeatable | **PASS** | PASS | {"diff_bytes": 110, "outside_noise": 0, "inside_noise": 110, "size_delta": 0} |
| case-repeat-e3 | repeatable | **PASS** | PASS | {"diff_bytes": 106, "outside_noise": 4, "inside_noise": 102, "size_delta": 0} |
| noise-budget | noise_budget | **PASS** | PASS | {"noise_bytes": 117, "share": 0.0098} |
| control-same-input-no-change | diff_outside_noise | **PASS** | PASS(按预期被判红) | {"diff_bytes": 0, "outside_noise": 0, "inside_noise": 0, "noise_share": 0.0} |
| control-shape-escapes-weight-zone | diff_within_zone | **PASS** | PASS(按预期被判红) | {"zone_bytes": 4096, "diff_bytes": 10308, "outside_noise": 10158, "inside_noise": 150, "escaped_zone": 6079} |
| control-quant-escapes-weight-zone | diff_within_zone | **PASS** | PASS(按预期被判红) | {"zone_bytes": 4096, "diff_bytes": 10040, "outside_noise": 9954, "inside_noise": 86, "escaped_zone": 5876} |
| case-e1-stable-byte | stable_diff | **PASS** | - | {"diff_bytes": 1, "stable_diff_bytes": 1, "reorder_bytes": 0, "signal_bytes": 1, "expect_offsets": [6343]} |
| case-e1b-stable-byte | stable_diff | **PASS** | - | {"diff_bytes": 1, "stable_diff_bytes": 1, "reorder_bytes": 0, "signal_bytes": 1, "expect_offsets": [6343]} |
| case-e2-stable-none | stable_diff | **PASS** | - | {"diff_bytes": 0, "stable_diff_bytes": 0, "reorder_bytes": 0, "signal_bytes": 0, "expect_offsets": null} |
| case-d-stable-weight-block | stable_diff | **PASS** | - | {"diff_bytes": 4080, "stable_diff_bytes": 4080, "reorder_bytes": 0, "signal_bytes": 4080, "expect_offsets": null} |
| case-f-gno-no-payload | flag_payload | **PASS** | - | {"size_delta": 10, "body_size_delta": 0, "payload_bytes": 0, "size_field_bytes": 7} |
| case-f-gno-all-no-payload | flag_payload | **PASS** | - | {"size_delta": 18, "body_size_delta": 0, "payload_bytes": 0, "size_field_bytes": 7} |
| case-g-meta-payload | flag_payload | **PASS** | - | {"size_delta": 185, "body_size_delta": 90, "payload_bytes": 8, "size_field_bytes": 2} |
| case-h-resize-changes-artifact | flag_payload | **PASS** | - | {"size_delta": -2, "body_size_delta": -16, "payload_bytes": 363, "size_field_bytes": 11} |
| case-p-dlb-levels | pack_size | **PASS** | - | {"ratio": 0.732} |

## 单个差分（噪声口径 / 稳定字节口径）

- [per-variant] fc_a vs fc_a：size 7635 -> 7635（+0），不同 0 B，噪声内 0 B，噪声外 0 B
- [union] fc_a vs fc_a：size 7635 -> 7635（+0），不同 0 B，噪声内 0 B，噪声外 0 B
- [per-variant] fc_a vs fc_b_n128：size 7635 -> 11987（+4352），不同 10308 B，噪声内 150 B，噪声外 10158 B
- [union] fc_a vs fc_b_n128：size 7635 -> 11987（+4352），不同 10308 B，噪声内 207 B，噪声外 10101 B
- [per-variant] fc_a vs fc_c_k128：size 7635 -> 11731（+4096），不同 9971 B，噪声内 86 B，噪声外 9885 B
- [union] fc_a vs fc_c_k128：size 7635 -> 11731（+4096），不同 9971 B，噪声内 145 B，噪声外 9826 B
- [per-variant] fc_a vs fc_d_w_cal：size 7635 -> 7635（+0），不同 4137 B，噪声内 59 B，噪声外 4078 B
- [per-variant] fc_a vs fc_d_w_swap：size 7635 -> 7635（+0），不同 4136 B，噪声内 56 B，噪声外 4080 B
- [union] fc_a vs fc_d_w_swap：size 7635 -> 7635（+0），不同 4136 B，噪声内 57 B，噪声外 4079 B
- [per-variant] fc_a vs fc_e_abs2：size 7635 -> 7635（+0），不同 101 B，噪声内 101 B，噪声外 0 B
- [union] fc_a vs fc_e_abs2：size 7635 -> 7635（+0），不同 101 B，噪声内 101 B，噪声外 0 B
- [per-variant] fc_a vs fc_e_fp16：size 7635 -> 11731（+4096），不同 10040 B，噪声内 86 B，噪声外 9954 B
- [union] fc_a vs fc_e_fp16：size 7635 -> 11731（+4096），不同 10040 B，噪声内 145 B，噪声外 9895 B
- [per-variant] fc_a vs fc_e_out16：size 7635 -> 7635（+0），不同 36 B，噪声内 35 B，噪声外 1 B
- [union] fc_a vs fc_e_out16：size 7635 -> 7635（+0），不同 36 B，噪声内 35 B，噪声外 1 B
- [per-variant] fc_a vs fc_e_out8：size 7635 -> 7635（+0），不同 106 B，噪声内 105 B，噪声外 1 B
- [union] fc_a vs fc_e_out8：size 7635 -> 7635（+0），不同 106 B，噪声内 105 B，噪声外 1 B

## 稳定字节口径（两侧都逐字节稳定的偏移上取值不同）

| a | b | 稳定字节差 | 整块重排 | 信号 | 常量区剔除 | 排布类 | 信号区间（前 8） |
|---|---|---|---|---|---|---|---|
| fc_a | fc_d_w_swap | 4080 | 0 | 4080 | 0 | 同 | [[1951, 1993], [1995, 2196], [2198, 2656], [2658, 3223], [3225, 3614], [3616, 3680], [3682, 3891], [3893, 4263]] |
| fc_a | fc_e_abs2 | 0 | 0 | 0 | 0 | 同 | [] |
| fc_a | fc_e_out16 | 1 | 0 | 1 | 0 | 同 | [[6343, 6343]] |
| fc_a | fc_e_out8 | 1 | 0 | 1 | 0 | 同 | [[6343, 6343]] |

