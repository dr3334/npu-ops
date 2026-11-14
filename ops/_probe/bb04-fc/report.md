# bb04-fc —— 差分探查自动报告

生成：/usr/bin/python3 tools/bbdiff.py --matrix tools/bbdiff-matrix-blackbox04.json（WSL /usr/bin/python3）
时间：2026-11-14T12:17:54+0800 · 每变体保留 5 次同类别编译（另跑 4 次做类别校验）· 编译入口 tools/compile.sh

## 变体

| 变体 | tflite B | 权重 B | .dla B（保留各次） | rc | 尝试/保留/丢弃 | 噪声内字节 |
|---|---|---|---|---|---|---|
| fc_a | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 39 |
| fc_b_n128 | 9456 | 8192 | 11987 / 11987 / 11987 / 11987 / 11987 | 0/0/0/0/0 | 9/5/4 | 60 |
| fc_c_k128 | 9200 | 8192 | 11731 / 11731 / 11731 / 11731 / 11731 | 0/0/0/0/0 | 9/5/4 | 130 |
| fc_d_w_cal | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 113 |
| fc_d_w_swap | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 104 |
| fc_e_out16 | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 60 |
| fc_e_abs2 | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 115 |
| fc_e_out8 | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 60 |
| fc_e_fp16 | 8896 | 8192 | 11731 / 11731 / 11731 / 11731 / 11731 | 0/0/0/0/0 | 9/5/4 | 119 |

## 已知噪声区（两次编译不稳定）

共 **251 B** / 134 段（占最大产物 2.1%）；由每变体前 4 次编译两两比对取并集（第 5 次留作 holdout，见 case2）。逐段见 noise.txt。

| zone | 字节 | 来源 |
|---|---|---|
| weight | 4096 | fc_a vs fc_d_w_cal（离噪） |

## 判定

| check | kind | 结果 | 关键数字 |
|---|---|---|---|
| case-b-n-changed | diff_outside_noise | **PASS** | {"diff_bytes": 10261, "outside_noise": 10110, "inside_noise": 151, "noise_share": 0.0147} |
| case-b-size-grows | size_delta | **PASS** | {"delta": 4352, "ratio": 0.531} |
| case-c-k-changed | diff_outside_noise | **PASS** | {"diff_bytes": 10116, "outside_noise": 9930, "inside_noise": 186, "noise_share": 0.0184} |
| case-c-size-grows | size_delta | **PASS** | {"delta": 4096, "ratio": 0.5} |
| case-d-weights-confined | diff_within_zone | **PASS** | {"zone_bytes": 4096, "diff_bytes": 4137, "outside_noise": 4079, "inside_noise": 58, "escaped_zone": 0} |
| case-e1-out-scale-changed | diff_outside_noise | **FAIL** | {"diff_bytes": 37, "outside_noise": 1, "inside_noise": 36, "noise_share": 0.973} |
| case-e1-quant-enters-artifact | diff_outside_noise | **FAIL** | {"diff_bytes": 37, "outside_noise": 1, "inside_noise": 36, "noise_share": 0.973} |
| case-e2-abs-scale-changed | diff_outside_noise | **FAIL** | {"diff_bytes": 56, "outside_noise": 0, "inside_noise": 56, "noise_share": 1.0} |
| case-e1b-out8-changed | diff_outside_noise | **FAIL** | {"diff_bytes": 61, "outside_noise": 1, "inside_noise": 60, "noise_share": 0.9836} |
| case-e3-fp16-differs | diff_outside_noise | **PASS** | {"diff_bytes": 10057, "outside_noise": 9908, "inside_noise": 149, "noise_share": 0.0148} |
| case-repeat-b | repeatable | **PASS** | {"diff_bytes": 57, "outside_noise": 0, "inside_noise": 57, "size_delta": 0} |
| case-repeat-c | repeatable | **PASS** | {"diff_bytes": 62, "outside_noise": 0, "inside_noise": 62, "size_delta": 0} |
| case-repeat-e1b | repeatable | **PASS** | {"diff_bytes": 107, "outside_noise": 1, "inside_noise": 106, "size_delta": 0} |
| case-repeat-e1 | repeatable | **PASS** | {"diff_bytes": 108, "outside_noise": 0, "inside_noise": 108, "size_delta": 0} |
| case-repeat-e2 | repeatable | **PASS** | {"diff_bytes": 103, "outside_noise": 0, "inside_noise": 103, "size_delta": 0} |
| case-repeat-e3 | repeatable | **PASS** | {"diff_bytes": 55, "outside_noise": 0, "inside_noise": 55, "size_delta": 0} |
| noise-budget | noise_budget | **PASS** | {"noise_bytes": 251, "share": 0.0209} |
| control-same-input-no-change | diff_outside_noise | **PASS** | {"diff_bytes": 0, "outside_noise": 0, "inside_noise": 0, "noise_share": 0.0} |
| control-shape-escapes-weight-zone | diff_within_zone | **PASS** | {"zone_bytes": 4096, "diff_bytes": 10261, "outside_noise": 10110, "inside_noise": 151, "escaped_zone": 6031} |
| control-quant-escapes-weight-zone | diff_within_zone | **PASS** | {"zone_bytes": 4096, "diff_bytes": 10057, "outside_noise": 9908, "inside_noise": 149, "escaped_zone": 5831} |

## 单个差分

- fc_a vs fc_a：size 7635 -> 7635（+0），不同 0 B，噪声内 0 B，噪声外 0 B -> diff/fc_a__vs__fc_a.txt
- fc_a vs fc_b_n128：size 7635 -> 11987（+4352），不同 10261 B，噪声内 151 B，噪声外 10110 B -> diff/fc_a__vs__fc_b_n128.txt
- fc_a vs fc_c_k128：size 7635 -> 11731（+4096），不同 10116 B，噪声内 186 B，噪声外 9930 B -> diff/fc_a__vs__fc_c_k128.txt
- fc_a vs fc_d_w_cal：size 7635 -> 7635（+0），不同 4186 B，噪声内 109 B，噪声外 4077 B -> diff/fc_a__vs__fc_d_w_cal.txt
- fc_a vs fc_d_w_swap：size 7635 -> 7635（+0），不同 4137 B，噪声内 58 B，噪声外 4079 B -> diff/fc_a__vs__fc_d_w_swap.txt
- fc_a vs fc_e_abs2：size 7635 -> 7635（+0），不同 56 B，噪声内 56 B，噪声外 0 B -> diff/fc_a__vs__fc_e_abs2.txt
- fc_a vs fc_e_fp16：size 7635 -> 11731（+4096），不同 10057 B，噪声内 149 B，噪声外 9908 B -> diff/fc_a__vs__fc_e_fp16.txt
- fc_a vs fc_e_out16：size 7635 -> 7635（+0），不同 37 B，噪声内 36 B，噪声外 1 B -> diff/fc_a__vs__fc_e_out16.txt
- fc_a vs fc_e_out8：size 7635 -> 7635（+0），不同 61 B，噪声内 60 B，噪声外 1 B -> diff/fc_a__vs__fc_e_out8.txt

