# bb03-fc —— 差分探查自动报告

生成：/usr/bin/python3 tools/bbdiff.py --matrix tools/bbdiff-matrix-blackbox03.json（WSL /usr/bin/python3）
时间：2026-11-14T12:09:53+0800 · 每变体保留 5 次同类别编译（另跑 4 次做类别校验）· 编译入口 tools/compile.sh

## 变体

| 变体 | tflite B | 权重 B | .dla B（保留各次） | rc | 尝试/保留/丢弃 | 噪声内字节 |
|---|---|---|---|---|---|---|
| fc_base | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 115 |
| fc_w_cal | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 115 |
| fc_w_swap | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 115 |
| fc_n128 | 9456 | 8192 | 11987 / 11987 / 11987 / 11987 / 11987 | 0/0/0/0/0 | 9/5/4 | 126 |
| fc_extract | 5104 | 4096 | 3813 / 3813 / 3813 / 3813 / 3813 | 0/0/0/0/0 | 9/5/4 | 119 |
| fc_k8_rejected | 1520 | 512 | - |  | 9/0/0 | - |

## 已知噪声区（两次编译不稳定）

共 **214 B** / 107 段（占最大产物 1.8%）；由每变体前 4 次编译两两比对取并集（第 5 次留作 holdout，见 case2）。逐段见 noise.txt。

| zone | 字节 | 来源 |
|---|---|---|
| weight | 4096 | fc_base vs fc_w_cal（离噪） |

## 判定

| check | kind | 结果 | 关键数字 |
|---|---|---|---|
| case1-n-changed | diff_outside_noise | **PASS** | {"diff_bytes": 10251, "outside_noise": 10117, "inside_noise": 134, "noise_share": 0.0131} |
| case2-repeatable | repeatable | **PASS** | {"diff_bytes": 34, "outside_noise": 0, "inside_noise": 34, "size_delta": 0} |
| case3-weights-confined | diff_within_zone | **PASS** | {"zone_bytes": 4096, "diff_bytes": 4181, "outside_noise": 4041, "inside_noise": 140, "escaped_zone": 0} |
| case4-extract-shrinks | size_delta | **PASS** | {"delta": 3822, "ratio": 0.933} |
| control-same-input-no-change | diff_outside_noise | **PASS** | {"diff_bytes": 0, "outside_noise": 0, "inside_noise": 0, "noise_share": 0.0} |
| control-shape-escapes-weight-zone | diff_within_zone | **PASS** | {"zone_bytes": 4096, "diff_bytes": 10251, "outside_noise": 10117, "inside_noise": 134, "escaped_zone": 6072} |
| case2b-noise-budget | noise_budget | **PASS** | {"noise_bytes": 214, "share": 0.0179} |

## 单个差分

- fc_base vs fc_base：size 7635 -> 7635（+0），不同 0 B，噪声内 0 B，噪声外 0 B -> diff/fc_base__vs__fc_base.txt
- fc_base vs fc_n128：size 7635 -> 11987（+4352），不同 10251 B，噪声内 134 B，噪声外 10117 B -> diff/fc_base__vs__fc_n128.txt
- fc_base vs fc_w_cal：size 7635 -> 7635（+0），不同 4135 B，噪声内 96 B，噪声外 4039 B -> diff/fc_base__vs__fc_w_cal.txt
- fc_base vs fc_w_swap：size 7635 -> 7635（+0），不同 4181 B，噪声内 140 B，噪声外 4041 B -> diff/fc_base__vs__fc_w_swap.txt

