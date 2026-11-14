# bb03-fc —— 差分探查自动报告

生成：/usr/bin/python3 tools/bbdiff.py --matrix tools/bbdiff-matrix-blackbox03.json（WSL /usr/bin/python3）
时间：2026-11-14T14:11:00+0800 · 每变体保留 5 次同类别编译（另跑 4 次做类别校验）· 编译入口 tools/compile.sh
噪声口径：判定用 **per-variant**；对照列同时给出 union（03/04 旧口径）

## 变体

| 变体 | tflite B | 权重 B | .dla B（保留各次） | rc | 尝试/保留/丢弃 | 自身噪声 B | 稳定字节 B |
|---|---|---|---|---|---|---|---|
| fc_base | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 114 | 7516 |
| fc_w_cal | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 115 | 7520 |
| fc_w_swap | 5104 | 4096 | 7635 / 7635 / 7635 / 7635 / 7635 | 0/0/0/0/0 | 9/5/4 | 114 | 7521 |
| fc_n128 | 9456 | 8192 | 11987 / 11987 / 11987 / 11987 / 11987 | 0/0/0/0/0 | 9/5/4 | 64 | 11857 |
| fc_extract | 5104 | 4096 | 3813 / 3813 / 3813 / 3813 / 3813 | 0/0/0/0/0 | 9/5/4 | 71 | 3742 |
| fc_k8_rejected | 1520 | 512 | - |  | 9/0/0 | 0 | None |

## 已知噪声区（两次编译不稳定）

并集口径 **220 B** / 109 段（占最大产物 1.8%）；由每变体前 4 次编译两两比对取并集（第 5 次留作 holdout）。判定已改用**每变体各自标定**：跨变体差分只豁免**参与这次差分的两个变体**自己的噪声，逐段见 noise.txt。

每变体噪声 B：{"fc_base": 114, "fc_w_cal": 115, "fc_w_swap": 114, "fc_n128": 64, "fc_extract": 71, "fc_k8_rejected": 0}

| zone | 字节 | 来源 |
|---|---|---|
| weight | 4096 | fc_base vs fc_w_cal（离噪） |

## 判定

「结果」按 per-variant 口径；「union 旧口径」列是同一份产物在旧口径下的判定（改口径前后对照）。

| check | kind | 结果 | union 旧口径 | 关键数字 |
|---|---|---|---|---|
| case1-n-changed | diff_outside_noise | **PASS** | PASS | {"diff_bytes": 10331, "outside_noise": 10218, "inside_noise": 113, "noise_share": 0.0109} |
| case2-repeatable | repeatable | **PASS** | PASS | {"diff_bytes": 103, "outside_noise": 5, "inside_noise": 98, "size_delta": 0} |
| case3-weights-confined | diff_within_zone | **PASS** | PASS | {"zone_bytes": 4096, "diff_bytes": 4124, "outside_noise": 4080, "inside_noise": 44, "escaped_zone": 0} |
| case4-extract-shrinks | size_delta | **PASS** | - | {"delta": 3822, "ratio": 0.933} |
| control-same-input-no-change | diff_outside_noise | **PASS** | PASS(按预期被判红) | {"diff_bytes": 0, "outside_noise": 0, "inside_noise": 0, "noise_share": 0.0} |
| control-shape-escapes-weight-zone | diff_within_zone | **PASS** | PASS(按预期被判红) | {"zone_bytes": 4096, "diff_bytes": 10331, "outside_noise": 10218, "inside_noise": 113, "escaped_zone": 6142} |
| case2b-noise-budget | noise_budget | **PASS** | PASS | {"noise_bytes": 115, "share": 0.0096} |

## 单个差分（噪声口径 / 稳定字节口径）

- [per-variant] fc_base vs fc_base：size 7635 -> 7635（+0），不同 0 B，噪声内 0 B，噪声外 0 B
- [union] fc_base vs fc_base：size 7635 -> 7635（+0），不同 0 B，噪声内 0 B，噪声外 0 B
- [per-variant] fc_base vs fc_n128：size 7635 -> 11987（+4352），不同 10331 B，噪声内 113 B，噪声外 10218 B
- [union] fc_base vs fc_n128：size 7635 -> 11987（+4352），不同 10331 B，噪声内 152 B，噪声外 10179 B
- [per-variant] fc_base vs fc_w_cal：size 7635 -> 7635（+0），不同 4182 B，噪声内 104 B，噪声外 4078 B
- [per-variant] fc_base vs fc_w_swap：size 7635 -> 7635（+0），不同 4124 B，噪声内 44 B，噪声外 4080 B
- [union] fc_base vs fc_w_swap：size 7635 -> 7635（+0），不同 4124 B，噪声内 82 B，噪声外 4042 B

