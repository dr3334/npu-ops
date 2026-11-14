# npu_lib 板端矩阵（自动生成，勿手改）

生成：`PYTHONPATH=tools /usr/bin/python3 -m npu_lib.matrix`（外层 `tools/devlock.sh run`）

| 算子 | 形状 | 元素 | requant | 同源逐字节 | golden 逐位 | 结果 | 证据 |
|---|---|---|---|---|---|---|---|
| ADD | [1, 64] | 64 | 1.0 | PASS | PASS | **PASS** | `artifacts/add_1x64_int8/evidence.json` |
| ADD | [1, 896] | 896 | 1.0 | PASS | PASS | **PASS** | `artifacts/add_1x896_int8/evidence.json` |
| ADD | [1, 96] | 96 | 1.0 | PASS | PASS | **PASS** | `artifacts/add_1x96_int8/evidence.json` |
| MUL | [1, 64] | 64 | 1.0 | PASS | PASS | **PASS** | `artifacts/mul_1x64_int8/evidence.json` |
| MUL | [1, 896] | 896 | 1.0 | PASS | PASS | **PASS** | `artifacts/mul_1x896_int8/evidence.json` |
| MUL | [1, 96] | 96 | 1.0 | PASS | PASS | **PASS** | `artifacts/mul_1x96_int8/evidence.json` |
| SUB | [1, 64] | 64 | 1.0 | PASS | PASS | **PASS** | `artifacts/sub_1x64_int8/evidence.json` |
| SUB | [1, 896] | 896 | 1.0 | PASS | PASS | **PASS** | `artifacts/sub_1x896_int8/evidence.json` |
| SUB | [1, 96] | 96 | 1.0 | PASS | PASS | **PASS** | `artifacts/sub_1x96_int8/evidence.json` |

共 9/9 组合 PASS。
