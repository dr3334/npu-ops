# 自建库的第 2 个算子：FULLY_CONNECTED（动态权重）

日期：2026-11-14 · 板 Titan 1 / MT6897 · Android 16 · ncc-tflite 7.3.15
入口：`tools/npu_lib`（`Graph.fc`） · 提交器：[`probe/npu_run.c`](../probe/npu_run.c) · 证据：`artifacts/fc_8x64x32_int8/evidence.json`

## 0. 一句话

矩阵乘级算子第一次走完「定义 -> 编译 -> 捕获 -> 自建提交 -> 两道门」：`x[8,64] × w[32,64] -> out[8,32]`，
权重是**运行时输入**，设备输出与厂商那次输出、与 CPU golden **三者 sha256 完全相同**。

## 1. 结果

| 项 | 值 |
|---|---|
| 形状 | `x[8,64] int8` × `w[32,64] int8`（TFLite FC 布局）→ `out[8,32] int8` |
| 量化 | `s_a = s_w = 1/127`，`s_out = s_a*s_w*32`（requant multiplier = 1/32） |
| 编译 | `EXIT=0`，`.dla` 3165 B |
| 计划 | `bytes_a=512`(激活) / `bytes_b=2048`(权重) / `bytes_out=256` —— **三个不同字节数** |
| 门① 同源 | **PASS**（喂抓取那对输入：`device_out == vendor_out`，逐字节） |
| 门② 数值 | **PASS**，0/256 失配（喂全新随机输入也是 0/256） |
| 三者相等 | `device == vendor == golden`，sha256 `e660b3146b10db1c…` |

## 2. 这一票新增了什么（相对 int8 逐元素那 9 个资产）

1. **提交器支持每块缓冲各自的字节数**：`plan.txt` 从单一 `n_elems` 扩到 `bytes_a/bytes_b/bytes_out`
   （`n_elems` 保留为兼容回退）。FC 是 `[8,64]`、`[32,64]`、`[8,32]` 三个尺寸，旧接口表达不了。
2. **权重是第二输入**（动态权重），不是烘进 `.dla` 的常量。理由有板端证据：常量权重在自建提交器上
   只有落在抓取那次的 IOVA 才对（[`docs/ISSUE-const-fc.md`](../docs/ISSUE-const-fc.md) 的 3/11），
   动态权重在探针里 0/512、0/1024（[`docs/OWN-SUBMITTER.md`](../docs/OWN-SUBMITTER.md) §6）。
3. **golden 契约**：`int64` 精确累加 + [`tools/quant_contract.py`](../tools/quant_contract.py) 的两步整数舍入，
   于是 `int8_tol = 0`（探针阶段记录的「1/64 差 1 码」是拿精确有理数当参考时的舍入边界，不是设备误差）。

## 3. 回归

`./tools/run_npu_lib_matrix.sh`（前后各一次 healthcheck，全程持锁）：**9/9 PASS**，
老资产的 `plan.txt` 仍是 `n_elems` 形式，新 runner 回退路径逐字节不变 —— 见 [`reports/npu-lib-matrix.md`](npu-lib-matrix.md)。

## 4. 复现

```bash
./tools/devlock.sh run -- wsl.exe -e bash -lc \
  "cd '/mnt/d/deepseek_ workspace/mtk/npu-ops' && NPU_LOCK_HELD=1 PYTHONPATH=tools \
   /usr/bin/python3 -c \"
import numpy as np, sys; sys.path.insert(0,'tools')
from npu_lib import Graph, DType
s = 1/127; g = Graph('fc_8x64x32_int8')
x = g.input('x', (8,64), DType.int8(s)); w = g.input('w', (32,64), DType.int8(s)); g.fc(x, w)
a = np.fromfile('artifacts/fc_8x64x32_int8/golden/x.bin', dtype=np.int8).reshape(8,64)
b = np.fromfile('artifacts/fc_8x64x32_int8/golden/w.bin', dtype=np.int8).reshape(32,64)
print(g.verify(g.run({'x': a, 'w': b})))\""
```

## 5. 边界（不夸大）

- 一条图仍只有一个算子；链式图要 N 缓冲 plan（未做）；
- 只做 int8 per-tensor、`zero_point=0`；int16/fp16 仍未实现；
- 权重每次运行都要从 host 推一遍（本形状 2 KB，可忽略；真实模型的 30 MB 权重是另一件事）；
- `bias` 用全 0 常量（与 `ops/fc_dynb_*` 同一条已验证约定），非零 bias 未测。
