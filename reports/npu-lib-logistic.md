# LOGISTIC（SwiGLU 的 sigmoid）：先走官方流程，再走自建提交器

日期：2026-11-14 · 板 Titan 1 / MT6897 · 形状取 block 里 SwiGLU 的实际形状 `[8,128]`（fp16）

## 0. 结论

| 走法 | 结果 | 数字 |
|---|---|---|
| **① 官方流程**（`tools/e2e.sh`：ncc 编译 → neuronrt 上板 → `compare.py` 对拍） | **PASS** | `max_abs 4.883e-04`（= 0.5 处 1 个 fp16 码）、rmse 4.20e-05、rel_l2 6.32e-05；`.dla` 2589 B；1075 FPS（3 次重复） |
| **② 自建提交器**（`npu_lib` 捕获 + `npu_run` 重放） | **PASS** | 门① 与厂商输出**逐字节相同**；门② 0/2048 超限（max_ulp 1，容差 1）；同输入重跑逐字节相同；`.dla` 2687 B |

这是最后一个「不知道行不行」的算子 —— 现在它只是一条普通证据。

## 1. 官方流程（先做，作为独立基线）

```
./tools/e2e.sh ops/logistic_8x128_fp16 --repeat 3
```

- 生成器：`tools/build_actreduce.py`（新增 `logistic_8x128_fp16` 配置项，形状/容差照抄同族条目）；
- 层清单：单个 `SigmoidLayer`，1 个执行步；
- 对拍：设备 vs 生成器给的 float64 sigmoid → fp16 参考，**PASS（atol 2e-3 / rtol 1e-2）**。

## 2. 自建提交器（再试，判据更严）

```python
g = Graph('logistic_8x128_fp16_lib')
x = g.input('x', (8,128), DType.fp16())
g.logistic(x); g.compile(); g.verify(g.run({'x': X}))
```

| 项 | 值 |
|---|---|
| 门① 同源 | **PASS**（`sha256_out == sha256_vendor`，逐字节） |
| 门② 数值 | **0/2048 超限**，`max_ulp = 1`（容差 1 个 fp16 码） |
| 换新随机输入 | 同样 0 超限，max_ulp 1 |
| 重放确定性 | 同输入连跑两次，输出逐字节相同 |
| plan | `bytes 2048/0/2048`、`n_inputs=1`、`expect_patches=2`（一元、无 LUT） |

## 3. 顺带说清 int8 那个悬案

探针阶段的 int8 LOGISTIC 记录是「提交成功但 64/64 失配」。这一轮没有去碰它：

- 它需要的判据与 fp16 不同（TFLite 的 int8 logistic 输出是 `1/256, zp=-128`，值是 LUT 定点的）；
- 而 block 走 fp16，int8 logistic 对本票没有价值；
- 所以 `Graph.logistic` **直接拒绝 int8**，并把这个理由写在代码里 —— 宁可显式不支持，也不让一个未定性的算子混进块。

## 4. 复现

```bash
./tools/e2e.sh ops/logistic_8x128_fp16 --repeat 3          # 官方流程
# 自建：见上面的 python 片段（外层 devlock 持锁）
```
