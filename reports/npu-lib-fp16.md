# fp16 那道门：逐元素 fp16 在我们的提交器上跑对了

日期：2026-11-14 · 板 Titan 1 / MT6897 · `tools/npu_lib` 的 `DType.fp16()` + `Graph.add/sub/mul`

## 0. 结论

| 算子 | 形状 | 门① 同源（逐字节） | 门② 数值（ULP ≤ 1） | 同输入重跑 |
|---|---|---|---|---|
| MUL | [8,64] fp16 | **PASS** | **0/1024**（max_ulp 0；换新数据 max_ulp 1，仍在 1 内） | 逐字节相同 |
| ADD | [8,64] fp16 | **PASS** | **0/1024**（max_ulp 0） | 逐字节相同 |
| SUB | [8,64] fp16 | **PASS** | **0/1024**（max_ulp 0） | 逐字节相同 |

三分之二的 `SameDLA` 都是 1 个执行步、`.dla` 3177 B，plan 里 `bytes 1024/1024/1024`（fp16 每元素 2 B）。
回归：int8 矩阵仍 **9/9 PASS**。

## 1. 这一票真正回答的问题：**fp16 重放是确定性的**

同源门（门①）只在「同一份命令流 + 同一个输入 → 同一个输出」成立时才有意义。
int8 那 9 个资产早就证明了这一点，fp16 没测过。这轮的实测是：
**同一资产、同一输入连跑两次，输出逐字节相同**（三个算子都成立）。
所以门① 在 fp16 上照样可判 —— 这也是后面 fp16 整块验收的基础。

## 2. 判据为什么改成 ULP

fp16 没有 scale，设备做的就是 fp16 算术，和 float32 参考之间本来就有舍入。所以：

- golden：`golden.reference_fp16` 用 **float32** 算再落回 fp16（与 flatbuffer 生成无关，规则 R3）；
- 判据：逐元素 **ULP 距离 ≤ 1**（fp16 尾数 10 位），用位模式算（负数取反保证单调）；
- 实测最大 1 ULP，绝大多数 0 ULP；没有用「相对误差」那种含糊口径。

## 3. 为 fp16 改的四处（都是 dtype 分派，没有新机制）

1. `DType.fp16()` 真实现（scale 固定 1.0，只为接口一致）；int16 仍抛。
2. `build_model.build_binop_fp16`：张量类型 `F16`、**没有量化参数**；输入里特意放了 ±1、0、100、1e-3 这些点。
3. `Output` / `verify` 按 dtype 读回（fp16 每元素 2 B，字节数不再是元素数）—— 这里踩了一次 `short read 512/1024`。
4. `run()` 的输入打包按 dtype 转（原来硬转 `int8`，会把 fp16 数据打烂）。

## 4. 对 block 意味着什么

| block 需要的 | 状态 |
|---|---|
| fp16 逐元素（MUL/ADD/SUB） | ✅ 本轮 |
| fp16 FC（动态权重） | 机制同 int8，未测 |
| fp16 SOFTMAX / MEAN / RSQRT / LOGISTIC | 未测（int8 版都通了，fp16 是同一套一元管道） |
| 整块 39 步 + 三道门 | 未开始 |

下一步自然就是 fp16 的 RMSNorm 半块：`MUL(x,x) → MEAN → RSQRT → MUL` —— **fp16 不需要 requant 契约**，
这正是当初把这个 gate 排在 int8 RMSNorm 前面的原因（int8 那条链卡在 requant ≠ 1，见 reports/npu-lib-mean-rsqrt.md §3）。

## 5. 复现

```python
g = Graph('mul_8x64_fp16')
a = g.input('a', (8,64), DType.fp16()); b = g.input('b', (8,64), DType.fp16())
g.mul(a, b); g.compile(); g.verify(g.run({'a': A, 'b': B}))
```
