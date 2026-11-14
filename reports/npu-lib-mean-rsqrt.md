# 自建库加两个一元算子：MEAN 与 RSQRT（int8）

日期：2026-11-14 · 板 Titan 1 / MT6897 · `tools/npu_lib` 的 `Graph.mean` / `Graph.rsqrt`

## 0. 结果

| 算子 | 形状 | 输出 | 门① 同源 | 门② 数值 | 关键配置 |
|---|---|---|---|---|---|
| MEAN | `[1,8,16]` int8 → `[1,8,1]` | 8 码 | **PASS**（逐字节） | **0/8** | 输出 scale = 输入 scale；命令流只有 **2 个地址字** |
| RSQRT | `[1,8,16]` int8 | 128 码 | **PASS**（逐字节） | **0/128** | 一元 + `scratch.bin`（LUT），3 个地址字 |

两个都喂过全新随机输入，数值门同样 0 失配。回归：`run_npu_lib_matrix.sh` 仍 **9/9 PASS**。

## 1. MEAN：先错在 scale 约定，改对后逐位一致

第一版把输出 scale 定成 `s_in / K`（想当然「码值 = 行和」），结果门① 过、门② 差 2/8（最多差 7 码）。
查厂商那条已上板的 `ops/mean_4x16_int8`：它的约定是 **输出 scale == 输入 scale**，于是
`multiplier = s_in/(K*s_out) = 1/K`、码值 = `round(行和 / K)`，golden 走仓库的两步整数舍入契约。
照这个约定改完再跑：**0/8，逐字节**。

> 教训：`int8_tol=1` 不是「先写上再说」的创可贴 —— 先确认 scale 约定，再谈容差。

## 2. 一元链路的两处新东西

1. **地址字个数不再是常数 3**：MEAN 没有额外缓冲，命令流里只有 A 与 C 两个地址（一开始报 `patched 2/3`）。
   现在 `compile()` 从 `iova.txt` 里非 0 的槽数推出 `expect_patches`（MEAN=2、RSQRT=3、二元算子=3）。
2. **没有 LUT 的一元算子**：`n_inputs=1`、`bytes_b=0`、不写 `extra_file`；有 LUT 的（RSQRT/SOFTMAX）走原来的路。

## 3. int8 的 RMSNorm 链走不通（这是算术结论，不是实现问题）

`norm = x * rsqrt(mean(x*x))` 在 int8 上需要两次 **requant ≠ 1**：

- `MUL(x,x)` 的输出 scale 被钉在 `s_x*s_x`（ratio = 1），码值 = `x码²`；x 码取 ±11 就已经逼近 ±127，
  再大就饱和 —— 而归一化的输入动态范围远大于此；
- 归一化乘 `x*rsqrt` 同理：两个码相乘动辄上千。

本库的逐元素只做 ratio = 1 的组合（`docs/OWN-SUBMITTER-LIB.md` §6.2 第 3 条本来就写明「没有板端证据」）。
**所以 int8 的 RMSNorm 不在这一票里**；block 走 fp16（map 的边界也这么定），
MEAN/RSQRT 到这里的作用是把一元管道、LUT 缓冲、地址字数都验证完。

## 4. 复现

```python
g = Graph('mean_1x8x16_int8'); x = g.input('x', (1,8,16), DType.int8(1/16)); g.mean(x); g.compile()
g = Graph('rsqrt_1x8x16_int8'); y = g.input('y', (1,8,16), DType.int8(1/127)); g.rsqrt(y, out_scale=1/64)
```
