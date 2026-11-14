# SOFTMAX 更正：原生 SoftMaxLayer 在我们的提交器上**是能跑对的**

日期：2026-11-14 · 板 Titan 1 / MT6897 · 模型 `ops/softmax_1x8x16_int8`（1 个执行步，`SoftMaxLayer<0>`）

## 0. 一句话

`docs/OWN-SUBMITTER.md` §8.1 记的「softmax 输出全 0」**已不成立**。用今天的抓取 + `probe/npu_run` 重放：
设备输出与厂商那次**逐字节相同**（0/128），与 golden `expected.bin` 也 0/128。
MDLA 有原生 SoftMaxLayer，不需要任何函数近似。

## 1. 证据

```
patch off=0x050 0x02563200 -> 0x02555900 (buf0)   # 输入
patch off=0x128 0x02586000 -> 0x0257fd00 (buf1)   # 输出
patch off=0x448 0x02586400 -> 0x0259cc00 (buf2)   # LUT/scratch（extra_file）
iova A=0x25559000 B=0x257fd000 C=0x259cc000 bytes=128/4096/128
RC=0 patches=3
```

| 对照 | 结果 |
|---|---|
| 我们 vs `golden/expected.bin` | **0/128** |
| 我们 vs 厂商设备输出 `out_device.bin` | **0/128** |
| 逐字节与厂商相同 | **True** |

模型用的是 `input scale 1/16`、`output scale 1/256`、`output_zero_point = -128`（TFLite 的 int8 softmax 惯例）——
重放不需要理解这些，它们已经在命令流里。

## 2. 旧结论错在哪（最可能的根因，未 100% 复现）

- 旧资产有两个目录 `probe/assets/softmax16` 与 `softmax16b`，两边的 `iova.txt` 里 **B/C 两槽是互换的** ——
  当时在猜「哪个缓冲是输出、哪个是 LUT」。
- 一元分支的 `mkasset.py` 里：输出取「非 A/B 的第一个 dump」，LUT 再用「其地址字出现在 cmd2 里」反查。
  两个候选都是 4 KiB，而 `.dla` 代码缓冲会因 **fd 复用**在 dump 里呈现同一份内容 —— 于是**角色认反**：
  输出地址被写进 LUT 槽、LUT 地址被写进输出槽，输出缓冲永远没人写 → 全 0。
- 今天 `mkasset.py` 已加判据：**候选必须能在 MAP 表里解析出 IOVA**，假候选（代码缓冲）被排除。
  同一套代码今天一次就跑出逐字节相同的结果。

## 2.5 已经做成一等算子（2026-11-14，同一天）

`g.softmax(x)` 现在走完整链路：**定义 → tflite → .dla（4647 B）→ 捕获 → plan → 自建提交 → 两道门**，
其中一元算子的三件事都自动了：

| 环节 | 现在的行为 |
|---|---|
| 捕获 | `b_path=None` → 只推 `a.bin`，`matrix` 侧不再需要假造第二个输入 |
| plan | 自动识别资产里的 `scratch.bin` → `n_inputs=2`、`in_b` 留空、`extra_file=scratch.bin`，`bytes 128/4096/128` |
| 验收 | ① 与厂商输出逐字节；② 与 golden 按 `int8_tol`（本算子登记 2）比，实际 **0/128** |

设备 / 厂商 / golden 三者 sha256 相同：`2c4e2e0b0d954ccc…`。喂全新随机输入同样 0/128。

做这条管道时又踩到两个坑，都已修：

1. **golden 必须先按张量形状 reshape**：一维码流直接算 `softmax` 会把整块当成一行（等于全张量归一化），
   49/128 失配 —— 看上去像设备错，其实是参考实现错。fc 早就有这个 reshape，softmax 漏了。
2. **spec 守卫不能把"后来新增的元数据"当成变更**：`compile()` 的一致性检查原来比较所有键，
   导致老资产（spec 里没有 `int8_tol`）一进矩阵就报"spec 不同"。现在只比两边都有的键。

回归：`run_npu_lib_matrix.sh` 仍 **9/9 PASS**（verify 改成"容差判据"后逐元素路径行为不变，tol=0 等价于逐位）。

## 3. 它对 block 意味着什么

attention 的 `softmax(scores)` 不再是一个「要修的 bug」，而是一个**有板端证据的算子**：
`[1,8,16]` 形状（= 8 个 token × 16 列）已验证；block 里用的是 `[T,T]`（T=8 → `[8,8]`，行 16 B，满足 16 B 行规则）。

仍未做：把它做成 `npu_lib` 的一等算子（`Graph.softmax`），那需要 golden 侧的 `output_zero_point = -128`
与 `int8_tol=2` 的登记（厂商那条 compare 用的是 tol=2）。**这是接口工作，不是可行性问题。**

## 4. 复现

```bash
# 1) 抓一次（一元：只推 a.bin）
./tools/devlock.sh run -- wsl.exe -e bash -lc "cd '<repo>' && NPU_LOCK_HELD=1 PYTHONPATH=tools \
  /usr/bin/python3 -c \"import sys;sys.path.insert(0,'tools');from npu_lib import capture;\
  capture.capture('sm_diag','artifacts/sm_diag','ops/softmax_1x8x16_int8/model.dla',\
  'ops/softmax_1x8x16_int8/golden/input.bin',None)\""
# 2) plan: n_inputs=2 + extra_file=scratch.bin + bytes_a=128/bytes_b=4096/bytes_out=128
# 3) 板上: npu_run plan.txt  -> 逐字节比 ops/softmax_1x8x16_int8/out_device.bin
```
