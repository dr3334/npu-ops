# OP-tier-matrix —— 真权重下的二维档位矩阵：8W16A 是唯一近标的档，但仍差 4e-5

> 票：`.scratch/quant-runtime/issues/04f-tier-matrix-real-weights.md`；spec `.scratch/quant-runtime/spec.md`；
> ADR `docs/adr/0005-quantization-tier-target.md`（第五次订正）。
> 上位证据：`docs/OP-w4a8-layer.md`（04）、`docs/OP-8w8a-layer.md`（04c）、`docs/OP-8w8a-real.md`（04e）、
> `docs/OP-w4a16-layer.md`（04d）、`docs/OP-groupwise-scale.md`（04b）。
> 分支 `impl/q04f-tier-matrix`，worktree `wt-q04f`。编译器 ncc-tflite **7.3.15**
> （`--arch=mdla5.0 --opt=3 --disallow-bridge --num-mdla=2 --l1-size-kb=3072`）；
> 设备 mt6897 / Android 16（`0123456789ABCDEF`），板端全部持 `/d/deepseek_ workspace/mtk/.npu-board.lock`，
> 每个板端块前后各一次 `healthcheck --quick`（全部 PASS）。
> 真权重 `/mnt/d/ai/Qwen2.5-0.5B-Instruct/model.safetensors`（bf16，**只读**，不入库、不复制）；
> 夹具 `npu-ops/weights/qwen25_frozen`（只读引用）。
> harness：`tools/build_w4a8.py`（**没有重写**；本轮只改了两处数学 + 一个护栏，见 §1）。

---

## 0. TL;DR

**票面的问题（8W16A 在真权重下是否 ≥0.9999）：否。设备 0.9998311（fp16 缝）/ 0.9998598（native 缝），
差 6.9e-5 / 4.0e-5。** 但它仍是四格里唯一接近判据的档，比第二名 4W16A（0.9733）高 0.027。

| 真权重 / 真输入，T=16，板端（f16 缝） | **4W8A** | **4W16A** | **8W8A** | **8W16A** |
|---|---|---|---|---|
| 量化 golden vs 独立 fp64 链 | 0.8841884 | 0.9732694 | 0.8792696 | **0.9998918** |
| **device vs 独立 fp64 链** | **0.8775767** | **0.9732673** | **0.8971126** | **0.9998311** |
| ｜Δcos(device, golden)｜ | 0.0066 | 0.0000 | 0.0178 | 0.0001 |
| 判据 `cosine_min=0.9999` | FAIL | FAIL | FAIL | **FAIL（差 6.9e-5）** |
| `scores` 段 device vs fp64 | 0.9997529 | 0.9999862 | 0.9997597 | 0.9999998 |
| `probs` 段 device vs fp64 | 0.7476799 | 0.9986388 | 0.7558834 | 0.9996958 |
| `out`（整层）device vs fp64 | 0.8775767 | 0.9732673 | 0.8971126 | 0.9998311 |
| 码级 device vs golden（max / mean） | 118 / 8.16 | 517 / 18.09 | 113 / 10.08 | 431 / 18.80 |
| `.dla` | 7,633,271 B | 7,725,923 B | 15,087,991 B | 15,104,511 B |
| Static / Code / 层数 / 执行步 | 7.2M / 32K / 61 / 1 | 7.3M / 42K / 61 / 1 | 14M / 32K / 61 / 1 | 14M / 34K / 61 / 1 |
| 板端单层（`-c 30`，6 次中位） | 1.43 ms | 2.20 ms | 1.81 ms | 2.16 ms |
| 噪声底（同 `.dla` 同输入两轮） | 0 码 | 0 码 | 0 码 | 0 码 |

**8W16A 的 4e-5 缺口拆开（1−cos 预算，native 缝）**：

| 项 | 1−cos | 占判据 1e-4 的 |
|---|---|---|
| 只量化 int8 权重、激活精确（`--wq-chain`，与 04e 逐位相同） | 8.43e-5 | **84%** |
| + int16 激活（量化 golden） | 9.43e-5 | +10% |
| + 器件舍入/LUT（device） | **1.402e-4** | +46% |
| 判据 | 1.0e-4 | — |

⇒ **8-bit per-channel 权重自己就用掉 84% 的预算**，整层要到 0.9999 只剩 1.6e-5 给激活与器件；
int16 激活（+1.0e-5）已经把激活侧压到几乎不可测，剩下的是器件舍入（+4.6e-5）。
**再往激活上加位宽不会有收益，要过 0.9999 只能动权重侧（16-bit/fp16 权重、group-wise scale、hessian）。**

---

## 1. 先修一个 harness 缺陷（它会改变四格的读数，也改变 04e 的结论）

第一次跑真权重时发现 `q3`（FC+RoPE）段只有 **0.9747**，而「只量化 int8 权重」是 0.9999157。
追下去是 **RoPE 的拼接写错了 scale**，不是激活位宽问题：

* `build_layer` 里 RoPE 的两半 `lo = q·cos − k·sin`、`hi = k·cos + q·sin` **各自**用 `pick_scale` 取 scale，
  然后 `CONCATENATION` 只声明**一个**输出 scale（`ins[0].s`），golden 又把两段码直接 `np.concatenate`。
* 真权重下 `|hi|max = 80.59`、`|lo|max = 51.13`（夹具 25.55 / 25.52），两半尺度差 **1.576 倍** ⇒
  hi 半被低估 1/1.576、0.22% 的元素被 int16 量程截断。夹具下尺度比 1.0003，所以 04c/04d 的夹具结论没被这条影响，
  **04e 的真权重数字被影响**。

三条实测（真权重 8W16A，`--upto q3`，`ops/_tier_matrix/q3_w8a16_fixed`）：

| 口径 | cos(q3, fp64) |
|---|---|
| 修复前 golden（两半按同一 scale 拼） | 0.9746969 |
| 修复前 **device** | 0.9933635 |
| 「把 hi requant 进输出 scale 并按 int16 截断」模型 | 0.9933636（与 device 逐码到 **4 码**） |
| 修复后 golden / device | **0.9999999 / 0.9999999**（码级 max 3） |

⇒ 设备按 TFLite 语义**是对的**（输入各自 requant 到输出 scale），旧 golden 没做这件事，所以
「device 比 golden 更准」这个 04e 的 UNKNOWN 有一半就是这条；另一半是**声明的输出 scale 装不下 hi 的 outlier**。

修复（两处，**图结构与层数不变**，只改 scale 取值）：

1. `build_layer` 的 RoPE 两半共用一条 `s_rope`（覆盖两者量程），`q_concat` 收到同一个 scale；
2. `q_concat` 的 golden 改为**按每个输入自己的 scale requant 到输出 scale**（与上面实测的设备语义一致；
   相等 scale 时仍是精确拷贝，`kv14` 的拼接不受影响）。

顺带一条：`_requant_check` 的界从 2^62 改回 int64 的**精确界 2^63**。新 scale 把**夹具** 4W16A 的
`scores` 累加器推到 `peak*m0 = 4.615e18 = 0.9999 × 2^62`，旧 cushion 误拒（产品仍在 int64 的 2 倍余量内）；
真溢出仍会被拦。

**夹具回归**（同一 builder，只换夹具权重）：4W16A 0.9509311 → 0.9509261、8W8A 0.9673771 → 0.9684761、
4W8A 0.9130/0.9134 → 0.9151901、4W16A `scores` 0.9999949 → 0.9999955。位移 ≤0.002，来源是 hi 半的 requant。
04c/04d 的夹具结论（0.9649 / 0.9510 / probs 0.9999993 的段边界）方向与量级都不变。

---

## 2. 事前登记（都在第一次板端运行之前写进 `model.json`）

| 项 | 登记值 | 依据 |
|---|---|---|
| 质量判据 | `cosine_min = 0.9999`（device vs 独立 fp64 链） | ADR 0005 原文 |
| 设备忠实 | ｜cos(dev) − cos(golden)｜ ≤ 0.003 | ADR 0005 2026-11-13 订正第 1 条 |
| 整层 `int8_tol`（码） | **2** | 与 04/04c/04d 同值同理由（同一 int8 RMSNorm 链 + 同一条 dtype 缝）；**未事后调整** |
| seam 段（`--upto probs`，fp16 输出） | `atol=1e-3`、`rtol=1e-2` | 1 ULP@1.0 = 9.77e-4；golden 在同一位置舍入 |
| 噪声底 | 实测，不预设 | 同 `.dla` 同输入两轮逐字节比较；**六张图全部 0 码** |

三个 FAIL 原样保留：整层四格全部 FAIL（三格距离判据 0.03–0.12，8W16A 差 6.9e-5）、
码级四格全部超 `int8_tol=2`（int16 域 1 码 = 9.3e-5，mean 13–19 码 = 1.2–1.8e-3 绝对，**码级不可跨位宽比较**）。

---

## 3. 做法：只换位宽两轴与权重来源，图不动

    wsl> /usr/bin/python3 tools/build_w4a8.py layer --w-bits {4,8} --a-bits {8,16} \
             [--seam {f16,native}] --weights real --input real --upto {scores,probs,out} --out ops/<name>

四格 = `--w-bits{4,8} × --a-bits{8,16}`，全部 `--weights real --input real`（真 embed 行的同一批 16 个 token id）、
T=16、同一 per-channel/int32-bias/两次整数舍入契约。每格建三张图（整层 / `--upto scores` / `--upto probs`），
共 12 张；另加两个 native 缝的整层与 8W16A native 的两个分段图。

---

## 4. 四格全表（板端）

### 4.1 编译

| 图 | 层数 | 执行步 | bridge | Static | Code | Total | `.dla` |
|---|---|---|---|---|---|---|---|
| `decoder_layer_t16_w4a8_real` | 61 | 1 | 无 | 7.2M | 32K | 7.3M | 7,633,271 B |
| `decoder_layer_t16_w4a16_real` | 61 | 1 | 无 | 7.3M | 42K | 7.4M | 7,725,923 B |
| `decoder_layer_t16_w4a16_real_native` | 60 | 1 | 无 | 7.3M | 43K | 7.4M | 7,727,061 B |
| `decoder_layer_t16_w8a8_real` | 61 | 1 | 无 | 14M | 32K | 14M | 15,087,991 B |
| `decoder_layer_t16_w8a16_real` | 61 | 1 | 无 | 14M | 34K | 14M | 15,104,511 B |
| `decoder_layer_t16_w8a16_real_native` | **60** | 1 | 无 | 14M | 34K | 14M | 15,105,649 B |

### 4.2 质量（device vs 独立 fp64 链）

| 量 | 4W8A | 4W16A | 8W8A | 8W16A（f16 缝） | 8W16A（native 缝） |
|---|---|---|---|---|---|
| cosine | 0.8775767 | 0.9732673 | 0.8971126 | **0.9998311** | **0.9998598** |
| rel_l2 | 239.05% | 23.263% | 261.98% | 1.8381% | 1.7308% |
| 量化 golden 的 cosine | 0.8841884 | 0.9732694 | 0.8792696 | 0.9998918 | 0.9999057 |
| ｜Δcos｜ | 0.0066 | 0.0000（2.1e-6） | 0.0178 | 0.0001（6.1e-5） | 0.0000（4.6e-5） |
| 1 码（真实量） | 2.391e-2 | 9.268e-5 | 2.391e-2 | 9.268e-5 | 9.268e-5 |
| 码级 device vs golden（不匹配 / max / mean） | 12931/14336、118、8.16 | 14045/14336、517、18.09 | 13049/14336、113、10.08 | 13975/14336、431、18.80 | 13835/14336、247、13.58 |

**设备忠实度**：4W16A（2.1e-6）、8W16A（6.1e-5 / 4.6e-5）、4W16A native（9.4e-5）都 ≤0.003 PASS；
两个 int8 档（0.0066 / 0.0178）FAIL —— 与 04e 同现象（int8 elementwise 段的 golden 偏悲观），
**在真权重上判断设备准不准要读 device 列**。

### 4.3 分段表（device vs fp64；golden 在括号里）

| stage | 4W8A | 4W16A | 8W8A | 8W16A | 8W16A native |
|---|---|---|---|---|---|
| `scores` | 0.9997529 (0.9997674) | 0.9999862 (0.9999863) | 0.9997597 (0.9997853) | **0.9999998** (0.9999998) | 0.9999998 |
| `probs` | 0.7476799 (0.7459077) | 0.9986388 (0.9984837) | 0.7558834 (0.7557585) | 0.9996958 (0.9997066) | **0.9998943** (0.9999504) |
| `out` | 0.8775767 (0.8841884) | 0.9732673 (0.9732694) | 0.8971126 (0.8792696) | 0.9998311 (0.9998918) | **0.9998598** (0.9999057) |

device-free 的十段（golden vs fp64，`--stages`）：

| stage | 4W8A | 4W16A | 8W8A | **8W16A** | 8W16A native |
|---|---|---|---|---|---|
| n1 | 0.9979915 | 0.9999998 | 0.9979915 | 0.9999998 | 0.9999998 |
| q3 | 0.9996204 | 0.9999847 | 0.9996304 | 0.9999999 | 0.9999999 |
| scores | 0.9997674 | 0.9999863 | 0.9997853 | 0.9999998 | 0.9999998 |
| probs | **0.7459077** | 0.9984837 | **0.7557585** | 0.9997066 | 0.9999504 |
| attn | 0.7593061 | 0.9520193 | 0.7883532 | 0.9997018 | 0.9998262 |
| r1 | 0.9476707 | 0.9903094 | 0.9434586 | 0.9999487 | 0.9999654 |
| n2 | 0.9302449 | 0.9888528 | 0.9257702 | 0.9999373 | 0.9999585 |
| mlp_y | 0.8525503 | 0.9714449 | 0.8267081 | 0.9998700 | 0.9998927 |
| down | 0.8883897 | 0.9731287 | 0.8822534 | 0.9998914 | 0.9999053 |
| **out** | 0.8841884 | 0.9732694 | 0.8792696 | **0.9998918** | **0.9999057** |
| softmax 翻转行 | 164/224 | 15/224 | 161/224 | 10/224 | **2/224** |

**读法**：8 位激活在真权重下把 `probs` 打到 0.75（1 码 = 98.5，softmax 近并列 row 翻转 161–164/224），
这是 4W8A/8W8A 都救不回来的地方（04e 的结论在修好 RoPE 后依然成立）。
int16 把 `probs` 抬到 0.9997–0.9999，掉码起点改成 `attn`（P·V 的 1 码 = 8.85e-6，在 |attn|max 0.29 上偏粗）
与 `mlp_y`（0.99989）。

---

## 5. 8W16A 的缺口在哪里（票面第二轮问题的直接回答）

按 1−cos 拆（native 缝，真权重）：

| 来源 | 1−cos | 说明 |
|---|---|---|
| int8 per-channel 权重（激活精确） | 8.43e-5 | `--wq-chain`：0.9999157（与 04e/04b 逐位相同） |
| int16 激活（量化 golden 相对 weights-only） | +1.00e-5 | `probs` 0.9999504、`attn` 0.9998262 贡献 |
| 器件舍入 / LUT（device 相对 golden） | +4.59e-5 | 码级 mean 13.58 码（=1.26e-3 绝对），`|Δcos|` 4.6e-5 |
| 合计 device | **1.402e-4**（0.9998598） | 判据 1.0e-4，**超 4.0e-5** |

**哪一段拖后腿**：不是 probs（native 缝下已 0.99989），而是 **int8 权重本身（84% 的预算）+ 器件的 int16
定点链路（46%）**。fp16 缝还会额外赔 2.4e-4 给 `probs`（0.9996958 vs native 0.9998943）—— 见 §6。

---

## 6. int16 还需要 fp16 缝吗（04d 的结论在真权重复核）

| 8W16A 真权重 | f16 缝 | native 缝 |
|---|---|---|
| 层数 | 61 | **60** |
| `probs` device | 0.9996958 | **0.9998943** |
| 整层 device | 0.9998311 | **0.9998598** |
| `|Δcos|` | 6.1e-5 | 4.6e-5 |
| `.dla` | 15,104,511 B | 15,105,649 B |
| 板端（6 次中位） | 2.16 ms | 2.16 ms |

**不需要 fp16 缝**：native 缝少 1 层、`probs` 高 2e-4、整层高 2.9e-5、体积与时间同档。
04d 的夹具结论（native 段边界 0.9999993）在真权重上同向复现。**建议主档用 native 缝。**

---

## 7. 8W8A 复核（票面要求「用同一判据复核一格」）

| 量 | 04e 记的 | **本票（修好 RoPE 后）** |
|---|---|---|
| device vs fp64 | 0.8801239 | **0.8971126** |
| golden vs fp64 | 0.8737825 | 0.8792696 |
| probs device | 0.7676888 | 0.7558834 |
| 码级 max / mean | 120 / — | 113 / 10.08 |
| `.dla` | 15,089,171 B | 15,087,991 B |

结论方向不变（8W8A 远低于 0.9999，差 0.10），**但 04e 的 0.8801 被 RoPE bug 压低了 0.017**；
「真权重比夹具更差」依然成立（真 0.8971 vs 夹具 0.9649/0.9685-golden），差距从 0.085 收敛到 0.068。

---

## 8. 代价（体积 / Static / 时间）

| 量 | 4W8A | 8W8A | 4W16A | 8W16A | 8W16A native |
|---|---|---|---|---|---|
| `.dla` | 7,633,271 | 15,087,991 | 7,725,923 | 15,104,511 | 15,105,649 |
| 相对 4W8A | 1.00× | 1.98× | 1.01× | 1.98× | 1.98× |
| Static | 7.2M | 14M | 7.3M | 14M | 14M |
| 板端单层（6 次中位 / 区间） | 1.43 / 1.11–1.61 | 1.81 / 1.42–2.20 | 2.20 / 1.99–2.27 | 2.16 / 1.82–2.48 | 2.16 / 2.12–2.45 |

* 全部单执行步、无 bridge；int16 只改激活张量（28K vs 14K）与 Code（+2K），**权重侧体积不变**。
* 时间只读成档位差：**4W8A 最快（~1.4 ms）**；8W8A / 4W16A / 8W16A 同档（1.8–2.2 ms）。
  同一份 `.dla` 在本会话内能测出 1.11 / 1.61 ms 两个值（r3 vs r4），跨轮漂移比档位差还大 ——
  与 04c/04e 记录一致，**没有做时钟/热态控制**。

---

## 9. 结论

1. **8W16A 不达 0.9999**：0.9998311（f16 缝）/ **0.9998598（native 缝）**，差 6.9e-5 / 4.0e-5。
   这是四格里唯一进入 1e-4 量级的档（4W16A 0.9733、8W8A 0.8971、4W8A 0.8776）。
2. **缺口不在激活**：int8 权重本身 0.9999157（84% 的 1e-4 预算），int16 激活只再加 1.0e-5，
   器件定点链路再加 4.6e-5。⇒ 想真正过 0.9999，只能动**权重侧**（fp16/16-bit 权重、group-wise、hessian）；
   这与 04b/04d 的结论一致，并且现在有了真权重的量化拆分。
3. **主档建议：8W16A + native 缝**（60 层、15.1 MB、2.16 ms）。它是唯一可用档；
   4W16A 在真权重下 0.9733 < 0.999，**不能当「更省内存的备选」**（票面给的那条路径不成立）。
4. **4-bit 彻底退出主档**：int4 per-channel 的权重天花板在真权重上是 0.9733031（`--wq-chain`），
   4W16A 实测 0.9732694 与它重合 —— 激活再精细也没用。
5. **判据本身需要 Lead 决定**：在 int8 per-channel 权重下，整层 0.9999 只剩 1.6e-5 的余量给激活与器件，
   而器件舍入一项就是 4.6e-5。要么把档位判据按实际改成 `≥0.9998`（并在 ADR 里写清它与「算子级 0.9999」的区别），
   要么回答「0.9999 必须配 16-bit/fp16 权重」。本票不改判据、只给数。

---

## 10. 与票面 checklist 对照

| 票面要求 | 状态 | 证据 |
|---|---|---|
| 四个组合补齐 4W8A / 4W16A / 8W16A，8W8A 同判据复核 | ✅ | §4（12 张图 + 2 个 native 缝整层，全部真权重/真输入） |
| 每格给整层 device vs fp64 cosine、分段（scores/probs/out）、码级、｜Δcos｜ | ✅ | §4.2 / §4.3 |
| 8W16A ≥0.9999？主档结论 | ✅ **否**（0.9998311 / 0.9998598）；主档建议 8W16A+native | §5 / §9 |
| 每格 `.dla` / Static / 板端单层时间 | ✅ | §8 |
| 结论写进 `docs/OP-tier-matrix.md` 与 ADR 0005 第五次订正 | ✅ | 本文 + ADR |
| 若某格编不过（seam），记录阶段与替代方案 | ✅ **没有编不过的格**；两个 seam 都 EXIT=0 | §6 |

---

## 11. UNKNOWN（不圆）

1. **器件为什么比量化 golden 差 4.6e-5**（8W16A）：没有逐算子对拍，只知码级 mean 13.58 码集中在
   elementwise/RSQRT/LOGISTIC/SOFTMAX 的定点表上（04d §4.3 已经量过 LUT 的绝对分辨率 ~2e-4 满量程）。
   本轮没有把这条拆到单个算子。
2. **0.9999 与 0.9998 的选择是产品决定**，不是测量结论；本票只证明了「int8 权重 + 任何激活位宽都在 1e-4 边缘」。
3. **时间没有时钟/热态控制**（同 04c/04e），只作档位粗比较；native 缝少 1 层但没有更快。
4. **只测 T=16 的 prefill 形态**；decode（T=1）不在本票。
5. **修好的 RoPE 拼接让 04e 的 8W8A 数字整体上移 0.017**；04e 正文没有改（历史证据保留），
   ADR 第五次订正里给出了订正读数。**是否回填 04e 文档由 Lead 决定。**

---

## 12. 复现

```bash
export QWEN25_WEIGHTS_DIR="/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen"
export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
WT="/mnt/d/deepseek_ workspace/mtk/wt-q04f"

# 1) 12 张图（编译放锁外）
for t in "w4a8 4 8" "w4a16 4 16" "w8a8 8 8" "w8a16 8 16"; do set -- $t
  for u in out scores probs; do
    /usr/bin/python3 tools/build_w4a8.py layer --w-bits $2 --a-bits $3 --weights real --input real \
        --upto $u --out ops/$1_$u ; done ; done
./tools/compile.sh ops/decoder_layer_t16_w8a16_real/model.tflite ops/decoder_layer_t16_w8a16_real/model.dla

# 2) 板端（一个锁会话；每张图前后 healthcheck）
./tools/devlock.sh run -- bash -c './tools/healthcheck.sh --quick && \
  ./tools/e2e_i16.sh ops/decoder_layer_t16_w8a16_real && ./tools/healthcheck.sh --quick'

# 3) 数
/usr/bin/python3 tools/build_w4a8.py --w-bits 8 --a-bits 16 --weights real \
  --device-report ops/decoder_layer_t16_w8a16_real ops/_tier_matrix/w8a16_scores ops/_tier_matrix/w8a16_probs
/usr/bin/python3 tools/build_w4a8.py --stages --w-bits 8 --a-bits 16 --weights real
/usr/bin/python3 tools/build_w4a8.py --wq-chain --weights real
/usr/bin/python3 tools/build_w4a8.py --weights-report --weights real
```

**可复现性**：同参数重建 `model.tflite` 与 `golden/*.bin` 逐字节稳定；`.dla` 不可复现（仓库既有约定）。
