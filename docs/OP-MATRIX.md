# OP-MATRIX —— 算子支持矩阵（W6 并行工作的合并结果）

采集：2026-11-12 · ncc-tflite 7.3.15 / mdla5.0 / MT6897(Titan 1) · 全部 `--disallow-bridge`。
回归命令：`./tools/run_all.sh`（合并后 **PASS=75 XFAIL=16 FAIL=0**）。

判定口径：**编译**（是否落 MDLA 且无 bridge） / **板端**（是否真跑） / **数值**（与独立 golden 比）。
三者分列，任何一项不能覆盖另一项。

## 1. 已确认可用（编译 + 板端 + 数值都有）

| 算子 | dtype | 落点（层） | 数值 | 记录 |
|---|---|---|---|---|
| FULLY_CONNECTED（静态权重） | int8 | FullyConnectedLayer | 逐位（small-codes 档）；全量程 ±1 LSB | S2/E-002/E-003 |
| FULLY_CONNECTED（动态权重） | int8 | 同上 | 逐位；换权重/换进程可复现 | S4/E-005 |
| FULLY_CONNECTED | fp16 | 同上 | 逐位 | E-004 |
| **ADD（自建提交器）** | int8 | MDLA（不经厂商 runtime） | 逐位 | E-106B-3 |
| ADD / MUL / SUB / MAXIMUM / MINIMUM | int8 | ElementWise*Layer | **逐位**（4352 codes 零误差） | OP-elementwise |
| RESHAPE / TRANSPOSE / CONCAT / SPLIT / PAD | int8、fp16 | Reshape/Transpose/Concat/Slice/Pad Layer | **逐位**（18 个 op dir） | OP-shapeops |
| RELU / LOGISTIC / TANH | int8、fp16 | ReLU/Sigmoid/Tanh Layer | 逐位（TANH fp16 差 1 code） | OP-actreduce |
| **SOFTMAX** | int8、fp16 | **SoftMaxLayer** | 逐位（int8 `[1,8,16]`）；fp16 差 1 code；**beta 生效** | OP-actreduce |
| MEAN / SUM | int8、fp16 | Mean / Sum Layer | 逐位（含非末轴） | OP-actreduce |
| CONV_2D（k1/k3、s1/s2、dilation 2、SAME/VALID、RELU6 融合） | int8 | Conv2DLayer（+ReLU6Layer） | 逐位；融合 RELU6 有 1/512 差 1 code（容差 1） | OP-convpool |
| DEPTHWISE_CONV_2D（mult 1/2） | int8 | DepthwiseConv2DLayer | 逐位 | OP-convpool |
| MAX_POOL_2D / AVG_POOL_2D | int8 | Max/AvgPool2DLayer | 逐位 | OP-convpool |
| 64³ matmul 用 1×1 conv 表达 | int8 | Conv2DLayer×2 + ConcatLayer | 逐位（但比 FC 慢 ~1.35×） | OP-convpool |

## 2. 边界规则（跨算子统一）

1. **16 字节行规则**：无桥编译要求 DRAM 边界张量的**最内层行字节省数 ≥16 且 %16==0**（输入侧最严）。
   它解释了 FC 的"K 必须是 16 的倍数"，也解释了 shapeops 的 `IN_GAP`/`OUT_GAP`。
2. **元素级算子**（MUL/SUB/MAX/MIN）除此之外还有 `K==4 且 (M*K)%16==0` 的例外（机制 **UNKNOWN**）。
3. **CONV/pool 的通道边界**：C=16 稳定；C=8 四个家族全失败；C=4 时好时坏；C≥32 可编译但会被分解成
   `Slice+op+Concat`（2×Conv），因此 1×1-conv matmul 比 FC 大 2.7×、慢 ~1.35×。
4. 失败原文统一为 `Bridging is disallowed but gap is detected between (Input|MDLA_5_0) and …`；
   **失败的模型仍会先打印 MDLA_5_0 的 exec plan**，必须看 `EXIT=` 判断。

## 3. 数值契约（重要，且已用"未见过的输入"验证过）

- **int8 requant 不是 round-half-away**。真实规则（从 dump 反推、再用未参与推导的 1/5、1/6 乘子验证）：

  ```
  (M0, shift) = QuantizeMultiplier(scale_ratio)
  out = RoundingDivideByPOT( round_half_up(acc * M0 / 2^31), shift )    # 两次舍入
  ```
  在 6912 个已记录值 / 6 个乘子上 0 失配；对未见乘子预测 0/512。这解释了 FC 全量程档的 ±1 LSB（E-003）。
- **AVG_POOL 用另一条规则**：对"有效元素个数"做一次 round-half-up（padding 不计入分母；
  把 padding 计入会在 432/1024 上被证伪）。
- `fp16` 层的差异普遍是"1 个 fp16 code"，不是算法差异。

## 4. XFAIL（已记录的反例，回归里单独计数）

16 个：shapeops 的 6 个（8 字节行宽边界）、act 的 `relu_1x13_fp16`、elem 的若干非 16 倍数形状、
convpool 的 9 个（C=4/8 的 pool 与 conv、dilation/stride 特定组合）。
它们**在 run_all 里按 commit 记录的 EXIT=1 复核**：仍失败 = XFAIL，突然能编过 = REGRESS。

## 5. 未覆盖 / 下一步

1. SOFTMAX 的 `-inf` 掩码（因果 attention 的真实前提）；4-D `[1,H,L,L]`、batch>1、L=256/1024；
2. CONV/pool 的 C=8 失败机制（UNKNOWN）与 eDPA relayout 政策（是否允许"非 MDLA 步但仍是 NPU"的桥）；
3. 逐通道 scale、fp16 卷积、per-channel 量化；
4. 把自建提交器从 ADD 泛化到其它算子（elem 已给出 `npu_binop` 框架，MUL/SUB/MAX/MIN 还缺各自的命令流抓取）。

---

## 9. 自建提交器覆盖（不经过厂商 runtime）

> 判定标准：**我们自己的进程**完成 分配 → 重定位 → 组装 `mdw_cmd_in` → `CMD RUN` → 读回，
> 且与独立 golden 逐字节一致。厂商 runtime 只在"抓一次合法提交"时出现，之后不再参与。

| 算子 | 入口 | 结果 |
|---|---|---|
| ADD | `npu_add64()` / `npu_binop(NPU_OP_ADD)` | 逐位 0/64 |
| MUL | `npu_binop(NPU_OP_MUL)` | 逐位 0/64 |
| SUB | `npu_binop(NPU_OP_SUB)` | 逐位 0/64 |
| MAXIMUM | `npu_binop(NPU_OP_MAXIMUM)` | 逐位 0/64 |
| MINIMUM | `npu_binop(NPU_OP_MINIMUM)` | 逐位 0/64 |

流程与资产格式、新增算子的三步法见 `docs/OWN-SUBMITTER.md`；抓取工具 `probe/mkasset.sh`。
**未覆盖**：一元算子（RELU/SOFTMAX，代码已支持 `n_inputs=1`，待实测）、FULLY_CONNECTED（需要动态权重与更多重定位处）、非 64 元素形状。

---

## 10. W6b 新增（2026-11-13，按"一算子一 agent"并行）

| 家族 | 算子 | 结果 | 关键结论 |
|---|---|---|---|
| 注意力矩阵乘 | **BATCH_MATMUL**（QKᵀ / PV，双动态输入） | 26/29 通过，int8 逐位、fp16 2.4e-4 | **两个操作数都是运行时张量也能上 NPU**；FC(dynW)/CONV_2D(dynF) 也接受运行时 B；换 B 不需重编译 |
| 归一化 | **RSQRT + RMSNorm** | 链式 6 层同一个执行步，fp16 2 ULP / int8 ±1 code | RSQRT 是**原生层**，不需要 CPU 求倒数；int8 会把 sub-LSB 的 eps 常量折掉 |
| 位置编码 | **RoPE（split-half）** | 10 变体全过，fp16 1 ULP | **sin/cos 可作运行时输入**（11 层），换位置不用重编译；4-D 形状炸到 35 层（rank 触发） |
| 门控激活 | **SiLU / gated MLP** | fp16 全过（1 code），int8 需 696→704 填充 | 本 schema **没有 SILU 算子**，只能 `LOGISTIC+MUL`；组合误差为零 |
| 整块 | 1 层 decoder block（无 Norm/RoPE） | ✅ 16 层同一 MDLA 步，max_abs 9.77e-4 | 见 `docs/OP-block.md` |
| 整块 | + RMSNorm 进图 | ❌ 卡在形状传播 | bisect 到 2 算子仍失败；下一步做两份 tflite 的 tensor 表 diff |

回归：**133 PASS / 0 FAIL / 26 XFAIL**（`./tools/run_all.sh`，前后 healthcheck PASS）。

---

## 11. 2026-11-13 收尾：三个里程碑

| # | 里程碑 | 证据 |
|---|---|---|
| 1 | **完整 block**（RMSNorm + RoPE 都在图内） | `ops/block_qwen25_1l_full_fp16`：42 层、单 MDLA 步、`.dla` 91.8 KB、板端 rmse 1.86e-04 → PASS |
| 1b | **T=128 prefill** | `ops/block_qwen25_1l_full_t128_fp16`：同样 42 层、PASS（rmse 1.90e-04） |
| 1c | 多 head | ❌ WIP：`--heads 2` 走 RESHAPE+BMM 路径，importer 报 `Fail to process the given model` |
| 2 | T=128 变体 | ✅ 同 1b |
| 3 | **自建提交器跑 FC（动态权重）** | `probe/npu_fc_demo`：`CMA RUN rc=0`，3 处重定位，1/64 元素差 1 code（与 `int8_tol=1` 契约一致），两次运行一致 |

回归：**136 PASS / 0 FAIL / 26 XFAIL**（`tools/run_all.sh`，前后 healthcheck PASS）。

### 定位 v2 bug 的方法（写进流程）
`probe/dump_tflite.py` 打印 tensor 表 + operator 表，两份模型 diff → 一眼看到
`RSQRT` 的输入是 **axis 张量**（`len(tensors)-1` 在 append 之后取值导致的索引错位）。
**"独立算子能跑、拼进整图跑不了"这类问题，tflite 表 diff 比手写试错快一个数量级。**

---

## 12. ✅ 自建提交器跑通完整 attention（2026-11-13）

`probe/npu_attn16_demo`，T=16、int8、**51 次提交**、同一进程、无 `libneuron`/`neuronrt`：

```
q,k,v = x·Wᵀ    ← 48 次 FULLY_CONNECTED(动态权重)
s     = q·kᵀ    ← BATCH_MATMUL(T=16)          mismatch 0/256
p     = softmax(s) ← SOFTMAX[1,16,16] + LUT    mismatch 0/256
o     = p·v     ← BATCH_MATMUL(PV T=16)        mismatch 0/1024
```

两次运行结果相同；前后 healthcheck PASS。

**定位方法（可复用）**：命令流字节 diff（我方 patch 后 vs 厂商抓取）证明"只有 3 个地址字不同"，
于是把怀疑从命令流转到缓冲 → 再比对缓冲内容，发现 `mkasset.py` 把资产目录上一层的
`vendor_out.bin` 当成了输入，导致 **LUT 与输出缓冲互换**（给 kernel 喂了全零 LUT）。

**链路里必须自己做的一步**：softmax 输出域 `(1/256, zp=-128)` → PV 输入域 `(1/127, zp=0)`
的 requant（`code = round(((c+128)/256)*127)`）。TFLite 图里这步由 ncc 插入。

---

## 13. 大维度 FC 实测（2026-11-13）

用 `tools/build_ops.py fc` 生成、ncc 编译、`tools/e2e.sh` 板端对拍（int8，独立 golden）：

| 形状 (M×K×N) | 权重 | 编译 | 板端数值 | 判定 |
|---|---|---|---|---|
| 8 × 1024 × 1024 | 1 MB | EXIT=0，1 层 | 5/8192 差 1 code | PASS |
| 64 × 1024 × 1024 | 1 MB | EXIT=0 | 41/65536 差 1 code | PASS |
| 1 × **4096 × 4096** | **16 MB** | EXIT=0 | **0/4096 逐位** | PASS |
| 1 × **8192 × 8192** | **64 MB** | EXIT=0 | 2/8192 差 1 code | PASS |

结论：**单算子 FC 在 64 MB 权重、8192×8192 的规模上依然能编译并在 MDLA 上正确执行**；
偏差全部是 ±1 code 的 requant 舍入（`int8_tol=1` 契约），没有结构性失败。

### 自建提交器跑大 FC：卡在映射域

- `probe/iolog2.c` 的抓取上限从 64 KiB 放宽到 **4 MiB**（大权重现在能被 dump）；
- 但 512×512（256 KiB）权重缓冲在驱动里拿到的地址是 **`0xcbbc0000`**，和其它张量的
  `0x25xxxxxx` 完全不在一个区段 —— 说明**大张量走了不同的映射路径**（可能是另一段 IOVA 域或 host 映射）。
- 因此 `mkasset` 目前还无法为大 FC 生成可用资产（地址语义未明）。这是下一步的明确问题：
  **驱动对大缓冲的 IOVA 分配/映射语义**。
