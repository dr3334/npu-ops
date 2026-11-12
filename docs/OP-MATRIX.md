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
