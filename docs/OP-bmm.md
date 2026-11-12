# OP-bmm —— attention 里的两个双重动态矩阵乘（QKᵀ / PV）

范围： transformer block 自注意力里的两个 matmul，**两个操作数都是运行时输入**。

```
scores   = Q · Kᵀ      Q,K: [1,T,64]  ->  [1,T,T]
attn_out = P · V       P: [1,T,T], V: [1,T,64] -> [1,T,64]
batched  = 2 heads     [1,2,T,64] × [1,2,T,64] -> [1,2,T,T]
```

T ∈ {8, 16, 32}；int8 与 fp16。工具链 ncc-tflite **7.3.15** / mdla5.0 / MT6897(Titan 1)，
全部 `--disallow-bridge`。生成器 `tools/build_bmm.py`（29 个变体），产物 `ops/bmm_*`, `ops/fc_bmm_*`,
`ops/conv_bmm_*`。`.tflite`/`.dla`/`.bin` 被 gitignore，用生成器重建即可（不依赖历史产物）。

---

## 1. 假设（先写，后测）

| 编号 | 假设 | 如果成立，预期观察 | 结果 |
|---|---|---|---|
| H1 | ncc 把 `BATCH_MATMUL` 映射成 MDLA 原生 matmul 层 | exec plan 出现 `BatchMatMulLayer` + `Target: MDLA_5_0`，`EXIT=0` | **成立**（单层，无额外转换） |
| H2 | `BATCH_MATMUL` 没有 MDLA 后端 → 桥接 CPU | `Bridging is disallowed…`，`EXIT=1` | **否定** |
| H3 | `FULLY_CONNECTED` 的动态权重通道可覆盖 QKᵀ | QKᵀ 无需转置（`out[m,n]=Σ_k a[m,k]W[n,k]`，W:=K） | **成立** |
| H4 | PV 用 FC 表达必须喂 **Vᵀ**（W 形状 `[N,K]=[64,T]`） | 直接喂 `[T,64]` 会算错；喂 Vᵀ 逐位对 | **成立**（并且 CONV 形式有同一个坑，见 §4） |
| H5 | `CONV_2D`(1×1) 无法做动态 B（filter 必须常量） | 转换期报错 | **否定** —— 接受且 filter 真按运行时读取 |

形状边界预测（16 字节行规则）：**3 条全部命中**（见 §3.2）。

---

## 2. 做法

`tools/build_bmm.py` 手搓 flatbuffers（follow `tools/build_ops.py` 的 FC 模式），golden 全部用 numpy
独立算（int8 → int64 累加；fp16 → float32 累加后落 fp16），与序列化代码零共享（R3）。

| 形式 | 图结构 | 备注 |
|---|---|---|
| `form=bmm` | `BATCH_MATMUL`，A/B 都是 subgraph input | QKᵀ 用 `adj_x=0, adj_y=1`；PV 用 `0/0` |
| `form=fc` | `FULLY_CONNECTED`，`weights` 是 subgraph input（S4/E-005 通道） | bias 为常量 0 |
| `form=conv` | `CONV_2D` 1×1，`filter` 是 subgraph input | NHWC，W=1 |

量化契约（**先登记，后跑**，写在 `model.json`）：

- int8 `codes=small`（码值 ∈ {−1,0,1}，`out_scale = s_in·s_w` → 乘子恰为 1）→ `int8_tol=0`（逐位）。
  ±1 码值就是 AGENTS §3.2 的"基向量判布局"：转置/偏移错误无法存活。
- int8 `codes=full`（全量程，乘子 1/32）→ 按 OP-MATRIX §3 的两次舍入规则登记 `int8_tol=1`。
- fp16 → `atol=1e-2, rtol=3e-2`（K 项 fp16 累加的相对误差上界 ≈ K·2⁻¹¹ = 64·4.9e-4 ≈ 3.1e-2）。

复现：

```bash
wsl -e bash -lc 'cd "/mnt/d/deepseek_ workspace/mtk/wt-bmm" && \
  for s in $(/usr/bin/python3 tools/build_bmm.py --list); do \
    /usr/bin/python3 tools/build_bmm.py "$s" --alt-b --out "ops/$s/model.tflite"; \
    ./tools/compile.sh "ops/$s/model.tflite" "ops/$s/model.dla"; done'
export MSYS_NO_PATHCONV=1
NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" ./tools/devlock.sh run -- ./tools/e2e.sh ops/bmm_pv_t16_int8
```

---

## 3. 结果

### 3.1 三种 op 形式 × 两个 matmul（编译 / 板端 / 数值，分列）

图例：层名 = exec plan 里落 MDLA_5_0 的层；数值 = 与独立 golden 比。

| op 形式 | matmul | T | dtype | 编译 | 落点（层） | 板端 | 数值 |
|---|---|---|---|---|---|---|---|
| **BATCH_MATMUL** | QKᵀ | 8 | int8 | OK | BatchMatMulLayer | 跑通 | **逐位 0/64** |
| | QKᵀ | 16 | int8 | OK | BatchMatMulLayer | 跑通 | **逐位 0/256** |
| | QKᵀ | 16 | int8 全量程 | OK | BatchMatMulLayer | 跑通 | 1/256 差 1（tol=1）|
| | QKᵀ | 32 | int8 | OK | BatchMatMulLayer | 跑通 | **逐位 0/1024** |
| | QKᵀ | 16 | fp16 | OK | BatchMatMulLayer | 跑通 | **逐位**（max_abs 0）|
| | P·V | 8 | int8 | **EXIT=1** | — | — | 见 §3.2 |
| | P·V | 16 | int8 | OK | BatchMatMulLayer | 跑通 | **逐位 0/1024** |
| | P·V | 16 | int8 全量程 | OK | BatchMatMulLayer | 跑通 | 6/1024 差 1（tol=1）|
| | P·V | 32 | int8 | OK | BatchMatMulLayer | 跑通 | **逐位 0/2048** |
| | P·V | 16 | fp16 | OK | BatchMatMulLayer | 跑通 | max_abs 2.44e-4（登记 1e-2/3e-2）|
| | P·V | 8 | **fp16** | OK | BatchMatMulLayer | 跑通 | **逐位**（8×16b=16B 行）|
| | QKᵀ 2头 `[1,2,16,64]` | 8/16 | int8 | OK | BatchMatMulLayer（单层）| 跑通 | **逐位 0/128、0/512** |
| | QKᵀ 2头 | 16 | fp16 | OK | BatchMatMulLayer（单层）| 跑通 | **逐位** |
| | P·V 2头 `[1,2,16,·]` | 16 | int8 | OK | **Slice+BMM+Slice+BMM+Concat** | 跑通 | **逐位 0/2048** |
| **FULLY_CONNECTED**(动态 W) | QKᵀ | 8 | int8 | OK | FullyConnectedLayer | 跑通 | **逐位 0/64** |
| | QKᵀ | 16 | int8 | OK | FullyConnectedLayer | 跑通 | **逐位 0/256** |
| | QKᵀ | 16 | int8 全量程 | OK | FullyConnectedLayer | 跑通 | 1/256 差 1（tol=1）|
| | QKᵀ | 32 | int8 | OK | FullyConnectedLayer | 跑通 | **逐位 0/1024** |
| | P·V | 16 | int8 | OK | FullyConnectedLayer | 跑通 | **逐位 0/1024**（W=Vᵀ）|
| | P·V | 8 | int8 | **EXIT=1** | — | — | 见 §3.2 |
| | QKᵀ | 16 | fp16 | OK | FullyConnectedLayer | 跑通 | **逐位** |
| | P·V | 16 | fp16 | OK | FullyConnectedLayer | 跑通 | max_abs 2.44e-4 |
| | QKᵀ 2头（并集）| 16 | int8 | OK | FullyConnectedLayer（`[32,64]×[32,64]`）| 跑通 | **逐位 0/1024**；对角块 = 每头 scores 逐位 |
| **CONV_2D 1×1**(动态 filter) | QKᵀ | 16 | int8 | OK | Conv2DLayer | 跑通 | **逐位 0/256** |
| | QKᵀ | 16 | fp16 | OK | Conv2DLayer | 跑通 | **逐位** |
| | P·V | 16 | int8 | OK | Conv2DLayer | 跑通 | **逐位 0/1024**（filter=Vᵀ）|
| | P·V | 8 | int8 | **EXIT=1** | — | — | 见 §3.2 |
| | P·V | 16 | fp16 | OK | Conv2DLayer | 跑通 | max_abs 2.44e-4 |

29 个变体：**26 编译通过并上板跑通，3 个按预测编译失败**（T=8 int8 的 P·V，三种形式全失败）。
设备状态：跑前/跑后 `tools/healthcheck.sh --quick` 均 **exit 0**（无污染）。

### 3.2 精确失败行（3 个 T=8 int8 P·V，三种形式**同一条**报错）

```
$ ./tools/compile.sh ops/bmm_pv_t8_int8/model.tflite ops/bmm_pv_t8_int8/model.dla
...
   ├ BatchMatMulLayer<0>{BATCH_MATMUL}
   └ Output: OpResult (external)

Bridging is disallowed but gap is detected between Input and MDLA_5_0
ERROR: Fail to compile ops/bmm_pv_t8_int8/model.tflite
EXIT=1   SIZE=0
```

`ops/fc_bmm_pv_t8_int8/`、`ops/conv_bmm_pv_t8_int8/` 的日志**逐字相同**（只差模型路径）。
注意 OP-MATRIX §2.4 的现象再次出现：**失败的模型仍先打印了 MDLA 执行计划**，必须看 `EXIT=`。

**判读**：失败跟着 **P 这个输入张量的最内层行宽**（int8 T=8 → 8 B < 16 B）走，与 op 形式无关
→ 这是全局的 DRAM 边界规则，不是 `BATCH_MATMUL` 的限制。三条对照支持这个判读：

| 观察 | 说明 |
|---|---|
| QKᵀ T=8 int8 **通过**（输出行 8 B）| 8 B 行宽在**输出侧**没触发 —— 与 OP-MATRIX "输入侧最严"一致 |
| P·V T=8 **fp16 通过** | 8 个 fp16 = 16 B/行 → 规则按**字节**算，不是按元素个数 |
| P·V T=8 int8 三种形式全失败 | 换 op 形式救不了输入行宽 |

### 3.3 双重动态的"真"证据：换 B 再跑一次（L3 re-bind）

同一个 `.dla`，A 不变、只把第二个输入换成另一组随机 B，与**为这组新 B 独立算的 golden** 对拍
（`ops/<name>/compare_altb.txt`）：

| op | 结果 |
|---|---|
| `bmm_qk_t16_int8` | 逐位 0/256 |
| `fc_bmm_qk_t16_int8` | 逐位 0/256 |
| `conv_bmm_qk_t16_int8` | 逐位 0/256 |
| `bmm_pv_t16_int8` | 逐位 0/1024 |
| `fc_bmm_pv_t16_int8` | 逐位 0/1024 |
| `conv_bmm_pv_t16_int8` | 逐位 0/1024 |

即：**编译产物与权重内容无关**，一份组件可服务任意 K/V（不用每次重编）。这与 S4/E-005 对
FC 静态权重的结论一致，现在覆盖到 BMM/CONV 两种形式和 PV。

### 3.4 每种形式的 B 侧内存布局（**踩过的坑，必须记**）

同一个数学，三种形式的 B 侧期望布局不同：

| 形式 | QKᵀ 的 B | P·V 的 B |
|---|---|---|
| BATCH_MATMUL | `K:[1,T,64]`，靠 `adj_y=1` 转置 | `V:[1,T,64]`，`adj=0/0`，**无需转置** |
| FULLY_CONNECTED | `W=[N,K]=[T,64]`（就是 K）| `W=Vᵀ=[64,T]` |
| CONV_2D 1×1 | `filter=[OC,1,1,IC]=[T,1,1,64]`（就是 K）| `filter=[64,1,1,T]`**即 Vᵀ** |

**失败知识（R4）**：第一版把 `V:[16,64]` 原样当 CONV 的 filter 推下去 —— 元素个数一样（1024），
模型照跑不报错，但数值 **922/1024 错**（max_abs 12）。用设备 dump 反查才定位：

```
device == clip(A · Fwᵀ)  where Fw = pushed_bytes.reshape(64,16)   -> 逐位相等 0/1024
device == 用 Vᵀ 布局的结果                                        -> False
```

即设备做的是"把我们推的字节按 `[OC,IC]` 解释"，是我喂错了布局，不是编译器/驱动的问题。
教训：**B 侧布局错了不会报错，只会静默算错** —— 每个 PV 变体都必须有独立 golden 才能挡住。
修好后 `conv_bmm_pv_t16_int8` 从 922/1024 错 → **逐位 0/1024**。

### 3.5 批（2 头）能力

| 形式 | 2 头是否原生 | 证据 |
|---|---|---|
| BATCH_MATMUL | **是**（QKᵀ 单层；P·V 被拆成 2×(Slice+BMM)+Concat）| `ops/bmm_qk_b2h_*`、`ops/bmm_pv_b2h_t16_int8` 逐位 |
| FULLY_CONNECTED | **否**，只能用"并集矩阵"：输入/权重都堆成 `[2T,64]`→`[2T,2T]`，只有 2 个对角 `[T,T]` 块是每头 scores，其余是跨头乘积（丢弃）→ **MAC 浪费 2×** | `ops/fc_bmm_qk_b2h_t16_int8` 逐位 0/1024，且对角块单独验证逐位等于 `Q_h·K_hᵀ`（`compare.txt` 记录的是整块并集）|
| CONV_2D 1×1 | **否**，结构性不可能：1×1 conv 的 filter 对所有空间位置/所有 batch 项共享，一个 filter 只能配一份 K | 未单独建模型（若照并集办法堆成 `[2T,1,1,64]` 就是 T=32 的 QKᵀ 形状，没有新的编译信息）；标 **UNKNOWN/未实测** |

### 3.6 五个结论分列（R7）

1. **编译**：26/29 落在 MDLA_5_0 且无 bridge；3 个按预测因 8 B 输入行宽失败。
2. **ioctl/运行**：26 个全部实际跑起来（`neuronrt -m hw`，`-c 3`）。
3. **数值**：int8 small-codes 全部**逐位**；全量程档 1/256 与 6/1024 差 1 LSB（登记 tol=1，符合
   OP-MATRIX §3 的两次舍入）；fp16 在 QKᵀ 上逐位，在 P·V 上 max_abs 2.44e-4（远小于登记容差）。
4. **设备状态**：批跑前后 `healthcheck --quick` 均 exit 0；没有发现污染。
5. **性能**：**UNKNOWN**。`run.log` 里的 0.49–1.34 ms/inf、~500–950 FPS 受 3 次重复的提交开销支配，
   kernel 只有 16–64 行，不能当吞吐结论；要测性能需要 L=256/512 与多算子成图。

---

## 4. 明确回答：QKᵀ 和 P·V 能不能在 NPU 上做，且**两个操作数都动态**？

**能，今天就能，三种 op 形式都接受动态 B；推荐 `BATCH_MATMUL`。**

成立范围（本机实测，逐位或登记容差内）：

| 运算 | 形状 | 精度 | 结论 |
|---|---|---|---|
| `Q·Kᵀ` | `[1,T,64]×[1,T,64]→[1,T,T]`，T=8/16/32 | int8、fp16 | **可以**（BATCH_MATMUL 单层；T=8 int8 也行）|
| `Q·Kᵀ` 2 头 | `[1,2,T,64]²→[1,2,T,T]`，T=8/16 | int8、fp16 | **可以**（单层 BatchMatMulLayer）|
| `P·V` | `[1,T,T]×[1,T,64]→[1,T,64]`，T=16/32 | int8、fp16 | **可以** |
| `P·V` | T=8 | int8 | **不可以**（P 行 8 B；三种形式全失败，报错 §3.2）|
| `P·V` | T=8 | fp16 | **可以**（16 B/行）|
| `P·V` 2 头 | `[1,2,16,16]×[1,2,16,64]` | int8 | **可以**，但被拆成 2×(Slice+BMM)+Concat |

选型建议（工程口径）：

- **首选 `BATCH_MATMUL`**：两个 matmul 都可能免转置（QKᵀ 用 `adj_y=1`，PV 用 `adj=0/0`），2 头原生，
  一份 `.dla` 服务任意权重。
- `FULLY_CONNECTED(dyn W)` 是等价的退路，但 P·V 需要调用方给 **Vᵀ**（多一次转置/一份换布局），
  且 2 头只能并集矩阵（2× MAC）。
- `CONV_2D(1×1)` 也能用动态 filter，但它对两个 matmul 都要求 B 以 `[OC,1,1,IC]` 布局喂入
  （PV 同样是 Vᵀ），批只能用并集，**没有优势**，不推荐。

---

## 5. 适用范围 / 剩余疑问（UNKNOWN）

已实测覆盖：T ∈ {8,16,32}，D=64，2 头，int8(small/full codes) 与 fp16，小整数/窄范围输入，
零 bias，单算子图。**超出这个盒子的一切都还没证据**：

1. **因果掩码**：`softmax(QKᵀ/√d + mask)` 的 `-inf` 掩码没有测（CAPABILITY §5 也列为未覆盖）；
   掩码要求 SOFTMAX 的输入含 −inf，属于下一个实验。
2. **更大的 T**：T=64/128/256/512 未测；T 不是 16 的倍数（如 T=24）在 int8 PV 上按行宽规则应失败，
   未实测。T=8 int8 PV 的绕法：改用 fp16（已证）或把 P 补到 16 列（未证）。
3. **D ≠ 64**（如 32/128）未测；K 维（D）不是 16 的倍数时会被行宽规则咬到，未实测。
4. **2 头以上**（H=4/8）与 `[1,H,L,L]` 4-D 的分解代价未测；目前只知道 H=2 的 PV 会拆成 2×(Slice+BMM)。
5. **int8 输出行宽 8 B 在输出侧为何能过**：只是观察到 QKᵀ T=8 能编过；是"输出侧无此要求"还是
   "内部 padding 后拷回"，机制 **UNKNOWN**（需要拆 `.dla` 或扫更大输出才能区分）。
6. **per-channel 量化**、非零/非对称 zero-point 未测（本实验全部 zero_point=0，out_scale 使乘子为 1 或 1/32）。
7. **成图**：QKᵀ→SOFTMAX→PV 的多算子图、以及 softmax 的 tanh 近似误差如何影响 PV 未测（单算子探针到此为止）。
8. **性能**：见 §3.6 第 5 条，UNKNOWN。
9. `BATCH_MATMUL` 的 `adj_x=1`（转置 A）未测；本实验只用 `adj_y`。

## 6. 需要 Lead 处理

- `tools/rebuild_ops.sh` 的 `FILE_BUILDERS` 是硬编码列表，加一个词 `build_bmm` 就能让这 29 个 op
  在产物缺失时被自动重建（`build_bmm.py` 已按 `--list` + `--out <dir>/model.tflite` 约定实现）。
  我按约定不改公共工具。
- `tools/e2e.sh` 在 `model.tflite` 缺失时只调用 `build_ops.py`，其它 builder 的 op（含本次 29 个）
  在干净 checkout 里需要先跑 `rebuild_ops.sh`。这是既有行为，不是本次引入的。
