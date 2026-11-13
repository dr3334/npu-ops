# OP-q06 —— prefill/decode 双图 + KV cache，共用一份外部权重 bin

> 票：.scratch/quant-runtime/issues/06-prefill-decode-kv.md；spec .scratch/quant-runtime/spec.md
> 前驱：docs/OP-24layer-shared.md（票 05：24 层单图 + 外部共享权重，prefill）
> ADR：docs/adr/0006-execution-form-shared-weights.md（含 q05 的四条实测补充）、
>      docs/adr/0005-quantization-tier-target.md 第六次订正（实现门 + 按档质量期望）
> 分支 impl/q06-decode-kv，worktree wt-q06；ncc-tflite 7.3.15
> （--arch=mdla5.0 --opt=3 --disallow-bridge --num-mdla=2 --l1-size-kb=3072）；
> 设备 mt6897 / Android 16（0123456789ABCDEF），板端全部持 /d/deepseek_ workspace/mtk/.npu-board.lock，
> 每个板端块前后各一次 healthcheck --quick（**全部 PASS**）。
> 权重夹具 npu-ops/weights/qwen25_frozen（只读；**24 层是同一套层 0 权重重复 24 次**，
> 即权重复用，不是真实 24 层的 Qwen2.5-0.5B）。
> 公共工具（e2e/compare/compile/devlock/healthcheck/build_ops/quant_contract/build_w4a8/build_24layer）**一行未改**；
> 本票新增 tools/build_q06.py、tools/e2e_q06.sh、tools/mem_q06.sh、tools/mem_q06_device.sh。

---

## 0. TL;DR

**prefill 图与 decode 图共用一份 bin 成立；decode 跨 8 个 token 的 KV cache 在板上逐 token 对拍通过实现门。**

| 量 | 结果 |
|---|---|
| prefill 图（T=16，24 层） | 1346 个算子 / **1 执行步** / EXIT=0 / .dla 18,079,138 B |
| decode 图（T=8 查询行 over C=2048，24 层） | 1396 个算子 / **1 执行步** / EXIT=0 / .dla 18,141,590 B |
| 共用 bin | **16,674,816 B**，381 个静态张量；prefill nonreuse **0**，decode nonreuse **2 个 / 8,224 B**（0.05%） |
| 抽取后图 | prefill 1,491,484 B、decode 1,554,144 B |
| decode 的 T | **T=1 编不过**（prefill 图报 SOFTMAX 的 Act1 shift；decode 图报 Input↔MDLA gap）→ **pad 到 T=8，只用第 0 行** |
| KV 形态 | K/V 各一张 [48, C, 64] int16 **打包张量**（48 = 24 层 × 2 KV head）；每 token 图**只输出新行** [48, T, 64]，**host 写回缓存并整块回传** |
| KV 大小 | K 12,582,912 B + V 12,582,912 B = **25,165,824 B（24.0 MiB）**；票面的 12.6 MB 是**只有 K** |
| 每 token 主机拷贝 | **25.2 MB 上行 + 0.2 MB 下行**（adb 实测 0.73–0.97 s / 0.26–0.35 s，CLI 形态） |
| 正确性（8 token 逐步） | ｜Δcos(dev, 量化 golden)｜ ≤ **8.2e-5**（门 0.003）、码级 max **370**（门 1200）、**噪声底 0 字节** |
| 质量（vs 独立 fp64 链） | hidden cos **0.9727–0.9985**、logits cos **0.9727–0.9985**、argmax 命中 **6/8** |
| 延迟（device，含常驻权重） | decode **68.5 ms/token**（C=2048）/ **35.1 ms/token**（C=64）→ **KV 宽度占 33.4 ms** |
| 常驻 | PSS **46.4 MB**（Anon 44.2）、Rss 50.3 MB、200 次 Run 的 rchar **冻结在 42.59 MB**、read_bytes **0** |

四条要点：

1. **mask / RoPE 提成输入之后，bin 与 T 解耦**：prefill(T=16)、decode(T=8, C=2048)、decode(T=8, C=64)
   三张图抽出的是**同一份权重**的 bin（体积都是 16,674,816 B）；第二张图 428/430 个静态张量按字节命中 reuse。
   这就是票 05 §4.3 那条建议的落地验证。
2. **T=1 在本栈编不过**（不是 16 B 行规则，是 SOFTMAX 的后端限制）。按要求 pad 到 T=8 只用第 0 行，
   代价是**每个 token 多算 7 行**（8× 的 query 行工作量），已计入 §6 的延迟。
3. **TFLite 是函数式，KV 写入只能是输出**：图每 token 只吐**新的一行**（48×T×64 int16），
   host 把它写进 [48, C, 64] 的第 m 行（O(1)，256 B/层），但**下一次 Run 必须把这 25.2 MB 整块再喂回去**。
   这就是 per-token 拷贝成本的来源，也是与厂商「KV 常驻设备、in-place 更新」的**结构性差别**。
4. **实现门全过、质量门按 logits 记录**：8 步 ｜Δcos｜ ≤ 8.2e-5、码级 ≤ 370、噪声 0；
   质量侧 logits cos 0.973–0.999、token 命中 6/8。后者是**夹具 + 24 层同权重**的数字，
   不能读成 Qwen2.5-0.5B 的端到端质量。

---

## 1. 票面要回答什么 / 做法

票面四件事：**①两张图共用一份 bin**、**②decode 图（T=1，必要时 pad）**、
**③KV cache 的图上进出形态 + per-token 拷贝成本**、**④连续 N≥8 个 token 逐步对拍 + 延迟分解**。

档位沿用票 05 锁死的 **8W16A + native 缝**（builder 里 set_tier(8,16) + SEAM="native"，不接受别的组合）。

### 1.1 相对票 05 图结构的四处改动（这是本票的全部新东西）

| 改动 | 为什么 |
|---|---|
| **causal mask 变成图输入** [H, T, L] int16（scale 0.1250038，码 -32767 = -4096） | T² 常量不再进 Static；bin 与 T 解耦（票 05 §4.3） |
| **RoPE cos/sin 变成图输入**（4 张：q 14 头、k 2 头各 cos/sin，int16，scale 1/32767） | 同上；decode 每步只喂**一行**表 |
| **KV 变成图输入/输出**（K、V 各一张 [48, C, 64] 进、[48, T, 64] 出） | 函数式图里唯一可行的 cache 形态 |
| **所有张量尺度来自一张冻结表** | 两张图的权重/偏置/gamma 常量因此**逐字节相同** → extract-shared 全部命中 reuse；也让「每步的量化 golden」与「设备图」用同一套尺度 |

「冻结尺度表」是这一步的关键工程结构：tools/build_q06.py 先用**prefill 批**标定一次（--calib），
把每个张量的输出尺度写进 ops/q06_tab.json（769 个）；此后**建图**与**每一步 golden 重建**
都从这张表取尺度，只有数据变。这样 golden 与设备图不会因为「两次不同的 pick_scale」而漂移，
device == 量化 golden 就真的只说明设备。

校准/评估分离（NOTES 11.2）：表来自 token id 0..15 的 prefill 批；报告里每个数字都是 position 16..23
的 decode 步（另一条激活路径）。

### 1.2 图里没有的部分（说明清楚，避免误读）

- **24 层是同一套夹具权重重复 24 次**：ncc 会把字节相同的静态张量合并，所以 Static 15M / bin 16.67 MB
  是**权重复用**的产物。真实 24 层不同权重按票 05 的 2 层控制实验外推 **≈366 MB**，
  内存预算必须按 366 MB 算，不能看 Static。
- **lm_head 不在图上**：Qwen2.5-0.5B 的 lm_head 与 embed.weight 绑定（151936×896）。本票的 logits 用
  host 侧 fp64 从 device hidden 算（h @ embed.T），只用来算 logits cosine 与 argmax 命中。
  图里没有 lm_head、没有词嵌入（与票 05 同）。

---

## 2. 双图共用一份 bin（票面第 1 条）

### 2.1 编译

| 图 | 算子 | 执行步 | bridge | Input/Output/Temp | Static | Code | DRAM | .dla |
|---|---|---|---|---|---|---|---|---|
| ops/q06_prefill_t16_24l | 1346 | **1** | 无 | 67K / 220K / 0 | 15M | 829K | 16M | 18,079,138 |
| ops/q06_decode_t8_c2048_24l | 1396 | **1** | 无 | 24M / 110K / 32M | 15M | 858K | 73M | 18,141,590 |

- 两张图都是 Graph[0] has 1 Execution Step(s)、TargetExecutionOrder 只有 MDLA_5_0、无 bridge 字样。
- decode 的 **Input 24M** 就是 KV cache（12.0 MiB × 2）——它现在是**图输入**，不再是静态数据。
- decode 的 **Temp 32M** 是 [14, 8, 2056] 的 scores/probs 与 head-repeat 的中间张量。
- tflite 本体 367 MB / 389 MB：24 份字节相同的权重逐份写进 flatbuffer，去重发生在 ncc 侧（票 05 §3.2）。

### 2.2 extract-shared（prefill 第一张，-e diff 从第一张抽）

    extract-shared -i ops/q06_prefill_t16_24l/model.dla,ops/q06_decode_t8_c2048_24l/model.dla \
                   -o ops/q06_prefill_t16_24l/model_shared.dla,ops/q06_decode_t8_c2048_24l/model_shared.dla \
                   -s ops/q06_shared.bin

| 项 | 数 |
|---|---|
| 导出的静态数据 | **381 个张量 / 16,674,816 B** |
| prefill 图 reuse / nonreuse | 430 / **0**（16,557,056 B / 0 B） |
| decode 图 reuse / nonreuse | 428 / **2 个（8,224 B）**（16,556,992 B / 8,224 B） |
| 抽取后图 | prefill 1,491,484 B、decode 1,554,144 B |

**这是本票最重要的一条**：decode 图 99.95% 的静态数据按字节命中了 prefill 图抽出的那份 bin。
只有 2 个 4,112 B（= 2,056 个 int16 = C+T）的张量没有命中——它们的形状随 cache 长度变，
内容也因此不同；它们留在 decode 自己的 .dla 里（8 KB），不影响「权重共用一份」这个结论。

对照票 05：T=8 与 T=16 两张 **prefill** 图配对时，第二张图有 1,637,152 B 的 nonreuse
（bias / RoPE 表 / mask 常量）；本票把 mask 与 RoPE 提成输入之后，这部分**降到 8,224 B**。
又：把 prefill(T=16)、decode(C=2048)、decode(C=64) 三张图两两配对，抽出的 bin 都是 **16,674,816 B**，
说明 bin 只装权重、与 T/缓存长度无关（三份 bin 的 sha256 不同，是因为第一张图不同 → 那 2 个
随 C 变的张量不同；见 §9 UNKNOWN 3）。

### 2.3 完整性三联（票 03 警告：设备**不校验** bin 内容）

| 物 | sha256 |
|---|---|
| bin（prefill+decode 配对） | 22d55a330a34d77920950d0b0b09cc8c981035d65c5bd3b44f3e7fccc654c6af |
| 图 ops/q06_prefill_t16_24l/model_shared.dla | 0d9f48de6dd7d97f4a204c3f31cfdd0d6e9abe16405dece76fffe2fb32921ac2 |
| 图 ops/q06_decode_t8_c2048_24l/model_shared.dla | 96802f01fe735f97a5097c3b8407533ffaf2cf976a87d7b76a2136d02db273ff |
| 抽取前的 .dla（prefill / decode） | 3392f0a0399b6623ad8a99f393aebc9e6a6b3d3ba087d504562feeba8a9d1050 / 951bed7ea19d209a805f9de1cb20cf65395973ee132d9abd6dfe482d89fda4cc |
| 输出（decode 最后一步 out_hidden，三次独立 Run） | aeab7397856600f0a5cbd76c0add20f0d95342df1802611881564854bc158d82（三次相同，ops/q06_seq/noise.txt） |

---

## 3. decode 图：T=1 编不过，pad 到 T=8（票面第 2 条）

### 3.1 二分记录

| 变体 | 结果 | ncc 的原话 |
|---|---|---|
| prefill T=1，1 层，无 cache | **EXIT=1** | OP[37]: SOFTMAX / MDLA: Act1 shift exceeds range of MDLA command field / EDPA: unsupported operation → Cannot find an execution plan because of unsupported operations |
| decode T=1，1 层，C=64 / 256 / 2048 | **EXIT=1** | Bridging is disallowed but gap is detected between Input and MDLA_5_0 |
| decode **T=8**，1 层，C=64 / 512 / 2048 | EXIT=0 | —（packed 与 per-layer 两种 KV 输入形态都 EXIT=0） |
| decode T=8，**24 层**，C=2048 | EXIT=0 | 1 执行步 |

结论：**T=1 在本栈是后端期失败**，不是票 04 那条 16 B 行规则（那一条在 int16 下已经不存在）。
最可能的原因是 SOFTMAX 在「一行只有 1 个元素」时动态范围退化，Act1 的 shift 超出 MDLA 命令字段；
decode 图的报错点不同（Input↔MDLA gap），但两者都只在 T=1 出现，改成 T=8 后都消失。
按票面允许的做法：**decode 图用 T=8 查询行，只用第 0 行**。

### 3.2 padding 的取舍与浪费

- **语义**：8 行查询全部喂同一个 token 的 embedding；第 r 行的 mask 在 cache 之外再放行 C+r 这一个槽位，
  所以第 0 行看到的是「cache 全部有效位 + 自己」，行 1..7 同理（值不同但都是被忽略的行）。
- **浪费**：只有第 0 行是结果，**7/8 的 query 行计算是浪费的**（attention 的 QK/PV、softmax、
  o_proj 与整条 MLP 都按 8 行算）。§6 的 35.1 ms（C=64）就是「带 8 行 padding」的价格；
  若 T=1 能编，量级上应接近 1/8 的 query 行工作量，但**没有测到**（编不过，无从测）。
- **余量**：把 T 从 1 改成 8 不改变 KV 形状、不改 bin、不改 mask 语义，只多 8×T×64 的输出行
  （49,152 B/张量），所以这是一个**只多算、不多存**的折中。

---

## 4. KV cache 的形态与 per-token 拷贝成本（票面第 3 条）

### 4.1 形态：两张打包张量 + host 持有缓存

    输入  k_in [48, C, 64] int16（scale S_K = 2.374719e-3）   v_in [48, C, 64]（S_V = 4.481891e-6）
    输出  k_out [48, T, 64] int16                             v_out [48, T, 64]
          48 = 24 层 × 2 个 KV head；进图后先 SPLIT 成 24 份 [2, C, 64]

- **为什么打包成一张**：per-layer 分开要给 decode 图 48 个输入/48 个输出（neuronrt 支持，但命令行与脚本会很啰嗦）。
  打包形态实测 EXIT=0；per-layer 形态也 EXIT=0（--kv-split perlayer），两条路都留在 builder 里。
- **为什么 K 与 V 分开**：两者动态范围差 **530 倍**（夹具 |k|max 77.8 vs |v|max 0.147）。合成一张张量就得共用
  一个 scale，V 会只剩 6 bit。分开之后 K、V 各用自己的 scale（校准期从 prefill 的 |k|max / |v|max 定）。
- **谁写缓存**：**host**。图每 token 只吐**新的一行**；host 把它写进 [48, C, 64] 的第 m 个槽位（O(1)，256 B/层），
  下一次 Run 再把整块 25.2 MB 喂回去。TFLite 没有 in-place，这是绕不开的。
- **mask 的语义**：槽位 [0, m) 是真实历史、槽位 C+r 是第 r 行的当前 token、其余 -4096。
  m 到 C 之后窗口滚动（本票 8 个 token 没触发；代码里有，见 §9 UNKNOWN 5）。

### 4.2 体积（订正票面的 12.6 MB）

| 项 | 计算 | 字节 |
|---|---|---|
| K cache | 24 层 × 2 KV head × 64 × 2048 × 2 B | **12,582,912（12.0 MiB）** |
| V cache | 同上 | **12,582,912** |
| 合计 | | **25,165,824（24.0 MiB）** |
| 每 token 图输出 | 2 × 48 × 8 × 64 × 2 | 98,304 B |

票面的「24 层 × 2 KV head × 64 × cache 2048 × int16 ≈ 12.6 MB」算的是**只有 K**；真实缓存是它的 2 倍。

### 4.3 per-token 拷贝成本（实测）

| 阶段 | 实测（8 步，每步一次） | 说明 |
|---|---|---|
| host→device：8 个输入文件（含 25.2 MB 缓存） | **725–970 ms** | adb push，约 27 MB/s |
| device→host：3 个输出（0.2 MB） | **259–350 ms** | adb pull |
| 一次 Run（device 侧时钟，单次） | **253–276 ms** | 含进程启动 + 图/bin 装载 |
| 一次 Run（-c 20 摊薄） | **68.5 ms** | 见 §6 |

**结构性结论**：这一版把 KV 放在 host 上，所以「每 token 搬 25 MB」是**形态的代价，不是实现的 bug**。
在同一个进程里（-c 200）这 25.2 MB 只被读**一次**：rchar 冻结在 42.59 MB、read_bytes = 0（§7）。
换句话说：**用 RuntimeAPI 把缓存指针常驻，per-token 拷贝可以从 25 MB 降到 O(256 B)**；
本票用 CLI 形态，量化的是 CLI 的代价。这一条是决定性的：厂商 decode 图 21.6 MB、
KV 常驻设备、无拷贝，我们这版差的就是这一块。

---

## 5. 正确性：连续 8 个 token 逐步对拍（票面第 4 条）

prompt = token_ids(16)（夹具 embed 行），接着 8 个 token = token_ids(24)[16:24]（teacher forcing，
不是采样，因为采样还要依赖 lm_head 的精度，见 §5.3）。

三列参考：

- **量化 golden**：把同一份算子序列用**同一张尺度表**在 numpy 里重放（build_q06.py --seq-absorb），
  与设备图逐算子同构 → 实现门用它。
- **独立 fp64 链**：从模型定义重写（ref_layer / ref_chain），KV 缓存与 RoPE 位置都在 fp64 里独立维护，
  只用来回答质量。它和票 05 的 build_24layer.fp64_chain 在 24 层 prefill 上 **cos = 1.00000000、
  最大绝对差 1.8e-14**（交叉验证：这条参考不是自说自话）。
- **logits**：h @ embed.T（fp64，host 侧），三种 hidden 各算一次。

### 5.1 prefill（T=16，一次）

| 量 | 值 |
|---|---|
| device vs 量化 golden | 码级 max **411** / mean 38.8 / 逐位 171/14336；cos **0.9999776** ⇒ **｜Δcos｜ = 2.24e-5** |
| device vs fp64 | cos **0.996148**（票 05 同档 0.9961469） |
| logits（device vs fp64） | cos **0.99548** |

### 5.2 decode：8 步逐步（每一步都记，不只比最后一步）

| step | pos | 码级 max | 码级 mean | ｜Δcos(dev,gold)｜ | cos(dev,gold) | cos(dev,fp64) | logits cos | cos(gold,fp64) | argmax 命中 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 370 | 84.56 | 8.13e-5 | 0.9999187 | 0.9962753 | 0.9962701 | 0.9963359 | 0 |
| 1 | 17 | 135 | 27.76 | 8.81e-6 | 0.9999912 | 0.9925773 | 0.9926247 | 0.9925927 | 1 |
| 2 | 18 | 143 | 36.90 | 1.34e-5 | 0.9999866 | 0.9930829 | 0.9931482 | 0.9932280 | 1 |
| 3 | 19 | 98 | 21.68 | 5.19e-6 | 0.9999948 | 0.9949258 | 0.9949061 | 0.9948966 | 1 |
| 4 | 20 | 144 | 34.31 | 1.45e-5 | 0.9999855 | 0.9945539 | 0.9945727 | 0.9946039 | 1 |
| 5 | 21 | 67 | 17.93 | 3.13e-6 | 0.9999969 | 0.9984568 | 0.9984640 | 0.9984461 | 1 |
| 6 | 22 | 152 | 36.43 | 1.47e-5 | 0.9999853 | 0.9727470 | 0.9727467 | 0.9729822 | 0 |
| 7 | 23 | 249 | 59.76 | 3.77e-5 | 0.9999623 | 0.9967719 | 0.9967722 | 0.9967921 | 1 |

- **每一步都与量化 golden 对齐**：｜Δcos｜ 最大 **8.2e-5**（门 0.003，余量 ~37×），码级最大 **370**（门 1200）。
  码级 mean 18–85，与票 05 单张 prefill 图的 32.5 同量级 ⇒ **8 步的缓存链没有把设备舍入放大**。
- **噪声底 = 0 字节**：同一组输入独立跑 3 次，out_hidden 的 sha256 相同（ops/q06_seq/noise.txt）。
- **注意**：device vs fp64 与 golden vs fp64 几乎逐位相同（step 6：0.9727470 vs 0.9729822），
  说明设备是**忠实**的，剩下的差距全来自量化本身（与 ADR 0005 的结论一致）。

### 5.3 判据对照（ADR 0005 第六次订正）

| 门 | 登记值 | 实测 | 判定 |
|---|---|---|---|
| 实现门：｜Δcos(device, 量化 golden)｜ | ≤ 0.003 | **≤ 8.2e-5**（8 步 + prefill 2.2e-5） | **PASS** |
| 实现门：码级 max | ≤ 1200（= √24 × 票 04f 单层 247） | **≤ 411**（prefill + 8 步） | **PASS** |
| 实现门：rel_l2(device, golden) | ≤ 0.05 | prefill **6.69e-3**（第二轮运行的 prefill 实测）；decode 各步未取到（见 §5.4） | **prefill PASS / decode 未测** |
| 质量期望（按档，int8 权重 ≥0.9998） | — | 整链 cos 0.973–0.998 **达不到 0.9998** | **FAIL（照实报）** |
| 最终质量门（整网 logits） | 无硬阈值，按 spec 记录 | logits cos **0.9727–0.9985**、argmax **6/8** | 记录 |

### 5.4 两件事要说清

1. **prefill 做了两次独立运行**：第一次 168 ms / 第二次 149 ms，golden 侧的码级与 cosine **逐位相同**
   （codes max 411 / mean 38.80 / cos 0.9999776），只有时间不同 ⇒ 设备侧结果时间无关。
2. **decode 各步的 rel_l2 没取到**：为了把 rel_l2 也记进 seq.jsonl，跑了第二次完整序列，
   但**设备在 prefill 之后掉线**（adb: no devices/emulators found，重启 adb server 与 reconnect 都没有恢复），
   所以只有 prefill 那一格的 rel_l2 = 6.69e-3。model.json 里登记的 rel_l2 门（0.05）**没有改动**；
   decode 侧的实现门由 ｜Δcos｜（≤8.2e-5）与码级 max（≤370）两个口径支撑。这条记在这里，不圆。

**这一档能说明什么 / 不能说明什么**：

- 能说明：**图搭对了、设备忠实、KV 链没坏**（实现门有 37× 余量 + 噪声 0 + 逐步不掉）。
- 能说明：**同一座 24 层塔在 decode 形态下的保真度与 prefill 形态同档**
  （prefill 0.996148 vs decode 0.9727–0.9985，同一量级）。
- **不能**说明 Qwen2.5-0.5B 的端到端质量。三个理由：(a) 24 层是**同一套层 0 权重**，
  误差不按真实层间差异累积；(b) 夹具的分数分布比真权重更极端（NOTES 11.3 的反向结论：
  夹具不是悲观估计，它是乐观的那一个）；(c) logits 是 host 侧 fp64 算的、图里没有 lm_head。
  argmax 6/8 也不能读成 75% 命中率——8 个样本、teacher forcing、且 step 0 与 6 的 top-1 只差一个近邻。

---

## 6. 延迟分解（票面第 5 条）

方法：device 侧 date +%s%N 包住一次 neuronrt 调用；-c 1 与 -c 20 两次，差值给出摊薄后的单次成本。
**neuronrt 7.3.15 对这两张图不打印任何东西**（rc=0、输出文件正常），所以本票的成败判据是
**退出码 + 输出文件大小**，不是 grep "Inference"（票 05 的图会打印，这里不会）。

| 量 | C=2048 | C=64 | 读法 |
|---|---|---|---|
| 一次 Run（-c 1，含装载） | **253–276 ms** | 155 ms | 装载（图 1.55 MB + bin 16.7 MB + 25.2 MB 输入）占大头 |
| 摊薄单次（-c 20 总时 / 20） | **68.5 ms** | **35.1 ms** | 这才是「权重常驻」后的 per-token |
| prefill 一次（T=16） | 168 ms（-c 1）/ 41.4 ms（-c 20） | — | 票 05 同一座塔（无 KV 输出）是 22.8 ms |

**per-token 分解（C=2048，68.5 ms）**：

| 成分 | 量 | 依据 |
|---|---|---|
| 权重带宽 | **≈0**（常驻） | bin 16.67 MB 是 dma-buf；200 次 Run 期间 rchar 冻结、read_bytes=0（§7） |
| KV 读 + attention 宽度 | **≈33.4 ms** | C=2048 与 C=64 的差（68.5 − 35.1）；含 2049 宽的 scores/softmax 与 head-repeat |
| 塔本体（8 行 padding + 24 层） | **≈35.1 ms** | C=64 的摊薄值；其中 7/8 的 query 行是 padding 浪费 |
| host 拷贝 | **25.2 MB 上行 + 0.2 MB 下行** | 实测 0.73–0.97 s / 0.26–0.35 s（CLI 形态） |

三条读法：

1. **对照票 05 的 prefill 17–23 ms**：本票 prefill 是 41.4 ms（摊薄）/168 ms（单次），比票 05 慢 1.8×。
   差别只在**多出两张 KV 输出**（每层一次 TRANSPOSE + 24 段 CONCAT，共 98 KB 输出）与 mask/RoPE 变成输入；
   层数、档位、bin 都没变。⇒ **KV 导出不是免费的**，这是把 prefill 变成「能给 decode 灌缓存」的代价。
2. **decode 比 prefill 贵 3 倍**（68.5 vs 41.4 摊薄，但 decode 只产 1 个 token，prefill 产 16 个）：
   根因是 **2049 宽的 attention**（每个 head 每层都要过 2049 个槽位的 scores/softmax/PV），
   而 prefill 的 T=16 只有 16 宽。厂商 decode 图 21.6 MB、1t2048c，量级上与我们同构，但他们 KV 在设备上、
   没有 25 MB 的搬运。
3. **不要把 68.5 ms 当性能结论**：没有时钟/热态控制（同 04c/04e/04f），同一天不同会话能差 1.3 ms；
   而且 8 行 padding 与 2049 宽都是「为了证明形态能跑」的选择，不是调优后的配置。

---

## 7. 常驻内存（decode 图，一次装载 200 次 Run）

方法同票 05 / docs/OP-vendor-runtime.md §5–6：tools/mem_q06.sh 200（device 侧 tools/mem_q06_device.sh），
每 0.5 s 采 /proc/pid/smaps_rollup + /proc/pid/maps 的 dmabuf 行 + /proc/pid/io。

| 量 | decode（C=2048） | 票 05 prefill（对照） |
|---|---|---|
| PSS 稳态 | **46.4 MB**（Anon **44.2** / File 2.13 / Shmem 0.11） | 21.6 MB（Anon 19.3） |
| Rss | 50.3 MB | 25.4 MB |
| rchar（装载后） | **42,594,329 B 冻结** | 17.2 MB 冻结 |
| read_bytes | **0** | 0 |
| dma-buf 里能认出的块 | bin **16,674,816 B**（16.10 MiB 页对齐）、K cache **12,582,912 B**、V cache 12,517,376 B、workspace 33,951,744 B | bin 16.10 MiB + 1.57 MiB 未共享静态 |

读法：

1. **权重常驻得到确认**：200 次 Run 期间 rchar 一个字节不涨、read_bytes 恒 0，说明 bin 与输入缓冲只在装载时读一次。
2. **KV 缓存落在 dma-buf（APUBUF0）**：K、V 各一块 ~12 MB，NPU 直接读；但**写**只能由 host 做（§4）。
   进程的 Anon 44.2 MB ≈ 25.2 MB 输入缓冲 + 工作区，与票 05 的 19.3 MB Anon 之差正好是缓存量级。
3. **不要横向比绝对值**：本票的图没有 lm_head、没有词嵌入，厂商空载 PSS 346–355 MB 里含 260 MB 的
   词嵌入堆拷贝；可比的是**形态**（外部权重走 dma-buf、PSS 不含权重）。

---

## 8. 与厂商形态、与票 05 的差距

| 项 | 厂商 | 票 05（prefill） | 本票（prefill + decode） |
|---|---|---|---|
| 图 | prefill 3.24 MB + decode 21.6 MB | 1.49 MB（T=16） | 1.49 MB + 1.55 MB |
| 权重 | 一份 250.7 MB bin（4-bit） | 一份 16.88 MB bin（int8） | **同左，且两张图共用** |
| KV | 常驻设备、in-place 更新 | 无 | **host 持有，每 token 整块回传 25.2 MB** |
| 每 token 主机搬运 | ≈0（只有输入 token） | — | **25.2 MB 上行** |
| ms/token | 0.68（prefill，128 token / 87 ms） | 1.43–1.51（T=16） | prefill 2.59（41.4/16）、**decode 68.5** |

**结论**：形态（两张图 + 一份外部 bin + mask/RoPE 作输入）与厂商**同构**；
差的是 **KV 的归属**（TFLite 函数式 vs 厂商的自定义 in-place KV），以及由此带来的 25 MB/token 搬运。
这一条是**下一轮该动的第一块砖**（运行时 API 常驻缓存），不是靠再调图能解决的。

---

## 9. UNKNOWN（不圆）

1. **T=1 到底为什么编不过**：只做了 T=1 / T=8 的对照与两条不同的报错，没有 bisect 到具体算子
   （报错点名 SOFTMAX 的 Act1 shift，但 decode 图给的是 Input↔MDLA gap，两者未必同因）。
   若下一轮要真 T=1，建议先做只含 softmax、行宽 1 的最小图复现。
2. **decode 图那 2 个 4,112 B 的 nonreuse 张量是什么**：只从形状反推是「随 C+T 变」的常量
   （4,112 B = 2,056 个 int16），没有从工具侧确认。
3. **三份 bin 的 sha256 不同**（prefill-first / decode2048-first / decode64-first）：体积都是 16,674,816 B，
   差异只可能来自那 2 个随 C 变的张量；没有逐字节比对。
4. **8 行 padding 的真实成本**：只有「T=8 能编、T=1 不能编」这个二值事实，没有「若能编，T=1 会快多少」的实测。
5. **窗口滚动没跑过**：8 个 token 没触到 m == C（2048），滚动分支只有代码、没有板端证据。
6. **per-layer KV scale 的收益没量**：K/V 各用一个全局 scale（跨 24 层），夹具下层间幅度差 1.0–1.3 倍，
   所以本票没吃亏；**真权重下层间差异会大得多**，那时 per-layer（或 per-head）scale 才有意义。
7. **logits 的 6/8 命中率**：8 个样本、teacher forcing、host 侧 fp64 lm_head，不能当端到端命中率。
8. **decode 各步的 rel_l2**：第二次运行在 prefill 之后设备掉线，只留下 prefill 那一格（6.69e-3）。
   设备恢复后重跑一次 `./tools/e2e_q06.sh --steps 8` 即可补齐（seq.jsonl 每步都带 rel_l2 字段）。

---

## 10. 复现

    export QWEN25_WEIGHTS_DIR="/mnt/d/deepseek_ workspace/mtk/npu-ops/weights/qwen25_frozen"
    export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"

    # 1) 冻结尺度表（一次；prefill 批标定，769 个尺度）
    /usr/bin/python3 tools/build_q06.py --calib --layers 24 --t 16 --tab ops/q06_tab.json

    # 2) 两张图（编译放锁外；各 2–3 分钟）
    /usr/bin/python3 tools/build_q06.py --mode prefill --t 16 --cache 0    --layers 24 --tab ops/q06_tab.json --out ops/q06_prefill_t16_24l
    /usr/bin/python3 tools/build_q06.py --mode decode  --t 8  --cache 2048 --layers 24 --tab ops/q06_tab.json --out ops/q06_decode_t8_c2048_24l
    ./tools/compile.sh ops/q06_prefill_t16_24l/model.tflite      ops/q06_prefill_t16_24l/model.dla
    ./tools/compile.sh ops/q06_decode_t8_c2048_24l/model.tflite  ops/q06_decode_t8_c2048_24l/model.dla

    # 3) 抽共享权重（一张 bin 给两张图）
    extract-shared -i ops/q06_prefill_t16_24l/model.dla,ops/q06_decode_t8_c2048_24l/model.dla \
                   -o ops/q06_prefill_t16_24l/model_shared.dla,ops/q06_decode_t8_c2048_24l/model_shared.dla \
                   -s ops/q06_shared.bin

    # 4) 无板自检（先跑这个：票 09 的教训——device-free 的逐步重放能抓到整类接错）
    /usr/bin/python3 tools/build_q06.py --dry --seq-init --steps 8 --tab ops/q06_tab.json \
      --op-prefill ops/q06_prefill_t16_24l --op-decode ops/q06_decode_t8_c2048_24l --seqdir ops/q06_dry
    for s in 0 1 2 3 4 5 6 7; do
      /usr/bin/python3 tools/build_q06.py --dry --seq-absorb $s --tab ops/q06_tab.json --seqdir ops/q06_dry
    done

    # 5) 板端（一个锁会话：prefill 一次 + 8 个 decode，逐步对拍）
    ./tools/devlock.sh run -- ./tools/healthcheck.sh --quick
    ./tools/devlock.sh run -- ./tools/e2e_q06.sh --steps 8
    ./tools/devlock.sh run -- ./tools/mem_q06.sh 200
    ./tools/devlock.sh run -- ./tools/healthcheck.sh --quick

可复现性：同参数重建 model.tflite / golden/*.bin 逐字节稳定；.dla 不可复现（仓库既有约定），
所以绑定靠 **bin / dla / 输出 三个 sha256**（§2.3）。

---

## 11. 与票面 checklist 对照

| 票面要求 | 状态 | 证据 |
|---|---|---|
| prefill 图与 decode 图共用一份 bin | ✅ | §2：bin 16,674,816 B；prefill nonreuse 0、decode nonreuse 2 个/8,224 B |
| decode 图 T=1（挡住则 pad 到 T=8/16 并记录） | ✅（**T=1 挡住，pad T=8**） | §3：两种报错原文 + 二分表 + 浪费说明 |
| KV cache：形态 + per-token 拷贝成本 | ✅ | §4：两张打包张量、25.2 MB、0.73–0.97 s 上行 |
| 连续 N≥8 token 逐步对拍，每步都记 | ✅ | §5.2：8 步逐步表（码级 / ｜Δcos｜ / logits / 命中） |
| 实现门 + 最终质量门（logits cosine / 命中率） | ✅ | §5.3：实现门 PASS；logits cos 0.973–0.999、命中 6/8；并写清能/不能说明什么 |
| 每 token 延迟分解（权重带宽 + KV 读写 + host 拷贝） | ✅ | §6：68.5 ms = 0（权重常驻）+ 33.4（KV 宽度）+ 35.1（塔）+ 25.2 MB host 拷贝 |
| 不改公共工具 / 不改 .scratch | ✅ | 只新增 tools/build_q06.py、e2e_q06.sh、mem_q06*.sh |
