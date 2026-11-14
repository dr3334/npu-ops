# P-05 — ncc 的外部/共享权重机制：能用 SDK 自带的 `extract-shared` 造「图 + 外部权重 bin」

> 票：`.scratch/quant-runtime/issues/03-shared-weights-mechanism.md`
> 结论先给：**能**。不需要厂商私有工具，NeuroPilot SDK 7.0.8 自带三个二进制就够；
> 已在板端端到端复现「两张图 + 一份权重 bin」，数值 1 个 fp16 ULP。**票 05 成立。**
> 证据产物在仓库外：`D:\deepseek_ workspace\mtk\.qwen25-impl\probe-shared\`（下载的厂商资产副本、探针图、bin、板端输出）。

## 1. 问题

厂商的部署形态是「两张小图 + 一个大权重文件」：

| 资产 | 大小 | 说明 |
|---|---|---|
| `..._24layer_128t2048c_0_shared.dla` | 3,238,205 B | prefill 图 |
| `..._24layer_1t2048c_0_shared.dla` | 21,587,592 B | decode 图 |
| `shared_weights_2048c_0.bin` | 250,737,664 B | 24 层 W4 权重，被上面**两张图共用** |
| `embedding_..._int16.bin` | 272.3 MB | 151936×896×2 B = 词嵌入表（int16），不进权重 bin |

`llm_sdk/config_*.yaml` 里 `dlaPromptPaths` / `dlaGenPaths` 指向两张 `.dla`，`sharedWeightsPaths` 指向那个 bin。
我们自己「每层一个 .dla」把权重编进图，24 层就是 24 次装载。所以必须先回答：ncc 有没有权重外置/多图共享常量的机制，
我们能不能自己造一份。

## 2. 假设与预测

- **H1（编译期开关）**：`ncc-tflite` 有「权重外置」开关，编出来的 `.dla` 引用外部文件。
- **H2（后处理工具）**：权重外置不是编译期做的，而是编译后用 SDK 的某个工具从 `.dla` 里抽出来的。
- **H3（厂商私有）**：抽取/打包是 GAI 工具链里的私有步骤，SDK 里没有，我们造不了。

预测：H2 成立则「编译普通 tflite → 两个 .dla → 一条命令抽出共享 bin → 板端两图跑同一 bin」这一串应当可复现，
且 `.dla` 的输入表里应当多出一个「大小 = bin 大小、size = 0」的输入。

## 3. 做法

1. `ncc-tflite --help` / `--target-help` 全量（WSL，`ncc-tflite 7.3.15`），以及 `host/bin/` 下每个二进制的 `--help` / 无参 usage。
2. 读 `host/include/neuron/api/`（DLAMuxer.h / RuntimeAPI.h）与 GAI 工具链里厂商自己的脚本和运行时源码
   （`GAI-Deployment-Toolkit-.../compile/*.sh`、`compile/extract-shared/extract-shared.sh`、
   `inference/jni/prebuilt/src/executor/{executor,llm_executor,shared_weights}.cpp`）。
3. 只读拉下厂商 `.dla` / `bin` 副本做静态分析（`adb pull`，设备上的文件一个字节没动）。
4. 自己造探针：两张**权重字节完全相同**的图（prefill T=8 / decode T=1，D=896，两层 FC+RELU，fp16），
   用 `tools/compile.sh` 编成两张 `.dla`，再跑 `extract-shared` 抽 bin。
5. 板端（持 `devlock`，前后各一次 `healthcheck --quick`）：两图 + 同一 bin 跑 `neuronrt`，与 numpy 参考对拍；
   另外做「错 bin / 截断 bin / 连跑两次」的完整性探针。
6. 对照实验：把 decode 图的权重换 seed（不共享）再抽一次，看工具怎么处理。

## 4. 结果

### 4.1 SDK 里就有这三个工具（H2 成立，H3 不成立）

| 工具 | 作用 | 关键参数 |
|---|---|---|
| `ncc-tflite --extract-static-data <file>` | 编译期把静态参数「变成输入张量」 | 见 §4.7：**这个不可用** |
| `extract-shared` | 后处理：从**多个** `.dla` 里抽出共享静态数据到一个 bin，并改写 `.dla` | `-i a.dla,b.dla -o a_shared.dla,b_shared.dla -s shared.bin [-e diff\|union\|reduce] [-a]` |
| `dla-packer` | 把多个 `.dla` + 外部权重打成 `.dlb` 包 | `--ext-static-data <bin> -o out.dlb a.dla b.dla` |
| `readdlb` | 读 `.dlb` 信息（DLA 列表、是否 extract-shared 产出） | — |

`extract-shared` 的 `-e` 三种模式（原文）：`diff` = 共享数据从第一张 DLA 抽（默认）；
`union` = 从所有 DLA 抽；`reduce` = 生成一组 DLB，进一步省内存。

**厂商自己就是这么干的**，脚本在本地工具链里：
`GAI-Deployment-Toolkit-.../compile/extract-shared/extract-shared.sh` 把每个 `x.dla` 变成 `x_shared.dla`，
`-s ./<STATIC_FILE>` 输出那个 bin —— 正好解释了资产命名 `_shared.dla` + `shared_weights_*.bin`。
运行侧源码 `executor.cpp:59 loadSharedWeights()` 的做法是：把 bin 当成**模型的最后 N 个输入**、
`memcpy` 进 buffer 后照常 register（`llm_executor.cpp:32 firstSharedWeightsInputIdx = getExpectedNumInputs() - numSharedWeights`），
prefill/decode 两个 Executor 共用同一批 preload buffer（`shared_weights.cpp:170 preloadUnion`）。

### 4.2 厂商 `.dla` 解剖：两张图都挂同一个「size=0 的最后一个输入」

板端只读 `neuronrt -d`（IO 表）：

| | prefill `128t2048c` | decode `1t2048c` |
|---|---|---|
| 输入个数 | 52 | 52 |
| 最后一个输入 | `Handle = 51, <250737664>, size = 0 bytes` | `Handle = 51, <250737664>, size = 0 bytes` |
| 其余输入 | 激活 229,376 B + mask 557,056 B + 24×524,288 B + 24×32,768 B | 激活 303,872 B + 24×524,288 B + 24×256 B + … |
| 输出 | `<1×1×128×151936>` = 38,895,616 B（整段 logits，lm_head 在图里） | — |

**两张图最后一个输入的字节数完全等于 bin 的字节数（250,737,664），且 size 标 0** —— 这就是「外部权重」在
DLA 里的样子：它不是普通输入（没有 per-run 大小），而是抽出来的静态张量，由 App 绑一次。这也是为什么
24 × `.dla` 只有 3.2 / 21.6 MB：权重根本不在图里。

`.dla` 里还嵌着厂商的完整编译命令行（`strings`，元数据区），例如 prefill：

```
--arch mdla5.0 --opt 3 --opt-footprint --opt-aggressive --stable-linearize --gno LTS --gno-exp
--gno-non-4d-tiling --fc-to-conv --broadcast-act-wgt --broadcast-flow-distance 63
--ncl-patch .../compile/llm.ncl --mdla-int16-lut --l1-size-kb 3072 --num-mdla 2
--suppress-input --suppress-output --show-memory-summary --model ...128t2048c_0.tflite
```
（decode 用 `--split-16a4w-conv-oc` 取代 `--broadcast-act-wgt`。）**注意：这里没有 `--extract-static-data`** ——
说明厂商是「先正常编译，再 `extract-shared` 后处理」，与 H2 一致。`llm.ncl` 是他们的 ncc patch（本地有副本）。

### 4.3 我们复现（板端 PASS）

探针（仓库外 `probe-shared/`）：两张权重字节完全相同的图，D=896，`FC → RELU → FC`，fp16，
T=8（prefill 形态）与 T=1（decode 形态）；权重合计 3,214,848 B。

| 步骤 | 产物 | 大小 |
|---|---|---|
| `tools/compile.sh` | `prefill_T8/model.dla` | 3,223,010 B（Static 3.1 M） |
| `tools/compile.sh` | `decode_T1/model.dla` | 3,223,478 B |
| `extract-shared -i …,… -o …,… -s ./probe_shared.bin` | `*_shared.dla` | **4,652 B / 5,120 B** |
| 同上 | `probe_shared.bin` | **3,219,456 B** |

`extract-shared` 的自述：`Total number of exported static data: 4 / Total size: 3219456`，
两张图各自 `reuse static data: 4 / 3,218,432 B`、`nonreuse: 0`。

板端（持锁，`neuronrt -m hw -c 3`，两图都喂 `-i 输入.bin -i probe_shared.bin`）：

| 图 | 输入表 | 运行 | 与 numpy 对拍 |
|---|---|---|---|
| prefill T=8 | `Handle1 = <3219456>, size = 0` | 1.32 ms/inf | `max_abs = 4.883e-4`（= 1 ULP @\|y\|≈1.16），rel 4.2e-4，逐位 4091/7168 |
| decode T=1 | `Handle1 = <3219456>, size = 0` | 0.973 ms/inf | `max_abs = 4.883e-4`（1 ULP），rel 4.3e-4，逐位 510/896 |

连跑两次输出**逐位相同**（`max|r1-r2| = 0`）。→ 两张图、一份外部权重、板端数值正确。

### 4.4 bin 的格式（对照实验，而不是猜）

因为探针权重是我们自己造的，可以直接验证：

- bin 里**没有任何头部**：偏移 0 开始就是数据（我们自己抽的 bin 首次 64 B 已是权重值）。
- bin 内容 = 4 个静态张量的字节，**顺序拼接 + 对齐填充**：把 w1/w2 的 u16 词表与 bin 的词表比对，
  802,816/802,816 全命中；但连续 32 B 片段一个都搜不到 → 段内**被重排过**（MDLA 的 tiling/reorder）。
- 大小 = 3,218,432（4 个张量原字节）+ 4,608（对齐）= 3,219,456。
- 所以：**bin 里没有路径、没有校验、没有张量目录**；偏移与大小记在改写后的 `_shared.dla` 里
  （`neuronrt` 报的 `<3219456>` 就是它）。谁提供 bin 由 App 决定，DLA 不知道文件路径。

厂商那个 bin 用同一工具产出（同 SDK 同版本族），实测首字节即 payload、无 magic；
熵约 6.4 bit/byte（我们 fp16 的对照 bin 是 7.1）→ 是打包/量化过的数据，不是 fp16。

**厂商 bin 的体积口径（推断，标 UNKNOWN 的边界见 §7）**：24 层参数 357,826,560 + lm_head 136,134,656
= 493,961,216；按 4 bit 打包 = 246,980,608 B，bin 是 250,737,664 B，多出 3,757,056 B（1.5%）——
正好是 fp16 分组 scale 的量级（group≈256）。即 bin = 4-bit 打包 + 分组 scale，词嵌入表单独走 int16 文件。
我试过按 G∈{16,32,64,128,256,512,1024} 的周期扫描「scale 插在 nibble 之间」的排列，全部退化为无结构
（熵 4.33 与随机基线同档）→ **scale 与 nibble 不是字节级交错**，应该是每个张量块内部各占一段。

### 4.5 完整性：长度校验有，内容校验没有（对票 05 重要）

| 探针 | 结果 |
|---|---|
| 把 bin 换成**全零**（长度相同） | **不报错**，照跑，输出与正确答案 max\|Δ\|=1.16、相对误差 1.0 → **静默算错** |
| 把 bin **截半**（1,609,728 B） | 明确报错 `The size of input[1] should be 3219456. Inference fail` |

→ 运行期只按 DLA 里记的**字节数**校验，没有任何内容校验/哈希/路径绑定。
所以票 05 必须自己记 `bin 的 sha256 + 图的 sha256 + 一次运行输出`（spec 的 Testing Decisions 已经这么写了），
配错 bin 在设备上**不会**有任何提示。

### 4.6 两张图权重不一致时会怎样（安全降级，不报错）

把 decode 图换成 different seed（权重不共享）再抽一次：

```
prefill:  reuse 4 / 3,218,432 B   nonreuse 0
decode :  reuse 0                 nonreuse 4 / 3,218,432 B   → d.dla 仍是 3,223,860 B
```

共享的那份进 bin，不共享的那份**留在自己的 DLA 里**，工具不报错、也不提示「你以为共享了其实没有」。
→ 票 05 的前提是：prefill 与 decode 两张图的**权重字节必须完全一致**（同一份量化产物，不能各量化一次）。

### 4.7 `ncc-tflite --extract-static-data` 单独评估：语义是「常量变输入」，不产出可用权重

- 128 维小模型：1.5 s 编完，`.dla` 只有 5,608 B（66 KB 权重不在图里），IO 表变成 3 个输入
  （激活 + 每个权重**各自一个规则张量** `<1×1×128×128>` = 32,768 B）；`ext.bin` 8,388,672 B 里
  **只有末尾 64 B 非零**（其余全零）→ 权重字节没写出来。
- 896 模型：120 s 内**没跑完**，期间写出 1.07 GB 仍在增长（被超时杀掉，无 `.dla`）。

→ 这条路要求 App 按编译器内部布局逐个喂权重张量，而那个布局不可得；**不要用**。
共享权重要用后处理的 `extract-shared`（一次一个 blob、大小精确、板端已验证）。

### 4.8 另一条路：`dla-packer` + `.dlb`（可用，但厂商没用）

`dla-packer --ext-static-data ./probe_shared.bin -o pair.dlb a_shared.dla b_shared.dla` → `pair.dlb` 3,229,848 B；
`readdlb pair.dlb` 报 `DLA List: #0 a_shared.dla / #1 b_shared.dla`，
`Legacy Static Data Extraction: ON`、`Generated by extracted-shared: OFF`、`DRAM Temporary Buffer Sharing: OFF`。
对应运行侧是 `NeuronDLAMuxer_createWithSharedData()`（DLAMuxer.h:61）/ `NeuronDLAMuxer_setStaticData()`
—— 它才是「一次装载多图 + 快速切 prefill↔decode」的官方 API，并会自动从 bundled bin 分配 static data。
厂商资产没用 `.dlb`（是两张独立 `.dla` + `sharedWeightsPaths`），说明他们走的是 per-model 输入绑定那条路。
**两条路我们都用得**；票 05 建议先用「独立 `.dla` + bin」（改动最小、与厂商形态一致），
「换图开销」这件事再交给票 06/后续用 `.dlb` + DLAMuxer 量。

## 5. 对票 05 的直接含义

**成立**，最小改动路线（全部是 SDK 自带工具，无私有依赖）：

```bash
# 1) 两张图各自编译（照旧）            注意：图里权重字节必须完全一致
./tools/compile.sh ops/q05_prefill/model.tflite ops/q05_prefill/model.dla
./tools/compile.sh ops/q05_decode/model.tflite  ops/q05_decode/model.dla
# 2) 抽共享权重（-e diff 即厂商默认；权重一致时 union 结果相同）
extract-shared -i ops/q05_prefill/model.dla,ops/q05_decode/model.dla \
               -o ops/q05_prefill/model_shared.dla,ops/q05_decode/model_shared.dla \
               -s ops/q05_shared_weights.bin
# 3) 运行：两图各自喂 [-i 激活 -i 权重bin]，或打成 pair.dlb 用 DLAMuxer
```

必须记住的三条：

1. **两张图的权重必须字节一致**（§4.6）。同一份量化 tflite 派生两张图；别对 prefill/decode 各量化一次。
2. **bin 没有内容校验**（§4.5）。把 `model_shared.dla` 与 `bin` 的 sha256 一起记进 `model.json`/报告。
3. **不要用 `--extract-static-data`**（§4.7）；用 `extract-shared`。

对 e2e/公共工具的诉求（请 Lead 定，我不改公共工具）：现有 `tools/e2e.sh` 只支持「一个 model.dla + 最多 2 个输入」，
共享权重形态需要「两个 .dla + 一份 bin」这条缝。可以不改 e2e，票 05 自己写一个只读调用 e2e 组件的脚本，
或者由 Lead 决定加一个 `--shared` 分支。

## 6. 适用范围

- 只对**静态张量**（权重/bias/常量表）有效；KV cache、激活这些运行时张量本来就该是输入（厂商的 48 个 KV 输入即此，见票 06）。
- 与量化档正交：fp16 / int8 / int4 都能抽，抽的是编译后的设备字节（本节实测的是 fp16 与厂商的 4-bit）。
- `num-mdla 2` 下也可用；运行时源码里 `numDlaChunks` 支持「每个 chunk 一组 bin」，
  多文件写法是 `sharedWeightsPaths` 给多个路径（我们只测了单文件，见 §7）。
- 不改公共工具、不动设备资产：本节所有板端操作都是 `-d` 只读或喂 `-o` 到 `/data/local/tmp/npuops/q03`。

## 7. 剩余疑问（UNKNOWN）

1. **厂商 bin 的精确布局**（分组 scale 的位置/组大小/是否 per-channel）——只有体积口径能对上（0.5075 B/param），
   没有独立权重可比对，所以组大小、nibble 顺序、zero-point 位置都未验证。
   *这一条不影响票 05*：我们不需要复刻它的格式，我们造自己的 bin。
2. **多文件 chunk**：`sharedWeightsPaths` 是列表、`SharedWeights` 按文件数量对应末尾 N 个输入，
   但「一个 bin 拆成多个 chunk」的切分规则（`-s a.bin,b.bin`）没测。单文件已够票 05。
3. **`--extract-static-data` 在 896 模型上为什么不停**（>1 GB、>2 min）——没查到底，只知道不可用。
4. **`llm.ncl`（厂商 ncc patch）在共享权重里扮演什么角色**：从命令行看它是自定义 op/优化 patch，
   与权重外置无关（外置是后处理），但没有逐条确认。
5. **`.dlb` + DLAMuxer 的换图开销**（厂商文档称比 `createWithSharedData` 更快的那条 API）未在板端量；
   这是票 05/06 的性能问题，不是可行性问题。

## 8. 复现

```bash
# 探针图（两张，权重字节相同）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/.qwen25-impl/probe-shared' && \
  /usr/bin/python3 mkprobe.py --T 8 --out prefill_T8 && /usr/bin/python3 mkprobe.py --T 1 --out decode_T1"
# 编译（在 wt-q03 里跑公共脚本）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-q03' && \
  ./tools/compile.sh ../.qwen25-impl/probe-shared/prefill_T8/model.tflite ../.qwen25-impl/probe-shared/prefill_T8/model.dla"
# 抽共享权重
wsl.exe -e bash -lc "export LD_LIBRARY_PATH='/mnt/d/deepseek_ workspace/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk/host/lib' && \
  cd '/mnt/d/deepseek_ workspace/mtk/.qwen25-impl/probe-shared' && \
  .../host/bin/extract-shared -i prefill_T8/model.dla,decode_T1/model.dla \
      -o prefill_T8/model_shared.dla,decode_T1/model_shared.dla -s ./probe_shared.bin"
# 板端（持锁；两图 + 同一 bin）
cd npu-ops && export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" \
  && ./tools/devlock.sh run -- ./tools/healthcheck.sh --quick \
  && ./tools/devlock.sh run -- bash "/d/deepseek_ workspace/mtk/.qwen25-impl/probe-shared/board.sh" \
  && ./tools/devlock.sh run -- ./tools/healthcheck.sh --quick
```
