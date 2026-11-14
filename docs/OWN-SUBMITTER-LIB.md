# OWN-SUBMITTER-LIB —— 用我们自己的库「定义 -> 编译 -> 在 NPU 上跑」逐元素算子

本文件是 `tools/npu_lib/` 这个包的说明书：接口、证据、适用范围、未解。
提交器本身的来历与逐项实测（ADD/MUL/SUB/MAXIMUM/MINIMUM/RELU/FC/BMM/softmax）在
[`docs/OWN-SUBMITTER.md`](OWN-SUBMITTER.md)；边界定义在
[`docs/adr/0002-self-built-means-execution-path.md`](adr/0002-self-built-means-execution-path.md)。

一句话：**定义在 Python 里（Graph），命令仍是厂商编译器产物（ncc + 一次录制），
执行路径完全是我们自己的（/dev/apusys ioctl）。** 运行期不 dlopen、不链接 libneuron/neuronrt。

---

## 1. 接口（票面定死，实现按这个形状）

```python
from npu_lib import Graph, DType

q = DType.int8(1 / 127)          # per-tensor；zero_point 目前只支持 0
g = Graph("add_1x896")
a = g.input("a", (1, 896), q)
b = g.input("b", (1, 896), q)
c = g.add(a, b)                  # 也支持 g.sub(a, b) / g.mul(a, b)

art = g.compile()                # ① tflite ② .dla ③ 资产捕获 ④ 落 artifacts/<name>/
out = g.run({"a": a_codes, "b": b_codes})    # 只走自建提交器
res = g.verify(out)              # ① 同源逐字节 ② CPU golden 逐位
res.ok                           # True 才算过
```

| 接口 | 语义 | 本票状态 |
|---|---|---|
| `DType.int8(scale, zero_point=0)` | per-tensor int8 | 已实现；`zero_point != 0` 显式抛 `UnsupportedError`（非对称量化无板端证据） |
| `DType.int16(scale)` | int16 | 构造入口在，抛 `NotImplementedError`（int16 的累加/requant 契约与 int8 不同） |
| `DType.fp16()` | fp16（无 scale） | **已实现**：`[8,64]` 上 MUL/ADD/SUB 全过，判据改成 ULP ≤ 1，见 [`reports/npu-lib-fp16.md`](../reports/npu-lib-fp16.md) |
| `Graph(name)` | 一张图一个名字；artifacts 目录用这个名字 | 已实现 |
| `g.input(name, shape, dtype)` | 静态形状；形状即命令流的一部分 | 已实现，带 16 B 行规则校验（int8 每行 >= 16 B 且 % 16 == 0） |
| `g.add/sub/mul(a, b)` | 同形状、无广播、同 dtype | 已实现；形状不同 -> `BroadCastError`（显式报错，不悄悄广播） |
| `g.fc(x, w)` | `x[M,K]` × `w[N,K]`（**权重是运行时输入**）-> `[M,N]`，`s_out = s_a*s_w*32`（可显式给） | 已实现（int8）；见 [`reports/npu-lib-fc.md`](../reports/npu-lib-fc.md) |
| `g.softmax(x)` | 沿最后一维；int8 输出 `(1/256, zp=-128)` | 已实现；一元 + LUT 缓冲（`extra_file`），见 [`reports/npu-lib-softmax.md`](../reports/npu-lib-softmax.md) |
| `g.mean(x, axis=-1)` | 整数归约 + 两步舍入；输出 scale 缺省 = 输入 scale（厂商约定） | 已实现；0/8，见 [`reports/npu-lib-mean-rsqrt.md`](../reports/npu-lib-mean-rsqrt.md) |
| `g.rsqrt(x)` | 一元；输入码值须 > 0 | 已实现；0/128，同上 |
| `g.logistic(x)` | 一元 sigmoid，**只支持 fp16**（int8 版 64/64 失配未定性，显式拒绝） | 已实现；`[8,128]` 门① 逐字节 + 0/2048 ULP，见 [`reports/npu-lib-logistic.md`](../reports/npu-lib-logistic.md) |
| `g.compile()` | 四步流水线，见 §2 | 已实现 |
| `g.run(inputs)` | dict{输入名: int8 码} -> `Output` | 已实现，只走 /dev/apusys |
| `g.verify(out)` | 两道门，见 §4 | 已实现，返回 `VerifyResult`（`.ok == 同源 且 数值在 `int8_tol` 内；tol=0 即逐位） |

出错类型：`NpuLibError`（编译/捕获/提交失败）、`BroadCastError`（形状不匹配）、
`UnsupportedError`（接口留了入口但没实现）。三者都是 `NpuLibError` 的子类，
`UnsupportedError` 同时是 `NotImplementedError`。

**运行位置**：库在 **WSL 的 `/usr/bin/python3`** 下跑（仓库里唯一带 numpy/flatbuffers/tflite
schema 的解释器，NOTES §2）。设备访问用 adb.exe（WSL interop，push 的源/目的路径过 `wslpath -w`），
编译与生成用 WSL 原生工具。

---

## 2. compile() 的四步（票面 ①②③④）

| 步 | 做什么 | 谁来做 | 产物 |
|---|---|---|---|
| ① | 按 Graph 的 shape/dtype/scale 写 `model.tflite`（+ `model.json`、`golden/`） | `npu_lib/build_model.py`（flatbuffers，借用 `tools/build_ops.py` 的 helper） | `artifacts/<name>/model.tflite` |
| ② | `tools/compile.sh` -> `.dla`（`--arch=mdla5.0 --disallow-bridge`） | ncc-tflite（WSL，SDK 7.0.8） | `model.dla` + `model.compile.log` |
| ③ | **用厂商 runtime 跑一次**，录下命令流/子命令/模板/三个旧 IOVA | `npu_lib/capture.py`（neuronrt + `probe/libiolog2.so`，一次） | `artifacts/<name>/asset/` + `capture/` |
| ④ | 资产、plan、golden、sha256 清单落盘 | `npu_lib/graph.py` | `plan.txt` / `manifest.json` / `capture.json` |

### 为什么要第 ③ 步（这是 ADR 0002 的边界，不是绕路）

命令流是**厂商编译器输出**，我们没有 MDLA5.0 ISA 的命令生成器
（`docs/adr/0002-self-built-means-execution-path.md` 原文：真正自生成命令是后续阶段）。
所以资产必须来自"一次合法提交"，而那一次只能由厂商 runtime 产生。
**这一步只在 compile() 里发生一次，run() 里不再出现**（见 §3.2 的证明）。

形状（含元素数）烘死在命令流里，所以**换形状 = 重新捕获一次资产**。这正是本包存在的理由：
把这个"每次都要人肉做的事"变成 `g.compile()` 里的自动一步，并且**不再需要重编 C**。

### 资产目录内容（`artifacts/<name>/asset/`）

| 文件 | 说明 |
|---|---|
| `cmd0/1/2.bin` | 三个 cmdbuf（32 B / 16 B / 368 B 这一组，顺序有语义：dir1 / dir2 / dir1） |
| `subcmd.bin` | `mdw_subcmd_info` 模板（80 B，提交时替换 `num_cmdbufs` 与数组指针） |
| `cmdin.bin` | `mdw_cmd_in` 模板（152 B，提交时替换子命令/adj/exec_infos） |
| `iova.txt` | 抓取那次运行 A/B/C 的 IOVA —— **与命令流配对**的重定位匹配键 |

---

## 3. run()：只走自建提交器（plan 驱动，不重编 C）

`g.run()` 做四件事：写输入文件 -> push（输入 + plan + 资产 + `npu_run`）-> 执行 `npu_run` -> 拉回输出。

### 3.1 plan.txt —— 提交器的唯一输入

`probe/npu_run.c` 是 `probe/npu_binop.c` 提交核心（握手/分配/重定位/提交/同步）的
**读 plan 版**，`npu_binop.c` 一行未改。plan 是简单 key=value 文本：

```text
asset_dir=/data/local/tmp/npuops/runs/add_1x896_int8/asset
old_a=0x25596000            # 抓取那次的 A/B/C IOVA（重定位匹配键）
old_b=0x258d3000
old_c=0x25587000
alloc_a=4096                # 每块的分配大小（抓取时 runtime 就是 4 KiB）
alloc_b=4096
alloc_c=4096
dir0=1                      # 三个 cmdbuf 的方向标志，顺序有语义
dir1=2
dir2=1
patch_idx=2                 # 地址字在哪个 cmdbuf 里
expect_patches=3            # 必须命中 3 处（A/B/C）
n_elems=896                 # 元素数（int8 => 字节数）
n_inputs=2
in_a=/data/local/tmp/npuops/runs/add_1x896_int8/a.bin
in_b=/data/local/tmp/npuops/runs/add_1x896_int8/b.bin
out=/data/local/tmp/npuops/runs/add_1x896_int8/out.bin
```

**换算子 / 换形状 / 换长度 = 改这份文本**，C 不用重编。这就是"顺序语义固定、形状烘死在命令流"
与"每次重新捕获"之间最省事的接口。

提交本身沿用已验证的结论（逐条见 `docs/MAPPING.md`）：
握手 BASIC×2 + DEV(1,2,4,6,7) + MEM(1,2,4,5) -> ALLOC + mmap + MEM MAP（flags=2, align=0x100）
-> 写内容 -> 命令缓冲与数据缓冲 `DMA_BUF_IOCTL_SYNC` -> 按值重定位（命令流里的地址字是 `iova >> 4`）
-> 组装 `cmdbuf[3] / subcmd_info / cmd_in`（exec_infos 必须是 MEM handle）-> `CMD RUN`
-> 等 200 ms -> `DMA_BUF_SYNC(READ)` -> 读回。

### 3.2 怎么证明 run() 没碰厂商库

三道独立证据（每次运行都记进 `artifacts/<name>/evidence.json` 的 `no_vendor_lib` 字段）：

1. **进程 maps**：让 `npu_run` 提交完成后多活 1.5 s（`NPU_RUN_HOLD_MS`），从另一个 shell 采
   `/proc/<pid>/maps`（42 行），按 `neuron|mdla` 过滤**一行都没有**；
2. **二进制层面**：`readelf -d probe/npu_run` 输出 `There is no dynamic section in this file`
   —— **静态链接，没有 DT_NEEDED**，不存在"链接到 libneuron"的可能；`strings` 里
   `libneuron|neuronrt` 出现 0 次（唯一的 `dlopen` 字样是 bionic 自己的
   `memtag_stack_dlopen_callback`，与厂商库无关）；
3. **源码层面**：`probe/npu_run.c` 只有 `open("/dev/apusys")` 与三个 ioctl；
   运行期路径（`graph.run` -> `env.dev_shell`）只执行 `npu_run`，不执行 `neuronrt`。

反过来，`compile()` 的第 ③ 步**确实**调了厂商 runtime —— 那是命令捕获，ADR 0002 明确划进
"自建"范围；它在 `capture.json` 里留了完整记录（跑了什么、产出了什么、日志在哪）。

---

## 4. verify()：两道门

| 门 | 判据 | 口径 |
|---|---|---|
| ① **同源等价** | 与抓资产那次厂商输出**逐字节相同** | `capture/vendor_out.bin` 是 neuronrt 那一次写出来的结果；只有"喂的就是抓取时那对输入"时才定义得出来，否则这一门记 `n/a` 并在 detail 里写清楚，不伪造 PASS |
| ② **数值正确** | 与独立 CPU golden **逐位相同** | `np.int64` 精确累加 + `tools/quant_contract.requant_int8`（gemmlowp 两步整数舍入），与 flatbuffer 生成完全分离（规则 R3） |

requant 口径（为什么这一票的 `int8_tol = 0`）：本库把输出 scale 定成
- ADD / SUB：`s_out = s_in`（real_multiplier == 1）
- MUL：`s_out = s_a * s_b`（real_multiplier == 1）

于是设备只是整数运算 + 钳位，golden 与设备应当**逐位**相等。
"逐元素 requant != 1"（例如 `s_out = 1/127` 的 MUL）**没有板端证据**，
本票不声称支持 —— 见 §6。

---

## 5. 板端证据（票面最小矩阵，9/9 PASS）

跑法一条命令（前后各一次 healthcheck，全程持 `NPU_LOCK`）：

```bash
./tools/run_npu_lib_matrix.sh --maps
```

一次连续运行的实测（2026-11-13，mt6897 / Android 16）：

| 算子 | 形状 | 元素 | requant | 同源逐字节 | golden 逐位 | 结果 | device_out sha256 (= vendor = golden) |
|---|---|---|---|---|---|---|---|
| ADD | [1,64]   | 64   | 1.0 | PASS | PASS | **PASS** | `2787bdeadb31` |
| ADD | [1,896]  | 896  | 1.0 | PASS | PASS | **PASS** | `6a498bcb02c1` |
| ADD | [1,96]   | 96   | 1.0 | PASS | PASS | **PASS** | `e6f98143ec4b` |
| MUL | [1,64]   | 64   | 1.0 | PASS | PASS | **PASS** | `80d2c055441e` |
| MUL | [1,896]  | 896  | 1.0 | PASS | PASS | **PASS** | `ca09750473e8` |
| MUL | [1,96]   | 96   | 1.0 | PASS | PASS | **PASS** | `012c02e17dc4` |
| SUB | [1,64]   | 64   | 1.0 | PASS | PASS | **PASS** | `9a246c7c5602` |
| SUB | [1,896]  | 896  | 1.0 | PASS | PASS | **PASS** | `a760cfd1c137` |
| SUB | [1,96]   | 96   | 1.0 | PASS | PASS | **PASS** | `30bd8f7e9517` |

要点：

* **896 = Qwen hidden 宽度，96 = 非 2 的幂**，两者都 PASS —— 形状没有烘死在 C 里
  （C 只读 `n_elems`，命令流由 compile() 现抓）。
* 每个组合的 `.dla` 都是 **1 个执行步、无 bridge**（`--disallow-bridge` 下 ncc 仍 `EXIT=0`），
  层清单是单层 `ElementWiseAddLayer/ElementWiseMulLayer/ElementWiseSubLayer`。
  896 的 DRAM 账：Input 1.8K（2×896）、Output 896、Code 368。
* `device_out == vendor_out == golden/expected` **三者 sha256 完全相同**：自建提交器不但
  算对了，而且与厂商 runtime 那次逐字节一致。
* 前后 `healthcheck.sh --quick` 都是 PASS（known-good `fc_1x64x64_int8` 回归 + 板子无残留），
  自建库跑完不污染后续任务。
* **每个组合跑两次**：第一次喂"抓取时那对输入"（于是同源门与 golden 门都能判），
  第二次喂一对**全新的随机输入**（同源门按定义不适用 —— 厂商那次只见过抓取输入），
  9/9 组合第二次也都 @@numeric_ok@@ 且 0 失配。这说明数值门不是"重放同一对输入"才成立。
  两次的输出 sha256 不同（@@device_out@@ vs @@device_out_alt@@），各自都等于对应的 golden。
* 单组合耗时：**冷跑**（编译 + 捕获 + 运行 + 验收）约 **6.3–7.6 s**，其中板端只占约 0.5 s，
  大头是 ncc 编译；**热跑**（资产已存在，只 run + 验收）约 **1.5–1.7 s**。

自动生成版（含每次运行的 plan 全文与全部 sha256）：
`reports/npu-lib-matrix.md` / `reports/npu-lib-matrix.json`；
单次运行的完整证据：`artifacts/<name>/evidence.json`（与 `plan.txt` / `manifest.json` / `capture.json` 同目录）。
注意 `.dla` 的 sha256 只用来标识产物，**不是**回归判据（NOTES §4：同一 tflite 连编三次哈希都不同）。

---

### 5.1 第二个算子：FULLY_CONNECTED（动态权重）

证据：`artifacts/fc_8x64x32_int8/evidence.json`；报告 [`reports/npu-lib-fc.md`](../reports/npu-lib-fc.md)。

| 项 | 值 |
|---|---|
| 形状 | `x[8,64] int8` × `w[32,64] int8`（权重是第二输入）-> `out[8,32]` |
| 量化 | `s_a=s_w=1/127`，`s_out=s_a*s_w*32`（requant = 1/32） |
| 编译 | `EXIT=0`，`.dla` 3165 B |
| 计划 | `bytes_a=512` / `bytes_b=2048` / `bytes_out=256` —— **三块缓冲三个字节数** |
| 门① 同源 | PASS（喂抓取那对输入，`device_out == vendor_out` 逐字节） |
| 门② 数值 | PASS，0/256（喂全新随机输入也 0/256） |
| 三者相等 | `device == vendor == golden`，sha256 `e660b3146b10db1c…` |

这一条同时把提交器的尺寸接口从"单一 `n_elems`"扩成"每块缓冲各自的字节数"（`n_elems` 保留为兼容回退，
老资产的 plan 一个字没改仍然 9/9 PASS）。

## 5.2 计时与统计（`npu_lib.stats`，2026-11-14）

测量不再散落在各脚本里：统一走 [`tools/npu_lib/stats.py`](../tools/npu_lib/stats.py)，
记录进 `reports/timings.json`（只追加），报告渲染成 [`reports/timings.md`](../reports/timings.md)。

| 接口 | 用途 |
|---|---|
| `stats.measure(fn, rounds=5, warmup=1)` | 跑 fn 若干轮，返回每轮毫秒（warmup 不计） |
| `stats.record(name, xs, kind=..., note=...)` | 落盘一条（带 provenance：主机/时间/口径/备注） |
| `stats.summarize(xs)` | n / min / p10 / **median** / mean / p90 / max / **MAD** / jitter% |
| `stats.paired(fn_a, fn_b, rounds=7)` | **配对交替**（A,B,A,B…），只认它的中位数比值 |
| `stats.parse_neuronrt(text)` | 解析厂商 `Total inference time = .. (.. ms/inf)` |
| `stats.parse_runner(text)` | 解析 `npu_run` 的 `TIMING iter=.. wait_ms=.. med=..` |
| `python -m npu_lib.stats --render` / `--list` | 出 `timings.md` / 列所有测量 |

纪律（写进模块 docstring）：**中位数是主指标、MAD 是离散度**；跨时间/跨设备比较**只认配对交替的比值**，
不拿两次绝对数相除（同一份 `.dla` 会漂 ±15%）。已经接上：`npu_lib.matrix` 的每次 `g.run()` 与冷跑总时长。

## 6. 适用范围与未解（不圆）

### 6.1 这一票确实做到的

* 逐元素 **ADD / SUB / MUL**，int8 per-tensor，同形状，任意长度（只要满足 16 B 行规则）；
* 定义 -> 编译 -> 捕获 -> 提交 -> 两道验收，**全链路一条 Python 调用**；
* 运行期零厂商库（§3.2 三重证据）；
* plan 驱动的 C runner：换算子/换形状/换长度**不重编 C**（`probe/npu_binop.c` 一行未改）。
* **矩阵乘级算子 FULLY_CONNECTED（动态权重）** 也进了库：`g.fc(x, w)`，`[8,64]×[32,64] -> [8,32]`，
  三块缓冲三个字节数，`device == vendor == golden`（§5.1）。

### 6.2 明确的未解 / 边界

1. **一条图只能有一个算子**。当前提交器重放**一份**命令流 + A/B/C 三块数据缓冲；
   链式图（`g.add -> g.mul`）需要 N 块缓冲的 plan（缓冲数随中间张量增长），
   现在 `g.mul(...)` 在已有算子的图上会显式抛 `UnsupportedError`。
   这是接口留好的下一步，不是本票范围。
2. **int16 / fp16 只有构造入口**（抛 `NotImplementedError` 并写明理由）。
   int16 的累加宽度、fp16 的噪声底与阈值登记都还没做（NOTES §4、docs/adr/0005）。
3. **requant != 1 的逐元素**没验证：本库把 out scale 定成让 multiplier == 1 的组合，
   于是 golden 与设备逐位相等。若要让 `add` 的两个输入用不同 scale、或 MUL 输出回 1/127，
   需要先出板端证据（仓库里 FC/BMM 的 ±1 code 结论不能直接套到逐元素上）。
4. **非零 zero_point（非对称量化）**未验证，`DType.int8(zp != 0)` 直接拒绝。
5. **广播**按票面要求不支持，形状不同显式报 `BroadCastError`。
6. **换形状仍要重新捕获资产**（形状烘在命令流里，ADR 0002 已写明这是本阶段的性质）。
   `compile()` 把它自动化了，但没有消除它；`artifacts/<name>/` 的数量会随形状增长。
7. **需要额外缓冲的算子**：`extra_file` 字段现在**有证据**了 ——
   SOFTMAX 的 LUT/scratch 缓冲（`[1,8,16]` int8，4 KiB）走 `extra_file=scratch.bin` 重放，
   与厂商设备输出**逐字节相同**（0/128），见 [`reports/npu-lib-softmax.md`](../reports/npu-lib-softmax.md)。
   softmax 已经是 `Graph` 的一等算子（`g.softmax`）：一元链路、LUT 缓冲、`output_zero_point=-128` 的 golden、
   `int8_tol` 登记都通了，实测 0/128 且与厂商逐字节相同。
   常量权重 FC 仍压着 OWN-SUBMITTER §10.3 那个 IoVA 契约未解。
8. **"命令自生成"仍未开始**：本包把"重放捕获的命令流"工程化了，没有生成命令。
9. **锁与并发**：库自己用目录锁（与 `tools/devlock.sh` 同一把，`NPU_LOCK_HELD=1` 表示外层已持锁）。
   板子只有一台，长跑矩阵仍应整体包在 `devlock.sh run` 里。

---

### 6.3 改动边界与证据体系（2026-11-14 补记）

- **`probe/` 由本库改，且只由 Lead 改。** 提交器本体就在 `probe/`（`npu_run.c` / `iolog2.c` / `mkasset.py`），
  本库不可能不改它们。按 [`docs/AGENT-BRIEF.md`](AGENT-BRIEF.md) §2「公共工具的行为变化由 Lead 统一改」，
  这些改动一律由 Lead 在本分支完成，并在对应 commit 与 `reports/npu-lib-*.md` 里留证据。
  **并行 agent 仍然不要碰 `probe/`。**
- **证据体系有两套，等价。** `ops/<name>/compare.txt` 是「一算子一目录」体系的板端判据；
  本库路径下的等价物是 `artifacts/<name>/evidence.json`（plan 全文 + 两道门 + 各文件 sha256 + no_vendor_lib 证据）。
  两者等价 —— 不要因为某算子没有 `compare.txt` 就判「无证据」。

## 7. 复现

```bash
# 0) 环境：WSL 有 /usr/bin/python3 (numpy/flatbuffers/tflite)；NDK 交叉编译 npu_run
./tools/build_npu_run.sh                         # -> probe/npu_run (aarch64 static)

# 1) 全矩阵（前后各一次 healthcheck --quick，全程持锁）
./tools/run_npu_lib_matrix.sh --maps

# 2) 或者只跑一个组合
./tools/devlock.sh run -- wsl.exe -e bash -lc \
  "cd '/mnt/d/deepseek_ workspace/mtk/wt-lib' && NPU_LOCK_HELD=1 PYTHONPATH=tools \
   /usr/bin/python3 -m npu_lib.matrix --ops add --lens 896"

# 3) 单条命令重跑某个产物（不重编、不重抓）
./tools/devlock.sh run -- wsl.exe -e bash -lc \
  "cd '/mnt/d/deepseek_ workspace/mtk/wt-lib' && NPU_LOCK_HELD=1 PYTHONPATH=tools \
   /usr/bin/python3 -c \"import numpy as np; from npu_lib import Graph;\
g=Graph('add_1x896_int8'); a=np.fromfile('artifacts/add_1x896_int8/golden/a.bin',dtype=np.int8);\
b=np.fromfile('artifacts/add_1x896_int8/golden/b.bin',dtype=np.int8);\
print(g.verify(g.run({'a':a,'b':b})))\""
```

重建一个产物（例如改了 scale）先删目录：`rm -rf artifacts/<name>`，再 `g.compile()`。
