# 黑盒研究 06：补四道规格缺口（dump⇄偏移 / debug·urate 可读性 / 点名的变量 / 噪声口径）

日期：2026-11-14 · SDK 7.0.8 / `ncc-tflite` 7.3.15 · 纯离线（只编译 + 文件差分，未上板）
票：`06-review-gaps.md` · 规格：[spec.md](../.scratch/compiler-blackbox/spec.md) ·
前序：[03](./compiler-blackbox-03.md)（工具与噪声方法）/ [04](./compiler-blackbox-04.md)（控制变量矩阵）

## 0. 一句话

四道缺口逐条补齐：**dump 文本只能对上「尺寸」不能对上「偏移」**（对不上的三项写清了原因）；
`--gen-debug-info` 写进去的是**可辨认但无官方读取器**的结构（含 ASCII 算子名），`--gen-urate-info` 写进去的是**明文 JSON**；
`mdla-bwhint -d <hint> <dla>` 跑通并给出真实 dump，还做了 encode→dump 往返；
点名的四个变量（`gno` / `--dla-metadata` / `.dlb` 等级 / `--resize`）全部进矩阵：**前两个里只有 metadata 有内容效应，gno 三个值都只多出命令行回显**；
噪声口径改成**每变体各自标定 + 稳定字节口径进工具**，`tools/bbdiff.py` 现在能独立复现 04 §4.2 那个 **1 B @6343** 的信号。

## 1. 怎么跑（可复现）

```bash
# 两个矩阵都用新的口径重跑（03 与 04 各一条命令）
wsl.exe -e bash -c "cd '/mnt/d/deepseek_ workspace/mtk/wt-bb06' && \
  /usr/bin/python3 tools/bbdiff.py --matrix tools/bbdiff-matrix-blackbox03.json"   # 退出码 0
wsl.exe -e bash -c "cd '/mnt/d/deepseek_ workspace/mtk/wt-bb06' && \
  /usr/bin/python3 tools/bbdiff.py --matrix tools/bbdiff-matrix-blackbox04.json"   # 退出码 1（原因同 04 §4.4）

# dump ⇄ 偏移（小项 1）
/usr/bin/python3 tools/bbmap.py --dla ops/_probe/bb06-dump/v0.dla \
    --tflite ops/_probe/bb06-dump/_build/model.tflite \
    --log ops/_probe/bb06-dump/v0.compile.log --json ops/_probe/bb06-dump/v0.regions.json
```

本报告里的所有数字都来自 **2026-11-14T13:55（+08:00）那两次运行**（04 矩阵 13:54:30→13:55:27，03 矩阵 13:55:30→13:55:57），
工具版本就是本提交里的 `tools/bbdiff.py`；重复编译本来就不是逐字节可复现的，重跑会得到同量级但不同的具体值。

产物：`ops/_probe/bb04-fc/`（15 变体 × 9 次编译）、`ops/_probe/bb03-fc/`（6 变体）、
`ops/_probe/bb06-dump/`（小项 1/2 的样本：`dumpmap.txt`、`debug-urate.txt`、`readers.txt`、各自 `.compile.log`）。
`.dla`/`.tflite`/`.dlb` 不入库（`.gitignore`），重跑上面的命令即可重建。

## 2. 小项 1：dump 文本 ⇄ 产物字节偏移（`fc_a` 一个样本，另加三个形状做交叉）

### 2.1 方法与实测

`tools/bbmap.py`（新，只读）：从 `.tflite` 取常量 buffer（跳过 buffer 0 = IO 占位），在 `.dla` 里**精确子串优先、否则字节多重集**定位；
再把编译日志里的 `DRAM Usage` 表按 ncc 的显示精度折回去比。`fc_a` 实测：

| dump 里的项 | 产物里实测 | 判定 |
|---|---|---|
| `Input 64` / `Output 64` | tflite 张量 `1x64` int8 = 64 B；产物里是空 buffer | 尺寸对得上；**dump 不给偏移** |
| `Temp 0` | 无临时缓冲 | — |
| `Static 4.2K` | bias 256 B @**1671..1926** + weights 4096 B @**1951..6046** = **4352 B** → `%.1f` → `4.2K` | **对得上** |
| `Code 400` | 产物里没有 400 B 的区段；同一模型 `--resize 2x64` 后 dump 变 **384**，产物仍是 7633 B | **对不上**（分配量，不是可定位区段） |
| `Total 4.8K` | Input+Output+Temp+Static+Code = **4880 B** → `4.8K` | **对得上** |
| `L1: 0` | `--show-l1-req` 的 help 原文：*Only effective when global buffer allocation is in effect*；单 FC 无全局缓冲分配 | **对不上**（没有 L1 区段；该开关连产物内容都不改，见 §4.2） |
| 执行计划层名 `FullyConnectedLayer<0>{FULLY_CONNECTED}` | 普通产物里搜不到 `FULLY_CONNECTED` / `weights` / `input`（唯一 ASCII 是尾部的命令行） | **对不上**（层名只有 `--gen-debug-info` 才写进产物，§3.2） |

四组形状的闭合（`ops/_probe/bb06-dump/dumpmap.txt` 原文）：

| 变体 | `Static`（dump） | 实测常量块 | `Total`（dump） | 五列之和 | `Code` |
|---|---|---|---|---|---|
| `fc_a`（1×64×64） | 4.2K | 4352 B → 4.2K | 4.8K | 4880 B → 4.8K | 400 |
| `fc_b_n128`（1×64×128） | 8.5K | 8704 B → 8.5K | 9.1K | 9296 B → 9.1K | 400 |
| `fc_c_k128`（1×128×64） | 8.2K | 8448 B → 8.2K | 8.8K | 9040 B → 8.8K | 400 |
| `fc_e_fp16` | 8.2K | 8448 B → 8.2K | 8.9K | 9104 B → 8.9K | 400 |
| `--resize 2x64` | 4.2K | 4352 B → 4.2K | 4.9K | 4992 B → 4.9K | **384** |

### 2.2 三条结论（含一条被自己抓到的假不一致）

1. **能对上的只有尺寸**：`Static` = 常量块字节数、`Total` = 五列之和，五组逐点闭合。
   换算必须按 ncc 的打印精度做（一位小数 KiB、round-half-even：4352 B → `4.2K`）。
   `bbmap.py` 初版把 `4.2K` 反算成 4301 B 去比 4352 B，报「对不上」——**假不一致**，改口径后闭合。
   这条写在这里是因为它正是本小项要防的错：拿一位小数的打印值做整数比较。
2. **dump 不给任何偏移**：`--show-memory-summary=1/2/3/4` 四种模式的输出**逐字节相同**（实测，产物也都 7581 B）；
   `--show-exec-plan` 只给步/层/target。所以「dump 优先」只能到「区段尺寸」这一层，偏移只能从产物量。
3. **不能对上的三项**（`Code` / `L1` / 层名）都给出了原因与推翻实验，没有含糊过去。

**能推翻结论 1 的实验**：找一个 `Static` 与「常量块字节数」不等的模型（例如带 `--extract-static-data`，权重变成输入张量）。
**能推翻结论 2 的实验**：用 `--show-memory-summary=<别的值>` 或更复杂的图（多 step）看是否印出偏移。

## 3. 小项 2：story 14/16 —— `--gen-debug-info` / `--gen-urate-info` 写进去的东西可读吗

### 3.1 先把「体积差」拆对

产物尾部嵌了**完整命令行**（03 §2.1），所以任何开关都会改产物大小。同一固定路径、同一模型、单次编译的对照（`ops/_probe/bb06-dump/`）：

| 变体 | `.dla` B | Δ | 开关文本长度（含分隔空格） | **内容增量** |
|---|---|---|---|---|
| `v0`（基线，`--show-exec-plan --show-memory-summary`） | 7561 | — | — | — |
| `--show-l1-req` | 7575 | +14 | 14 | **0** |
| `--gen-debug-info` | 8117 | +556 | 17 | **+539** |
| `--gen-urate-info` | 7847 | +286 | 17 | **+269** |
| `--dla-metadata note:<file>` | 7689 | +128 | 37 | **+91** |
| `--gno NDF` | 7571 | +10 | 10 | **0** |
| `--gno NDF,SMP,BMP` | 7579 | +18 | 18 | **0** |

-> OP-toolchain 里记的「+556 / +286」里各有 17 B 是**命令行回显**，真正写进去的是 539 / 269 B。

### 3.2 `--gen-debug-info`：可辨认、但**没有官方读取器**

插入段实测：`v0[6711]` 处插入 **539 B**（`debug-urate.txt` 有完整 hexdump）。结构长这样（原文）：

```text
001a37  17 02 00 00 08 00 00 00 ec 15 00 00 00 00 00 00 07 02 00 00 ff 01 00 00 00 00 00 00 fb 01 00 00
001a97  0f 00 00 00 00 00 00 00 46 55 4c 4c 59 5f 43 4f 4e 4e 45 43 54 45 44 08 00 00 00 00 00 00 00 00
         ^^^^^^^^^^^^^^^^^^^^ "FULLY_CONNECTED"（ASCII 明文）
```

- 5 条记录，每条都带 `FULLY_CONNECTED` 明文 + 一组小整数（`S=0x53` / `0x37` / `0x2b` / `0x17` / `0x0f` 反复出现），
  形状像**每层/每张量的调试记录**（vtable 风格的头 + 定长字段）；
- `strings` 直接就能读出算子名 —— **结构可辨认**；但字段语义没有 schema，**没有 host 侧读取器**：
  - `readdlb`（对打包后的 `.dlb`）**完全看不到**这段：三个产物（基线 / debug / urate）的 readdlb 输出**逐字相同**，只报容器级三行状态；
  - `neuronrt -d` 能正常吃这份 `.dla`（离线打印 `Input 64 / Output 64`，没有报错），说明这段不影响运行时契约，但它**只印 I/O，不印调试记录**；
  - 官方定位（`--help` 原文）：*Runtime can work with this info for profiling* —— 读者是运行时，不是 CLI。

分级：**内容存在且部分可读（ASCII 算子名 + 定长数值字段）= 已验证**；**字段语义 = 未知**；**无官方 host 读取器 = 已验证（三条否定证据）**。
推翻实验：拿到 `neuronrt` 的 profiling 输出，或反出这段的 schema；若某条记录能对上一个层的编译期属性，则「语义未知」被推翻。

### 3.3 `--gen-urate-info`：进展比 debug 大 —— 写进去的是**明文 JSON**

插入段实测：142 B @944 + 280 B @5771 + 92 B @6811 + 177 B @7670。其中后两段是 JSON 原文：

```text
5b 7b 22 66 6c 6f 77 73 22 3a 20 5b 7b 22 69 64 65 61 6c 5f 75 72 61 74 65 22 ...
[{"flows": [{"ideal_urate": [{"Conv": "100.00"}]}]}]
...
{"format_version": 1, "graphs": [{"subgraphs": [{"codes": [{"flows": [{"ideal_urate": [{"Conv": "100.00"}]}]}], "target": "MDLA_5_0"}]}]}
```

- 它是 **hint 文件格式（`format_version: 1`）的 JSON**，与 `mdla-bwhint` 的格式同族；`ideal_urate` 三个算子的名字写成了 `"Conv"`（不是 `FullyConnected`）—— 这是编译器内部把 FC 当 Conv 的痕迹，值得记一笔；
- 92 B 那段是「同一份 JSON 的片段」，177 B 那段是完整的 hint 文件；两段都在尾部元数据区（`.dla` 变大 269 B 里 177+92 = 269，正好对上）；
- 分级：**可读（明文 JSON）= 已验证**。

### 3.4 `mdla-bwhint -d`：跑通了，而且做了 encode→dump 往返

正确用法（help 原文 `mdla-bwhint [OPTION...] hint dla`）：**第一个位置参数是 hint 文件，第二个才是 `.dla`**；`-d` 时 hint 是**输出**。

```bash
$ mdla-bwhint -d /tmp/v0.hint ops/_probe/bb06-dump/v0.dla   # rc=0，115 B
{"format_version": 2, "graphs": [{"subgraphs": [{"codes": [{"flows": [{"bw_index": 0}]}], "target": "MDLA_5_0"}]}]}
```

- 基线 / `--gen-debug-info` / `--gen-urate-info` 三个产物 dump 出来的内容**完全相同**（都是 `bw_index: 0`）——
  说明 `-d` 读的是**带宽提示段**，不是 debug/urate 段；
- **往返验证**（这才是「可读」的硬证据）：把 `bw_index` 改成 7 写回、再 dump 回来，读到的就是 7：

```bash
$ echo '{"format_version": 2, "graphs": [{"subgraphs": [{"codes": [{"flows": [{"bw_index": 7}]}], "target": "MDLA_5_0"}]}]}' > /tmp/hint.json
$ mdla-bwhint /tmp/hint.json v0.dla -o /tmp/encoded.dla     # encode rc=0，仍是 7561 B（改的是段内字段，不改大小）
$ mdla-bwhint -d /tmp/back.json /tmp/encoded.dla            # rc=0
{"format_version": 2, "graphs": [{"subgraphs": [{"codes": [{"flows": [{"bw_index": 7}]}], "target": "MDLA_5_0"}]}]}
```

-> `.dla` 里**确实有一段官方可读可写的 JSON 元数据**（bw hint），读写都通。`readdlb` 的退出码是 **134（SIGABRT）**，与 OP-toolchain 记的一致，输出本身完整。

## 4. 小项 3：点名的四个变量进矩阵

### 4.1 `gno`（NDF / SMP / BMP）：**没有内容效应**

`fc_f_gno_ndf` / `_smp` / `_bmp` / `_all` 四个变体（`fc_a` 的模型，只多一个开关），judged by `flag_payload`：

| 变体 | `.dla` B | Δ | `body_size_delta` | `payload_bytes` | 判定 |
|---|---|---|---|---|---|
| `fc_f_gno_ndf` | 7645 | +10 | **0** | 0 | 只多出命令行回显 |
| `fc_f_gno_smp` | 7645 | +10 | **0** | 0 | 同上 |
| `fc_f_gno_bmp` | 7645 | +10 | **0** | 0 | 同上 |
| `fc_f_gno_all` | 7653 | +18 | **0** | 0 | 同上 |

7 个随长度变化的字节（偏移 0 / 7019 / 7312 / 7328 / 7332 / 7340 / 7344）的差值**都恰好等于 size delta**，
说明它们是「文件多大 / 相对文件尾的偏移」这类书签字段（与 04 §4.1 观察到的那类字段同一批），不是内容。
分级：**`gno` 对单算子 FC 的产物内容零影响 = 已验证**。推翻实验：换多算子图（NDF/SMP/BMP 是**图优化**），若仍零影响则「gno 只是默认开」也成立。

### 4.2 `--dla-metadata`：**有内容效应**（这条是对照，证明上一条不是恒绿）

`fc_g_meta`（`--dla-metadata note:{build}/note.txt`，文件内容 `bb06 metadata note\n` = 19 B）：`.dla` 7635 → **7820 B**（+185），
其中命令行回显 +95、**产物本体 +90 B**。90 B 里含 19 B 的 note 原文 + key 名 + 定长头 → 元数据确实被编码进产物。
分级：**已验证（写进产物，90 B）**；**段内布局 = 未知**。

### 4.3 `.dlb` 压缩等级：**level 9 比 level 0 小 26.8%**

新增 `tools/pack.sh`（与 `compile.sh` 同规格：只有一条入口、每次记录 cmd/rc/size/sha，并顺带跑 `readdlb` 把官方读取结果留在 `.pack.log` 里）。
`fc_a` 的同一份 `.dla`（7635 B）：

| 等级 | `.dlb` B | 相对 `.dla` | 相对 level 0 |
|---|---|---|---|
| 0 | **8003** | 1.048× | — |
| 9 | **5858** | 0.767× | **0.732×** |

判定 `case-p-dlb-levels`（要求 ratio ≤ 0.85）**PASS**。另测：level 3 / 5 / 9 在 7561 B 的样本上给出**同一个大小**（5756 B），
说明这档压缩到 level 3 就饱和；**容器层的变量不进 `.dla` 字节差分**，所以它只能到这层（大小 + readdlb 输出）。

### 4.4 `--resize`：**改内容**，且 dump 会跟着变

`fc_h_resize`（`--resize 2x64`）：`.dla` 7635 → **7633 B**（−2 = 命令行 +14、本体 −16），`payload_bytes` 363 → 判定「不是纯开关」**按预期 FAIL**（`expect: fail`）。
配套 dump 读数（同一模型 `--resize 2x64` / `--resize 1x128` 两次都相同）：`Input 128 / Output 128 / Temp 0 / Static 4.2K / Code 384 / Total 4.9K`
—— 形状换到了 128，`Code` 从 400 变 384，`Static` 不变（权重还是 4096 B）。**02 票「编译期换形状」在结构上又对了一次**。

## 5. 小项 4：噪声口径改造（每变体各自标定 + 稳定字节口径进工具）

### 5.1 为什么要改

旧口径 = 所有变体噪声的**并集**当一张全局掩码。它有两个反方向的毛病，本轮都实测到了：

1. **跨变体豁免真信号**：某变体在偏移 X 上抖过一次，其它所有变体的真实变化在 X 上被一起免掉（票面点名的 04 §5.3 的 1 B 就是靠 `allow_outside_bytes: 8` 的余量遮住的）；
2. **反过来也会「兜底」**：单个变体自己 4 次标定没覆盖的抖动，被别的变体的抖动**顺手覆盖**，于是「该红」的项在原口径下变绿。
   本轮开发中就撞到一次：`fc_e_abs2` 的 holdout 有 **21 B** 落在自己噪声之外、但落在全局并集之内，`case-repeat-e2` 于是 union 口径 PASS、per-variant 口径 FAIL。

结论：**噪声是每变体的属性，不是整份实验的属性**。

### 5.2 新口径做了什么（`tools/bbdiff.py`）

| 改动 | 内容 | 缺省行为 |
|---|---|---|
| `noise_scopes` / `default_noise_scope`（矩阵字段） | 判定用 `per-variant`；`union` 仍算，作为报告里的对照列 | 缺省 `per-variant`，两列都输出 |
| `mask_for(a,b,scope)` | `union` = 全局并集（旧）；`per-variant` = **只豁免参与这次差分的两个变体**各自的噪声 | 同左 |
| `stable_offsets(runs)` | 「稳定字节」口径内置：一个变体所有保留编译里取值逐字节一致的偏移 | — |
| `stable_diff`（新 check kind） | 只在两侧都稳定的偏移上取值比较；`expect_offsets` 可直接钉死偏移 | — |
| `split_reorder` | 把「同一组字节换了位置」的大段从信号里剔出去（多重集距离 ≤ max(64, 2%)） | — |
| `constant_layout` / 排布类 | 用 tflite 常量 buffer 在产物里的落址做**可比较的排布签名**；两侧排布类不同则剔除常量区并记 `excluded_area_bytes` | — |
| `flag_payload`（新） | 只加开关的变体：抹掉命令行回显后比 `body_size_delta`，并把「随长度位移的书签字段」单列 | — |
| `pack_size`（新） | `.dlb` 等级压缩比 | — |

### 5.3 顺带发现：**常量块顺序会整块翻转**（同一个模型、同一台机器、连编多次）

`fc_a` 连编 6 次，用同一份 tflite 的常量去定位：

```text
attempt1  weights@1671  bias@5791        attempt2  bias@1671  weights@1951
attempt3  weights@1671  bias@5791        attempt5  bias@1671  weights@1951
attempt4  weights@1671  bias@5791        attempt6  bias@1671  weights@1951
```

**同一份模型、同样的开关，两种落址**（产物都是 7635 B）。03 记的「偶发整块换排布 ≈4.4 KB」在 04 §4.1 被当成「K 决定顺序」的解释，
现在看**至少还有一半是每次编译自己的随机性**：FC-C 的「K 翻转常量块顺序」这条要降级（见 §6 分级表第 6 行）。
本轮实测到一次跨变体排布类不同：`fc_a` 与 `fc_e_out8` 的差分从 106 B 变成 **4458 B**（`ops/_probe/bb04-fc/report-legacy-union-2026-11-14.txt` 里那次运行）。

分级：**「常量块落址在编译之间会翻转」= 已验证**；**翻转的触发条件 = 未知**（推翻实验：固定 `gno`/单核连编 20 次，看是否恒定）。

### 5.4 改口径前后对照（**同一份产物**，两条口径同时算）

这不是两次运行相减 —— 新工具在**一次运行**里把每个 check 按两种口径各算一遍，所以下面每行都是同一批 `.dla`。

| 指标（04 矩阵） | union（旧口径） | per-variant（新口径） |
|---|---|---|
| 噪声字节（并集 / 自身最大） | **217 B** / 92 段 / 1.8% | 最大 **117 B**（每变体 39…117 B） |
| `fc_a` vs `fc_b_n128` 噪声外 | 10101 B | **10158 B** |
| `fc_a` vs `fc_c_k128` 噪声外 | 9826 B | **9885 B** |
| `fc_a` vs `fc_e_fp16` 噪声外 | 9895 B | **9954 B** |
| `fc_a` vs `fc_e_out16` 噪声外 | 1 B | 1 B（两口径一致：真信号不在任何噪声区里） |
| 判定 | 25/29 符合预期（4 红） | **25/29 符合预期（同一组 4 红）**：新增的 4 条稳定字节判定全绿；红的仍是 04 预注册被推翻的两条 + 两条 `noise_share` 口径不适用的。**结论没被口径改变，变的是数字** |

| 指标（03 矩阵） | union | per-variant |
|---|---|---|
| 噪声字节 | **214 B** / 104 段 / 1.8% | 最大 **127 B** |
| `fc_base` vs `fc_n128` 噪声外 | 10121 B | **10163 B** |
| 判定 | 7/7 符合预期 | **7/7 符合预期**（退出码 0） |

**方向是预期的**：新口径豁免得更少 → 噪声字节变小、噪声外字节变大 → 更不容易把真信号当噪声。
**但要说清**：本轮最终这次运行里，两口径对 29 条 check 的**判定完全一致**（都 4 红）。口径差在**数字**上（每个差分被豁免的字节少 50~60 B）。
判定真正翻过来的例子出现在开发中的一次运行（§5.1 第 2 条，`case-repeat-e2`：union PASS / per-variant FAIL，21 B），
以及 04 原报告那次 `fc_e_out8` 的 1 B（靠 `allow_outside_bytes: 8` 的余量遮住）——两者都不是「新口径更宽松」，而是「旧口径更宽松」。
**没有为了让 check 变绿而调阈值**：04 那 4 条红**原样保留**（两条是预注册被推翻：`min_bytes: 8` 的 e1、e2 的「绝对 scale 进产物」；两条是 `max_noise_share` 对 1 B 信号不适用）。

### 5.5 `bbdiff.py` 现在能独立复现 04 §4.2 的 1 B 信号

```text
| case-e1-stable-byte  | stable_diff | **PASS** | {"diff_bytes": 1, "signal_bytes": 1, "expect_offsets": [6343]} |
| case-e1b-stable-byte | stable_diff | **PASS** | {"diff_bytes": 1, "signal_bytes": 1, "expect_offsets": [6343]} |
| case-e2-stable-none  | stable_diff | **PASS** | {"diff_bytes": 0, "signal_bytes": 0} |
| case-d-stable-weight-block | stable_diff | **PASS** | {"diff_bytes": 4080, "signal_bytes": 4080} |
```

矩阵里就是这两行（**数据**，不是代码）：

```json
{"id": "case-e1-stable-byte", "kind": "stable_diff", "a": "fc_a", "b": "fc_e_out16",
 "min_bytes": 1, "expect_offsets": [6343]}
```

`expect_offsets` 是硬钉子：**信号的偏移集合必须恰好等于 {6343}**，多一个字节就判红。04 §4.2 那次是用未入库的临时脚本算的，现在是工具的一条 check。

## 6. 结论分级总表（本票新增/修订）

| # | 结论 | 证据 | 分级 | 能推翻它的实验 |
|---|---|---|---|---|
| 1 | dump 的 `Static` = 常量块字节和、`Total` = 五列之和（按一位小数 KiB 口径） | 五组形状逐点闭合（§2.1） | **已验证** | 用 `--extract-static-data` 编一次（权重变输入张量），若 `Static` 仍等于常量字节和则推翻 |
| 2 | dump **不提供任何偏移**；`--show-memory-summary` 的四种 mode 输出相同 | 实测四种 mode 同字节；`--show-exec-plan` 只有层序 | **已验证** | 换成多 step 图看是否印出偏移；或找到未文档化的 mode |
| 3 | `Code` 是分配量、不是产物区段（400 → `--resize` 后 384） | 产物里无等长区段；两个 resize 变体都 384 | **推测**（机制未知） | 找到一个模型的产物里存在恰好等长的可解释区段 |
| 4 | `--gen-debug-info` 写入 539 B 结构化记录（含 ASCII 算子名），**无官方 host 读取器** | hexdump + readdlb/neuronrt 三条否定证据（§3.2） | 存在与可辨认 **已验证**；字段语义 **未知** | 反出 schema，或拿到 `neuronrt` profiling 输出 |
| 5 | `--gen-urate-info` 写入 269 B，其中 177 B 是完整 **JSON hint 文件**（`format_version: 1`，`ideal_urate`） | 明文 JSON 原文 | **已验证** | 用 `mdla-bwhint -v 1` 编码同一份 JSON 若产物不再变大，则这段就是 hint 段 |
| 6 | `.dla` 有一段**官方可读写**的 JSON 元数据（bw hint，`format_version: 2`） | `mdla-bwhint` encode→dump 往返（bw_index 7） | **已验证** | encode 后 dump 读回别的值即推翻 |
| 7 | `gno`（NDF/SMP/BMP）对单算子 FC 的产物内容**零影响** | 四个变体 `body_size_delta = 0` | **已验证** | 多算子图重做；若仍 0，则「gno 是图级优化」也成立 |
| 8 | `--dla-metadata` 把元数据**写进产物**（+90 B 本体） | `body_size_delta = 90` | **已验证** | 换更长的 note，看 90 是否随之线性增长 |
| 9 | `.dlb` level 9 = level 0 的 **0.732×**，level 3 起饱和 | 8003 → 5858 B；3/5/9 同尺寸 | **已验证**（单样本） | 换更大的 `.dla`（多算子）看饱和点是否移动 |
| 10 | 同一模型的**常量块落址会在编译之间翻转**（bias/weights 互换），产物等长 | 6 次编译两种落址（§5.3） | **已验证** | 固定 `gno`/单核连编 20 次，若恒定则推翻 |
| 11 | 04 的「K 决定常量块顺序」需要**降级**：其中至少一部分是编译间随机性 | 同一 K=64 也能出现两种顺序 | **推测**（两者可能叠加） | 固定 K=64 连编 20 次统计翻转比例，再看 K=128 是否显著更高 |
| 12 | 噪声是**每变体属性**；全局并集既会豁免真信号也会兜底假绿 | 两口径同产物对照（§5.4） | **已验证** | 若某变体的自身噪声集合恒等于全局并集，则「每变体」没有额外信息 |

## 7. 遗留（明确不做/做不了）

- **对齐差分**（04 §5.1 提的）：本轮加了排布类识别与常量区剔除，但**没有做真正的块对齐差分**。改形状的变体（+4352 / +4096）仍然只能回答「有没有变」。
- **`neuronrt` 的 profiling 输出**：需要设备侧，本票纯离线，没做。
- **debug 段的 schema**：只到「结构可辨认」；5 条记录对应什么（层？张量？step？）未判定。
- **`--dla-metadata` 段内布局**：只知道本体 +90 B 且 note 原文在内，键值与长度编码未解。
- **`.dlb` 等级饱和点**：只有两个样本（7561 / 7635 B 的 `.dla`），没扫大产物。
- 上板：全程离线。「产物变了 ⇒ 设备行为也变」依然没有证据。

## 8. 验收口径

- 03 矩阵：退出码 **0**，7/7 符合预期；04 矩阵：退出码 **1**，红的 4 条**与本票前完全一致**（两条预注册被推翻 + 两条口径不适用），**没有新增红**；
- `tools/bbdiff.py` 能独立复现 04 §4.2 的 1 B 信号（`case-e1-stable-byte` / `case-e1b-stable-byte` 的 `expect_offsets: [6343]`）；
- 四个点名的变量都在矩阵里且各有实测判定；`gno` 与 `--dla-metadata` 互为对照（前者零内容、后者有内容）；
- 每条「对不上」都写了原因与推翻实验（§2.1 表、§3.2 三条否定证据）。

