# 黑盒研究 03：差分探查工具 + 四个有牙齿的用例

日期：2026-11-14 · SDK 7.0.8 / `ncc-tflite` 7.3.15 · 纯离线（只编译 + 文件差分，未上板）
票：[03-diff-harness.md](../.scratch/compiler-blackbox/issues/03-diff-harness.md) · 规格：[spec.md](../.scratch/compiler-blackbox/spec.md) ·
上一票：[02](./compiler-blackbox-02.md)

## 0. 一句话

新接缝是**一个**工具 `tools/bbdiff.py` + **一个**数据矩阵 `tools/bbdiff-matrix-blackbox03.json`：跑编译 → 收集产物 →
两次编译一致性检查（差异记为「已知噪声区」）→ 差分 → 出报告。四个用例**全绿**，两组对照**按预期被判红**（证明判定不是恒真）。
顺带勘明三件对后续有用的事：**产物里嵌了绝对路径 + 完整命令行 + 每次编译都不同的 UUID**、**权重在产物里是连续 4096 B 明文块**、
**`--extract-static-data` 的缩量 3822 B ≈ 0.933 × 权重字节**。

## 1. 怎么跑

```bash
# WSL（/usr/bin/python3 里有 numpy/flatbuffers/tflite schema）
wsl.exe -e bash -c "cd '/mnt/d/deepseek_ workspace/mtk/wt-bb03' && \
  /usr/bin/python3 tools/bbdiff.py --matrix tools/bbdiff-matrix-blackbox03.json"
```

退出码：0 = 所有 check 都符合预期（`expect` 默认 `pass`）；1 = 有 check 不符；矩阵里写 `expect: fail` 的对照项，
「被判 FAIL」才是符合预期。常用旋钮：`--only id1,id2`、`--repeats N`、`--out-root`、`--report <path>`。

产物（一个实验一个目录，沿用 `ops/` 的既有证据形态）：

```
ops/_probe/bb03-fc/
  manifest.json          每次编译的 rc/size/sha256、类别分簇、噪声区、zone、全部 check 的原始数字
  noise.txt              已知噪声区的全部区间 + 每变体的噪声字节数
  zone-weight.txt        权重区的全部区间 + 标定来源
  report.md              自动报告（本报告 §3 的表格就是它）
  diff/<a>__vs__<b>.txt  单次差分的区间清单（离噪后）
  <variant>/
    model.json           tflite 的形状/量化/权重来源/字节数
    evidence.txt         9 次尝试的类别与保留情况 + 每次的命令行/rc/sha256
    attempt1..9.compile.log   每次编译的完整日志（失败变体也在这）
    attempt1..9.dla      （不入库，可由同一条命令重建）
```

`.dla` / `.tflite` / `*.bin` / `_build/` 不入库（`.gitignore`），重跑一条命令即可重建；**判定依据（日志、json、diff 区间）入库**。

### 矩阵是数据

| 键 | 含义 |
|---|---|
| `experiment` | 实验名 = 产物目录名（`ops/_probe/<experiment>/`） |
| `repeats` / `class_slack` | 每变体**保留**几次同类别编译 / 额外跑几次做类别校验 |
| `class_threshold_share` | 判定「同一种产物类别」的差分距离上限（占产物比例） |
| `zones.<name>` | `{from, to, gap}`：用一对变体的离噪差分区来标定一个 zone（`gap` = 小间隙合并阈值） |
| `variants[]` | `{id, note, model{op,m,k,n,weights{kind,seed|code}}, flags[]}` —— **加变体只加这一行** |
| `checks[]` | `{id, kind, ...参数, expect}` —— **加判定也只加这一行**（kind 见工具 docstring） |

## 2. 两个环境事实（它们决定了工具的形态）

### 2.1 产物里嵌了路径、命令行、编译期元数据和 UUID

`ops/_probe/bb03-fc/fc_base/attempt1.dla` 尾部原文（可 `strings` 直接看到）：

```text
{"Compiler": "/mnt/d/deepseek_ workspace/mtk/wt-bb03/../neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk/host/bin/ncc-tflite 7.3.15", "Neuron SHA1": "0e64467e79"}
--arch mdla5.0 --opt 3 --disallow-bridge --num-mdla 2 --l1-size-kb 3072 --show-exec-plan --show-memory-summary  --model /mnt/d/.../wt-bb03/ops/_probe/bb03-fc/_build/model.tflite --dla-file /mnt/d/.../wt-bb03/ops/_probe/bb03-fc/_build/model.dla
```

另有一段 36 字符的 UUID，**每次编译都不同**（同一份模型连编两次：`468e16cd-4f45-4a69-b5fa-95e99bf6efe4` / `53421d7e-22bd-45f4-8767-ae94f713d48c`）。

后果：**同一个模型从不同路径编译，产物大小都不一样**（实测 7526 / 7527 / 7635 B 三个值，只差输入路径长度与描述串）。
所以工具让所有变体**共用一条固定编译路径** `<exp>/_build/model.tflite` → `<exp>/_build/model.dla`，
跨变体差分里就不会混进「路径字符串变了」这种假信号。本报告一律**不跨报告比绝对尺寸**（02 票的 7545 B 与本表 7635 B 不可直接相减）。

### 2.2 重复编译不是逐字节可复现的（所以「噪声区」不是可选项）

同一份输入连编多次：账面上只有 UUID + 头部几个索引字节在动 —— 噪声区 **214 B / 107 段，占产物 1.8%**
（`noise.txt`，逐段列出）。另有一个低频现象：偶尔整块换一种排布，两份产物差 **≈4.4 KB**（占一半）。
把那种差异当噪声，噪声区会被撑到半个文件、后面所有差分都失去意义 —— 所以工具**先按差分距离分簇**（并查集），
只保留主类别的 `repeats` 次，跨类的尝试单列进 `evidence.txt`（本次每变体 9 次尝试里 4 次落在另一类）。
主类别内部再取「前 `repeats-1` 次」定噪声区，**最后一次留作 holdout**（否则「除噪声区外相同」是同义反复）。

## 3. 四个用例（实际输出）

被测矩阵：FC(1,64,64) int8 常量权重为基准（与票 01/02 同一份夹具的同一量化档），每次只动一个维度。
每变体保留 5 次同类别编译（共 9 次尝试）。自动报告原文：`ops/_probe/bb03-fc/report.md`。

| 变体 | tflite B | 权重 B | .dla B（保留各次） | rc | 尝试/保留/丢弃 |
|---|---|---|---|---|---|
| `fc_base`（基准） | 5104 | 4096 | 7635 ×5 | 0 | 9/5/4 |
| `fc_w_cal`（只换权重，圈定权重区） | 5104 | 4096 | 7635 ×5 | 0 | 9/5/4 |
| `fc_w_swap`（只换权重，用来验 zone） | 5104 | 4096 | 7635 ×5 | 0 | 9/5/4 |
| `fc_n128`（只改 N） | 9456 | 8192 | 11987 ×5 | 0 | 9/5/4 |
| `fc_extract`（+`--extract-static-data`） | 5104 | 4096 | 3813 ×5 | 0 | 9/5/4 |
| `fc_k8_rejected`（故意失败：K=8） | 1520 | 512 | – | **1** | 9/0/0 |

| check | 结果 | 关键数字 |
|---|---|---|
| `case1-n-changed` | **PASS** | 差分 10251 B；噪声外 10117 B（98.7%），噪声内 134 B（1.3%） |
| `case2-repeatable` | **PASS** | holdout 第 5 次 vs 第 1 次：差 34 B，**噪声外 0 B** |
| `case2b-noise-budget` | **PASS** | 噪声区 214 B = 产物的 1.8%（上限 5%） |
| `case3-weights-confined` | **PASS** | 权重区 **4096 B**；差分 4181 B，噪声外 4041 B，**逃出权重区 0 B** |
| `case4-extract-shrinks` | **PASS** | 7635 → 3813 B，缩 **3822 B = 0.933 × 4096**（窗口 0.5×~1.5×） |
| `control-same-input-no-change` | **PASS（按预期被红）** | 同一份产物差分 0 B → 「该变」判据不成立（证明判定不是恒真） |
| `control-shape-escapes-weight-zone` | **PASS（按预期被红）** | 改 N 的差分逃出权重区 6072 B → 「收敛」判据不成立 |

### ① 只改 N（`fc_base` → `fc_n128`）

```text
size: 7635 -> 11987 (delta +4352)        # +4352 B = 权重 4096 + bias 256，正好是 N 翻倍的常量增长
diff bytes=10251   inside noise=134   outside noise=10117
```

判定：**有变化，且变化不能被噪声解释**（噪声内占比 1.3% ≤ 10%，噪声外 10117 B ≥ 1% 产物）。

**诚实标注**：严格意义的「变化区与噪声区完全不相交」在**尺寸变化**的变体上做不到 ——
噪声区就是产物头部的几个索引字节和尾部那段 UUID，而改 N 会合法地改到同一批位置（头部的长度/索引字段）。
工具因此把判据写成「噪声解释不了这些变化」，并把**噪声内/外字节数都打印出来**；真要「零重叠」，只能比较
**同尺寸**的两个变体（③ 就是这种）。

### ② 形状与权重都不变（`fc_base` 内部：第 5 次 vs 第 1 次）

```text
diff bytes=34   inside noise=34   outside noise=0
```

判定：**除噪声区外逐字节相同** —— 噪声区外的 0 B 差异，而第 5 次编译没有参与噪声区标定（holdout）。
配套 `case2b` 把噪声区本身钉在 ≤5%（实测 1.8%）：如果哪天编译器的不确定性涨到吃掉半份产物，这条会先红。

### ③ 只换权重（`fc_base` → `fc_w_swap`；zone 由**另一对**权重 `fc_w_cal` 标定）

```text
zone weight: 4096 B, runs=[[1951, 6046]]            # 恰好等于权重字节数 4096，1.0000x
fc_base vs fc_w_swap: diff bytes=4181  outside noise=4041  escaped zone=0
```

判定：**变化被限制在「权重区」内**，且权重区恰好 4096 B —— 权重在产物里是**连续、原样、每字节只出现一次**的一块，
既没被压缩也没被重排（这条对常量区/L4 那条线直接有用：权重是明文可定位的）。

*注*：zone 用 `gap=64` 把「两个不同权重恰好同值」的孤立字节并进区间 —— 否则区间会被这些巧合切成 29 段；
并进后**总长仍然精确等于 4096 B**，说明这不是橡皮筋。

### ④ `--extract-static-data`（`fc_base` → `fc_extract`）

```text
size: 7635 -> 3813 (delta 3822)   ratio 0.933 x weight_bytes(4096)
side file static.bin = 262178 B
```

判定：**产物显著缩小，缩量与权重字节同量级**（0.933×，窗口 0.5×~1.5×）。
与 02 票（7545 → 3666，−3879 B）同一量级、同一结论；侧文件 262,178 B 也与 02 票完全一致 —— 那个「侧文件远大于权重」的未解项仍在。

### 失败实验也留证据（`fc_k8_rejected`）

K=8 违反 16 B 行规则，被 `--disallow-bridge` 拒绝（`ops/_probe/bb03-fc/fc_k8_rejected/attempt1.compile.log` 原文）：

```text
Bridging is disallowed but gap is detected between Input and MDLA_5_0
ERROR: Fail to compile /mnt/d/.../ops/_probe/bb03-fc/_build/model.tflite
EXIT=1
SIZE=0
```

9 次尝试全部 rc=1，全部日志留在目录里；`manifest.json` 里 `ok:false`、`error:"attempt 1 没有产出 .dla（rc=1）"`。
**没有一个 check 引用它**，所以它不会把整轮判红 —— 失败实验是证据，不是判据。

## 4. 附带结论（本票顺带勘明，供 04 票直接用）

| # | 结论 | 证据 | 分级 |
|---|---|---|---|
| 1 | 权重在产物里是连续 4096 B 明文块（未压缩、未重排） | 权重区恰好 4096 B；只换权重时离噪差分 4041 B 全在区内 | 已验证（本票对照实验） |
| 2 | 产物尾部带「编译器绝对路径 + 版本 + Neuron SHA1 + 完整命令行」+ 每次不同的 UUID | `strings attempt1.dla`、两次编译 UUID 不同 | 已验证（原文在 §2.1） |
| 3 | `--extract-static-data` 缩量 0.933×权重字节，侧文件 262,178 B | §3 ④；与 02 票一致 | 已验证（两票一致） |
| 4 | 改 N 时产物增长 = 权重 +4096 B、bias +256 B（正比，无额外填充） | 7635 → 11987（+4352） | 已验证 |
| 5 | 噪声区（重复编译差异）集中在产物头部索引字节 + 尾部 UUID，占 1.8% | `noise.txt` 107 段 | 已验证 |
| 6 | 偶尔整块换排布（跨类差 ≈4.4 KB，同尺寸） | `evidence.txt` 的类别分簇（9 次尝试里 4 次） | **推测**：机制未知；推翻它的实验 = 固定 `--opt`/`gno`/单核再编 20 次，看跨类是否消失 |

## 5. 未做 / 留给下一票

- **板端**：本票全程离线（编译 + 文件差分）。本工具**没有上板路径**，任何「产物变了 ⇒ 设备行为也变」的推断都还没证据。
  `--extract-static-data` 的数值等价 02 票已用 `neuronrt` 验过，但那不是本工具做的。**上板验证留给下一票**。
- 矩阵目前只覆盖 **int8 FC 常量权重**：同形状换**量化配置/精度档**、`.dlb` 打包、`--resize`/`--runtime-dynamic-shape`
  都还没进矩阵（加一行 variant 即可，但模型 builder 要先支持对应 dtype/开关）。
- 噪声区里那几处**头部索引字节**（偏移 152/200/248/296…，步长 48）语义未知；整块换类的机制未知。
- 产物尺寸仍随权重内容小幅变化（早期旁测 ±48 B，本次矩阵内同尺寸未复现）—— 未定性。

## 6. 复现与验收

```bash
wsl.exe -e bash -c "cd '/mnt/d/deepseek_ workspace/mtk/wt-bb03' && \
  /usr/bin/python3 tools/bbdiff.py --matrix tools/bbdiff-matrix-blackbox03.json"   # 退出码 0 = 七个 check 都符合预期
cat ops/_probe/bb03-fc/report.md ops/_probe/bb03-fc/noise.txt
```

判据（也是本票的验收口径）：四条 case 全 PASS、两条 control 按预期被红、`manifest.json` 里 `ok:true`、退出码 0。
