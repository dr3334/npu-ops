# Q02 · int8 golden 修到逐位（两次舍入 + 整乘子 M0）

时间：2026-11-13 · ncc-tflite 7.3.15 / mdla5.0 / MT6897 · 分支 `impl/q02-int8-bitexact`。
本页是票 02 的证据记录；量化契约本身见 `docs/OP-MATRIX.md` §3 与 `docs/OP-convpool.md` §5（CP-05）。

## 1. 结论

`int8_tol = 1` 对 FC / BMM 的两个码档**都不再需要**。原因是两处实现债，都已还清：

| 债 | 表现 | 现在 |
|---|---|---|
| (a) golden 用 float64 单次 `np.round(acc·ratio)` | 半 LSB 边界上与设备差 1 码 | 按契约两次整数舍入（`tools/quant_contract.py`） |
| (b) M0 从 float32 scale 乘积反算（= 1073741840） | 再多 0.03–0.2% 的 ±1 | M0 取声明乘子的精确有理数（1 → 2³⁰ / shift +1，1/32 → 2³⁰ / shift −4） |

同一批板端 dump、同一批 INT64 累加器，四种规则的失配数（**35 个 int8 FC/BMM op、145 296 个值**）：

| 规则 | 失配 |
|---|---|
| A 单次舍入，ratio 取 float32 scale 乘积（0.031250000345…） | **55** |
| D 单次舍入，ratio 取精确 1/32（= 仓库此前真正发出去的 golden） | **127** |
| C 两次舍入，但 M0 从 float32 scale 乘积反算（1073741840） | **69** |
| **B 两次舍入 + 整乘子 M0=2³⁰（= 现在的 golden）** | **0** |

D` 列在每个 op 上都与**修复前** commit 的 `golden/expected.bin` 逐位相同（表格末列 `D=gold` 全 0；
重放是在主检出 `npu-ops` 上跑的，那里还是旧 golden），所以上表不是"另一套推导"，
而是**同一份 golden 的四种写法**。反过来在自己的 worktree 里跑同一张表，`D=gold` 列会正好
等于 `D` 列对设备的失配数 —— 因为新 golden 就是 B，B 与 D 只在 D 错的地方不同。

## 2. 做了什么

1. `tools/quant_contract.py`（新）：契约的唯一实现
   ```
   out = RoundingDivideByPOT( round_half_up(acc·M0 / 2^31), shift )
       = TFLite MultiplyByQuantizedMultiplier(acc, M0, shift)
   ```
   `shift` 用 TFLite 的符号约定（+ 为高乘之前的左移、− 为之后的右移），所以乘子恰为 1 时是**恒等**，
   `codes=small` 档的 golden 因此逐字节不变。ratio 经 `Fraction.limit_denominator()` 吸附到短有理数，
   这样 builder 手里那个 float64 的 `b_scale/o_scale`（= 1/32·(1+1.1e-8)）也能落到精确的 2³⁰。
2. 三个 builder 换实现（golden 侧，序列化侧一行未动）：
   `tools/build_ops.py`（FC）、`tools/build_bmm.py`（BATCH_MATMUL 与 FC-form）、`tools/build_dynb.py`（动态权重 FC）。
   `int8_tol` 的默认值随之从 1 收到 0。
3. `tools/verify_int8_requant.py`（新，不占板端）：读 `model.tflite` 的 operator / shape / scale，
   用 `golden/input*.bin` 与内嵌权重重建 INT64 累加器，对 `out_device.bin` 同时算上表四种规则，
   并报告 `expected.bin` 是否等于 D。零板端周期、零写入。
4. 17 个受影响的 op 目录：`model.json` 的 `int8_tol` 1 → 0，`compare.txt` 重算。

## 3. 怎么验的

### 3.1 离线重放（零板端）

```bash
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/npu-ops' && \
  /usr/bin/python3 '/mnt/d/deepseek_ workspace/mtk/wt-q02/tools/verify_int8_requant.py' --all-int8"
# ---- ops=35 values=145296  A=55  D=127  B=0  C=69
```

重放之前先证明 **dump 属于这个模型**：用打过补丁的 builder 重新生成 31 个 op 的 `model.tflite`，
与记录 dump 时那份**逐字节相同**（30 个直接命中；`fc_dynb_1x64x64_int8` 要用它自己的
`build_dynb.py` 重建，也一样相同）。所以"旧 dump 也能当新证据"，本票**没有为规则修复花一分钟板时**。

### 3.2 板端复核（新跑，持锁，前后各一次 healthcheck）

| op | 旧 golden | 新 golden | 编译 | 板端 |
|---|---|---|---|---|
| `fc_64x1024x1024_int8` | 41/65536 | **0/65536** | exit=0 / 1 层 / 1 步 / 1 055 874 B | 0.625 ms/inf（-c 3） |
| `bmm_pv_t16_int8_full` | 6/1024 | **0/1024** | exit=0 / 3 084 B | 0.650 ms/inf |
| `bmm_qk_t16_int8_full` | 1/256 | **0/256** | exit=0 / 3 084 B | 0.645 ms/inf |

三次 `HEALTHCHECK: PASS`（跑前、两次之间、跑后）。全部 `exact: YES mismatched=0`、
`max_abs_diff=0`、`int8_tol=0`。

### 3.3 回归面

- **fp16/fp32 golden 一位没动**：10 个 `*_fp16`/`*_fp32` spec 重建后 `model.tflite` 与
  `golden/expected.bin` 都与仓库里的逐字节相同。
- **23 个 `tools/build_*.py` 全部仍可 import**（`--list` 冒烟）。
- 乘子为 1 的 18 个 op（`codes=small`）：新旧 golden 逐字节相同（恒等乘子），`int8_tol` 本来就是 0。

## 4. 受影响 op 清单（27 个 `int8_tol: 1`，见 §5 未覆盖）

本票处理并验证的 17 个（13 FC + 3 BMM full + 1 dynB，132 688 个值，`compare.txt` 全部 `0/N` / PASS）：

| op | 旧 `compare.txt` | 新 |
|---|---|---|
| fc_128x128x128_int8 | 24/16384 | 0/16384 |
| fc_1x128x128_int8 | 0/128 | 0/128 |
| fc_1x128x21704_int8 | 35/21704 | 0/21704 |
| fc_1x16x64_int8 | 0/64 | 0/64 |
| fc_1x4096x4096_int8 | 0/4096 | 0/4096 |
| fc_1x64x128_int8 | 0/128 | 0/128 |
| fc_1x64x64_int8 | 0/64 | 0/64 |
| fc_1x64x8_int8 | 0/8 | 0/8 |
| fc_1x8192x8192_int8 | 2/8192 | 0/8192 |
| fc_64x1024x1024_int8 | 41/65536 | 0/65536 |
| fc_8x1024x1024_int8 | 5/8192 | 0/8192 |
| fc_8x128x128_int8 | 4/1024 | 0/1024 |
| fc_8x128x696_int8 | 7/5568 | 0/5568 |
| bmm_pv_t16_int8_full | 6/1024 | 0/1024 |
| bmm_qk_t16_int8_full | 1/256 | 0/256 |
| fc_bmm_qk_t16_int8_full | 1/256 | 0/256 |
| fc_dynb_1x64x64_int8 | 1/64 | 0/64 |
| **合计** | **127 / 132 688** | **0 / 132 688** |

这 17 个里 3 个（`fc_64x1024x1024_int8`、`bmm_pv_t16_int8_full`、`bmm_qk_t16_int8_full`）是**新跑的板端**，
其余 14 个的 `compare.txt` 是从记录 dump 重算的：`e2e.json` 一律没动（它记的是板端那一次），
只有 `compare.txt`（记的是对拍）变了；dump 与模型的对应关系由 §3.1 的字节相同保证。

## 5. 仍需 ±1（或未验证）的例外

| 类别 | op | 为什么 |
|---|---|---|
| conv，旧 rational golden | `conv_1x8x8x4_f3_full_int8`、`_relu6_int8`、`_div3_int8` | 这三份 golden 是**被证伪前**的写法，历史失败记录保留；同形状的 `*_devrule_int8` 已是 0/512 且 `int8_tol=0`。要收口就重跑这三份（可用已有 dump，0 板端），或直接以 `_devrule` 版本为准 |
| 另一个单元（不是本契约） | `avgpool_1x8x8x4_*`（3）、`mean_4x16_int8` | AVG_POOL 用单次 round-half-up（`build_convpool.avg_divice`），MEAN 同理；不属于 `MultiplyByQuantizedMultiplier` |
| 无 dump，无法验证 | `fc_dynb_1x1024x1024_int8`、`fc_dynb_1x512x512_int8`、`_fc_ref_1x64x64` | 板端从来没跑过这三份 |

契约**没被证据覆盖**的三件事（维持 `int8_tol ≥ 1` 的默认，遇到就要重新出证据）：

1. **per-channel / per-axis scale**：所有实测都是张量级单乘子；
2. **非对称 zero point**（`zero_point ≠ 0`）：所有实测 zp 都是 0，acc 里没有 zp 补偿项；
3. **int16 累加器**：int64 累加在 35 个 op 上 0 失配，但 MDLA 实际位宽可能小于 int64，
   大 K 时若溢出/饱和，这条规则解释不了。

另外一个**新证据**值得记进 CP-05：ratio 恰为 1/32（短二进制）时，设备用的 M0 是理想值 2³⁰，
**不是**从 float32 scale 乘积反算出来的 1073741840（后者 69 个失配）。也就是说 ncc 的乘子
归一化在"活代数可精确表示"时是理想的；而 CP-05 里留的 UNKNOWN（ratio 不能被精确表示时 ncc 挑哪个 M0）
依然成立。

## 6. 剩余工作量（若要全量收口）

| 项 | op 数 | 板端 |
|---|---|---|
| 上面 17 个 | 0（已完成） | 3 个新跑（已跑），其余用已有 dump |
| conv 旧 rational 3 个 | 3 | **0**：重算 golden + 复用 dump，约 3 分钟 CPU |
| avgpool 3 + mean 1 | 4 | 0，但要先定它们的规则（已是 `avg_divice`） |
| 无 dump 的 3 个 | 3 | 每个一次 e2e，约 1–2 分钟/个 |
| 若**不**信任历史 dump（要求全部重跑板端） | 20 | 20 × 1–3 分钟 ≈ **40–60 分钟板时** |

## 7. 复现

```bash
# 1) 离线重放（零板端）：旧/新/反算 四种规则对 35 个 int8 FC/BMM op 的板端 dump
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/npu-ops' && \
  /usr/bin/python3 '/mnt/d/deepseek_ workspace/mtk/wt-q02/tools/verify_int8_requant.py' --all-int8"

# 2) 重算某个 op 的 compare.txt（复用已有 dump，不占板端）
wsl.exe -e bash -lc "cd '/mnt/d/deepseek_ workspace/mtk/wt-q02' && \
  /usr/bin/python3 tools/compare.py ops/fc_64x1024x1024_int8 > ops/fc_64x1024x1024_int8/compare.txt"

# 3) 板端（持锁）
cd "/d/deepseek_ workspace/mtk/wt-q02"
export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
./tools/devlock.sh run -- ./tools/healthcheck.sh --quick
./tools/devlock.sh run -- ./tools/e2e.sh ops/fc_64x1024x1024_int8
./tools/devlock.sh run -- ./tools/healthcheck.sh --quick
```
