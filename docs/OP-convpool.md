# OP-convpool —— CONV_2D / DEPTHWISE_CONV_2D / MAX_POOL_2D / AVERAGE_POOL_2D on MDLA 5.0

> Scope: int8 NHWC, constant filters, single-op TFLite models compiled with the fixed policy
> (`--arch=mdla5.0 --opt=3 --disallow-bridge --num-mdla=2 --l1-size-kb=3072`).
> Owner: conv-agent (worktree `wt-convpool`, branch `feat/convpool`).
> Local experiment ids `CP-01 … CP-12` are used below; the Lead can renumber them into
> `docs/EXPERIMENTS.md` (that file is outside this scope).

## 0. Evidence chain (R1)

| item | value |
|---|---|
| device | `mt6897/16`, adb serial `0123456789ABCDEF`, neuronrt hw mode |
| compiler | ncc-tflite **7.3.15**, NeuroPilot 7.0.8 SDK |
| base commit of this worktree | `bdd7668` (feat/convpool) + this branch's commits |
| model builder | `tools/build_convpool.py` (specs listed by `--list`, seed `20261112`) |
| golden #1 | vectorised numpy int64 accumulator + explicit SAME/VALID padding rule |
| golden #2 | `tools/verify_convpool.py` — plain nested-loop Python, exact `Fraction` quantisation |
| device runs | `NPU_LOCK=/d/deepseek_ workspace/mtk/.npu-board.lock ./tools/devlock.sh run -- ./tools/e2e.sh ops/<name>` |
| `.dla` sha256 | in each `ops/<name>/model.compile.log` — **identity only**, not a reproducibility check (per AGENTS.md §4: `.dla` is not byte-reproducible, two compiles of the same model differ) |
| outputs compared | **17 168** device values across 21 conv/pool ops (+64 for the FULLY_CONNECTED reference, §7) |

Every model input is a runtime tensor with an empty data buffer (AGENTS.md §7.3: an input
with data would be constant-folded into `IdentityLayer on NVM` and the op would not run).
Filter and bias are constants inside the `.tflite`; they are also exported to
`ops/<name>/golden/filter.bin|bias.bin` only so the second golden can recompute them — the
device never receives them as inputs (`e2e.sh` pushes `golden/input.bin` only).

## 1. Method: three quantisation tiers, and why the third one exists

int8 requant is where "the operator is supported" quietly turns into "the operator is off
by a few LSB". To separate *layout/indexing* bugs from *rounding* bugs, every shape was run
in more than one tier (`model.json:"codes"`):

| tier | activations / weights | output scale | what it isolates | declared tol |
|---|---|---|---|---|
| `small` | codes ∈ {−1,0,1}, zero bias | `in·w` (multiplier exactly 1) | pure layout, stride, padding, depthwise mapping | 0 |
| `pow2` | full int8 codes, random bias | `in·w·2^k`, k picked from the actual accumulator range so **nothing saturates** | real int8 dynamic range, no rounding at all | 0 |
| `div` | codes ∈ [−3,3] | `in·w·d`, d ∈ {3,5,6} | the requant **rounding rule** | 0 (see §5) |

`pow2` deliberately avoids saturation: with the "standard" `N`-independent scale
(`out = in·w·32`, as used by the FC probes) a 3×3×4 int8 conv saturates most of its outputs
and the comparison stops being informative.

## 2. Results summary (all numbers measured in one session)

`mem` columns are the `DRAM Usage` row of `model.compile.log`:
Input | Output | Temp | Static | Code | Total (bytes).

| op dir | compile | emitted layers (MDLA step) | mem (in/out/temp/static/code/total) | device run | numerics |
|---|---|---|---|---|---|
| `conv_1x8x8x4_f3_int8` (small) | ok | Conv2DLayer | 256/512/0/768/368/1.9K | ok | **0/512 exact** |
| `conv_1x8x8x4_f3_full_int8` (rational golden) | ok | Conv2DLayer | 256/512/0/800/400/1.9K | ok | 2/512 off-by-1 → **FAIL vs tol 0** (kept, §5) |
| `conv_1x8x8x4_f3_full_devrule_int8` | ok | Conv2DLayer | 256/512/0/800/400/1.9K | ok | **0/512 exact** |
| `conv_1x8x8x4_f1x1_int8` | ok | Conv2DLayer | 256/512/0/288/400/1.4K | ok | **0/512 exact** |
| `conv_1x8x8x4_f3_relu6_int8` (rational) | ok | Conv2DLayer, **ReLU6Layer** | 256/512/0/800/512/2.0K | ok | 1/512 → FAIL (kept) |
| `conv_1x8x8x4_f3_relu6_devrule_int8` | ok | Conv2DLayer, ReLU6Layer | 256/512/0/800/512/2.0K | ok | **0/512 exact** |
| `conv_1x8x8x4_f3_div3_int8` (rational) | ok | Conv2DLayer | 256/512/0/768/368/1.9K | ok | 167/512 off-by-1, tol 1 → PASS |
| `conv_1x8x8x4_f3_div3/div5/div6_devrule_int8` | ok | Conv2DLayer | same as above | ok | **0/512 exact** each |
| `conv_1x16x16x16_f3s2v_int8` (stride 2, VALID) | ok | Conv2DLayer | 4.0K/784/0/2.3K/400/7.5K | ok | **0/784 exact** |
| `conv_1x8x8x16_f3d2_int8` (dilation 2) | ok | Conv2DLayer | 1.0K/1.0K/0/2.3K/384/4.7K | ok | **0/1024 exact** |
| `conv_1x16x16x8_f3s2v_int8` (C=8) | **FAIL** | Conv2DLayer *(then aborted)* | — | — | compile-time, §4 |
| `conv_1x8x8x4_f3d2_int8` (C=4) | **FAIL** | Conv2DLayer *(then aborted)* | — | — | compile-time, §4 |
| `dwconv_1x8x8x4_f3_int8` (mult 1) | ok | DepthwiseConv2DLayer | 256/256/0/784/400/1.7K | ok | **0/256 exact** |
| `dwconv_1x8x8x16_f3m2_int8` (mult 2) | ok | DepthwiseConv2DLayer | 1.0K/2.0K/0/1.4K/400/4.8K | ok | **0/2048 exact** |
| `dwconv_1x8x8x4_f3m2_int8` (C=4) | **FAIL** | DepthwiseConv2DLayer *(then aborted)* | — | — | compile-time, §4 |
| `maxpool_1x8x8x16_2x2s2_int8` | ok | MaxPool2DLayer | 1.0K/256/0/**0**/272/1.5K | ok | **0/256 exact** |
| `maxpool_1x8x8x16_3x3s1_same_int8` | ok | MaxPool2DLayer | 1.0K/1.0K/0/**0**/272/2.3K | ok | **0/1024 exact** (SAME pad = −128 never wins) |
| `avgpool_1x8x8x16_2x2s2_int8` | ok | AvgPool2DLayer | 1.0K/256/0/0/272/1.5K | ok | **0/256 exact** |
| `avgpool_1x8x8x16_2x2s2_same_int8` | ok | AvgPool2DLayer | 1.0K/256/0/0/272/1.5K | ok | **0/256 exact** |
| `avgpool_1x8x8x16_3x3s1_same_int8` | ok | AvgPool2DLayer | 1.0K/1.0K/0/0/272/2.3K | ok | **0/1024 exact** (counts {4,6,9}) |
| `avgpool_1x8x8x16_2x2s1_same_int8` | ok | AvgPool2DLayer | 1.0K/1.0K/0/0/272/2.3K | ok | **0/1024 exact** (counts {4,2,1}, out-of-sample) |
| `maxpool_1x8x8x4_*`, `avgpool_1x8x8x4_*` (C=4, 6 ops) | **FAIL** | MaxPool2DLayer / AvgPool2DLayer *(then aborted)* | — | — | compile-time, §4 |
| `conv_matmul_1x64x64_int8` (bonus) | ok | Conv2DLayer, Conv2DLayer, ConcatLayer | 4.0K/4.0K/0/4.2K/800/13K | ok | **0/4096 exact** |

Every compile log also shows `===Graph[0] has 1 Execution Step(s)=== … Target: MDLA_5_0`,
i.e. **no CPU/Neuron fallback**: the whole op set is one MDLA execution step (R6).
`--disallow-bridge` was on for all of them.

Pools have `Static = 0` and `Code = 272` — no weights, smaller footprint than any conv
(conv `Static` 288–4.2K depending on filter size, `Code` 368–800).

## 3. Per operator

### 3.1 CONV_2D — CP-01

**Hypothesis.** MDLA is a conv engine, so a NHWC int8 `CONV_2D` with constant OHWI filter,
bias, SAME/VALID padding, stride and dilation maps 1:1 onto one `Conv2DLayer`.

**Prediction (written before compiling).** Single `Conv2DLayer` on MDLA_5_0, no relayout,
bit-exact in the `small` tier; in the full-range tier bit-exact *if* the requant is a plain
round-half-away of `acc·scale_ratio` (prediction CP-01b, later **falsified**, §5).

**Method.** input [1,8,8,4] / filter [8,3,3,4] / bias [8] → [1,8,8,8]; then variants for
1×1, stride 2 + VALID (C=16), dilation 2 (C=16), fused RELU6, and four requant multipliers.

**Result.** `Conv2DLayer` alone (or `Conv2DLayer + ReLU6Layer` for RELU6), one MDLA step.
Bit-exact everywhere once the golden uses the device requant model (§5): 512+512+512+512+512
+784+1024 = 4 368 values, `max_abs_diff = 0`. Memory 1.4K–7.5K total for 8×8×4 … 16×16×16.

**Applicability.** int8 NHWC conv with constant filter, SAME/VALID, stride 1/2, dilation
1/2, k=1/3, C=4/16/64, OC=4…64; multipliers 1, 1/2^k, 1/3, 1/5, 1/6; fused NONE/RELU6.
Out of evidence: float/fp16 conv, per-channel weight scales, C=8 (does not compile here),
very large tensors (L1 3072 KB limit not probed), asymmetric zero points.

**Open questions.** (a) Is the `ReLU6Layer` a real second hardware pass or just a codegen
artifact of one kernel? Its `Code` grows 400→512 B and `Static` stays 800 B, and no Temp
buffer is added — consistent with activation-in-kernel, but not proven. (b) MDLA tile/CBUF
limits for large H,W,C. (c) Why C=8 needs a relayout while C=4 does not (§4).

### 3.2 DEPTHWISE_CONV_2D — CP-02

**Hypothesis.** TFLite keeps the depthwise filter transposed (`[1,FH,FW,C*mult]`); MDLA has
a dedicated depthwise path, so `DepthwiseConv2DLayer` handles multiplier 1 and 2 natively.

**Prediction.** One `DepthwiseConv2DLayer`, bit-exact after applying the device requant;
the `mult=2` case maps output channel `oc` to input channel `oc // mult`.

**Method.** [1,8,8,4] filter [1,3,3,4] (mult 1) and [1,8,8,16] filter [1,3,3,32] (mult 2).

**Result.** Single `DepthwiseConv2DLayer`, **0/256** and **0/2048** mismatches. Memory
1.7K / 4.8K total. The `mult=2` output is 32 channels and was *not* split, unlike
`conv C=32` which was split in two — depthwise does not hit the channel-split path.

**Applicability.** int8, mult ∈ {1,2}, k=3, s=1, SAME, C=4/16. Out of evidence: mult>2,
stride 2, dilation, fp16.

**Open questions.** Does `mult>2` also stay single-layer? Does depthwise stride 2 need the
eDPA relayout like regular conv at small C?

### 3.3 MAX_POOL_2D — CP-03

**Hypothesis.** Max pooling is a compare/select, so with `--disallow-bridge` it is
order-independent and must be bit-exact, *provided* the SAME-padding fill is the int8
minimum (−128 s.t. it can never win the max).

**Prediction.** Single `MaxPool2DLayer`, bit-exact for both VALID and SAME, i.e. SAME and
VALID outputs identical for shapes where the tiling is the same.

**Method.** [1,8,8,16], 2×2/s2 VALID and 3×3/s1 SAME (the latter has real padding at the
border), input codes full int8 range.

**Result.** Single `MaxPool2DLayer`, **0/256** and **0/1024** mismatches; `Static = 0`.
The 3×3 SAME case confirms the pad value is the dtype minimum, not 0 (a 0-pad would have
produced 0-dominated borders and a large mismatch count).

**Applicability.** int8, k=2 s=2 VALID, k=3 s=1 SAME, C=16. Not evidenced: C=4/8 (compile
boundary, §4), C≥32 (extra Slice/Concat layers, untested on device), global pooling,
dilation, fp16.

**Open questions.** Should `--disallow-bridge` even reject a relayout DMA? See §4.

### 3.4 AVERAGE_POOL_2D — CP-04

**Hypothesis (two competing ones).**
*H1*: the divisor is the SAME-padding-included filter area (`k²`), as in "pad with zeros".
*H2*: the divisor is the number of **valid** elements (TFLite semantics), so border windows
differ from H1 by ~1.5×, not by ±1.

**Prediction.** H2 (valid-count). If H1 were true the 3×3 SAME borders would be off by a
large multiplicative factor; if H2 were true the only residuals would be rounding (±1).

**Method.** [1,8,8,16]; 2×2/s2 VALID (count always 4), 3×3/s1 SAME (counts {9,6,4}),
2×2/s1 SAME (counts {4,2,1}, out-of-sample for the rounding rule found below).

**Result.** Single `AvgPool2DLayer`, `Static = 0`.
- H1 is **falsified**: a padding-included divisor gives 432/1024 mismatch (counts 9 vs 6
  at edges), while valid-count gives ≤110/1024 (`tools/diagnose_avgpool.py`).
- H2 accepted, but the *rounding* needed one more iteration: the first valid-count golden
  used the conv two-step fixed-point rule and missed 60/256 and 110/1024 by ±1. The
  contingency table (`tools/diagnose_avgpool_table.py`) shows the real rule is a **single
  rounding on the exact ratio, ties toward +∞**:

  ```
  out = floor( (2*sum + count) / (2*count) )        # = round-half-up of sum/count
  ```

  i.e. −38.5 → −38 (tie toward zero for negatives) while −38.75 → −39; +38.5 → +39.
  Two-step fixed-point would have given −39 for the tie.
- With that rule: **0/256, 0/256, 0/1024, 0/1024** mismatches, including the out-of-sample
  window counts {4,2,1}.

**Applicability.** int8, k=2 s=1/s=2, k=3 s=1, SAME and VALID, C=16; divisor = valid count;
rounding = round-half-up on the exact rational average. Not evidenced: count > 9 (e.g. 4×4
window, 7×7), stride > 2, fp16.

**Open questions.** Is the rule `floor((2s+c)/(2c))` or "round-half-away implemented with a
+1 fudge in the fixed-point product"? Both give identical results for all count/frac pairs
seen so far; a 4×4 (count 16) or 7×7 probe would separate them. **UNKNOWN.**

## 4. Compile-time failures (exact errors) — the eDPA relayout boundary

Nine of the 30 probes do not compile under the fixed policy. All four op families show the
same error shape: ncc *does* map the op to the right MDLA layer and then refuses because a
relayout step sits between the tensor and the MDLA step.

```
# ops/maxpool_1x8x8x4_2x2s2_int8/model.compile.log  (same text for the other five)
===Graph[0] has 1 Execution Step(s)===
Graph[0] Step[0]
 ├ Target: MDLA_5_0
 └ Subgraph:
   ├ MaxPool2DLayer<0>{MAX_POOL_2D}
   └ Output: OpResult (external)

Bridging is disallowed but gap is detected between Input and MDLA_5_0
ERROR: Fail to compile ops/maxpool_1x8x8x4_2x2s2_int8/model.tflite
EXIT=1
```

| op dir | layer ncc picked | gap location |
|---|---|---|
| `maxpool_1x8x8x4_2x2s2_int8` | MaxPool2DLayer | Input and MDLA_5_0 |
| `maxpool_1x8x8x4_2x2s2_same_int8` | MaxPool2DLayer | Input and MDLA_5_0 |
| `maxpool_1x8x8x4_3x3s1_same_int8` | MaxPool2DLayer | Input and MDLA_5_0 |
| `avgpool_1x8x8x4_2x2s2_int8` | AvgPool2DLayer | Input and MDLA_5_0 |
| `avgpool_1x8x8x4_2x2s2_same_int8` | AvgPool2DLayer | Input and MDLA_5_0 |
| `avgpool_1x8x8x4_3x3s1_same_int8` | AvgPool2DLayer | Input and MDLA_5_0 |
| `conv_1x16x16x8_f3s2v_int8` | Conv2DLayer | Input and MDLA_5_0 |
| `conv_1x8x8x4_f3d2_int8` | Conv2DLayer | MDLA_5_0 and **Output** |
| `dwconv_1x8x8x4_f3m2_int8` | DepthwiseConv2DLayer | Input and MDLA_5_0 |

**What fills the gap** (diagnostic with bridging *allowed*, `tools/diag_bridge.sh`, output
only in `/tmp`): an **`eDPA 1.0` "Relayout"** step —

```
WARNING: Relayout is having short burst transaction: 4
DRAM Usage:
       Target       Input  Output  Temp  Static  Code  Total
 [ 0]: eDPA 1.0     256    0       0     0       64    320
 [ 1]: MDLA 5.0     0      64      0     0       272   336
```

eDPA is the MDLA-side DMA engine, i.e. the fallback is *still NPU*, not CPU — but
`--disallow-bridge` rejects it. The burst warning reports 4 bytes for the C=4 pools and the
C=4 dw-mult2 case, 8 bytes for the C=8 stride-2 conv **and also** for the C=4 dilation case,
so the burst length tracks the offending per-pixel access rather than C alone. What exactly
triggers the relayout is **UNKNOWN**; only the pass/fail map below is measured.

**Boundary map (`ops/_scan/convpool_scan.tsv`, `tools/scan_convpool.py`, 36 configs):**

| C | pools | conv s1 | conv s2 | conv dil2 | dw m1 | dw m2 |
|---|---|---|---|---|---|---|
| 4 | **fail** | ok | ok | **fail** | ok | **fail** |
| 8 | **fail** | **fail** | **fail** | **fail** | **fail** | **fail** |
| 16 | ok | ok | ok | ok | ok | ok |
| 32 | ok (Slice+Pool+Concat) | ok (2×Conv+Concat) | ok (2×Conv+Concat) | ok (2×Conv+Concat) | ok (split) | ok |

So: **C=16 is the safe common denominator**, C=32 always works but is decomposed into two
16-channel halves plus Slice/Concat, and C=4/8 are hit-and-miss. The *mechanism* behind the
C=8 hole is **UNKNOWN** — it is not a simple alignment rule (C=4 works for conv s1/s2 but
fails for pools/dilation/dw-m2, C=8 fails for everything). Per R10 this is recorded as an
observation, not an explanation.

## 5. The requant rule: a falsified prediction, then an out-of-sample test — CP-05

This is the most transferable result of this worktree.

**v1 prediction (CP-01b).** "`out = round_half_away(acc · in_scale · w_scale / out_scale)`;
a power-of-two ratio makes the requant exact, so tol 0." Device said **no**:

```
# ops/conv_1x8x8x4_f3_full_int8/compare.txt   (ratio 1/1024, tol 0)
exact     : NO   mismatched=2/512
  at (0,3,2,5): expected=8 device=9
  at (0,6,1,2): expected=18 device=19
# ratio 1/3 (tol 1) is off by 1 on 167/512; RELU6 op: 1/512
```

**v1 is falsified.** Diagnostics (`tools/diagnose_requant*.py`) show the device rounds up at
`frac = 0.499` for ratio 1/1024 and flips *both ways* at `frac = 1/3` and `2/3` for ratio
1/3 — impossible for any single rounding of `acc·ratio`, so the device must be doing two
rounding steps. That is exactly TFLite/gemmlowp's fixed-point requant:

```
(M0, shift) = QuantizeMultiplier(ratio)        # ratio = M0/2^31 · 2^-shift, M0 ∈ [2^30,2^31)
out = RoundingDivideByPOT( round_half_up(acc·M0/2^31), shift )
```

**v2 validation (in-sample, 3 multipliers, 1 536 device values).** `tools/validate_devrule.py`
replays the recorded dumps with the two-step rule: **0 mismatches** for ratio 1/1024 (0/512),
1/2048 (0/512), 1/3 (0/512), 1 (0/512), 1/512 (0/512 + 0/256 dwconv), 1/2048 matmul (0/4096).
The naive rational reference misses 2, 1 and 167 of those.

**v2 out-of-sample test (CP-05).** Ratios **1/5** and **1/6** were never used to derive the
rule. Prediction recorded before compiling: bit-exact, tol 0. Result: `div5` **0/512**,
`div6` **0/512** — the rule predicts new multipliers, it does not merely fit.

**Applicability of the rule.** int8 conv / depthwise / 1×1 conv, one tensor-level multiplier
(ratio = in·w/out), any ratio expressible as `M0·2^-31-shift`. It does **not** apply to
`AVERAGE_POOL_2D` (that unit uses a single round-half-up, §3.4). NOT evidenced: per-channel
scales, asymmetric zero points, `int16` accumulators, and the exact `M0` normalisation ncc
picks when the ratio is not exactly representable (only ratios with short binary expansions
were tested).

**Consequence for the repo.** A `tol = 1` int8 contract hides a *systematic* ±1 that is now
explained. For a chain of N int8 ops the expected worst case is N LSB and not a fixed 1;
if a future op must be bit-exact, the golden should use the two-step rule (§5 formula), not
`round-half-away`.

**Status after ticket 02 (2026-11-13, `impl/q02-int8-bitexact`).** The rule now lives in
`tools/quant_contract.py` and is what the FC / BMM / dynB builders ship, so those tiers are
`int8_tol = 0` (35 ops, 145 296 recorded device values, 0 mismatches —
`reports/Q02-int8-bitexact.md`). Two things this page should be read with:

- the residual "conv golden still uses the old rational rule" cases are exactly three op dirs —
  `conv_1x8x8x4_f3_full_int8`, `_relu6_int8`, `_div3_int8` — kept as the *historical*
  FAIL records of the falsified prediction; their `*_devrule_int8` twins are the authoritative
  0/512 versions (`div3`'s 167/512 in the §3 table is that record, not a live contract).
- the "exact M0 normalisation when the ratio is not representable" UNKNOWN gets one more data
  point: when the ratio **is** exactly 1/32 the device's M0 is the ideal 2^30, **not** the value
  reverse-engineered from the float32 scale product (1073741840 — 69 mismatches on the same dump).
  So ncc normalises ideally when it can; the non-representable case stays open.

## 6. Device-state hygiene (R7 / §3.4)

`./tools/healthcheck.sh` was run after the 21-op batch (device lock held):

```
device present OK / device identity mt6897/16 / /dev/apusys node OK / neuronrt present OK
known-good op (fc_1x64x64_int8) PASS / repeatability (2nd run) PASS / board state OK
HEALTHCHECK: PASS
```

The conv/pool kernels do not pollute subsequent executions.

## 7. Bonus — 64×64×64 matmul as a 1×1 conv vs the FULLY_CONNECTED path

**Hypothesis.** The docs mention "express matmul as 1×1 conv" as a fallback. A 1×1 conv with
`H=M, W=1, C=K, OC=N` computes exactly `[M,K]·[K,N]`, so it should work on MDLA; the open
question is whether it is *competitive* with the native FC path.

**Method.** `ops/conv_matmul_1x64x64_int8`: input [1,64,1,64] (M=64 rows, W=1, C=64),
filter [64,1,1,64] → output [1,64,1,64]; dev-rule golden; compared against
`ops/_fc_ref_1x64x64` (built with the repo's `tools/build_ops.py fc_1x64x64_int8`, i.e. the
same 64×64×64 int8 matmul). Both compiled with the same policy, both run with
`--repeat 20`, **interleaved 5×** to average out the session noise.

**Result.**

| path | emitted layers | DRAM total | numerics | ms/inf (5 interleaved runs) |
|---|---|---|---|---|
| 1×1 conv | Conv2DLayer, Conv2DLayer, **ConcatLayer** | 13K | **0/4096 exact** | 0.661, 0.649, 0.745, 0.726, 0.649 (median 0.649) |
| FULLY_CONNECTED | FullyConnectedLayer | 4.8K | 0/64 exact | 0.432, 0.407, 0.628, 0.482, 0.627 (median 0.482) |

- The 1×1 conv form **is** viable and bit-exact, but the compiler splits C=64 into two
  32-channel convs + `ConcatLayer` (§4 C≥32 decomposition), so the DLA is 2.7× larger and
  ~1.35× slower (median) than the FC path.
- Session-to-session noise is large (FC ranged 0.41–0.63 ms for identical work), so the
  ~35% gap is a *trend over 5 interleaved pairs*, not a precise number.
- Both are single MDLA execution steps with no bridge.

**Applicability.** 64×64×64 int8 only. The conv layout `[1,M,1,K]` is a *fake* spatial
tensor: any real 2-D matmul (M,K both >1) has to be flattened into the H/W axes, which fits
`H·W = M` — fine for one operand, but it means a second operand must stay constant, exactly
like the FC path.

**Open questions.** Would `W>1` (e.g. `[1,8,8,K]`, M=64) avoid the channel split or the
`ConcatLayer`? Is there an `N` (OC) threshold where the conv form wins (the FC path has a
known high-N slowdown in `fc_1x128x21704_int8`)?

## 8. Failure knowledge index (R4) — do not re-learn these

1. **Shape/option mistakes in the flatbuffer are silent until compile.** A conv whose
   operator input indices were `[filter, bias, input]` produced
   `ERROR: The shape is wrong. Maybe remove "--no-verify" ?` / `ERROR: Fail to read the
   input model`. The tensor list order must be `input, filter, bias, output` = operator
   indices 0..3.
2. **int8 conv does NOT saturate like the FC probes do.** Reusing the FC scale scheme
   (`out = in·w·32`) saturates most outputs; `pow2` tier picks the shift from the actual
   accumulator range instead.
3. **C=8 is a dead spot** for all four op families (see §4). Use C=16 for probes.
4. **A pooling/conv op can be mapped correctly and still not compile** — the layer name in
   the exec plan is not a support verdict; the `gap` line is.
5. **`--disallow-bridge` also rejects the eDPA relayout**, which is *NPU* hardware, not CPU.
6. `AVERAGE_POOL_2D` rounding ≠ conv rounding (§3.4). Do not reuse the conv golden.
7. `RELU6` is not folded into `Conv2DLayer` in the exec plan (a separate `ReLU6Layer`
   appears) although the numerics are bit-exact and no extra `Temp` buffer is allocated
   (only `Code` grows 400 → 512 B). Interpretation: same MDLA pass, separate "layer" node in
   the plan — not proven.

## 9. Reproduction

```bash
# build + golden (WSL python with the tflite schema package)
/usr/bin/python3 tools/build_convpool.py --list
/usr/bin/python3 tools/build_convpool.py conv_1x8x8x4_f3_int8 --out ops/conv_1x8x8x4_f3_int8/model.tflite
# cross-check the golden with the second implementation, then compile
/usr/bin/python3 tools/verify_convpool.py ops/conv_1x8x8x4_f3_int8
./tools/compile.sh ops/conv_1x8x8x4_f3_int8/model.tflite ops/conv_1x8x8x4_f3_int8/model.dla
# device (shared board lock!)
NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" ./tools/devlock.sh run -- \
    ./tools/e2e.sh ops/conv_1x8x8x4_f3_int8
# helpers: requant-rule inference / validation, avg-pool rule, boundary scan
/usr/bin/python3 tools/validate_devrule.py ops/conv_1x8x8x4_f3_full_int8
/usr/bin/python3 tools/diagnose_avgpool_table.py ops/avgpool_1x8x8x16_3x3s1_same_int8
/usr/bin/python3 tools/scan_convpool.py
```

## 10. Needs from the Lead / open questions for the next owner

1. **Policy question (blocking 12 probe configs):** does `--disallow-bridge` mean "no CPU
   fallback" or "one MDLA step"? The rejected configs fall back to `eDPA 1.0 Relayout`, which
   is MDLA's own DMA engine. If eDPA relayout is acceptable, pools/stride/dilation at C=4/8
   become available; if not, C=16 is the documented workaround. This is a `tools/compile.sh`
   / policy decision, outside this worktree's scope.
2. **`tools/build_ops.py` only knows FC specs**, so `e2e.sh`'s auto-build step
   (`tools/build_ops.py <NAME> --out …`) cannot build any conv/pool op. I pre-build with
   `tools/build_convpool.py` and e2e.sh then skips the step — fine, but if the Lead wants a
   uniform flow, `build_ops.py` needs a dispatch hook (public tool, not touched here).
3. **`compare.py` has one global `int8_tol`** per op. Ops whose reference is exact for most
   values but ±1 on a tie would benefit from reporting a mismatch histogram; today only the
   first 5 mismatches are printed. Not blocking.
4. **Numbering:** CP-01…CP-05 above are ready to be folded into `docs/EXPERIMENTS.md`.
5. **Next experiments worth board time:** (a) 4×4 avg-pool (count 16) to decide between the
   two candidate formulas in §3.4; (b) fp16 conv/pool to see whether the int8-only limits in
   §4 are dtype-dependent; (c) per-channel weight scales (the common case in real models);
   (d) an `[1,H,W,K]` matmul-as-conv with W>1 to see whether the C≥32 split disappears.
