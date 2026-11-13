#!/usr/bin/env python3
"""PV (3-D BMM) + O projection + residual — Qwen2.5-0.5B real shapes, T=8, fp16.

Ticket 06 (feature qwen25-attention). Evidence dir: ops/pv_o_residual_t8_fp16/.

Graph (--variant main):

    in_pv [H,T,T+HD] fp16   (packed: per (head,token) row = p_row(8) | v_row(64), see D2)
      --SLICE begin=[0,0,0]  size=[H,T,T] --> p  [H,T,T]
      --SLICE begin=[0,0,T]  size=[H,T,HD]--> v  [H,T,HD]
    BATCH_MATMUL(p, v)  adj_x=0 adj_y=0  --> a3 [H,T,HD]        # 3-D => 1 native layer
    TRANSPOSE(a3, perm=[1,0,2])          --> at [T,H,HD]        # NOTES §10.2: reshape alone is wrong
    RESHAPE(at, [T,H*HD])                --> a  [T,896]
    FULLY_CONNECTED(a, W=o_proj[896,896], bias=zeros[896]) --> o [T,896]   # o_proj has NO bias
    ADD(in_x [T,896], o)                 --> out [T,896]        # residual

Why p and v are packed into one runtime input: tools/e2e.sh only feeds golden/input.bin +
golden/input1.bin (2 runtime inputs; NOTES §3/§8: public tools must not change).  The graph needs
three logical runtime tensors (p, v, x).  The residual x must stay *dynamic* (a constant-operand ADD
risks being folded into the preceding FC's bias, which would remove the layer and the evidence), so
p|v share one input and are split in-graph by two SLICEs.  The two SLICEs are transport plumbing
outside the math; the "PV is one native layer" claim is counted by layer *name*
(BatchMatMulLayer), never by total layer count.

p is NOT hand-made noise: it comes from a full attention pass on the frozen fixture weights
(embed row -> RMSNorm(input_layernorm) -> q/k/v (+bias) -> RoPE(theta=1e6) -> scores/sqrt(HD) ->
causal mask -> float32 softmax), see docs/OP-pv-o.md D3.  v uses the same v_proj+bias with the GQA
2->14 replication (head h uses KV head h//7).

Other variants (same builder => reproducible):
    --variant bmm3d          p [H,T,T] x v [H,T,HD]     -> a3 [H,T,HD]   1 BatchMatMulLayer
    --variant bmm4d          p [1,H,T,T] x v [1,H,T,HD] -> a4 [1,H,T,HD] 5 layers (E-112)
    --variant main-badlayout the same graph WITHOUT the TRANSPOSE (negative control: compiles,
                             runs in one step, rel_l2~0.92 -> proves the golden catches it)
    --variant main-alt       the spread-p input set only (main already writes it, see below)

--variant main also writes three extra runtime-input sets for the SAME compiled graph
(they are how the dynamic-operand evidence was produced; each is a golden/input*.bin swap):
    golden/{input,input1,expected}_alt.bin    p flattened (temperature 0.02) -> PV accumulates
    golden/{input,input1,expected}_altx.bin   different block input (embed rows 8..15) -> p,v,x all change

    /usr/bin/python3 tools/build_pv_o_residual.py --variant main --out ops/pv_o_residual_t8_fp16
    /usr/bin/python3 tools/build_pv_o_residual.py --variant bmm3d --out ops/pv_bmm3d_control_t8_fp16
    /usr/bin/python3 tools/build_pv_o_residual.py --variant bmm4d --out ops/pv_bmm4d_control_t8_fp16
    /usr/bin/python3 tools/build_pv_o_residual.py --variant main-badlayout --out ops/pv_o_badlayout_control_t8_fp16

Golden (golden/expected.bin) is pure numpy on the *logical* tensors and shares no code with the
flatbuffer writer (R3).  golden/expected_fp16sim.bin is the same math with fp16 rounding at every
layer boundary; its distance to the golden is the device-independent fp16 lower bound that the
registered tolerance is derived from (NOTES §10.1).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_ops as bo            # noqa: E402  (flatbuffer toolbox, read-only)
import qwen25_weights as qw       # noqa: E402  (frozen fixture loader)

SEED = 20261112                   # only used for the synthetic alt input
D = 896                           # hidden width
H = 14                            # query heads
KV = 2                            # kv heads (GQA 14Q-2KV)
HD = 64                           # head_dim
T = 8                             # static sequence length


# ------------------------------------------------------------------ data prep
def _rmsnorm(x32, gamma32, eps=1e-6):
    return x32 / np.sqrt((x32 * x32).mean(axis=-1, keepdims=True) + eps) * gamma32


def _rope(t2, nheads, theta=1e6):
    """Qwen rotate_half (split-half) RoPE on [T, nheads*HD] float32."""
    n = t2.shape[0]
    t3 = t2.reshape(n, nheads, HD)
    half = HD // 2
    inv = theta ** (-2.0 * np.arange(half, dtype=np.float32) / HD)
    ang = np.arange(n, dtype=np.float32)[:, None] * inv[None, :]                 # [T, half]
    cos, sin = np.cos(ang), np.sin(ang)
    a, b = t3[..., :half], t3[..., half:]
    ra = a * cos[:, None, :] - b * sin[:, None, :]
    rb = b * cos[:, None, :] + a * sin[:, None, :]
    return np.concatenate([ra, rb], axis=-1).reshape(n, nheads * HD)


def attention_inputs(temperature=1.0, x_rows=(0, T)):
    """Fixed, reproducible attention operands from the frozen fixture.

    Returns dict(p16 [H,T,T], v16 [H,T,HD], x16 [T,D], stats).
    temperature < 1 flattens the softmax (synthetic alt input only).
    x_rows selects which fixture embed rows act as the block input (i.e. which "tokens").
    """
    a, b = x_rows
    x16 = np.asarray(qw.get("embed.weight")[a:b], dtype=np.float16)
    x32 = x16.astype(np.float32)
    g = qw.get_fp32("layer0.input_layernorm.weight")
    n1 = _rmsnorm(x32, g)

    def proj(name, n_out):
        W = qw.get_fp32(f"layer0.attn.{name}_proj.weight")               # [n_out, D]
        b = qw.get_fp32(f"layer0.attn.{name}_proj.bias")                 # [n_out]
        return n1 @ W.T + b                                              # [T, n_out]

    q = _rope(proj("q", H * HD), H)                                      # [T, H*HD]
    k = _rope(proj("k", KV * HD), KV)                                    # [T, KV*HD]
    v = proj("v", KV * HD)                                               # [T, KV*HD]

    qh = q.reshape(T, H, HD).transpose(1, 0, 2)                          # [H,T,HD]
    kh = k.reshape(T, KV, HD).transpose(1, 0, 2)
    vh = v.reshape(T, KV, HD).transpose(1, 0, 2)
    rep = H // KV
    kh = np.repeat(kh, rep, axis=0)                                      # head h -> kv h//rep
    vh = np.repeat(vh, rep, axis=0)

    scores = np.einsum("htd,hsd->hts", qh, kh) / np.float32(np.sqrt(HD))
    if temperature != 1.0:
        scores = scores * np.float32(temperature)
    causal = np.triu(np.ones((T, T), dtype=bool), k=1)
    masked = np.where(causal[None, :, :], np.float32(-np.inf), scores)
    m = masked.max(axis=-1, keepdims=True)
    e = np.exp(masked - m)
    p = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)

    p16 = p.astype(np.float16)
    v16 = vh.astype(np.float16)
    stats = {
        "temperature": temperature,
        "p_max": float(p16.max()), "p_min": float(p16.min()),
        "p_argmax_frac_gt_0.99": float((p > 0.99).mean()),
        "p_rowsum_max_abs_err": float(np.abs(p.astype(np.float64).sum(-1) - 1).max()),
        "p_entropy_bits_mean": float(-(p * np.log2(np.maximum(p, 1e-30))).sum(-1).mean()),
        "scores_absmax_unmasked": float(np.abs(np.where(causal[None], 0.0, scores)).max()),
        "v_absmax": float(np.abs(v16).max()), "x_absmax": float(np.abs(x16).max()),
        "x_std": float(x16.astype(np.float32).std()),
    }
    return dict(p16=p16, v16=v16, x16=x16, stats=stats)


# --------------------------------------------------------------- flatbuffer graph
class G:
    """Minimal single-subgraph fp16 graph writer on top of build_ops helpers."""

    def __init__(self):
        self.b = bo.flatbuffers.Builder(1 << 20)
        self.S = bo.S
        self.buffers = [bo.buf_empty(self.b)]      # buffer 0 = runtime tensors
        self.tensors, self.opcodes, self.operators = [], [], []
        self.idx = {}

    def act(self, name, shape):
        self.tensors.append(bo.tensor(self.b, name, list(shape), bo.F16, 0))
        self.idx[name] = len(self.tensors) - 1
        return self.idx[name]

    def const(self, name, arr, ttype=bo.F16):
        arr = np.ascontiguousarray(arr)
        self.buffers.append(bo.buf_data(self.b, arr))
        t = bo.tensor(self.b, name, list(arr.shape), ttype, len(self.buffers) - 1)
        self.tensors.append(t)
        self.idx[name] = len(self.tensors) - 1
        return self.idx[name]

    def oc(self, builtin, version=1):
        self.opcodes.append(bo.opcode(self.b, builtin, version))
        return len(self.opcodes) - 1

    def slice(self, inp, out, begin, size):
        b, S = self.b, self.S
        self.const(f"{out}_begin", np.asarray(begin, np.int32), bo.I32)
        self.const(f"{out}_size", np.asarray(size, np.int32), bo.I32)
        t_o = self.act(out, size)
        S.SliceOptionsStart(b)
        so = S.SliceOptionsEnd(b)
        self.operators.append(bo.operator(b, self.oc(bo.BuiltinOperator.SLICE),
                                          [self.idx[inp], self.idx[f"{out}_begin"],
                                           self.idx[f"{out}_size"]], [t_o],
                                          bo.BuiltinOptions.SliceOptions, so))
        return t_o

    def bmm(self, a_name, b_name, out, shape, adj_y=False):
        b, S = self.b, self.S
        t_o = self.act(out, shape)
        S.BatchMatMulOptionsStart(b)
        S.BatchMatMulOptionsAddAdjX(b, False)
        S.BatchMatMulOptionsAddAdjY(b, bool(adj_y))
        S.BatchMatMulOptionsAddAsymmetricQuantizeInputs(b, False)
        o = S.BatchMatMulOptionsEnd(b)
        self.operators.append(bo.operator(b, self.oc(bo.BuiltinOperator.BATCH_MATMUL),
                                          [self.idx[a_name], self.idx[b_name]], [t_o],
                                          bo.BuiltinOptions.BatchMatMulOptions, o))
        return t_o

    def transpose(self, inp, out, perm, in_shape):
        b, S = self.b, self.S
        self.const(f"{out}_perm", np.asarray(perm, np.int32), bo.I32)
        t_o = self.act(out, [in_shape[p] for p in perm])
        S.TransposeOptionsStart(b)
        o = S.TransposeOptionsEnd(b)
        self.operators.append(bo.operator(b, self.oc(bo.BuiltinOperator.TRANSPOSE),
                                          [self.idx[inp], self.idx[f"{out}_perm"]], [t_o],
                                          bo.BuiltinOptions.TransposeOptions, o))
        return t_o

    def reshape(self, inp, out, shape):
        b, S = self.b, self.S
        self.const(f"{out}_shape", np.asarray(shape, np.int32), bo.I32)
        t_o = self.act(out, shape)
        nsv = b.CreateNumpyVector(np.asarray(shape, np.int32))   # order-preserving (NOTES §5.4)
        S.ReshapeOptionsStart(b)
        S.ReshapeOptionsAddNewShape(b, nsv)
        o = S.ReshapeOptionsEnd(b)
        self.operators.append(bo.operator(b, self.oc(bo.BuiltinOperator.RESHAPE),
                                          [self.idx[inp], self.idx[f"{out}_shape"]], [t_o],
                                          bo.BuiltinOptions.ReshapeOptions, o))
        return t_o

    def fc(self, inp, out, w, bias, m, k, n):
        b, S = self.b, self.S
        ti_w = self.const(f"{out}_W", np.asarray(w, np.float16))          # [N,K]
        ti_b = self.const(f"{out}_bias", np.asarray(bias, np.float16))    # [N] (zeros: no bias)
        t_o = self.act(out, (m, n))
        S.FullyConnectedOptionsStart(b)
        S.FullyConnectedOptionsAddFusedActivationFunction(b, 0)
        S.FullyConnectedOptionsAddWeightsFormat(b, 0)                     # DEFAULT [N,K]
        S.FullyConnectedOptionsAddKeepNumDims(b, True)
        S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(b, False)
        o = S.FullyConnectedOptionsEnd(b)
        self.operators.append(bo.operator(b, self.oc(bo.BuiltinOperator.FULLY_CONNECTED),
                                          [self.idx[inp], ti_w, ti_b], [t_o],
                                          bo.BuiltinOptions.FullyConnectedOptions, o))
        return t_o

    def add(self, a_name, b_name, out, shape):
        b, S = self.b, self.S
        t_o = self.act(out, shape)
        S.AddOptionsStart(b)
        S.AddOptionsAddFusedActivationFunction(b, 0)
        o = S.AddOptionsEnd(b)
        self.operators.append(bo.operator(b, self.oc(bo.BuiltinOperator.ADD),
                                          [self.idx[a_name], self.idx[b_name]], [t_o],
                                          bo.BuiltinOptions.AddOptions, o))
        return t_o

    def finish(self, inputs, outputs, desc):
        return bo.finish(self.b, self.buffers, self.opcodes, self.tensors, self.operators,
                         inputs, outputs, desc)


# ---------------------------------------------------------------- numpy goldens
def golden(p16, v16, x16, Wo16):
    """Float64 math on the logical tensors (no device simulation)."""
    p64, v64 = p16.astype(np.float64), v16.astype(np.float64)
    a3 = np.einsum("hts,hsd->htd", p64, v64)                         # [H,T,HD]
    a = a3.transpose(1, 0, 2).reshape(T, H * HD)                     # [T,896]
    o = a @ np.asarray(Wo16, np.float64).T                           # no bias
    out = x16.astype(np.float64) + o
    stats = {"max_abs_out": float(np.abs(out).max()),
             "max_abs_attn_merged": float(np.abs(a).max()),
             "max_abs_o_proj": float(np.abs(o).max())}
    return out.astype(np.float16), stats


def golden_fp16sim(p16, v16, x16, Wo16):
    """Same math with an fp16 rounding at every layer boundary (device-independent bound)."""
    p32, v32 = p16.astype(np.float32), v16.astype(np.float32)
    a3 = np.einsum("hts,hsd->htd", p32, v32).astype(np.float16)       # BMM output rounded
    a = a3.astype(np.float32).transpose(1, 0, 2).reshape(T, H * HD)  # pure movement
    o = (a @ np.asarray(Wo16, np.float32).T).astype(np.float16)      # FC output rounded
    out = (x16.astype(np.float32) + o.astype(np.float32)).astype(np.float16)   # ADD output
    return out


# ------------------------------------------------------------------- variants
def build_main(temperature=1.0, bad_layout=False, x_rows=(0, T)):
    """bad_layout=True is the NEGATIVE control: skip TRANSPOSE (NOTES §10.2's trap)."""
    inp = attention_inputs(temperature, x_rows)
    p16, v16, x16 = inp["p16"], inp["v16"], inp["x16"]
    pack = np.concatenate([p16, v16], axis=-1).astype(np.float16)     # [H,T,T+HD]
    Wo16 = np.asarray(qw.get("layer0.attn.o_proj.weight"), dtype=np.float16)  # [896,896]

    g = G()
    g.act("in_pv", (H, T, T + HD))
    g.act("in_x", (T, D))
    g.slice("in_pv", "p", [0, 0, 0], [H, T, T])
    g.slice("in_pv", "v", [0, 0, T], [H, T, HD])
    g.bmm("p", "v", "a3", (H, T, HD))                                # 3-D PV
    if bad_layout:
        # NEGATIVE control: reshape straight from [H,T,HD] (no TRANSPOSE). This is exactly the
        # trap NOTES §10.2 records: it compiles, runs in one step, and is mathematically wrong.
        g.reshape("a3", "a", (T, H * HD))
    else:
        g.transpose("a3", "at", [1, 0, 2], (H, T, HD))               # [H,T,HD] -> [T,H,HD]
        g.reshape("at", "a", (T, H * HD))                            # -> [T,896]
    g.fc("a", "o", Wo16, np.zeros(D, np.float16), T, H * HD, D)      # o_proj, NO bias
    g.add("in_x", "o", "out", (T, D))                                # residual
    blob = g.finish([g.idx["in_pv"], g.idx["in_x"]], [g.idx["out"]],
                    f"pv_o_residual T{T} H{H} HD{HD} D{D} fp16 PV3D")

    exp, gstats = golden(p16, v16, x16, Wo16)
    sim = golden_fp16sim(p16, v16, x16, Wo16)
    bound = float(np.abs(sim.astype(np.float32) - exp.astype(np.float32)).max())
    max_out = gstats["max_abs_out"]
    atol = rtol = 4.0 * 2 ** -11 * max_out
    meta = {
        "op": "PV_3D_BMM__O_PROJ__RESIDUAL",
        "shape": {"T": T, "H": H, "KV": KV, "HD": HD, "D": D, "PV_rank": 3},
        "dtype": "float16",
        "inputs": [{"name": "in_pv", "shape": [H, T, T + HD], "type": "float16",
                    "bytes": int(pack.nbytes),
                    "layout": "per (head,token) row: p row (8) | v row (64); split in-graph by "
                              "two SLICEs. Packed only because e2e.sh feeds 2 runtime inputs."},
                   {"name": "in_x", "shape": [T, D], "type": "float16", "bytes": int(x16.nbytes),
                    "layout": "residual stream = fixture embed.weight rows 0..7"}],
        "output": {"name": "out", "shape": [T, D], "type": "float16", "bytes": int(exp.nbytes)},
        "weights": {"tensor": "layer0.attn.o_proj.weight", "shape": [D, D],
                    "bias": "none in Qwen2.5 (a zero-filled [D] fp16 tensor is still fed, "
                            "NOTES §5.1: FC wants a third input)",
                    "fixture_weights_digest": "898fa3195d627a4ce055cbdb6fc762a12e74665b431e3ae09b530b4abdad18a4"},
        "graph": ["SLICE(in_pv)->p [H,T,T]", "SLICE(in_pv)->v [H,T,HD]",
                  "BATCH_MATMUL(p,v) adj=0/0 3-D", "TRANSPOSE perm=[1,0,2] -> [T,H,HD]",
                  "RESHAPE [T,H*HD] -> [T,896]", "FULLY_CONNECTED(o_proj,bias=0)",
                  "ADD(in_x,o) residual"],
        "p_source": ("fixed attention pass on fixture weights: embed rows -> RMSNorm("
                     "input_layernorm, eps=1e-6) -> q/k/v (with bias) -> RoPE(theta=1e6) -> "
                     "scores/sqrt(HD) -> causal mask -> float32 softmax; v replicated 2->14 KV "
                     "heads (head h uses kv h//7)."
                     + (" ALT: softmax temperature 0.02 to flatten p (synthetic; layout/"
                        "accumulation discrimination only)." if temperature != 1.0 else "")),
        "p_stats": inp["stats"],
        "reference": "numpy float64 on logical tensors (R3: independent of the flatbuffer writer)",
        "fp16sim_lower_bound": {
            "max_abs_sim_vs_golden": bound,
            "rel": (bound / max_out) if max_out else 0.0,
            "method": "fp16 rounding at BMM/FC/ADD boundaries, fp32 inside; numpy only"},
        "golden_stats": gstats,
        "atol": atol, "rtol": rtol,
        "tolerance_revision": 1,
        "tolerance_reason": ("NOTES §10.1 form: atol = rtol = 4*2^-11*max|out| = {:.3e} "
                             "(max|out| = {:.4g}); device-independent fp16-simulation bound = "
                             "{:.3e} max_abs, registered threshold is {:.1f}x that bound. The "
                             "repo default 2e-2 is rejected a priori: it is larger than max|out| "
                             "itself here, i.e. it could never catch a layout error."
                             ).format(atol, max_out, bound,
                                      atol / bound if bound else float("inf")),
        "seed": SEED,
    }
    return blob, meta, {"in_pv": pack, "in_x": x16}, exp, sim


def build_bmm_control(rank):
    inp = attention_inputs(1.0)
    p16, v16 = inp["p16"], inp["v16"]
    g = G()
    if rank == 3:
        g.act("p", (H, T, T))
        g.act("v", (H, T, HD))
        g.bmm("p", "v", "a3", (H, T, HD))
        oname, oshape = "a3", [H, T, HD]
        desc = f"pv bmm control 3-D [H,T,T]x[H,T,HD] T{T} H{H} HD{HD} fp16"
    else:
        g.act("p", (1, H, T, T))
        g.act("v", (1, H, T, HD))
        g.bmm("p", "v", "a4", (1, H, T, HD))
        oname, oshape = "a4", [1, H, T, HD]
        desc = f"pv bmm control 4-D [1,H,T,T]x[1,H,T,HD] T{T} H{H} HD{HD} fp16"
    blob = g.finish([g.idx["p"], g.idx["v"]], [g.idx[oname]], desc)
    p64, v64 = p16.astype(np.float64), v16.astype(np.float64)
    a3 = np.einsum("hts,hsd->htd", p64, v64).astype(np.float16)
    exp = a3 if rank == 3 else a3[None]
    meta = {
        "op": f"BATCH_MATMUL_PV_CONTROL_{rank}D",
        "shape": {"T": T, "H": H, "HD": HD, "rank": rank},
        "dtype": "float16",
        "purpose": ("compile-only control for ticket 06: count native layers for 3-D vs 4-D PV "
                    "(3-D must be ONE BatchMatMulLayer; 4-D is lowered into 5 layers, E-112). "
                    "No board run: the claim under test is a compile-time layer count."),
        "inputs": [{"name": "p", "shape": [H, T, T] if rank == 3 else [1, H, T, T],
                    "type": "float16"},
                   {"name": "v", "shape": [H, T, HD] if rank == 3 else [1, H, T, HD],
                    "type": "float16"}],
        "output": {"name": oname, "shape": oshape, "type": "float16"},
        "atol": 1e-2, "rtol": 3e-2,
        "p_source": "same fixed attention pass as the main model (temperature 1.0)",
        "reference": "numpy float64 einsum on logical tensors",
        "seed": SEED,
    }
    return blob, meta, {"p": p16, "v": v16}, exp, exp


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",
                    choices=["main", "main-alt", "main-badlayout", "bmm3d", "bmm4d"],
                    default="main")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = os.path.abspath(args.out)
    gd = os.path.join(d, "golden")
    os.makedirs(gd, exist_ok=True)

    if args.variant in ("main", "main-alt", "main-badlayout"):
        alt = args.variant == "main-alt"
        bad = args.variant == "main-badlayout"
        blob, meta, ins, exp, sim = build_main(temperature=0.02 if alt else 1.0, bad_layout=bad)
        sfx = "_alt" if alt else ""
        if bad:
            meta["op"] = "PV_3D_BMM__O_PROJ__RESIDUAL_NEGATIVE_CONTROL"
            meta["purpose"] = (
                "NEGATIVE control (intentionally WRONG math): the PV output [H,T,HD] is RESHAPEd "
                "straight to [T,896] with NO TRANSPOSE (NOTES §10.2 trap). Compiles, runs in one "
                "execution step, and shows a large rel_l2 -> proves the main PASS is not vacuous "
                "and that the golden catches head-layout errors.")
            meta["expected_result"] = "compare must FAIL by a large margin (rel_l2 ~ 1e-1)"
            meta["graph"] = ["SLICE(in_pv)->p [H,T,T]", "SLICE(in_pv)->v [H,T,HD]",
                             "BATCH_MATMUL(p,v) adj=0/0 3-D",
                             "RESHAPE [H,T,HD] -> [T,896]   <-- NO TRANSPOSE (the bug)",
                             "FULLY_CONNECTED(o_proj,bias=0)", "ADD(in_x,o) residual"]
        open(os.path.join(d, "model.tflite"), "wb").write(blob)
        ins["in_pv"].tofile(os.path.join(gd, f"input{sfx}.bin"))
        ins["in_x"].tofile(os.path.join(gd, f"input1{sfx}.bin"))
        exp.tofile(os.path.join(gd, f"expected{sfx}.bin"))
        sim.tofile(os.path.join(gd, "expected_fp16sim_alt.bin" if alt else "expected_fp16sim.bin"))
        meta["tflite"] = "model.tflite"
        meta["tflite_bytes"] = len(blob)
        if not alt and not bad:
            # extra runtime-input variants for the SAME compiled graph (dynamic-operand evidence):
            # e2e.sh feeds one input pair, so each variant is a swap of golden/input*.bin+expected
            # and a re-run of the identical .dla (compare_altp.txt / compare_altx.txt).
            Wo16 = np.asarray(qw.get("layer0.attn.o_proj.weight"), dtype=np.float16)
            extras = []
            for tag, kw, purpose in (
                ("alt", dict(temperature=0.02),
                 "spread p (softmax temperature 0.02): PV accumulates over all valid positions "
                 "instead of an almost 1-hot selection"),
                ("altx", dict(x_rows=(T, 2 * T)),
                 "different block input (fixture embed rows 8..15): p, v AND the residual x all "
                 "change -> all three runtime operands are really read at run time"),
            ):
                inp2 = attention_inputs(**kw)
                pack2 = np.concatenate([inp2["p16"], inp2["v16"]], axis=-1).astype(np.float16)
                exp2, _ = golden(inp2["p16"], inp2["v16"], inp2["x16"], Wo16)
                pack2.tofile(os.path.join(gd, f"input_{tag}.bin"))
                inp2["x16"].tofile(os.path.join(gd, f"input1_{tag}.bin"))
                exp2.tofile(os.path.join(gd, f"expected_{tag}.bin"))
                extras.append({"tag": tag,
                               "files": [f"golden/input_{tag}.bin", f"golden/input1_{tag}.bin",
                                         f"golden/expected_{tag}.bin"],
                               "purpose": purpose, "p_stats": inp2["stats"],
                               "golden_max_abs": float(np.abs(exp2.astype(np.float32)).max())})
            meta["alt_inputs"] = {
                "note": "same compiled .dla, different runtime operands; each variant was run by "
                        "swapping golden/input*.bin + golden/expected.bin and re-running "
                        "tools/e2e.sh, then restoring the main files",
                "variants": extras}
        if alt:
            meta["alt_input"] = {
                "files": ["golden/input_alt.bin", "golden/input1_alt.bin",
                          "golden/expected_alt.bin"],
                "purpose": "spread p (temperature 0.02) on the SAME compiled .dla: dynamic-operand "
                           "check + PV accumulation discrimination (compare_altp.txt)"}
        # the alt variant must not overwrite the main contract written by --variant main
        json.dump(meta, open(os.path.join(d, "model.alt.json" if alt else "model.json"), "w"),
                  indent=2)
        print(f"{d}: {len(blob)} B  p{ins['in_pv'].shape} x{ins['in_x'].shape} -> {exp.shape}  "
              f"atol=rtol={meta['atol']:.3e}  fp16sim bound={meta['fp16sim_lower_bound']['max_abs_sim_vs_golden']:.3e}")
        print("  p_stats:", json.dumps(meta["p_stats"], ensure_ascii=False))
        print("  golden_stats:", json.dumps(meta["golden_stats"]))
        return

    rank = 3 if args.variant == "bmm3d" else 4
    blob, meta, ins, exp, _ = build_bmm_control(rank)
    open(os.path.join(d, "model.tflite"), "wb").write(blob)
    # rank 4 control keeps the real 4-D operand layout [1,H,T,*] (compile-only: layer count is
    # the claim under test, but the stored inputs match the declared tensor shapes anyway)
    ins["p"].tofile(os.path.join(gd, "input.bin")) if rank == 3 else \
        ins["p"][None].tofile(os.path.join(gd, "input.bin"))
    ins["v"].tofile(os.path.join(gd, "input1.bin")) if rank == 3 else \
        ins["v"][None].tofile(os.path.join(gd, "input1.bin"))
    exp.tofile(os.path.join(gd, "expected.bin"))
    meta["tflite"] = "model.tflite"
    meta["tflite_bytes"] = len(blob)
    json.dump(meta, open(os.path.join(d, "model.json"), "w"), indent=2)
    print(f"{d}: {len(blob)} B  {meta['op']} p{ins['p'].shape} v{ins['v'].shape} -> {exp.shape}")


if __name__ == "__main__":
    main()
