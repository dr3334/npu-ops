#!/usr/bin/env python3
"""Build ops/lm_head_t8_fp16 -- Qwen2.5-0.5B LM head (tied output projection), T=8, fp16.

    /usr/bin/python3 tools/build_lm_head.py                      # single FC (default)
    /usr/bin/python3 tools/build_lm_head.py --chunks 4           # fallback: 4 FC + CONCAT(axis=1)

Graph (chunks=1)
----------------
    x [8,896] fp16 --FULLY_CONNECTED(W = embed.weight [151936,896], b = 0)--> logits [8,151936] fp16

Graph (chunks=C)
----------------
    x [8,896] --FC_i(W = embed.weight[v0:v1], b = 0)--> y_i [8,151936/C]  (i = 0..C-1)
    CONCAT(y_0..y_{C-1}, axis=1) --> logits [8,151936]
    (same graph, same single weight copy -- just split into C constant buffers)

WHY these choices (this is experiment design, not incidental code):

* **tied**: the real model has tie_word_embeddings=true, i.e. the LM head *is*
  model.embed_tokens.weight. We load exactly one array through the fixture loader
  (tools/qwen25_weights.py) and assert that the "lm_head.weight" alias returns the
  *same object*; the model file therefore carries one 272,269,312 B weight buffer.
  Materialising it twice would waste ~260 MiB on device (docs/OP-weights.md 8.1).
* **bias must exist**: ncc rejects a 2-input FULLY_CONNECTED at *load* time
  ("Fail to process the given model", NOTES 5.1). The real LM head has no bias, so
  we feed an all-zero [151936] fp16 tensor (303,872 B) -- numerically a no-op.
* **weights layout**: TFLite FULLY_CONNECTED with weights_format=DEFAULT is [N,K],
  which is exactly embed.weight's [vocab, hidden] layout -> logits = x . W^T.
  No transpose is needed (and a transpose would be a silent numerical bug).
* **input provenance**: x is 8 rows of a *different* fixture tensor
  (layer0.attn.o_proj.weight, rows 0..7), each row scaled to RMS = 1. That mimics the
  final RMSNorm output of the real model (row RMS ~ 1) without inventing a random
  stream, and puts the logits at the real model's scale:
  ||x|| * std(embed) = sqrt(896)*0.015249 ~ 0.456.
  WHY NOT embed rows: x taken from embed.weight makes logits[t,t] = 896*rms_t ~ 13.68
  (a 30-sigma self-dot product), and a max|out|-based tolerance (NOTES 10.1) would then
  be 4*2^-11*13.68 = 2.67e-2 -- 6x looser than the bulk of the output needs, and above
  NOTES 4's 2e-2 cap. The spike is a property of *this* input, not of the operator, so we
  use a source decorrelated from the weights. (Measured: see docs/OP-lm-head.md 2.4.)
* **golden independence (AGENTS.md R3)**: golden/expected.bin is plain numpy
  float32 matmul of the *stored fp16 values*, computed in vocab chunks so peak RSS
  stays ~100 MB. It shares no code with the flatbuffer construction.

Contract written to model.json: |dev - exp| <= atol + rtol*|exp| with the tolerance
registered *before* the first device run under the NOTES 10.1 exception rule
(outputs are not O(1) -- see the tolerance_revision block).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import flatbuffers
import numpy as np
import tflite as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_ops as bo                      # flatbuffer toolbox (read-only use)
import qwen25_weights as qw                 # frozen fixture loader (sha256-verified)

OPDIR = "ops/lm_head_t8_fp16"
M, K, N = 8, 896, 151936                    # hidden batch x hidden size x vocab
INPUT_TENSOR = "layer0.attn.o_proj.weight"  # decorrelated fixture rows used as x
INPUT_ROWS = list(range(M))                 # rows 0..7 of that tensor
GOLDEN_CHUNK = 8192                         # vocab rows per matmul chunk (RSS bound)
FP16_ULP_REL = 2.0 ** -11                   # 4.8828125e-4, the fp16 mantissa step
TOL_K = 4.0                                 # NOTES 10.1: atol = rtol = k * 2^-11 * max|out|


# --------------------------------------------------------------- flatbuffers
class Model:
    """Tiny model accumulator; buffer 0 stays the (empty) input placeholder."""

    def __init__(self, initial_size=1 << 16):
        self.b = flatbuffers.Builder(initial_size)
        self.buffers = [bo.buf_empty(self.b)]
        self.tensors = []
        self.opcodes = []
        self.ops = []
        self.buffer_sizes = [0]            # recorded for the "one weight copy" evidence

    def buf(self, arr):
        self.buffers.append(bo.buf_data(self.b, arr))
        self.buffer_sizes.append(int(arr.nbytes))
        return len(self.buffers) - 1

    def empty_buf(self):
        self.buffers.append(bo.buf_empty(self.b))
        self.buffer_sizes.append(0)
        return len(self.buffers) - 1

    def const(self, name, shape, ttype, arr):
        self.tensors.append(bo.tensor(self.b, name, shape, ttype, self.buf(arr)))
        return len(self.tensors) - 1

    def act(self, name, shape, ttype, own_buffer=True):
        bi = self.empty_buf() if own_buffer else 0
        self.tensors.append(bo.tensor(self.b, name, shape, ttype, bi))
        return len(self.tensors) - 1

    def add_op(self, builtin, ins, outs, opt_type=None, opt=None):
        self.opcodes.append(bo.opcode(self.b, builtin))
        self.ops.append(bo.operator(self.b, len(self.opcodes) - 1, ins, outs, opt_type, opt))

    def finish(self, ins, outs, desc):
        return bo.finish(self.b, self.buffers, self.opcodes, self.tensors, self.ops,
                         ins, outs, desc)


def fc_options(bb):
    S.FullyConnectedOptionsStart(bb)
    S.FullyConnectedOptionsAddFusedActivationFunction(bb, 0)   # NONE
    S.FullyConnectedOptionsAddWeightsFormat(bb, 0)             # DEFAULT -> [N,K]
    S.FullyConnectedOptionsAddKeepNumDims(bb, True)
    S.FullyConnectedOptionsAddAsymmetricQuantizeInputs(bb, False)
    return S.FullyConnectedOptionsEnd(bb)


def concat_options(bb, axis):
    S.ConcatenationOptionsStart(bb)
    S.ConcatenationOptionsAddAxis(bb, axis)
    S.ConcatenationOptionsAddFusedActivationFunction(bb, 0)    # NONE
    return S.ConcatenationOptionsEnd(bb)


# ------------------------------------------------------------------- data
def load_tied_weights(weights_dir=None):
    """embed.weight (tied LM head) -- one array, alias must be the same object."""
    t0 = time.time()
    w_embed = qw.get("embed.weight", path=weights_dir)
    w_alias = qw.get("lm_head.weight", path=weights_dir)
    tied_identity = w_alias is w_embed
    if not tied_identity:
        raise SystemExit("FIXTURE BUG: lm_head.weight is not the same object as embed.weight")
    if w_embed.dtype != np.float16 or list(w_embed.shape) != [N, K]:
        raise SystemExit("embed.weight shape/dtype %s %s != [%d,%d] fp16"
                         % (w_embed.shape, w_embed.dtype, N, K))
    entry = qw.spec("embed.weight", path=weights_dir)
    info = {
        "tensor": "embed.weight",
        "alias": "lm_head.weight",
        "tied_identity_check": bool(tied_identity),
        "shape": list(w_embed.shape),
        "dtype": "float16",
        "bytes": int(w_embed.nbytes),
        "sha256": entry["sha256"],
        "fixture_dir": str(qw.resolve_dir(weights_dir)),
        "weights_digest": qw.manifest(weights_dir).get("weights_digest"),
        "load_seconds": round(time.time() - t0, 2),
    }
    return w_embed, info


def make_input(weights_dir=None):
    """8 rows of a decorrelated fixture tensor, each scaled to row RMS = 1.

    Stand-in for the final RMSNorm output (row RMS ~ 1). Deliberately NOT embed rows:
    an embed row dotted with its own row gives 896*rms ~ 13.68, which would dominate a
    max|out|-based tolerance (see module docstring + docs/OP-lm-head.md 2.4).
    """
    src = INPUT_TENSOR
    rows = np.asarray(qw.get(src, path=weights_dir)[INPUT_ROWS, :], dtype=np.float64)
    if rows.shape != (M, K):
        raise SystemExit("%s rows %s give %s != (%d,%d)" % (src, INPUT_ROWS, rows.shape, M, K))
    rms = np.sqrt(np.mean(rows * rows, axis=1, keepdims=True))      # per-row RMS
    x = (rows / rms).astype(np.float16)
    got = np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=1))
    stats = {
        "provenance": "%s rows %s, each row scaled to RMS=1" % (src, list(INPUT_ROWS)),
        "source_tensor": src,
        "source_sha256": qw.spec(src, path=weights_dir)["sha256"],
        "rows": list(INPUT_ROWS),
        "row_rms": [round(float(v), 6) for v in got],
        "row_norm": [round(float(np.linalg.norm(x[i].astype(np.float64))), 4) for i in range(M)],
        "abs_max": float(np.abs(x.astype(np.float64)).max()),
    }
    return x, stats


def golden(x, w_embed):
    """Independent numpy float32 reference: expected = fp16(x_f32 @ W_f32^T + 0)."""
    t0 = time.time()
    xf = x.astype(np.float32)                                     # [8,896]
    exp = np.empty((M, N), dtype=np.float16)
    for v0 in range(0, N, GOLDEN_CHUNK):
        v1 = min(v0 + GOLDEN_CHUNK, N)
        w = np.asarray(w_embed[v0:v1], dtype=np.float32)           # [chunk,896]
        acc = xf @ w.T                                             # [8,chunk] fp32 (BLAS)
        exp[:, v0:v1] = acc.astype(np.float16)                     # + zero bias (no-op)
    e = exp.astype(np.float64)
    stats = {
        "seconds": round(time.time() - t0, 2),
        "chunk_rows": GOLDEN_CHUNK,
        "abs_max": float(np.abs(e).max()),
        "min": float(e.min()),
        "max": float(e.max()),
        "std": float(e.std()),
        "bytes": int(exp.nbytes),
    }
    return exp, stats


def tolerance(golden_stats, noise_floor=0.0, revision=1):
    """Register the contract BEFORE the device run (NOTES 10.1)."""
    mx = golden_stats["abs_max"]
    if mx > 1.0:
        # outputs are not O(1) -> 10.1 exception: atol = rtol = k * 2^-11 * max|out|
        atol = rtol = TOL_K * FP16_ULP_REL * mx
        rule = "NOTES 10.1 exception: atol=rtol=%.1f*2^-11*max|exp|" % TOL_K
        why = ("|logits| max=%.3f > 1, so NOTES 4's absolute 2e-2 would be only %.2e relative; "
               "threshold derived from the fp16 mantissa step (2^-11) x %g x max|exp| -- "
               "device-independent lower bound, no device data used to pick it"
               % (mx, 2e-2 / mx, TOL_K))
    else:
        atol = rtol = 1e-3
        rule = "NOTES 4 degenerate rule: noise floor 0 -> fp16 last-digit scale"
        why = "|logits| max=%.3f <= 1 so the O(1) branch applies" % mx
    block = {
        "revision": revision,
        "registered_before_run": True,
        "policy": rule,
        "reason": why,
        "formula": "atol = rtol = %g * 2^-11 * max|exp| = %.6g" % (TOL_K, atol),
        "max_abs_expected": mx,
        "fp16_ulp_rel": FP16_ULP_REL,
        "rel_equivalent": "max_abs/max|exp| <= %.3e" % (atol / mx + rtol),
        "under_2e-2_cap": bool(atol + rtol * mx < 2e-2),
        "noise_floor_used": noise_floor,
        "note": ("top-1/argmax is NOT a criterion (ADR 0003): under frozen random weights "
                 "top-1 carries no semantics and one fp16 ULP flips near-ties. argmax is "
                 "frozen until the real-weight stage (last step)."),
    }
    return float(atol), float(rtol), block


# ------------------------------------------------------------------ model
def build_model(x, w_embed, chunks):
    """Return (tflite_bytes, model_accumulator) for chunks=1 (single FC) or >1 (FC+CONCAT)."""
    if N % chunks:
        raise SystemExit("vocab %d is not divisible by chunks=%d" % (N, chunks))
    n_chunk = N // chunks
    w_bytes_total = int(w_embed.nbytes) + chunks * n_chunk * 2   # weights + zero bias
    m = Model(initial_size=w_bytes_total + (16 << 20))           # avoid doubling copies

    t_in = m.act("input", [M, K], bo.F16)
    outs = []
    for i in range(chunks):
        v0, v1 = i * n_chunk, (i + 1) * n_chunk
        w = np.ascontiguousarray(w_embed[v0:v1])                 # view/copy of the ONE array
        t_w = m.const("weights_%d" % i if chunks > 1 else "weights", [n_chunk, K], bo.F16, w)
        t_b = m.const("bias_%d" % i if chunks > 1 else "bias", [n_chunk], bo.F16,
                      np.zeros(n_chunk, dtype=np.float16))
        t_y = m.act("fc_out_%d" % i if chunks > 1 else "output", [M, n_chunk], bo.F16)
        m.add_op(bo.BuiltinOperator.FULLY_CONNECTED, [t_in, t_w, t_b], [t_y],
                 bo.BuiltinOptions.FullyConnectedOptions, fc_options(m.b))
        outs.append(t_y)

    if chunks == 1:
        t_out = outs[0]
    else:
        t_out = m.act("output", [M, N], bo.F16)
        m.add_op(bo.BuiltinOperator.CONCATENATION, outs, [t_out],
                 bo.BuiltinOptions.ConcatenationOptions, concat_options(m.b, 1))

    desc = "lm_head T=%d K=%d vocab=%d fp16 tied%s" % (M, K, N,
                                                       "" if chunks == 1 else " chunks=%d" % chunks)
    blob = m.finish([t_in], [t_out], desc)
    return blob, m


def describe_tflite(path):
    """Read the model back with the tflite schema: prove one weight copy is materialised."""
    buf = open(path, "rb").read()
    mod = S.Model.GetRootAsModel(buf, 0)
    sg = mod.Subgraphs(0)
    tensors = [sg.Tensors(i) for i in range(sg.TensorsLength())]
    bufs = []
    for i in range(mod.BuffersLength()):
        b = mod.Buffers(i)
        n = b.DataLength()
        bufs.append(int(n) if n else 0)
    ops = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        code = mod.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        ops.append(int(code))
    big = sorted(bufs, reverse=True)[:3]
    return {
        "tensor_count": len(tensors),
        "buffer_count": len(bufs),
        "buffer_sizes_top3": big,
        "constant_bytes": sum(bufs),
        "operators": ops,
        "sha256": hashlib.sha256(buf).hexdigest(),
    }


# ------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the LM head (tied embedding output projection) op.")
    ap.add_argument("--out", default=OPDIR + "/model.tflite")
    ap.add_argument("--chunks", type=int, default=1, help="1 = single FC (default); e.g. 4 = FC+CONCAT fallback")
    ap.add_argument("--weights-dir", default=None, help="fixture dir (default $QWEN25_WEIGHTS_DIR)")
    ap.add_argument("--noise-floor", type=float, default=0.0,
                    help="measured max|r1-r2| (0.0 at registration time, before any device run)")
    args = ap.parse_args(argv)

    d = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(os.path.join(d, "golden"), exist_ok=True)

    w_embed, winfo = load_tied_weights(args.weights_dir)
    x, xstats = make_input(args.weights_dir)
    exp, gstats = golden(x, w_embed)
    atol, rtol, tol = tolerance(gstats, args.noise_floor)

    t0 = time.time()
    blob, m = build_model(x, w_embed, args.chunks)
    build_secs = round(time.time() - t0, 2)

    with open(args.out, "wb") as f:
        f.write(blob)
    x.tofile(os.path.join(d, "golden", "input.bin"))
    exp.tofile(os.path.join(d, "golden", "expected.bin"))

    tinfo = describe_tflite(args.out)
    weight_bytes = int(w_embed.nbytes)
    overhead = int(tinfo["constant_bytes"]) - weight_bytes
    want_bias = 303872 * args.chunks
    single_copy = (int(tinfo["constant_bytes"]) == weight_bytes + want_bias)

    meta = {
        "op": "FULLY_CONNECTED" if args.chunks == 1 else "FULLY_CONNECTED+CONCATENATION",
        "graph": ("x[%d,%d] --FC(W=embed.weight tied, b=0)--> logits[%d,%d] fp16" % (M, K, M, N)
                  if args.chunks == 1 else
                  "x[%d,%d] --FC_i(W=embed.weight[%d:%d], b=0)--> y_i[%d,%d] x%d, CONCAT(axis=1) --> logits[%d,%d] fp16"
                  % (M, K, 0, N // args.chunks, M, N // args.chunks, args.chunks, M, N)),
        "shape": {"M": M, "K": K, "N": N},
        "dtype": "fp16",
        "chunks": args.chunks,
        "chunk_n": N // args.chunks,
        "input": {"name": "input", "shape": [M, K], "type": "float16", "bytes": int(x.nbytes),
                  "provenance": xstats["provenance"], "row_rms": xstats["row_rms"]},
        "output": {"name": "output", "shape": [M, N], "type": "float16", "bytes": int(exp.nbytes)},
        "quant": {"input": None, "weights": None, "bias": None, "output": None},
        "weights_shape": [N, K],
        "weights_source": winfo,
        "tied_materialised_once": bool(single_copy),
        "constant_bytes": int(tinfo["constant_bytes"]),
        "weight_overhead_bytes": int(overhead),
        "bias": {"value": "zeros", "elements": N, "bytes": want_bias,
                 "why": "ncc requires a 3rd FULLY_CONNECTED input (NOTES 5-1); real LM head has no bias"},
        "reference": "float32",
        "atol": atol,
        "rtol": rtol,
        "tolerance_revision": tol,
        "int8_tol": 0,
        "golden": gstats,
        "tflite": os.path.basename(args.out),
        "tflite_bytes": len(blob),
        "tflite_tensors": tinfo["tensor_count"],
        "tflite_buffers": tinfo["buffer_count"],
        "tflite_buffer_sizes_top3": tinfo["buffer_sizes_top3"],
        "tflite_sha256": tinfo["sha256"],
        "build_seconds": build_secs,
        "builder": os.path.basename(__file__),
    }
    with open(os.path.join(d, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log = [
        "op=%s chunks=%d" % (meta["op"], args.chunks),
        "fixture=%s weights_digest=%s" % (winfo["fixture_dir"], winfo["weights_digest"]),
        "embed.weight sha256=%s shape=%s bytes=%d (tied alias lm_head.weight: same object=%s)"
        % (winfo["sha256"], winfo["shape"], winfo["bytes"], winfo["tied_identity_check"]),
        "input=%s row_rms=%s" % ([M, K], xstats["row_rms"]),
        "golden: abs_max=%.6f min=%.6f max=%.6f std=%.6f  (%.2f s)"
        % (gstats["abs_max"], gstats["min"], gstats["max"], gstats["std"], gstats["seconds"]),
        "tflite=%d B  tensors=%d buffers=%d constant_bytes=%d weight=%d overhead=%d (bias+zp=%d)"
        % (len(blob), tinfo["tensor_count"], tinfo["buffer_count"], tinfo["constant_bytes"],
           weight_bytes, overhead, want_bias),
        "one-weight-copy check: %s" % ("PASS" if single_copy else "FAIL"),
        "tolerance (registered BEFORE device run): atol=%.6g rtol=%.6g  [%s]"
        % (atol, rtol, tol["policy"]),
        "build %.2f s" % build_secs,
    ]
    with open(os.path.join(d, "build.log"), "w") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    if not single_copy:
        raise SystemExit("FATAL: constant bytes %d != one weight copy + bias (%d)"
                         % (tinfo["constant_bytes"], weight_bytes + want_bias))
    return 0


if __name__ == "__main__":
    sys.exit(main())