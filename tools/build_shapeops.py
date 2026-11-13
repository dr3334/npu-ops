#!/usr/bin/env python3
"""Shape / layout operators: RESHAPE, TRANSPOSE, CONCATENATION, SPLIT, PAD.

    /usr/bin/python3 tools/build_shapeops.py --list
    /usr/bin/python3 tools/build_shapeops.py <spec> [--out ops/<name>]

Every spec writes the same contract as the other probes:
    model.tflite  model.json  golden/input.bin (+ input1.bin)  golden/expected.bin

WHY these models look the way they do (this is experiment design, not incidental code):

* The activations carry **no** quantization-induced arithmetic: input scale == output
  scale == 1/127, zero point 0.  All five operators are pure data movement in TFLite
  semantics, so if the device reorders/misplaces/pads bytes wrongly the comparison in
  tools/compare.py is an exact integer test (int8_tol = 0) - there is no requant
  rounding to hide behind (contrast with FC, see reports/S2-*.md).
* Input codes are a *seeded random permutation* of 0..N-1 reduced mod 255.  For every
  model smaller than 255 elements that makes every code unique -> any positional error
  (transpose instead of reshape, swapped concat halves, split halves exchanged, pad
  inserted at the wrong edge) shows up as a large mismatch count instead of a lucky hit.
* PAD uses codes that are never 0, so the padded border (constant 0) is unambiguous:
  a wrong pad layout cannot masquerade as "the data was already there".
* RESHAPE is fed the new shape twice (shape tensor + ReshapeOptions.new_shape) exactly
  like the TFLite converter does, so the compiler sees a normal model rather than a
  hand-crafted one.

Golden data is produced with plain numpy (reshape/transpose/concatenate/split/pad) and
shares no code with the flatbuffer construction (AGENTS.md R3).
"""
import argparse
import json
import os

import flatbuffers
import numpy as np
import tflite as S

import build_ops as bo

SEED = 20261112
S8 = 1.0 / 127.0            # single activation scale for in AND out (no requant)
I8, I32 = bo.I8, bo.I32


def codes(n, seed):
    """Seeded permutation of 0..n-1 mapped into int8 - unique for n <= 255."""
    rng = np.random.default_rng(seed)
    return ((rng.permutation(n) % 255) - 127).astype(np.int8)


def nonzero_codes(n, seed):
    c = codes(n, seed)
    return np.where(c == 0, np.int8(-42), c).astype(np.int8)


class M:
    """Tiny flatbuffer model accumulator (buffer 0 stays the empty placeholder)."""

    def __init__(self):
        self.b = flatbuffers.Builder(8192)
        self.buffers = [bo.buf_empty(self.b)]
        self.tensors = []
        self.opcodes = []
        self.ops = []

    def buf(self, arr):
        self.buffers.append(bo.buf_data(self.b, arr))
        return len(self.buffers) - 1

    def const(self, name, shape, ttype, arr):
        """A tensor that lives in the model file (weights / shape / perm / paddings)."""
        self.tensors.append(bo.tensor(self.b, name, shape, ttype, self.buf(arr)))
        return len(self.tensors) - 1

    def io(self, name, shape, ttype, quant=True):
        q = bo.quant_tensor(self.b, [S8], [0]) if quant else None
        self.tensors.append(bo.tensor(self.b, name, shape, ttype, 0, q))
        return len(self.tensors) - 1

    def out(self, name, shape, ttype, quant=True):
        # separate empty buffer, like tools/build_add.py does for its output
        self.buffers.append(bo.buf_empty(self.b))
        bi = len(self.buffers) - 1
        q = bo.quant_tensor(self.b, [S8], [0]) if quant else None
        self.tensors.append(bo.tensor(self.b, name, shape, ttype, bi, q))
        return len(self.tensors) - 1

    def add_op(self, builtin, ins, outs, opt_type=None, opt=None):
        self.opcodes.append(bo.opcode(self.b, builtin))
        idx = len(self.opcodes) - 1
        self.ops.append(bo.operator(self.b, idx, ins, outs, opt_type, opt))

    def finish(self, ins, outs, desc):
        return bo.finish(self.b, self.buffers, self.opcodes, self.tensors, self.ops,
                         ins, outs, desc)


def emit(out_dir, blob, meta, inputs, expected):
    os.makedirs(os.path.join(out_dir, "golden"), exist_ok=True)
    with open(os.path.join(out_dir, "model.tflite"), "wb") as f:
        f.write(blob)
    for i, (nm, arr) in enumerate(inputs):
        fn = "input.bin" if i == 0 else f"input{i}.bin"
        arr.tofile(os.path.join(out_dir, "golden", fn))
    expected.tofile(os.path.join(out_dir, "golden", "expected.bin"))
    meta["tflite"] = "model.tflite"
    meta["tflite_bytes"] = len(blob)
    meta["seed"] = SEED
    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{out_dir}: {len(blob)} B  in={[a.shape for _, a in inputs]} "
          f"out={expected.shape} dtype={meta['dtype']}")


# ------------------------------------------------------------------ builders
def build_reshape(in_shape, out_shape, out_dir):
    x = codes(int(np.prod(in_shape)), 11).reshape(in_shape)
    y = x.reshape(out_shape)                       # numpy: independent reference

    m = M()
    t_in = m.io("input", list(in_shape), I8)
    t_shape = m.const("new_shape", [len(out_shape)], I32,
                      np.asarray(out_shape, np.int32))
    t_out = m.out("output", list(out_shape), I8)

    b = m.b
    nsv = b.CreateNumpyVector(np.asarray(out_shape, np.int32))   # before the table
    S.ReshapeOptionsStart(b)
    S.ReshapeOptionsAddNewShape(b, nsv)
    opt = S.ReshapeOptionsEnd(b)
    m.add_op(S.BuiltinOperator.RESHAPE, [t_in, t_shape], [t_out],
             S.BuiltinOptions.ReshapeOptions, opt)
    blob = m.finish([0], [2], f"reshape {in_shape}->{out_shape} int8")

    meta = {"op": "RESHAPE", "shape": {"in": list(in_shape), "out": list(out_shape)},
            "dtype": "int8", "input": {"name": "input", "shape": list(in_shape), "type": "int8"},
            "output": {"name": "output", "shape": list(out_shape), "type": "int8"},
            "quant": {"input": S8, "output": S8}, "int8_tol": 0,
            "reference": "numpy_exact", "axis_semantics": "flat row-major (no reorder)"}
    emit(out_dir, blob, meta, [("input", x)], y)


def build_transpose(in_shape, perm, out_dir, dtype="int8"):
    """dtype="fp16" exists only to discriminate *why* an 8-channel boundary tensor is
    bridged: if the rule is "16 int8 bytes per channel row", fp16 C=8 (also 16 B) would
    compile while int8 C=8 does not; if the rule is "16 channels", fp16 C=8 also fails.
    Values are small integers, exactly representable in fp16, so the comparison stays exact.
    """
    x = codes(int(np.prod(in_shape)), 21).reshape(in_shape)
    y = np.transpose(x, perm)                      # numpy: independent reference

    int8 = dtype == "int8"
    ttype = I8 if int8 else bo.F16
    xin = x if int8 else x.astype(np.float16)
    yexp = y if int8 else y.astype(np.float16)

    m = M()
    b = m.b
    t_in = m.io("input", list(in_shape), ttype, quant=int8)
    t_perm = m.const("perm", [len(perm)], I32, np.asarray(perm, np.int32))
    t_out = m.out("output", list(y.shape), ttype, quant=int8)
    S.TransposeOptionsStart(b)
    opt = S.TransposeOptionsEnd(b)
    m.add_op(S.BuiltinOperator.TRANSPOSE, [t_in, t_perm], [t_out],
             S.BuiltinOptions.TransposeOptions, opt)
    blob = m.finish([0], [2], f"transpose {in_shape} perm={perm} {dtype}")

    meta = {"op": "TRANSPOSE", "shape": {"in": list(in_shape), "perm": perm,
                                         "out": list(y.shape)},
            "dtype": dtype,
            # tools/compare.py keys its dtype table on exactly these names
            "input": {"name": "input", "shape": list(in_shape),
                      "type": "int8" if int8 else "float16"},
            "output": {"name": "output", "shape": list(y.shape),
                       "type": "int8" if int8 else "float16"},
            "quant": {"input": None if not int8 else S8,
                      "output": None if not int8 else S8},
            "int8_tol": 0, "reference": "numpy_exact",
            "note": "fp16 variant only for the channel-boundary discriminator"}
    emit(out_dir, blob, meta, [("input", xin)], yexp)


def build_concat(a_shape, b_shape, axis, out_dir):
    na, nb = int(np.prod(a_shape)), int(np.prod(b_shape))
    a = codes(na, 31).reshape(a_shape)
    b = (codes(nb, 32).astype(np.int16) + np.int16(1)).clip(-127, 127).astype(np.int8)
    b = b.reshape(b_shape)
    y = np.concatenate([a, b], axis=axis)          # numpy: independent reference

    m = M()
    bb = m.b
    t_a = m.io("a", list(a_shape), I8)
    t_b = m.io("b", list(b_shape), I8)
    t_out = m.out("out", list(y.shape), I8)
    S.ConcatenationOptionsStart(bb)
    S.ConcatenationOptionsAddAxis(bb, axis)
    S.ConcatenationOptionsAddFusedActivationFunction(bb, 0)   # NONE
    opt = S.ConcatenationOptionsEnd(bb)
    m.add_op(S.BuiltinOperator.CONCATENATION, [t_a, t_b], [t_out],
             S.BuiltinOptions.ConcatenationOptions, opt)
    blob = m.finish([0, 1], [2], f"concat axis={axis} int8")

    meta = {"op": "CONCATENATION", "shape": {"a": list(a_shape), "b": list(b_shape),
                                             "axis": axis, "out": list(y.shape)},
            "dtype": "int8",
            "inputs": [{"name": "a", "shape": list(a_shape), "type": "int8"},
                       {"name": "b", "shape": list(b_shape), "type": "int8"}],
            "output": {"name": "out", "shape": list(y.shape), "type": "int8"},
            "quant": {"input": S8, "output": S8}, "int8_tol": 0,
            "reference": "numpy_exact"}
    # tools/e2e.sh feeds golden/input.bin and golden/input1.bin in that order
    emit(out_dir, blob, meta, [("a", a), ("b", b)], y)


def build_split(in_shape, axis, num, part, out_dir):
    """One model per output slice: the graph output is slice #part, the other slice
    stays an unused operator output. tools/e2e.sh only reads ONE output file."""
    x = codes(int(np.prod(in_shape)), 41).reshape(in_shape)
    parts = np.split(x, num, axis=axis)            # numpy: independent reference
    y = parts[part]

    m = M()
    b = m.b
    t_axis = m.const("axis", [1], I32, np.asarray([axis], np.int32))
    t_in = m.io("input", list(in_shape), I8)
    t_o = [m.out(f"out{i}", list(y.shape), I8) for i in range(num)]
    S.SplitOptionsStart(b)
    S.SplitOptionsAddNumSplits(b, num)
    opt = S.SplitOptionsEnd(b)
    m.add_op(S.BuiltinOperator.SPLIT, [t_axis, t_in], t_o,
             S.BuiltinOptions.SplitOptions, opt)
    # inputs of the subgraph: the axis tensor is a graph input in TFLite models emitted
    # by the converter only when it is dynamic; here it is a constant, so the only
    # subgraph input is the data tensor.
    blob = m.finish([1], [2 + part], f"split axis={axis} n={num} int8")

    meta = {"op": "SPLIT", "shape": {"in": list(in_shape), "axis": axis, "num_splits": num,
                                     "graph_output": part, "out": list(y.shape)},
            "dtype": "int8", "input": {"name": "input", "shape": list(in_shape), "type": "int8"},
            "output": {"name": f"out{part}", "shape": list(y.shape), "type": "int8"},
            "quant": {"input": S8, "output": S8}, "int8_tol": 0,
            "reference": "numpy_exact",
            "note": f"slice {part} of {num}; the other slice(s) are unused operator outputs"}
    emit(out_dir, blob, meta, [("input", x)], y)


def build_pad(in_shape, paddings, out_dir):
    x = nonzero_codes(int(np.prod(in_shape)), 51).reshape(in_shape)
    y = np.pad(x, paddings, mode="constant", constant_values=0)   # numpy reference

    m = M()
    b = m.b
    t_in = m.io("input", list(in_shape), I8)
    t_pad = m.const("paddings", [len(paddings), 2], I32,
                    np.asarray(paddings, np.int32).reshape(-1, 2))
    t_out = m.out("output", list(y.shape), I8)
    S.PadOptionsStart(b)
    opt = S.PadOptionsEnd(b)
    m.add_op(S.BuiltinOperator.PAD, [t_in, t_pad], [t_out],
             S.BuiltinOptions.PadOptions, opt)
    blob = m.finish([0], [2], f"pad {in_shape} int8")

    meta = {"op": "PAD", "shape": {"in": list(in_shape), "paddings": paddings,
                                   "out": list(y.shape)},
            "dtype": "int8", "input": {"name": "input", "shape": list(in_shape), "type": "int8"},
            "output": {"name": "output", "shape": list(y.shape), "type": "int8"},
            "quant": {"input": S8, "output": S8}, "int8_tol": 0,
            "reference": "numpy_exact", "pad_value": 0,
            "note": "input codes are never 0, so the constant border is unambiguous"}
    emit(out_dir, blob, meta, [("input", x)], y)


def build_chain_concat_pad(a_shape, b_shape, axis, paddings, out_dir):
    """CONCAT -> PAD in one graph.

    A single-op graph *forces* the compiler to keep the layer, so it cannot answer the
    question the transformer block actually asks: does a shape op survive when it sits
    between two real layers, or is it folded away?  Two pure data-movement ops chained
    together answer it without any requant (same scale in/out, exact integer contract).
    """
    na, nb = int(np.prod(a_shape)), int(np.prod(b_shape))
    a = codes(na, 61).reshape(a_shape)
    b = codes(nb, 62).reshape(b_shape)
    mid = np.concatenate([a, b], axis=axis)                      # numpy reference
    y = np.pad(mid, paddings, mode="constant", constant_values=0)

    m = M()
    bf = m.b
    t_a = m.io("a", list(a_shape), I8)
    t_b = m.io("b", list(b_shape), I8)
    t_mid = m.out("concat_out", list(mid.shape), I8)
    t_pad = m.const("paddings", [len(paddings), 2], I32,
                    np.asarray(paddings, np.int32).reshape(-1, 2))
    t_out = m.out("output", list(y.shape), I8)

    S.ConcatenationOptionsStart(bf)
    S.ConcatenationOptionsAddAxis(bf, axis)
    S.ConcatenationOptionsAddFusedActivationFunction(bf, 0)
    copt = S.ConcatenationOptionsEnd(bf)
    m.add_op(S.BuiltinOperator.CONCATENATION, [t_a, t_b], [t_mid],
             S.BuiltinOptions.ConcatenationOptions, copt)

    S.PadOptionsStart(bf)
    popt = S.PadOptionsEnd(bf)
    m.add_op(S.BuiltinOperator.PAD, [t_mid, t_pad], [t_out],
             S.BuiltinOptions.PadOptions, popt)
    blob = m.finish([0, 1], [4], "concat axis3 -> pad int8")

    meta = {"op": "CONCAT+PAD", "shape": {"a": list(a_shape), "b": list(b_shape),
                                          "axis": axis, "out": list(y.shape)},
            "dtype": "int8",
            "inputs": [{"name": "a", "shape": list(a_shape), "type": "int8"},
                       {"name": "b", "shape": list(b_shape), "type": "int8"}],
            "output": {"name": "output", "shape": list(y.shape), "type": "int8"},
            "quant": {"input": S8, "output": S8}, "int8_tol": 0,
            "reference": "numpy_exact",
            "note": "chain CONCAT->PAD: tests whether shape ops survive between layers"}
    emit(out_dir, blob, meta, [("a", a), ("b", b)], y)


# --------------------------------------------------------------------- specs
SPECS = {
    # flat-identical reshape, rank 2 -> rank 3
    "reshape_1x64_to_1x4x16_int8":
        lambda d: build_reshape([1, 64], [1, 4, 16], d),
    # 4D -> 4D, same element count, channel size unchanged (8): still flat-identical,
    # so any data movement here would be pure layout bookkeeping by the compiler
    "reshape_1x4x8x8_to_1x8x4x8_int8":
        lambda d: build_reshape([1, 4, 8, 8], [1, 8, 4, 8], d),
    # NHWC (H,W,C) -> NWHC: a genuine reorder, cannot be folded to a memcpy
    "transpose_1x4x8x16_int8":
        lambda d: build_transpose([1, 4, 8, 16], [0, 2, 1, 3], d),
    # channel concat: the case a real transformer block needs (q/k/v -> one tensor)
    "concat_axis3_int8":
        lambda d: build_concat([1, 4, 8, 8], [1, 4, 8, 8], 3, d),
    # split: two models so each output slice is checked separately
    "split_axis3_part1_int8":
        lambda d: build_split([1, 4, 8, 16], 3, 2, 0, d),
    "split_axis3_part2_int8":
        lambda d: build_split([1, 4, 8, 16], 3, 2, 1, d),
    # zero pad on H and W (border pixels), channel untouched
    "pad_1x4x8x16_int8":
        lambda d: build_pad([1, 4, 8, 16], [[0, 0], [1, 1], [1, 1], [0, 0]], d),

    # ---- boundary probes for the three compile failures above -----------------
    # (b1) same H/W swap as reshape_1x4x8x8_to_1x8x4x8 but with C=16: discriminates
    #      "spatial reorder is not free" from "channel must be a multiple of 16"
    "reshape_1x4x8x16_to_1x8x4x16_int8":
        lambda d: build_reshape([1, 4, 8, 16], [1, 8, 4, 16], d),
    # (b2) rank 2 -> rank 4, output channel 8 (no spatial swap at all): if this fails
    #      while reshape_1x64_to_1x4x16 passes, the blocker is C=8, not the reorder
    "reshape_1x4x64_to_1x4x8x8_int8":
        lambda d: build_reshape([1, 4, 64], [1, 4, 8, 8], d),
    # (b3/b4) concat with C=16 (axis 3 = channel, axis 1 = H)
    "concat_axis3_c16_int8":
        lambda d: build_concat([1, 4, 8, 16], [1, 4, 8, 16], 3, d),
    "concat_axis1_c16_int8":
        lambda d: build_concat([1, 4, 8, 16], [1, 4, 8, 16], 1, d),
    # (b5/b6) split of a 32-channel tensor: slices are C=16
    "split_axis3_c16_part1_int8":
        lambda d: build_split([1, 4, 8, 32], 3, 2, 0, d),
    "split_axis3_c16_part2_int8":
        lambda d: build_split([1, 4, 8, 32], 3, 2, 1, d),
    # (b7) same op as transpose_1x4x8x16_int8 (which compiles) but the external input
    #      has C=8: if this one fails with the same "gap between Input and MDLA_5_0",
    #      the blocker is the *input tensor layout*, not the operator
    "transpose_1x4x8x8_int8":
        lambda d: build_transpose([1, 4, 8, 8], [0, 2, 1, 3], d),
    # (b8) input C=16 (fine) but the *output* last dim is 8: perm (0,3,1,2) gives
    #      [1,4,8,16] -> [1,16,4,8].  If this fails with "gap ... between MDLA_5_0 and
    #      Output", the multiple-of-16 rule also applies to external outputs.
    "transpose_1x4x8x16_perm0312_int8":
        lambda d: build_transpose([1, 4, 8, 16], [0, 3, 1, 2], d),
    # (b9) two shape ops chained: CONCAT(axis3, C=16) -> PAD.  Answers "is a shape op
    #      folded away when it sits between layers?" - a single-op graph cannot.
    "chain_concat_pad_int8":
        lambda d: build_chain_concat_pad([1, 4, 8, 16], [1, 4, 8, 16], 3,
                                         [[0, 0], [1, 1], [1, 1], [0, 0]], d),
    # (b10) fp16 with C=8: 8 channels x 2 B = 16 B per row.  Compiles => the rule is
    #       "16 bytes per channel row"; fails like int8 C=8 => the rule is "16 channels".
    "transpose_1x4x8x8_fp16":
        lambda d: build_transpose([1, 4, 8, 8], [0, 2, 1, 3], d, dtype="fp16"),
    # (b11) fp16 with C=4: only 8 B per channel row.  Predict FAIL if the threshold is
    #       16 bytes; combined with (b7)/(b10) this brackets the boundary from both sides.
    "transpose_1x4x8x4_fp16":
        lambda d: build_transpose([1, 4, 8, 4], [0, 2, 1, 3], d, dtype="fp16"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.list:
        print("\n".join(SPECS))
        return
    if a.spec not in SPECS:
        raise SystemExit("unknown spec; use --list")
    SPECS[a.spec](a.out or f"ops/{a.spec}")


if __name__ == "__main__":
    main()
