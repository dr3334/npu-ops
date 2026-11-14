"""独立 CPU golden —— int64 精确累加 + 仓库的量化契约。

为什么必须走 tools/quant_contract.py：设备做的是 gemmlowp 的**两步整数舍入**
（SaturatingRoundingDoublingHighMul 然后 RoundingDivideByPOT），不是对精确有理数舍一次
（tools/quant_contract.py 的 docstring 有 145296 个值的对照证据）。这里把 acc 算成 int64，
再交给契约模块，所以 int8 结果是逐位的。

和 flatbuffer 生成完全分开：build_model.py 只写形状/scale，不碰数值（规则 R3）。
"""
import numpy as np

from . import _tools

_tools.ensure()
import quant_contract as qc  # noqa: E402  (公共工具，只 import)

def _fp16_bits(a):
    """fp16 码 -> 单调可比的 int16（负数取反），用来算 ULP 距离。"""
    b = np.asarray(a, dtype=np.float16).view(np.uint16).astype(np.int32)
    return np.where(b & 0x8000, 0x8000 - (b & 0x7FFF), b)


def ulp_diff(got_fp16, exp_fp16):
    """逐元素 ULP 距离（同号内用位模式差；跨零按位模式差也算得很保守）。"""
    return np.abs(_fp16_bits(got_fp16) - _fp16_bits(exp_fp16))


def reference_fp16(op, a_codes, b_codes, s_a=1.0, s_b=1.0, s_out=1.0, zp_out=0, axis=None):
    """fp16 的独立参考：用 float32 算再落回 fp16（规则 R3：与 flatbuffer 生成无关）。"""
    a = np.asarray(a_codes, dtype=np.float32)
    b = None if b_codes is None else np.asarray(b_codes, dtype=np.float32)
    if op == "add":
        y = a + b
    elif op == "sub":
        y = a - b
    elif op == "mul":
        y = a * b
    elif op == "mean":
        y = a.mean(axis=(-1 if axis is None else int(axis)), keepdims=True)
    elif op == "rsqrt":
        y = 1.0 / np.sqrt(a)
    else:
        raise ValueError("reference_fp16: unsupported op %r" % (op,))
    return np.asarray(y, dtype=np.float16)


# 每个算子：acc 用什么、乘数是什么（TFLite 的量化语义）
#   ADD/SUB: acc 在 input scale 的格子里 -> multiplier = s_in / s_out
#   MUL    : acc = a*b，单位是 s_a*s_b        -> multiplier = s_a*s_b / s_out
#   FC     : acc = a @ w.T（W 是 [N,K]，第二个输入），单位是 s_a*s_w
OP_ACC = ("add", "sub", "mul", "fc", "softmax", "mean", "rsqrt")


def ratio_for(op, s_a, s_b, s_out):
    if op in ("add", "sub"):
        return s_a / s_out
    if op in ("mul", "fc"):
        return (s_a * s_b) / s_out
    if op in ("softmax", "mean", "rsqrt"):
        return 1.0          # 这几个不是"整数累加 + requant"，没有 multiplier 这个概念
    raise ValueError("golden: unsupported op %r" % (op,))


def reference(op, a_codes, b_codes, s_a, s_b, s_out, zp_out=0, axis=None):
    """-> (expected int8 array, int64 accumulator, ratio)

    FC 的 a 是 [M,K]、b 是 [N,K]（TFLite 的权重布局），输出按行展平是 [M,N]。
    """
    if op == "mean":
        # 整数归约 + 仓库的两步舍入契约（与 ops/mean_4x16_int8 那条已上板的约定一致：
        # 输出 scale == 输入 scale，于是 multiplier = 1/K，码值 = round(sum/K)）
        a = np.asarray(a_codes, dtype=np.int64)
        ax = -1 if axis is None else int(axis)
        acc = a.sum(axis=ax, keepdims=True)
        k = a.shape[ax]
        ratio = s_a / (k * s_out)
        codes = qc.requant_int8(acc, ratio)
        if int(zp_out):
            codes = np.clip(codes.astype(np.int64) + int(zp_out), -128, 127)
        return codes.astype(np.int8), acc, ratio
    if op == "rsqrt":
        x = np.asarray(a_codes, dtype=np.float64) * s_a
        if (x <= 0).any():
            raise ValueError("golden rsqrt: 输入必须全为正（码值 > 0）")
        codes = np.clip(np.rint((1.0 / np.sqrt(x)) / s_out), -128, 127)
        return codes.astype(np.int8), None, 1.0
    if op == "softmax":
        # 浮点算再量化（设备是 LUT 定点实现，按登记容差比，不要求逐位）
        x = np.asarray(a_codes, dtype=np.float64) * s_a          # 输入 zp 约定为 0
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        p = e / e.sum(axis=-1, keepdims=True)
        codes = np.clip(np.rint(p / s_out).astype(np.int64) + int(zp_out), -128, 127)
        return codes.astype(np.int8), None, 1.0
    a = np.asarray(a_codes, dtype=np.int64)
    b = np.asarray(b_codes, dtype=np.int64)
    if op == "add":
        acc = a + b
    elif op == "sub":
        acc = a - b
    elif op == "mul":
        acc = a * b
    elif op == "fc":
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("golden fc: expected 2-D codes, got %r and %r"
                             % (a.shape, b.shape))
        acc = a @ b.T
    else:
        raise ValueError("golden: unsupported op %r" % (op,))
    ratio = ratio_for(op, s_a, s_b, s_out)
    return qc.requant_int8(acc, ratio), acc, ratio
