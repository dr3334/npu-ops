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

# 每个算子：acc 用什么、乘数是什么（TFLite 的量化语义）
#   ADD/SUB: acc 在 input scale 的格子里 -> multiplier = s_in / s_out
#   MUL    : acc = a*b，单位是 s_a*s_b        -> multiplier = s_a*s_b / s_out
#   FC     : acc = a @ w.T（W 是 [N,K]，第二个输入），单位是 s_a*s_w
OP_ACC = ("add", "sub", "mul", "fc", "softmax")


def ratio_for(op, s_a, s_b, s_out):
    if op in ("add", "sub"):
        return s_a / s_out
    if op in ("mul", "fc"):
        return (s_a * s_b) / s_out
    if op == "softmax":
        return 1.0          # softmax 不是"整数累加 + requant"，没有 multiplier 这个概念
    raise ValueError("golden: unsupported op %r" % (op,))


def reference(op, a_codes, b_codes, s_a, s_b, s_out, zp_out=0):
    """-> (expected int8 array, int64 accumulator, ratio)

    FC 的 a 是 [M,K]、b 是 [N,K]（TFLite 的权重布局），输出按行展平是 [M,N]。
    """
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
