"""量化数据类型。

票面要求：
  * DType.int8(scale, zero_point=0) —— 本票只做这一档；
  * DType.int16(scale) / DType.fp16() —— 先留构造入口，未实现就抛 NotImplementedError。

为什么 int16/fp16 先不做：MTK 的 int16/fp16 路径有各自的 requant 契约（docs/OP-MATRIX.md、
docs/adr/0005-quantization-tier-target.md），本票的验收只在 int8 上有逐位证据。
"""
from .errors import UnsupportedError

INT8 = "int8"
INT16 = "int16"
FP16 = "fp16"

_ITEMSIZE = {INT8: 1, INT16: 2, FP16: 2}


class DType:
    """一个张量的量化描述：kind + scale + zero_point（per-tensor）。"""

    __slots__ = ("kind", "scale", "zero_point")

    def __init__(self, kind, scale, zero_point=0):
        if kind not in _ITEMSIZE:
            raise UnsupportedError("unknown dtype kind: %r" % (kind,))
        if not (scale > 0):
            raise ValueError("scale must be > 0, got %r" % (scale,))
        self.kind = kind
        self.scale = float(scale)
        self.zero_point = int(zero_point)

    # ---- 构造入口 -------------------------------------------------------
    @staticmethod
    def int8(scale, zero_point=0):
        """int8，per-tensor。zero_point 目前只支持 0（非对称量化未验证）。"""
        if int(zero_point) != 0:
            raise UnsupportedError(
                "DType.int8(zero_point=%d): 非对称量化还没有板端证据，本票只做 zp=0"
                % int(zero_point))
        return DType(INT8, scale, 0)

    @staticmethod
    def int8_asym(scale, zero_point=-128):
        """非对称 int8：只在**有板端证据**的地方用。

        目前唯一一处是 SOFTMAX 的输出（TFLite 惯例 s_out = 1/256、zero_point = -128）——
        [1,8,16] 上我们自己重放的结果与厂商设备输出逐字节相同（reports/npu-lib-softmax.md）。
        逐元素算子的非零 zp 仍然没有证据，所以 `DType.int8(zp != 0)` 继续拒绝。
        """
        zp = int(zero_point)
        if zp == 0:
            raise UnsupportedError("int8_asym(zp=0) 没有意义，用 DType.int8(scale)")
        return DType(INT8, scale, zp)

    @staticmethod
    def int16(scale):
        raise UnsupportedError(
            "DType.int16 还没实现：int16 的累加/requant 契约与 int8 不同"
            "（见 docs/OP-MATRIX.md、docs/adr/0005-quantization-tier-target.md），"
            "本票只交付 int8。构造入口保留，实现时替换这里即可。")

    @staticmethod
    def fp16():
        raise UnsupportedError(
            "DType.fp16 还没实现：逐元素 fp16 需要噪声底实测与阈值登记（NOTES §4），"
            "本票只交付 int8。构造入口保留。")

    # ---- 属性 -----------------------------------------------------------
    @property
    def itemsize(self):
        return _ITEMSIZE[self.kind]

    @property
    def name(self):
        return self.kind

    def replace(self, kind=None, scale=None, zero_point=None):
        return DType(kind or self.kind,
                     self.scale if scale is None else scale,
                     self.zero_point if zero_point is None else zero_point)

    def __eq__(self, other):
        return (isinstance(other, DType) and self.kind == other.kind
                and self.scale == other.scale and self.zero_point == other.zero_point)

    def __hash__(self):
        return hash((self.kind, self.scale, self.zero_point))

    def __repr__(self):
        return "DType.%s(scale=%r, zero_point=%d)" % (self.kind, self.scale, self.zero_point)
