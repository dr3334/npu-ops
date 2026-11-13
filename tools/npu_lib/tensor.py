"""图里的张量：只描述形状/类型/来源，不做计算。"""


class Tensor:
    __slots__ = ("name", "shape", "dtype", "_producer")

    def __init__(self, name, shape, dtype, producer=None):
        self.name = name
        self.shape = tuple(int(d) for d in shape)
        self.dtype = dtype
        self._producer = producer          # None = 图输入；否则为 Op

    @property
    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self):
        return self.numel * self.dtype.itemsize

    @property
    def is_input(self):
        return self._producer is None

    @property
    def producer(self):
        return self._producer

    def __repr__(self):
        return "Tensor(%r, shape=%r, dtype=%s)" % (self.name, self.shape, self.dtype)
