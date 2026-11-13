"""npu_lib 的异常类型（都是 NpuLibError 的子类，便于调用方一把抓）。"""


class NpuLibError(RuntimeError):
    """库级错误（编译/捕获/提交失败）。"""


class BroadCastError(ValueError, NpuLibError):
    """形状不同 -> 本库不支持广播，显式报错（票面要求）。"""


class UnsupportedError(NotImplementedError, NpuLibError):
    """接口留了入口但还没实现的路径（int16 / fp16 / 多算子图 / 非零 zero_point ...）。"""
