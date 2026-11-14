"""npu_lib -- 用我们自己的库定义 -> 编译 -> 在 NPU 上运行逐元素算子.

    from npu_lib import Graph, DType

    q = DType.int8(1/127)
    g = Graph("add_1x64")
    a = g.input("a", (1, 64), q)
    b = g.input("b", (1, 64), q)
    c = g.add(a, b)
    g.compile()                       # model.tflite -> .dla -> 资产(命令流捕获) -> artifacts/<name>/
    out = g.run({"a": a_codes, "b": b_codes})   # 只走自建提交器 (/dev/apusys)
    g.verify(out)                     # 两道门：同源逐字节 + CPU golden 逐位

运行位置：**WSL 的 /usr/bin/python3**（仓库里唯一带 numpy/flatbuffers/tflite schema 的解释器，
见 .qwen25-impl/NOTES.md §2）。设备访问用 adb.exe（WSL interop），编译/生成用 WSL 原生工具。

术语（对齐 docs/adr/0002）：compile() 里的第 3 步会**用厂商 runtime 跑一次**来捕获命令流资产
-- 这是 ADR 0002 明确允许的「命令捕获」。run() 里绝不调用 libneuron / neuronrt。
"""
from .dtype import DType
from .errors import NpuLibError, BroadCastError, UnsupportedError
from .graph import Graph, Output, VerifyResult
from .tensor import Tensor

__all__ = [
    "Graph", "DType", "Tensor", "Output", "VerifyResult",
    "NpuLibError", "BroadCastError", "UnsupportedError",
]
__version__ = "0.1.0"
