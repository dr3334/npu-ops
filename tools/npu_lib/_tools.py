"""把仓库的 tools/ 放进 sys.path，以便 import 公共模块（quant_contract / build_ops）。

规则（.qwen25-impl/NOTES.md §8）：只 import、不修改公共工具。
"""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]


def ensure():
    p = str(TOOLS)
    if p not in sys.path:
        sys.path.insert(0, p)
    return TOOLS
