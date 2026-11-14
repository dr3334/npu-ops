#!/usr/bin/env python3
"""device-free 自检：把票面定死的接口形状与报错路径跑一遍（不碰板子、不编译）。

    PYTHONPATH=tools /usr/bin/python3 -m npu_lib.selftest
"""
import sys

import numpy as np

from . import DType, Graph, Tensor
from .errors import BroadCastError, UnsupportedError


def expect(exc, fn, what):
    try:
        fn()
    except exc as e:
        print("  ok   %-42s -> %s" % (what, type(e).__name__))
        return True
    except Exception as e:                     # noqa: BLE001
        print("  FAIL %-42s -> wrong exception %s: %s" % (what, type(e).__name__, e))
        return False
    print("  FAIL %-42s -> no exception" % what)
    return False


def main():
    ok = True
    q = DType.int8(1.0 / 127.0)
    print("[dtype]")
    ok &= repr(q) == "DType.int8(scale=0.007874015748031496, zero_point=0)"
    ok &= (DType.int8(1 / 127) == DType.int8(1 / 127))

    print("[unsupported entry points]")
    ok &= expect(NotImplementedError, lambda: DType.int16(1 / 32767.0), "DType.int16(scale)")
    ok &= expect(NotImplementedError, lambda: DType.fp16(), "DType.fp16()")
    ok &= expect(UnsupportedError, lambda: DType.int8(1 / 127, zero_point=-3),
                 "DType.int8(zp != 0)")

    print("[row rule / static shape]")
    ok &= expect(ValueError, lambda: Graph("g").input("a", (1, 100), q),
                 "input(1,100) violates 16 B row rule")
    ok &= expect(ValueError, lambda: Graph("g").input("a", (0, 64), q),
                 "input with a zero dim")

    print("[no broadcasting]")
    g = Graph("t")
    a = g.input("a", (1, 64), q)
    b = g.input("b", (1, 96), q)
    ok &= expect(BroadCastError, lambda: g.add(a, b), "add [1,64] + [1,96]")

    print("[one op per graph]")
    g2 = Graph("t2")
    x = g2.input("x", (1, 64), q)
    y = g2.input("y", (1, 64), q)
    z = g2.add(x, y)
    ok &= expect(UnsupportedError, lambda: g2.mul(z, y), "chained mul on an add graph")

    print("[graph ownership]")
    g3, g4 = Graph("t3"), Graph("t4")
    p = g3.input("p", (1, 64), q)
    r = g4.input("r", (1, 64), q)
    ok &= expect(ValueError, lambda: g4.add(p, r), "tensor from another graph")

    print("[dtype/shape of outputs]")
    ok &= (z.shape == (1, 64)) and (z.dtype == DType.int8(1 / 127))
    gm = Graph("t5")
    m1 = gm.input("m1", (1, 64), q)
    m2 = gm.input("m2", (1, 64), q)
    mo = gm.mul(m1, m2)
    ok &= bool(abs(mo.dtype.scale - q.scale * q.scale) < 1e-18)

    print("\nSELFTEST: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
