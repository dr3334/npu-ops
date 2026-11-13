#!/usr/bin/env python3
"""Dump a TFLite model's tensors + operators, for diffing two hand-built graphs.

    /usr/bin/python3 probe/dump_tflite.py <model.tflite> [--ops]

Print one line per tensor:  idx name shape type buffer  (buffer<>0 => has data)
Print one line per op:      idx OPCODE inputs -> outputs
"""
import sys

import tflite as S
from tflite import Model, TensorType


def main(path, show_ops=True):
    buf = open(path, "rb").read()
    m = Model.GetRootAsModel(buf, 0)
    sg = m.Subgraphs(0)
    print(f"== {path}  ({len(buf)} B, {sg.TensorsLength()} tensors, {sg.OperatorsLength()} ops)")
    tnames = []
    print("-- tensors")
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        name = t.Name().decode() if t.Name() else ""
        tnames.append(name)
        shape = [t.Shape(j) for j in range(t.ShapeLength())]
        ttype = str(int(t.Type()))
        print(f"  [{i:2d}] {name:14s} {str(shape):16s} {ttype:8s} buf={t.Buffer()}")
    if not show_ops:
        return
    print("-- ops")
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        oc = m.OperatorCodes(op.OpcodeIndex())
        code = oc.BuiltinCode()
        ins = [op.Inputs(j) for j in range(op.InputsLength())]
        outs = [op.Outputs(j) for j in range(op.OutputsLength())]
        opt = op.BuiltinOptionsType()
        print(f"  [{i:2d}] op={code} opt={opt} in={ins} out={outs}")
    print("-- inputs:", [sg.Inputs(j) for j in range(sg.InputsLength())],
          " outputs:", [sg.Outputs(j) for j in range(sg.OutputsLength())])


if __name__ == "__main__":
    main(sys.argv[1], "--no-ops" not in sys.argv)
