import sys
from tflite import Model
def dump(p):
    m = Model.GetRootAsModel(open(p, "rb").read(), 0)
    sg = m.Subgraphs(0)
    out = []
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        out.append((t.Name().decode(), list(t.ShapeAsNumpy()), t.Type()))
    codes = [m.OperatorCodes(i).BuiltinCode() for i in range(m.OperatorCodesLength())]
    ops = []
    for k in range(sg.OperatorsLength()):
        o = sg.Operators(k)
        ops.append((codes[o.OpcodeIndex()], [o.Inputs(j) for j in range(o.InputsLength())],
                    [o.Outputs(j) for j in range(o.OutputsLength())]))
    return out, ops
a, ao = dump(sys.argv[1])
b, bo = dump(sys.argv[2])
print("tensors  A=%d B=%d   ops A=%d B=%d" % (len(a), len(b), len(ao), len(bo)))
for i in range(max(len(a), len(b))):
    ta = a[i] if i < len(a) else None
    tb = b[i] if i < len(b) else None
    if ta != tb:
        print("tensor[%d] A=%s B=%s" % (i, ta, tb))
for i in range(max(len(ao), len(bo))):
    oa = ao[i] if i < len(ao) else None
    ob = bo[i] if i < len(bo) else None
    if oa != ob:
        print("op[%d] A=%s B=%s" % (i, oa, ob))
