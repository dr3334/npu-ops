import sys, os
sys.path.insert(0, "tools")
import numpy as np
import build_decoder_layer as bdl

def build(kind, T):
    bdl.configure(T)
    P = bdl.fixture()
    D, H, HD = bdl.D, bdl.H, bdl.HD
    g = bdl.G()
    x = g.act("x", (T, D))
    z = lambda n: g.const(n, np.zeros(D, np.float16))
    if kind == "fc_only":                       # control: FC -> graph output
        n1 = bdl.rmsnorm_nodes(g, x, "n1", P["g1"])
        out = g.fc(g.add(n1, z("z"), "byp", (T, D)), "q_fc", P["Wq"], P["bq"], D)
    elif kind == "fc_reshape":                  # FC -> RESHAPE -> graph output
        n1 = bdl.rmsnorm_nodes(g, x, "n1", P["g1"])
        y = g.fc(g.add(n1, z("z"), "byp", (T, D)), "q_fc", P["Wq"], P["bq"], D)
        out = g.reshape(y, "q_3d", (T, H, HD), H * HD)
    elif kind == "fc_add_reshape":              # FC -> ADD(0) -> RESHAPE
        n1 = bdl.rmsnorm_nodes(g, x, "n1", P["g1"])
        y = g.fc(g.add(n1, z("z"), "byp", (T, D)), "q_fc", P["Wq"], P["bq"], D)
        a = g.add(y, z("z2"), "byp2", (T, D))
        out = g.reshape(a, "q_3d", (T, H, HD), H * HD)
    elif kind == "fc_reshape_rope":             # FC -> RESHAPE -> full RoPE -> output
        n1 = bdl.rmsnorm_nodes(g, x, "n1", P["g1"])
        y = g.fc(g.add(n1, z("z"), "byp", (T, D)), "q_fc", P["Wq"], P["bq"], D)
        t3 = g.reshape(y, "q_3d", (T, H, HD), H * HD)
        c16, s16 = bdl.tables(H)
        tc = g.const("cos", c16); ts = g.const("sin", s16)
        a, b = g.split(t3, "sp", 2, 2, (T, H, HD // 2))
        p1 = g.mul(a, "p1", b=tc, shape=(T, H, HD // 2))
        p2 = g.mul(b, "p2", b=ts, shape=(T, H, HD // 2))
        p3 = g.mul(b, "p3", b=tc, shape=(T, H, HD // 2))
        p4 = g.mul(a, "p4", b=ts, shape=(T, H, HD // 2))
        import build_ops as bo
        S = bo.S
        lo = g.act("lo", (T, H, HD // 2))
        S.SubOptionsStart(g.b); S.SubOptionsAddFusedActivationFunction(g.b, 0)
        g.op(bo.BuiltinOperator.SUB, [p1, p2], [lo], bo.BuiltinOptions.SubOptions, S.SubOptionsEnd(g.b))
        hi = g.act("hi", (T, H, HD // 2))
        S.AddOptionsStart(g.b); S.AddOptionsAddFusedActivationFunction(g.b, 0)
        g.op(bo.BuiltinOperator.ADD, [p3, p4], [hi], bo.BuiltinOptions.AddOptions, S.AddOptionsEnd(g.b))
        out = g.concat([lo, hi], "rope", 2, (T, H, HD))
    else:
        raise SystemExit("unknown kind " + kind)
    return g.finish([x], [out], "probe " + kind)

kind = sys.argv[1]; T = int(sys.argv[2]); d = sys.argv[3]
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "model.tflite"), "wb").write(build(kind, T))
print("built", kind, T, d)
