import sys, json
sys.path.insert(0, "tools")
import numpy as np
import build_decoder_layer as bdl
meta = json.load(open("ops/decoder_layer_t128_fp16/model.json"))
bdl._configure_from_meta(meta)
P = bdl.fixture(); x16 = bdl.input_hidden()
g = np.asarray(bdl.forward(x16, P, np.float32, False, "out")["probs"], np.float64)
s = np.asarray(bdl.forward(x16, P, np.float32, True, "out")["probs"], np.float64)
ag, asim = g.argmax(-1), s.argmax(-1)
print("probs argmax golden vs sim: %d/%d rows agree (%.2f%%)" % ((ag == asim).sum(), ag.size, 100.0*(ag==asim).mean()))
m = bdl.causal_mask()
gm = np.asarray(bdl.forward(x16, P, np.float32, False, "out")["probs"], np.float64)
print("masked positions max prob (golden):", float(gm[m == np.float16(bdl.MASK)].max()))
# how many rows are "single visible position" (row 0 of each head) => prob exactly 1
print("row0 prob per head:", gm[:, 0, 0][:3], "... all 1.0:", bool(np.all(gm[:, 0, 0] == 1.0)))
