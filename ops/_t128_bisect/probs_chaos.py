import sys, json
sys.path.insert(0, "tools")
import numpy as np
import build_decoder_layer as bdl
meta = json.load(open("ops/decoder_layer_t128_fp16/model.json"))
bdl._configure_from_meta(meta)
P = bdl.fixture(); x16 = bdl.input_hidden()
full = bdl.forward(x16, P, np.float32, False, "out")
sim = bdl.forward(x16, P, np.float32, True, "out")
g = np.asarray(full["probs"], np.float64); s = np.asarray(sim["probs"], np.float64)
d = np.abs(g - s)
idx = np.unravel_index(int(d.argmax()), d.shape)
print("probs: max|golden-sim| = %.4g" % d.max())
print("worst element (head,row,pos) =", idx, "golden=%.4f sim=%.4f" % (g[idx], s[idx]))
rowmax = d.reshape(-1, d.shape[-1]).max(axis=1)
print("rows with max diff > 1e-2: %d / %d (%.1f%%)" % ((rowmax > 1e-2).sum(), rowmax.size,
      100.0 * (rowmax > 1e-2).mean()))
sc = np.asarray(full["scores"], np.float64) * bdl.SCALE
h, r = idx[0], idx[1]
print("that row's top-3 scaled scores:", np.round(np.sort(sc[h, r])[::-1][:3], 2))
grain = float(np.spacing(np.float16(5164.0)))
print("fp16 grain of raw scores at |5164| = %.3f -> %.4f after MUL(1/8)" % (grain, grain / 8))
top = np.sort(sc, axis=-1)[..., ::-1][..., :2]
has_tie = (top[..., 0] - top[..., 1]) < (grain / 8)
print("rows whose top1-top2 gap < one score fp16 step: %d / %d (%.1f%%)"
      % (has_tie.sum(), has_tie.size, 100.0 * has_tie.mean()))
print("these chaotic rows account for %d of the %d probs entries with diff > 1e-2"
      % (int(((d > 1e-2) & np.repeat(has_tie[..., None], d.shape[-1], -1)).sum()), int((d > 1e-2).sum())))
