import json, os
import numpy as np
D = os.path.abspath("ops/lm_head_t8_fp16")
meta = json.load(open(os.path.join(D, "model.json")))
shape = tuple(meta["output"]["shape"])
exp = np.fromfile(os.path.join(D, "golden", "expected.bin"), dtype=np.float16).reshape(shape)
dev = np.fromfile(os.path.join(D, "out_r1.bin"), dtype=np.float16).reshape(shape)
e = exp.astype(np.float64); a = dev.astype(np.float64)
d = np.abs(a - e)
mask = d > 0
print("differing:", int(mask.sum()), "/", d.size)
print("abs(exp) percentiles of differing elems:", np.percentile(np.abs(e[mask]), [0, 25, 50, 75, 100]).round(4))
edges = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
print("%-14s %8s %12s %12s %12s" % ("|exp| bin", "n", "max_err", "median_err", "max_rel"))
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = mask & (np.abs(e) >= lo) & (np.abs(e) < hi)
    if sel.sum() == 0: continue
    dd = d[sel]
    print("%-14s %8d %12.3e %12.3e %12.3e" % ("[%g,%g)" % (lo, hi), int(sel.sum()), dd.max(), np.median(dd), (dd / np.abs(e[sel])).max()))
print("\nmax |dev| =", np.abs(a).max(), " max |exp| =", np.abs(e).max())
print("elements whose |dev| exceeds fp16 range check:", int((np.abs(a) > 65504).sum()))
print("exact-equal fraction: %.4f%%" % (100 * (~mask).sum() / d.size))
# is the error a fixed absolute quantum?
u, c = np.unique(d[mask], return_counts=True)
print("distinct abs-diff values (top 6):", [(float(x), int(y)) for x, y in sorted(zip(u, c), key=lambda t: -t[1])[:6]])
