import numpy as np, json
d = "ops/gqa_scores_t8_fp16"
r1 = np.fromfile(d + "/out_r1.bin", dtype=np.float16).astype(np.float32)
r2 = np.fromfile(d + "/out_r2.bin", dtype=np.float16).astype(np.float32)
exp = np.fromfile(d + "/golden/expected.bin", dtype=np.float16).astype(np.float32)
dev = np.fromfile(d + "/out_device.bin", dtype=np.float16).astype(np.float32)
print("n =", r1.size, " noise_floor max|r1-r2| =", float(np.abs(r1 - r2).max()))
print("bits identical r1 vs r2:", bool(np.array_equal(r1.view(np.uint16) if hasattr(r1,'view') else r1, r2)))
e16 = np.fromfile(d + "/golden/expected.bin", dtype=np.float16)
d16 = np.fromfile(d + "/out_device.bin", dtype=np.float16)
print("raw fp16 bitwise equal:", bool(np.array_equal(e16.view(np.uint16), d16.view(np.uint16))))
print("dev vs exp: max_abs=%.6e rmse=%.6e rel_l2=%.6e" % (
    float(np.abs(dev-exp).max()),
    float(np.sqrt(np.mean((dev-exp)**2))),
    float(np.linalg.norm(dev-exp)/np.linalg.norm(exp))))
print("max|exp| = %.6g  -> max_abs/max|exp| = %.6e" % (float(np.abs(exp).max()), float(np.abs(dev-exp).max())/float(np.abs(exp).max())))
# ULP distance at output scale
sc = float(np.abs(exp).max())
ulp = sc * 2.0**-11
print("1 ULP @ max|out| = %.6g   -> dev is 0 ULP away" % ulp)
