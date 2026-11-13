import numpy as np, json, os, sys
sys.path.insert(0, "tools")
import build_gqa_scores as bgs
# Fixture directory: $QWEN25_WEIGHTS_DIR, else the in-repo copy (no hardcoded worktree path).
WEIGHTS = os.environ.get("QWEN25_WEIGHTS_DIR") or "weights/qwen25_frozen"
q, k, prov = bgs.make_qk(WEIGHTS)
exp = bgs.reference_scores(q, k).astype(np.float64)
dev = np.fromfile("ops/gqa_scores_t8_fp16/out_device.bin", dtype=np.float16).astype(np.float64)
# alternative KV-head mappings: how far would a WRONG mapping be from the golden?
qt = np.transpose(q.astype(np.float32), (1, 0, 2))
kt = np.transpose(k.astype(np.float32), (1, 0, 2))
def scores_with(idx):
    kre = kt[idx]
    return np.einsum("htd,hsd->hts", qt, kre).astype(np.float64)
correct = np.repeat(np.arange(2), 7)          # h//7
interleaved = np.arange(14) % 2               # h%2
blockwrong = np.where(np.arange(14) < 7, 1, 0)  # groups swapped
for name, idx in (("h//7 (chosen)", correct), ("h%2 interleaved", interleaved),
                  ("groups swapped", blockwrong)):
    s = scores_with(idx)
    print("%-18s vs chosen mapping: max_abs=%.4g rel_l2=%.4g" % (
        name, float(np.abs(s - exp).max()),
        float(np.linalg.norm(s - exp) / np.linalg.norm(exp))))
dev = dev.reshape(exp.shape)
print("device vs golden: max_abs=%.3g  rel_l2=%.3g  bitwise=%s" % (
    float(np.abs(dev - exp).max()),
    float(np.linalg.norm(dev - exp) / np.linalg.norm(exp)),
    bool(np.array_equal(np.fromfile("ops/gqa_scores_t8_fp16/out_device.bin", dtype=np.float16).view(np.uint16),
                        np.fromfile("ops/gqa_scores_t8_fp16/golden/expected.bin", dtype=np.float16).view(np.uint16)))))
