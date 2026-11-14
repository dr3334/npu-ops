#!/usr/bin/env python3
"""Noise floor for ops/decoder_layer_t8_fp16 (ticket 09, NOTES 4 protocol).

    /usr/bin/python3 ops/decoder_layer_t8_fp16/noise_floor.py

The two dumps are two e2e runs of the SAME .dla with the SAME input, taken inside
one devlock session (second run's dump is saved as out_r1.bin BEFORE the replay):

    export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"
    ./tools/devlock.sh run -- bash -c '
      ./tools/healthcheck.sh --quick &&
      ./tools/e2e.sh ops/decoder_layer_t8_fp16 &&
      cp ops/decoder_layer_t8_fp16/out_device.bin ops/decoder_layer_t8_fp16/out_r1.bin &&
      ./tools/e2e.sh ops/decoder_layer_t8_fp16 &&
      cp ops/decoder_layer_t8_fp16/out_device.bin ops/decoder_layer_t8_fp16/out_r2.bin &&
      ./tools/healthcheck.sh --quick'

Reads   out_r1.bin, out_r2.bin (falls back to out_device.bin as run 2), the
        .dla/.tflite they came from, golden/expected.bin (independent numpy
        reference) and e2e.json.  Writes noise_floor.json.

Nothing here is typed in by hand: every number is computed from those files, and
each input file's sha256 is recorded so the JSON can be tied back to the dumps.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
META = json.load(open(os.path.join(D, "model.json")))
SHAPE = tuple(META["output"]["shape"])


def path(name):
    return os.path.join(D, name)


def sha256(name):
    p = path(name)
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(name):
    p = path(name)
    if not os.path.exists(p):
        return None
    st = os.stat(p)
    return {"bytes": st.st_size, "sha256": sha256(name),
            "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))}


def load(name):
    p = path(name)
    if not os.path.exists(p):
        return None
    a = np.fromfile(p, dtype=np.float16)
    if a.size != int(np.prod(SHAPE)):
        sys.exit("%s: %d fp16 values, model.json says %s" % (p, a.size, SHAPE))
    return a.reshape(SHAPE)


def stats(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "max_abs": float(d.max()),
        "n_diff": int((d > 0).sum()),
        "n_total": int(d.size),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "bitwise_identical": bool(np.array_equal(a.view(np.uint16), b.view(np.uint16))),
    }


r1 = load("out_r1.bin")
if r1 is None:
    sys.exit("out_r1.bin missing: run e2e once and copy out_device.bin to out_r1.bin first")
r2 = load("out_r2.bin")
if r2 is None:
    r2 = load("out_device.bin")
exp = load(os.path.join("golden", "expected.bin"))
if r2 is None:
    sys.exit("neither out_r2.bin nor out_device.bin found: the second e2e run is missing")

nf = stats(r1, r2)
mx = float(np.abs(exp.astype(np.float64)).max())
dev = stats(r1, exp)
tol = float(META["atol"]) + float(META["rtol"]) * mx

e2e = json.load(open(path("e2e.json"))) if os.path.exists(path("e2e.json")) else {}

out = {
    "op": "decoder_layer_t8_fp16",
    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "protocol": ("NOTES 4: same .dla, same input, two e2e runs; the second run started "
                 "from the file left by the first (e2e.sh does not rebuild model.tflite/"
                 "model.dla/golden when they exist)"),
    "output_shape": list(SHAPE),
    "output_bytes": int(exp.nbytes),
    "files": {
        "out_r1.bin": fingerprint("out_r1.bin"),
        "out_r2.bin": fingerprint("out_r2.bin"),
        "out_device.bin": fingerprint("out_device.bin"),
        "golden/input.bin": fingerprint(os.path.join("golden", "input.bin")),
        "golden/expected.bin": fingerprint(os.path.join("golden", "expected.bin")),
        "model.dla": fingerprint("model.dla"),
        "model.tflite": fingerprint("model.tflite"),
    },
    "e2e_metadata": {k: e2e.get(k) for k in ("op", "device", "repeat", "ts", "dla_sha256")
                     if k in e2e},
    "artifact_note": ("model.dla is NOT byte-reproducible: this is the file the two runs above "
                      "actually used; model.compile.log records the SIZE/SHA256 of its own "
                      "compile and can differ by a few bytes. Regression compares numbers, never "
                      "the .dla hash."),
    "noise_floor": dict(nf),
    "noise_floor_value": nf["max_abs"],
    "noise_floor_is_zero": nf["max_abs"] == 0.0,
    "device_vs_golden": dict(dev, max_abs_relative=dev["max_abs"] / mx,
                             max_abs_expected=mx, threshold=tol,
                             verdict="PASS" if dev["max_abs"] <= tol else "FAIL"),
    "conclusion": ("noise floor 0 (bit-identical dumps) -> the device is deterministic for this "
                   ".dla and input; the registered tolerance therefore rests on the fp16 floor, "
                   "not on run-to-run jitter"
                   if nf["max_abs"] == 0.0 else
                   "noise floor %.3e != 0 -> the registered tolerance must be >= 3x this value"
                   % nf["max_abs"]),
}

json.dump(out, open(path("noise_floor.json"), "w"), indent=2)

print("dumps            : %s vs %s  (%s)" % ("out_r1.bin", "out_r2.bin" if os.path.exists(path("out_r2.bin")) else "out_device.bin", list(SHAPE)))
print("sha256 r1        : %s" % out["files"]["out_r1.bin"]["sha256"])
print("sha256 r2        : %s" % (out["files"]["out_r2.bin"] or out["files"]["out_device.bin"])["sha256"])
print("noise floor      : max_abs=%.3e  differ=%d/%d  bitwise_identical=%s"
      % (nf["max_abs"], nf["n_diff"], nf["n_total"], nf["bitwise_identical"]))
print("device vs golden : max_abs=%.3e  differ=%d/%d  max|exp|=%.6f  threshold=%.4e  %s"
      % (dev["max_abs"], dev["n_diff"], dev["n_total"], mx, tol, out["device_vs_golden"]["verdict"]))
print("-> noise_floor.json written")
