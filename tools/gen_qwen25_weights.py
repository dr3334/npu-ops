#!/usr/bin/env python3
"""Generate the frozen random weight fixture for the Qwen2.5-0.5B decoder work.

Why this exists
---------------
Tickets 03-10 compare "board output vs independent numpy golden". If both sides read
*different* weights, every numeric mismatch becomes unattributable. This script
produces one frozen set of weights so the only variable left in a comparison is the
implementation.

What is generated
-----------------
19 tensors covering one decoder layer (twice: layer0 for tickets 03-09, layer1 so
ticket 10 can chain two layers) plus the tied embed / LM head:

    embed.weight [151936,896]  (alias: lm_head.weight - same bytes, one materialisation)
    layerN.attn.{q,k,v}_proj.{weight,bias}, layerN.attn.o_proj.weight (no bias)
    layerN.mlp.{gate,up,down}_proj.weight
    layerN.input_layernorm.weight, layerN.post_attention_layernorm.weight
    N in {0, 1}

Distribution
------------
Normal with **per-tensor (mean, std) measured from the real Qwen2.5-0.5B-Instruct
bf16 weights on this machine**, not from config.json's initializer_range=0.02.
The measured values live in MEASURED below (provenance: SOURCE); re-derive them with
    python3 tools/gen_qwen25_weights.py --measure --safetensors <model.safetensors>
The task ticket allowed an initializer_range=0.02 fallback only if no real weights
existed locally. Real weights do exist (see docs/OP-weights.md section 3), so the
fallback is not used - measurement supersedes it.

Reproducibility
---------------
* per-tensor seed = sha256("<seed>:<name>")[:8] as big-endian int  -> adding a tensor
  later (e.g. layer2..23) never changes any already-frozen tensor's bytes;
* standard normals are drawn with Box-Muller from Generator.random_raw() (the raw
  PCG64 uint64 stream, which numpy *does* guarantee to be stable) instead of
  Generator.standard_normal() (numpy explicitly gives no cross-version guarantee for
  the distribution methods);
* values are stored as float16; manifest statistics describe the *stored* fp16 values.

sha256 semantics
----------------
Every tensor's sha256 is over the raw array bytes (C order, little-endian fp16,
contiguous) - NOT over the .npz container. The npz container hash is recorded
separately as npz_file_sha256 for instrumentation only.

Usage
-----
    python3 tools/gen_qwen25_weights.py                 # generate + write manifest
    python3 tools/gen_qwen25_weights.py --verify        # regenerate in memory, compare to manifest
    python3 tools/gen_qwen25_weights.py --list          # canonical tensor list
    python3 tools/gen_qwen25_weights.py --measure --safetensors /path/model.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

SCHEMA = "qwen25-frozen-weights/1"
VERSION = "1"
DEFAULT_SEED = 20261113  # fixture build date, 2026-11-13

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "weights" / "qwen25_frozen"

CHUNK_ELEMS = 1 << 22  # 4 Mi elements per block; keeps peak RSS ~ 100 MB
M53 = 2.0 ** -53
HALF_M53 = 0.5 * M53
TWO_PI = 2.0 * math.pi

# Real weights used to derive the distribution. Facts below were measured on
# 2026-11-13 (see docs/OP-weights.md section 3-4).
SOURCE = {
    "model": "Qwen2.5-0.5B-Instruct (Qwen2ForCausalLM, transformers 4.43.1, torch_dtype=bfloat16)",
    "path_wsl": "/mnt/d/ai/Qwen2.5-0.5B-Instruct/model.safetensors",
    "path_windows": r"D:\ai\Qwen2.5-0.5B-Instruct\model.safetensors",
    "safetensors_sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "config_json_sha256": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "tensor_dtype": "bfloat16",
    "num_tensors": 290,
    "measured_at": "2026-11-13",
    "measured_with": "torch+safetensors, chunked float64 accumulation, std is population (ddof=0)",
    "config_facts": {
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "num_hidden_layers": 24,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "initializer_range": 0.02,
        "note": "initializer_range=0.02 is recorded for provenance only; it is NOT used as"
                " the sampling scale because real weights exist locally (trained std differs"
                " from the init scale by 1.1x-4x, see docs/OP-weights.md section 4).",
    },
}

# canonical fixture tensor order: (fixture name, shape, real source tensor name)
TENSORS = [
    ("embed.weight", (151936, 896), "model.embed_tokens.weight"),
]
for _L in (0, 1):
    _p = "model.layers.%d." % _L
    TENSORS += [
        ("layer%d.attn.q_proj.weight" % _L, (896, 896), _p + "self_attn.q_proj.weight"),
        ("layer%d.attn.q_proj.bias" % _L, (896,), _p + "self_attn.q_proj.bias"),
        ("layer%d.attn.k_proj.weight" % _L, (128, 896), _p + "self_attn.k_proj.weight"),
        ("layer%d.attn.k_proj.bias" % _L, (128,), _p + "self_attn.k_proj.bias"),
        ("layer%d.attn.v_proj.weight" % _L, (128, 896), _p + "self_attn.v_proj.weight"),
        ("layer%d.attn.v_proj.bias" % _L, (128,), _p + "self_attn.v_proj.bias"),
        ("layer%d.attn.o_proj.weight" % _L, (896, 896), _p + "self_attn.o_proj.weight"),
        ("layer%d.mlp.gate_proj.weight" % _L, (4864, 896), _p + "mlp.gate_proj.weight"),
        ("layer%d.mlp.up_proj.weight" % _L, (4864, 896), _p + "mlp.up_proj.weight"),
        ("layer%d.mlp.down_proj.weight" % _L, (896, 4864), _p + "mlp.down_proj.weight"),
        ("layer%d.input_layernorm.weight" % _L, (896,), _p + "input_layernorm.weight"),
        ("layer%d.post_attention_layernorm.weight" % _L, (896,), _p + "post_attention_layernorm.weight"),
    ]

# aliases: tied embeddings -> one materialisation, two names
ALIASES = {"lm_head.weight": "embed.weight"}

# per-tensor (mean, std, abs_max) measured from SOURCE on 2026-11-13.
# abs_max is recorded for provenance / fp16 headroom only; it is not used for sampling.
MEASURED = {
    "embed.weight": (5.1431658361666124e-05, 0.015249471186505636, 0.201171875),
    "layer0.attn.q_proj.weight": (-1.6903982052884675e-05, 0.06674057322078238, 1.2265625),
    "layer0.attn.q_proj.bias": (-0.15199419856071472, 7.852643548685071, 79.0),
    "layer0.attn.k_proj.weight": (0.0001346356299011729, 0.08059654655395221, 1.0234375),
    "layer0.attn.k_proj.bias": (3.870380401611328, 32.21766970519115, 130.0),
    "layer0.attn.v_proj.weight": (-8.0177496195282e-06, 0.01147203444728321, 0.12109375),
    "layer0.attn.v_proj.bias": (0.001978270709514618, 0.014814663943677088, 0.09814453125),
    "layer0.attn.o_proj.weight": (1.859124503056364e-05, 0.01312900771421886, 0.5625),
    "layer0.mlp.gate_proj.weight": (-3.1943358967389256e-05, 0.02304633476314177, 0.4921875),
    "layer0.mlp.up_proj.weight": (-2.9259440850811106e-06, 0.018255463834534864, 0.279296875),
    "layer0.mlp.down_proj.weight": (-3.3242997141090408e-06, 0.018157518639496675, 0.3046875),
    "layer0.input_layernorm.weight": (0.032157489231654575, 0.09240944326762969, 0.7265625),
    "layer0.post_attention_layernorm.weight": (0.6781703404017857, 0.150931880956101, 1.9921875),
    "layer1.attn.q_proj.weight": (7.276373250448968e-05, 0.0287229134604443, 0.345703125),
    "layer1.attn.q_proj.bias": (0.14250622476850236, 1.1673632804581766, 10.5625),
    "layer1.attn.k_proj.weight": (8.775661557690131e-06, 0.04056347067290655, 0.28515625),
    "layer1.attn.k_proj.bias": (1.6681711673736572, 18.419964468969987, 147.0),
    "layer1.attn.v_proj.weight": (5.300466828016072e-05, 0.011527703695346746, 0.08251953125),
    "layer1.attn.v_proj.bias": (-0.021303579211235046, 0.24076252682103988, 2.703125),
    "layer1.attn.o_proj.weight": (-1.5464313030122412e-05, 0.014866886438549639, 0.2060546875),
    "layer1.mlp.gate_proj.weight": (0.00011780581709281161, 0.02327334840733284, 0.412109375),
    "layer1.mlp.up_proj.weight": (3.5130970457562438e-06, 0.017048485923197746, 0.2734375),
    "layer1.mlp.down_proj.weight": (9.157514777350892e-06, 0.01699749100524363, 0.359375),
    "layer1.input_layernorm.weight": (0.6842929295131138, 0.3227377143407279, 3.25),
    "layer1.post_attention_layernorm.weight": (1.006254040236984, 0.2518294473105806, 1.6796875),
}


# --------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------

def tensor_seed(seed: int, name: str) -> int:
    """Per-tensor sub-seed: stable, independent of tensor order and of the tensor list."""
    digest = hashlib.sha256(("%d:%s" % (seed, name)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rng(seed: int, name: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(tensor_seed(seed, name)))


def _box_muller(rng: np.random.Generator, n: int) -> np.ndarray:
    """n standard normals from the raw PCG64 uint64 stream (Box-Muller, float64).

    Stable across numpy versions because only the PCG64 bit generator's raw uint64 stream
    is consumed (Generator.standard_normal() carries no cross-version stream guarantee);
    log/cos/sin still come from the platform libm (see docs/OP-weights.md section 11).
    """
    m = (n + 1) // 2
    raw = rng.bit_generator.random_raw(2 * m)  # raw PCG64 uint64 stream (stable across numpy versions)
    u1 = (raw[0::2] >> np.uint64(11)).astype(np.float64) * M53 + HALF_M53
    u2 = (raw[1::2] >> np.uint64(11)).astype(np.float64) * M53 + HALF_M53
    radius = np.sqrt(-2.0 * np.log(u1))
    theta = TWO_PI * u2
    out = np.empty(2 * m, dtype=np.float64)
    out[0::2] = radius * np.cos(theta)
    out[1::2] = radius * np.sin(theta)
    return out[:n]


def _sha256_bytes(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()


def generate_tensor(name, shape, seed=DEFAULT_SEED):
    """Return (fp16 array, stats dict) for one tensor. Same seed+name => same bytes."""
    if name not in MEASURED:
        raise KeyError("no measured statistics for %r (add it to MEASURED)" % name)
    mean, std, _ = MEASURED[name]
    shape = tuple(int(d) for d in shape)
    total = int(np.prod(shape))
    row = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    rows = total // row
    chunk_rows = max(1, CHUNK_ELEMS // row)

    out = np.empty(total, dtype=np.float16)
    digest = hashlib.sha256()
    rng = _rng(seed, name)
    s = 0.0
    ss = 0.0
    abs_max = 0.0

    for r0 in range(0, rows, chunk_rows):
        r1 = min(rows, r0 + chunk_rows)
        count = (r1 - r0) * row
        z = _box_muller(rng, count)
        block = (mean + std * z).astype(np.float16)
        if not np.all(np.isfinite(block)):
            raise FloatingPointError("non-finite sample in %r (fp16 overflow?)" % name)
        out[r0 * row:r1 * row] = block
        f = block.astype(np.float64)
        s += float(f.sum())
        ss += float((f * f).sum())
        abs_max = max(abs_max, float(np.abs(f).max()))
        digest.update(memoryview(np.ascontiguousarray(out[r0 * row:r1 * row])).cast("B"))

    arr = out.reshape(shape)
    m = s / total
    var = ss / total - m * m
    stats = {
        "mean": m,
        "std": math.sqrt(max(var, 0.0)),
        "abs_max": abs_max,
        "sha256": digest.hexdigest(),
        "bytes": int(arr.nbytes),
        "shape": list(shape),
    }
    return arr, stats


# --------------------------------------------------------------------------------------
# manifest helpers
# --------------------------------------------------------------------------------------

def _script_sha256(path: Path):
    try:
        return _sha256_bytes(np.frombuffer(path.read_bytes(), dtype=np.uint8))
    except OSError:
        return None


def _weights_digest(entries) -> str:
    h = hashlib.sha256()
    for e in sorted(entries, key=lambda x: x["name"]):
        h.update(("%s %s\n" % (e["name"], e["sha256"])).encode("utf-8"))
    return h.hexdigest()


def _command_line() -> str:
    return "%s %s" % (Path(sys.executable).as_posix(), " ".join(a.replace("\\", "/") for a in sys.argv))


def _build_entry(name, shape, source_tensor, arr, stats, container_sha256, out_dir):
    measured = MEASURED[name]
    entry = {
        "name": name,
        "shape": list(shape),
        "dtype": "float16",
        "bytes": stats["bytes"],
        "sha256": stats["sha256"],
        "std": stats["std"],
        "mean": stats["mean"],
        "abs_max": stats["abs_max"],
        "file": "%s.npz" % name,
        "npz_key": name,
        "npz_file_sha256": container_sha256,
        "aliases": sorted(a for a, target in ALIASES.items() if target == name),
        "source_tensor": source_tensor,
        "measured": {
            "mean": measured[0],
            "std": measured[1],
            "abs_max": measured[2],
            "dtype": SOURCE["tensor_dtype"],
        },
    }
    return entry


def cmd_generate(out_dir: Path, seed: int, quiet: bool) -> int:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    total_bytes = 0

    for name, shape, source_tensor in TENSORS:
        arr, stats = generate_tensor(name, shape, seed)
        npz_path = out_dir / ("%s.npz" % name)
        np.savez(npz_path, **{name: arr})
        container = _sha256_bytes(np.frombuffer(npz_path.read_bytes(), dtype=np.uint8))
        entries.append(_build_entry(name, shape, source_tensor, arr, stats, container, out_dir))
        total_bytes += stats["bytes"]
        if not quiet:
            rel = "" if MEASURED[name][1] == 0 else " (std rel dev %+.3f%%)" % (
                100.0 * (stats["std"] - MEASURED[name][1]) / MEASURED[name][1])
            print("  %-42s %-14s %7.2f MB  std=%.6g%s"
                  % (name, "x".join(str(d) for d in shape), stats["bytes"] / 1e6, stats["std"], rel),
                  flush=True)

    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "seed": seed,
        "seed_derivation": 'per tensor: int.from_bytes(sha256("<seed>:<name>").digest()[:8], "big")',
        "bit_generator": "numpy.random.PCG64",
        "sampler": "Box-Muller on Generator.random_raw() uint64 stream (float64), then"
                   " x = float16(mean + std * z); per-tensor (mean, std) measured from real weights",
        "dtype": "float16",
        "layout": "C order, little-endian, contiguous",
        "hash_of": "raw array bytes (C order, little-endian fp16); NOT the .npz container",
        "stats_method": "mean/std/abs_max computed over the stored fp16 values, float64 accumulation,"
                        " std is population (ddof=0)",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator_command": _command_line(),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "generator": {
            "script": "tools/gen_qwen25_weights.py",
            "sha256": _script_sha256(Path(__file__)),
        },
        "loader": {
            "script": "tools/qwen25_weights.py",
            "sha256": _script_sha256(Path(__file__).resolve().parent / "qwen25_weights.py"),
        },
        "model": {
            "fixture_for": "Qwen2.5-0.5B decoder block (T=8, fp16, 14Q-2KV GQA, head_dim=64)",
            "config": SOURCE["config_facts"],
            "source_weights": {k: SOURCE[k] for k in (
                "model", "path_wsl", "path_windows", "safetensors_sha256",
                "config_json_sha256", "tensor_dtype", "num_tensors", "measured_at", "measured_with")},
        },
        "tied": {"embed.weight": ["lm_head.weight"],
                 "note": "tie_word_embeddings=true in source config; one materialisation,"
                         " lm_head.weight resolves to the same array"},
        "tensor_count": len(entries),
        "total_bytes": total_bytes,
        "weights_digest": _weights_digest(entries),
        "tensors": entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    if not quiet:
        print("\nwrote %d tensors, %.1f MB, weights_digest=%s"
              % (len(entries), total_bytes / 1e6, manifest["weights_digest"]))
        print("manifest: %s" % manifest_path.as_posix())
    return 0


def cmd_verify(out_dir: Path, seed: int, quiet: bool) -> int:
    """Regenerate everything in memory and compare sha256 against the manifest."""
    out_dir = Path(out_dir).resolve()
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        print("ERROR: manifest not found: %s" % manifest_path.as_posix(), file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {t["name"]: t for t in manifest["tensors"]}
    errors = []
    seen_digest = []
    for name, shape, _src in TENSORS:
        if name not in by_name:
            errors.append("%s: missing from manifest" % name)
            continue
        _arr, stats = generate_tensor(name, shape, seed)
        ref = by_name[name]
        if stats["sha256"] != ref["sha256"]:
            errors.append("%s: sha256 MISMATCH regen=%s manifest=%s"
                          % (name, stats["sha256"][:16], ref["sha256"][:16]))
        elif not quiet:
            print("  OK  %-42s %s" % (name, stats["sha256"][:16]), flush=True)
        seen_digest.append({"name": name, "sha256": stats["sha256"]})
    digest = _weights_digest(seen_digest)
    if digest != manifest["weights_digest"]:
        errors.append("weights_digest MISMATCH regen=%s manifest=%s"
                      % (digest[:16], manifest["weights_digest"][:16]))
    if errors:
        print("\n--verify FAILED (%d problem(s)):" % len(errors), file=sys.stderr)
        for e in errors:
            print("  - " + e, file=sys.stderr)
        return 1
    print("\n--verify OK: %d/%d tensors byte-identical on regeneration, weights_digest=%s"
          % (len(TENSORS), len(TENSORS), digest))
    return 0


def cmd_list() -> int:
    total = 0
    for name, shape, source in TENSORS:
        els = int(np.prod(shape))
        total += els
        print("%-42s %-14s fp16  %8.2f MB  <- %s"
              % (name, "x".join(str(d) for d in shape), els * 2 / 1e6, source))
    for alias, target in sorted(ALIASES.items()):
        print("%-42s %-14s alias of %s (tied, no extra bytes)" % (alias, "same", target))
    print("\n%d tensors, %d elements, %.1f MB" % (len(TENSORS), total, total * 2 / 1e6))
    return 0


def cmd_measure(safetensors_path: str) -> int:
    """Re-derive the (mean, std, abs_max) table from a real Qwen2.5 safetensors file."""
    try:
        import torch  # noqa: F401
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - environment dependent
        print("ERROR: --measure needs torch + safetensors in this python: %r" % (exc,), file=sys.stderr)
        return 2

    print("# measured from %s" % safetensors_path)
    print("MEASURED = {")
    for name, shape, source in TENSORS:
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            t = f.get_tensor(source)
        flat = t.reshape(-1)
        n = flat.numel()
        s = ss = amax = 0.0
        for i in range(0, n, 65536):
            c = flat[i:i + 65536].to(torch.float64)
            s += float(c.sum())
            ss += float((c * c).sum())
            amax = max(amax, float(c.abs().max()))
        mean = s / n
        std = math.sqrt(max(ss / n - mean * mean, 0.0))
        if name in MEASURED:
            rm, rs, _ = MEASURED[name]
            tag = "  # matches embedded (dmean=%+.2e, dstd=%+.2e)" % (mean - rm, std - rs)
        else:
            tag = "  # NEW"
        print('    %r: (%.17g, %.17g, %.17g),%s' % (name, mean, std, amax, tag), flush=True)
    print("}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate / verify the frozen Qwen2.5-0.5B weight fixture.")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="output directory (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="fixture seed (default: %(default)s)")
    ap.add_argument("--verify", action="store_true",
                    help="regenerate in memory and compare sha256 with the existing manifest")
    ap.add_argument("--list", action="store_true", help="print the canonical tensor list")
    ap.add_argument("--measure", action="store_true", help="re-measure statistics from a real safetensors file")
    ap.add_argument("--safetensors", default=SOURCE["path_wsl"], help="path for --measure")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        return cmd_list()
    if args.measure:
        return cmd_measure(args.safetensors)
    if args.verify:
        return cmd_verify(Path(args.out), args.seed, args.quiet)
    return cmd_generate(Path(args.out), args.seed, args.quiet)


if __name__ == "__main__":
    sys.exit(main())
