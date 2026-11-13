#!/usr/bin/env python3
"""Load the frozen Qwen2.5-0.5B weight fixture (fp16) by tensor name, verifying sha256.

This module is the *only* interface tickets 03-10 (and the numpy goldens) should use to
get weights. Two properties matter and both are enforced here:

1. **Same bytes everywhere.** Every array is checked against the sha256 recorded in
   weights/qwen25_frozen/manifest.json. A mutated fixture fails loudly instead of
   silently changing the numbers under a comparison.
2. **Same names everywhere.** The tensor names / shapes in the manifest are frozen
   (see docs/OP-weights.md section 11). The tied LM head is an alias of embed.weight,
   so it costs no extra memory.

Usage
-----
    import sys; sys.path.insert(0, "<repo>/tools")
    import qwen25_weights as qw

    w  = qw.get("layer0.attn.q_proj.weight")   # np.float16 [896, 896], read-only
    b  = qw.get("layer0.attn.q_proj.bias")     # np.float16 [896]
    e  = qw.get("lm_head.weight")              # same object as qw.get("embed.weight")
    wf = qw.get_fp32("layer0.mlp.gate_proj.weight")   # float32 copy for numpy goldens
    qw.verify_all()                            # -> {"ok": True, "checked": 19, ...}

Weights directory resolution: explicit path -> $QWEN25_WEIGHTS_DIR -> <repo>/weights/qwen25_frozen.
Downstream worktrees that have not generated their own copy can point the env var at a
generated one; regenerating locally yields identical sha256 (see
    python3 tools/gen_qwen25_weights.py --verify
).

CLI
---
    python3 tools/qwen25_weights.py --list
    python3 tools/qwen25_weights.py --verify
    python3 tools/qwen25_weights.py --stats layer0.attn.q_proj.weight
    python3 tools/qwen25_weights.py --sha256 embed.weight
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

SCHEMA = "qwen25-frozen-weights/1"
ENV_VAR = "QWEN25_WEIGHTS_DIR"
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "weights" / "qwen25_frozen"
MANIFEST_NAME = "manifest.json"

_manifest_cache = {}
_array_cache = {}


class FixtureError(RuntimeError):
    """Raised when the fixture is missing, inconsistent, or fails its sha256 check."""


def resolve_dir(path=None) -> Path:
    """Directory search order: explicit path -> $QWEN25_WEIGHTS_DIR -> <repo>/weights/qwen25_frozen."""
    if path is not None:
        return Path(path).expanduser().resolve()
    env = os.environ.get(ENV_VAR)
    return Path(env).expanduser().resolve() if env else DEFAULT_DIR.resolve()


def _missing_dir_message(d: Path) -> str:
    return (
        "frozen weights not found at %s\n"
        "  generate them (about 30 s, 332 MB) with:\n"
        "      python3 tools/gen_qwen25_weights.py\n"
        "  or point %s at an existing generated copy (another checkout, a shared cache):\n"
        "      export %s=/path/to/weights/qwen25_frozen\n"
        "  (regeneration is byte-identical; 'gen_qwen25_weights.py --verify' proves it)"
        % (d.as_posix(), ENV_VAR, ENV_VAR)
    )


def manifest(path=None) -> dict:
    d = resolve_dir(path)
    key = str(d)
    if key not in _manifest_cache:
        f = d / MANIFEST_NAME
        if not f.is_file():
            raise FixtureError(_missing_dir_message(d))
        m = json.loads(f.read_text(encoding="utf-8"))
        if m.get("schema") != SCHEMA:
            raise FixtureError("manifest schema %r != expected %r (%s)"
                               % (m.get("schema"), SCHEMA, f.as_posix()))
        _manifest_cache[key] = m
    return _manifest_cache[key]


def _entries(path=None) -> dict:
    return {t["name"]: t for t in manifest(path)["tensors"]}


def _aliases(path=None) -> dict:
    alias = {}
    for t in manifest(path)["tensors"]:
        for a in t.get("aliases") or ():
            alias[a] = t["name"]
    return alias


def names(path=None) -> list:
    """Frozen canonical tensor names (aliases excluded)."""
    return [t["name"] for t in manifest(path)["tensors"]]


def alias_names(path=None) -> list:
    return sorted(_aliases(path))


def canonical(name: str, path=None) -> str:
    """Map an alias (e.g. lm_head.weight) to the materialised tensor name."""
    if name in _entries(path):
        return name
    alias = _aliases(path)
    if name in alias:
        return alias[name]
    raise FixtureError("unknown tensor %r; known: %s (aliases: %s)"
                       % (name, ", ".join(names(path)), ", ".join(sorted(alias)) or "-"))


def spec(name: str, path=None) -> dict:
    """Manifest entry for a tensor or alias (aliases return their target's entry plus 'alias_of')."""
    canon = canonical(name, path)
    entry = dict(_entries(path)[canon])
    if canon != name:
        entry["alias_of"] = canon
        entry["requested_name"] = name
    return entry


def _sha256_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()


def get(name: str, path=None, verify: bool = True, mutable: bool = False) -> np.ndarray:
    """Return the frozen fp16 array for NAME (read-only by default), sha256-verified.

    Aliases resolve to the same cached object, so lm_head.weight does not allocate a
    second 259.6 MiB array. Pass mutable=True if the caller needs to write into it.
    """
    d = resolve_dir(path)
    canon = canonical(name, d)
    key = (str(d), canon)
    arr = _array_cache.get(key)
    if arr is None:
        entry = _entries(d)[canon]
        f = d / entry["file"]
        if not f.is_file():
            raise FixtureError("missing %s for tensor %r (%s)"
                               % (entry["file"], canon, _missing_dir_message(d)))
        with np.load(f) as z:
            if entry["npz_key"] not in z:
                raise FixtureError("%s has no key %r" % (f.as_posix(), entry["npz_key"]))
            arr = np.ascontiguousarray(z[entry["npz_key"]])
        if arr.dtype != np.float16:
            raise FixtureError("%s: dtype %s != float16" % (f.as_posix(), arr.dtype))
        if list(arr.shape) != list(entry["shape"]):
            raise FixtureError("%s: shape %s != manifest %s" % (f.as_posix(), arr.shape, entry["shape"]))
        if verify:
            got = _sha256_array(arr)
            if got != entry["sha256"]:
                raise FixtureError(
                    "%s: sha256 %s != manifest %s - the fixture was modified or is from a "
                    "different seed; regenerate with tools/gen_qwen25_weights.py"
                    % (f.as_posix(), got, entry["sha256"]))
        arr.setflags(write=False)
        _array_cache[key] = arr
    if mutable:
        return np.array(arr, copy=True)
    return arr


def get_fp32(name: str, path=None, verify: bool = True) -> np.ndarray:
    """float32 copy of a tensor (writable) - convenient for numpy goldens."""
    return get(name, path=path, verify=verify).astype(np.float32)


def get_many(*tensor_names, path=None, verify=True) -> dict:
    """Convenience: {'name': array} for several tensors (aliases kept under the asked name)."""
    return {n: get(n, path=path, verify=verify) for n in tensor_names}


def verify(name: str, path=None) -> dict:
    """Verify one tensor on disk; returns {'name','sha256','ok'} and raises on mismatch."""
    entry = spec(name, path)
    arr = get(name, path=path, verify=True)
    return {"name": entry["name"], "sha256": entry["sha256"], "shape": list(arr.shape), "ok": True}


def verify_all(path=None) -> dict:
    """Verify every tensor file + every alias. Never raises for data problems."""
    d = resolve_dir(path)
    errors = []
    checked = 0
    aliased = 0
    try:
        m = manifest(d)
    except FixtureError as exc:
        return {"ok": False, "dir": d.as_posix(), "checked": 0, "aliases": 0, "errors": [str(exc)]}
    for entry in m["tensors"]:
        try:
            verify(entry["name"], d)
            checked += 1
        except FixtureError as exc:
            errors.append(str(exc))
    for alias in alias_names(d):
        try:
            a = get(alias, path=d)
            b = get(canonical(alias, d), path=d)
            if a is not b:
                errors.append("alias %r did not reuse the materialised array" % alias)
            elif _sha256_array(a) != spec(alias, d)["sha256"]:
                errors.append("alias %r sha256 mismatch" % alias)
            else:
                aliased += 1
        except FixtureError as exc:
            errors.append(str(exc))
    digest = m.get("weights_digest")
    recomputed = hashlib.sha256()
    for e in sorted(m["tensors"], key=lambda x: x["name"]):
        recomputed.update(("%s %s\n" % (e["name"], e["sha256"])).encode("utf-8"))
    if recomputed.hexdigest() != digest:
        errors.append("weights_digest mismatch: manifest=%s recomputed=%s"
                      % (digest, recomputed.hexdigest()))
    return {
        "ok": not errors,
        "dir": d.as_posix(),
        "checked": checked,
        "aliases": aliased,
        "tensor_count": len(m["tensors"]),
        "total_bytes": m.get("total_bytes"),
        "weights_digest": digest,
        "errors": errors,
    }


def summary(path=None) -> str:
    d = resolve_dir(path)
    m = manifest(d)
    lines = [
        "frozen fixture: %s" % d.as_posix(),
        "schema=%s seed=%s dtype=%s tensors=%d total=%.1f MB"
        % (m["schema"], m["seed"], m["dtype"], m["tensor_count"], m["total_bytes"] / 1e6),
        "generated_at=%s weights_digest=%s" % (m.get("generated_at_utc"), m.get("weights_digest")),
        "distribution: normal, per-tensor (mean, std) measured from %s"
        % m["model"]["source_weights"]["model"],
        "",
        "%-42s %-14s %10s %12s %10s" % ("tensor", "shape", "std", "mean", "abs_max"),
    ]
    for e in m["tensors"]:
        lines.append("%-42s %-14s %10.5g %12.4g %10.4g"
                     % (e["name"], "x".join(str(x) for x in e["shape"]), e["std"], e["mean"], e["abs_max"]))
    for alias, target in sorted(_aliases(d).items()):
        lines.append("%-42s %-14s (tied alias of %s)" % (alias, "same", target))
    return "\n".join(lines)


def selftest(path=None) -> dict:
    """Prove the guard rails actually fire (tamper + missing file + unknown name).

    Works on a temp dir holding only the manifest and one small tensor, so it never
    touches the real fixture and never writes through a link into it.
    """
    import shutil
    import tempfile

    src = resolve_dir(path)
    target = "layer0.attn.v_proj.bias"
    entry = _entries(src)[target]
    tmp = Path(tempfile.mkdtemp(prefix="qwen25-fixture-selftest-"))
    checks = []

    def check(label, fn, expect_ok, expect_in=""):
        try:
            fn()
            ok, detail = expect_ok, "no exception"
        except FixtureError as exc:
            ok = (not expect_ok) and (expect_in in str(exc))
            detail = str(exc).splitlines()[0]
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            ok, detail = False, "%s: %s" % (type(exc).__name__, exc)
        checks.append({"label": label, "ok": bool(ok), "detail": detail})

    try:
        shutil.copy2(src / MANIFEST_NAME, tmp / MANIFEST_NAME)
        shutil.copy2(src / entry["file"], tmp / entry["file"])

        check("clean copy loads and verifies sha256", lambda: get(target, path=tmp), True)

        def _tamper():
            with np.load(tmp / entry["file"]) as z:
                arr = z[entry["npz_key"]].copy()
            before = arr.flat[0]
            arr.flat[0] = np.nextafter(before, np.float16(np.inf)).astype(np.float16)
            if arr.flat[0] == before:  # float16 nextafter fallback
                arr.flat[0] = (before + np.float16(0.001)).astype(np.float16)
            np.savez(tmp / entry["file"], **{entry["npz_key"]: arr})
            _array_cache.pop((str(tmp.resolve()), target), None)
            get(target, path=tmp)

        check("1-ULP mutation is rejected (sha256)", _tamper, False, "sha256")

        def _remove():
            (tmp / entry["file"]).unlink()
            _array_cache.pop((str(tmp.resolve()), target), None)
            get(target, path=tmp)

        check("missing .npz is rejected (not silent)", _remove, False, "missing")

        check("unknown tensor name is rejected",
              lambda: get("layer0.attn.nope.weight", path=tmp), False, "unknown tensor")
        check("alias lm_head.weight resolves to embed.weight",
              lambda: get("lm_head.weight", path=src), True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {"ok": all(c["ok"] for c in checks), "checks": checks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify / inspect the frozen Qwen2.5-0.5B weight fixture.")
    ap.add_argument("--dir", default=None, help="fixture directory (default: $%s or <repo>/weights/qwen25_frozen)" % ENV_VAR)
    ap.add_argument("--list", action="store_true", help="list tensor names/shapes")
    ap.add_argument("--verify", action="store_true", help="verify every tensor + alias against the manifest")
    ap.add_argument("--summary", action="store_true", help="print the fixture summary table")
    ap.add_argument("--stats", default=None, help="print one tensor's manifest entry as JSON")
    ap.add_argument("--sha256", default=None, help="print one tensor's sha256 (after verifying)")
    ap.add_argument("--selftest", action="store_true",
                    help="prove the tamper/missing/unknown-name guard rails fire (temp dir, cheap)")
    ap.add_argument("--json", action="store_true", help="machine-readable output for --verify/--list")
    args = ap.parse_args(argv)
    d = args.dir

    try:
        if args.list:
            if args.json:
                print(json.dumps([{"name": e["name"], "shape": e["shape"], "aliases": e.get("aliases") or []}
                                  for e in manifest(d)["tensors"]], indent=2))
            else:
                for e in manifest(d)["tensors"]:
                    print("%-42s %-14s %10d B  %s"
                          % (e["name"], "x".join(str(x) for x in e["shape"]), e["bytes"], e["sha256"][:16]))
                for alias, target in sorted(_aliases(d).items()):
                    print("%-42s %-14s (tied alias of %s)" % (alias, "same", target))
            return 0
        if args.stats:
            print(json.dumps(spec(args.stats, d), indent=2))
            return 0
        if args.sha256:
            print("%s  %s" % (verify(args.sha256, d)["sha256"], canonical(args.sha256, d)))
            return 0
        if args.summary:
            print(summary(d))
            return 0
        if args.selftest:
            res = selftest(d)
            for c in res["checks"]:
                print("%-4s %-46s %s" % ("OK" if c["ok"] else "FAIL", c["label"], c["detail"]))
            print("RESULT: %s" % ("PASS" if res["ok"] else "FAIL"))
            return 0 if res["ok"] else 1
        if args.verify:
            res = verify_all(d)
            if args.json:
                print(json.dumps(res, indent=2))
            else:
                print("fixture: %s" % res["dir"])
                if res.get("checked"):
                    print("tensors verified : %d/%d" % (res["checked"], res["tensor_count"]))
                    print("aliases verified : %d" % res["aliases"])
                    print("total bytes      : %d" % res["total_bytes"])
                    print("weights_digest   : %s" % res["weights_digest"])
                for err in res["errors"]:
                    print("ERROR: " + err, file=sys.stderr)
                print("RESULT: %s" % ("PASS" if res["ok"] else "FAIL"))
            return 0 if res["ok"] else 1
    except FixtureError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
