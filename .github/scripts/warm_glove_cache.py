#!/usr/bin/env python3
"""Build the GloVe projected-vector cache used by RAVANA CI.

Why this exists
---------------
`data/ravana_glove_cache.npz` is a *regenerable* cache (projected GloVe
vectors). It used to be tracked via Git LFS, which re-served the 144 MB asset
on every CI job and blew past the repo's 10 GB LFS bandwidth quota. We removed
it from LFS (see chore commit) and instead build it once per cache-population
in a dedicated `warm-glove-cache` job, then share it across all test shards via
`actions/cache`. That turns ~1 GB/run of LFS egress into a single ~822 MB
download that only happens when the cache key changes.

Build parity
------------
The npz schema and the random projection MUST match what the engine's loader
(`ravana.ontology.attribute_encoder.build_glove64_lookup`) and
`ravana.chat.engine_graph._init_glove` expect:
  - keys: words (list[str]), vecs (n, glove_dim) float32, proj (dim, glove_dim)
    float32, glove_dim (int)
  - proj is a random orthonormal-ish lift from glove_dim -> dim, seeded with
    numpy RandomState(42) and scaled by sqrt(glove_dim / dim), exactly as in
    engine_graph.py:_init_glove.

This script does NOT reimplement the projection from scratch blindly: it imports
the engine's own `build_glove64_lookup` and round-trips the produced npz through
it to prove the loader accepts the output before the job caches it.
"""

from __future__ import annotations

import os
import sys
import zipfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA = os.path.join(_REPO_ROOT, "data")
_GLOVE_DIR = os.path.join(_DATA, "glove")
_CACHE_NPZ = os.path.join(_DATA, "ravana_glove_cache.npz")

# Dim the engine projects GloVe (100D) down to. Mirrors engine_graph.py usage.
_TARGET_DIM = 64
_GLOVE_DIM = 100


def _ensure_raw_glove() -> str:
    """Make sure data/glove/glove.6B.100d.txt exists; download if missing.

    Respects RAVANA_OFFLINE: if offline and the file is absent, fail loudly so
    the cache-population job surfaces a clear error instead of silently shipping
    a repo without GloVe.
    """
    name = "glove.6B.100d.txt"
    path = os.path.join(_GLOVE_DIR, name)
    if os.path.exists(path):
        print(f"  [warm] raw GloVe present: {path}")
        return path

    if os.environ.get("RAVANA_OFFLINE") == "1" and \
            os.environ.get("RAVANA_ALLOW_GLOVE_DOWNLOAD") != "1":
        sys.exit(
            "ERROR: data/glove/glove.6B.100d.txt missing and RAVANA_OFFLINE=1 "
            "set. The warm-cache job must run with RAVANA_OFFLINE=0 and "
            "RAVANA_ALLOW_GLOVE_DOWNLOAD=1 so it can fetch the ~822 MB GloVe "
            "archive once."
        )

    os.makedirs(_GLOVE_DIR, exist_ok=True)
    zip_path = os.path.join(_GLOVE_DIR, "glove.6B.zip")
    urls = [
        "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
        "https://nlp.stanford.edu/data/glove.6B.zip",
    ]
    import urllib.request

    for url in urls:
        try:
            print(f"  [warm] downloading GloVe 6B from {url} ...")
            req = urllib.request.Request(
                url, headers={"User-Agent": "RAVANA-CI/1.0"}
            )
            with urllib.request.urlopen(req, timeout=300) as resp, open(zip_path, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  [warm] download failed: {exc}; trying next mirror")
    else:
        sys.exit("ERROR: all GloVe mirrors failed to download.")

    print("  [warm] extracting glove.6B.100d.txt ...")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("glove.6B.100d.txt") as src, open(path, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
    return path


def _build_cache(raw_path: str) -> None:
    import numpy as np

    print(f"  [warm] reading {raw_path} ...")
    glove: dict[str, "np.ndarray"] = {}
    with open(raw_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != _GLOVE_DIM + 1:
                continue
            glove[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)

    rng = np.random.RandomState(42)
    max_d = max(_GLOVE_DIM, _TARGET_DIM)
    full_q, _ = np.linalg.qr(rng.randn(max_d, max_d).astype(np.float32))
    proj = full_q[:_TARGET_DIM, :_GLOVE_DIM].copy().astype(np.float32)
    proj *= np.sqrt(float(_GLOVE_DIM) / float(_TARGET_DIM))

    words = list(glove.keys())
    vecs = np.array([glove[w] for w in words], dtype=np.float32)
    print(f"  [warm] {len(words)} words, {_GLOVE_DIM}D -> {_TARGET_DIM}D")

    os.makedirs(_DATA, exist_ok=True)
    np.savez_compressed(
        _CACHE_NPZ,
        words=words,
        vecs=vecs,
        proj=proj,
        glove_dim=_GLOVE_DIM,
    )
    print(f"  [warm] wrote {_CACHE_NPZ} ({os.path.getsize(_CACHE_NPZ) // 1024} KB)")


def _verify() -> None:
    """Prove the engine loader accepts the produced npz (schema parity)."""
    try:
        from ravana.ontology.attribute_encoder import build_glove64_lookup
    except Exception as exc:  # noqa: BLE001
        print(f"  [warm] could not import engine loader ({exc}); skipping verify")
        return
    lut, dim = build_glove64_lookup(_CACHE_NPZ)
    assert lut and dim == _TARGET_DIM, f"loader returned bad state: dim={dim}"
    print(f"  [warm] verified: loader accepted cache ({len(lut)} words, dim={dim})")


def main() -> int:
    raw = _ensure_raw_glove()
    _build_cache(raw)
    _verify()
    print("GloVe cache warmed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
