"""Shared BLAS/OpenMP threading bootstrap for RAVANA.

Motivation (numpy #27989, #11826): on Windows, native *access violation*
crashes occur when a BLAS routine (OpenBLAS / MKL / Accelerate) is touched from a
worker thread while the main thread is also inside BLAS. The web learner runs
``ThreadPoolExecutor`` fetch workers that call GloVe cosine / coherence scoring
(numpy dot / matmul) concurrently with the main-thread decoder/inference. The
interleaved, un-bound BLAS calls across threads are the trigger.

The fix recommended by the numpy maintainers is to pin every BLAS backend to a
single thread *before* ``import numpy`` is ever evaluated, optionally reinforced
with ``threadpoolctl.threadpool_limits``. We centralise that here so every
entrypoint (engine.py, web/learner.py, any ``__main__``) imports this module
*first*, guaranteeing the env vars land ahead of numpy's C-extension import.

This is a module-side-effect bootstrap: importing it for its side effects is the
intended use. It deliberately does NOT re-export numpy.
"""

import os

# Cross-backend thread caps. Must be set before numpy/OpenBLAS/MKL are imported.
# KMP_INIT_AT_FORK / OMP_DYNAMIC are OpenMP-family stabilisers that prevent the
# fork/thread-init races numpy #27989 describes.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    # setdefault: never override an explicit user override.
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

# Reinforce the env caps with threadpoolctl when available (it drives the
# actual OpenBLAS/MKL runtime thread pools, not just the env vars). Best-effort:
# threadpoolctl is an optional dev dependency, so a missing install is fine.
try:
    import threadpoolctl  # noqa: F401  (import has side effects on some builds)
    threadpoolctl.threadpool_limits(limits=1)
except Exception:
    pass

# faulthandler turns a silent Windows access violation into a readable Python
# traceback (instead of an opaque process abort), so any residual BLAS crash is
# diagnosable rather than mysterious.
try:
    import faulthandler

    if not faulthandler.is_enabled():
        faulthandler.enable()
        # NOTE: we deliberately do NOT call dump_traceback_later here. A 30s
        # threshold fires on legitimate (slow) live-web lookups and obscures
        # real results with noise; the AV protection is the BLAS env-var pin
        # above, and faulthandler.enable() already converts any residual native
        # crash into a readable traceback instead of a silent abort.
except Exception:
    pass
