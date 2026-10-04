"""Build a matrix across PROCESSES, chunked by source element.

WHY PROCESSES AND NOT THREADS. The per-source loop looks like an obvious
``prange``, and it is not. Threading it measures 1.28x at two threads and then
gets WORSE -- 1.05x at four, 0.83x at six -- because the per-source body is
many small numpy calls and the GIL is held almost continuously. The writes are
genuinely race-free (each source owns a disjoint column slice, which is why
``kernels/assemble.py`` loops over sources), but no-race was never the binding
constraint. Processes measure 8.5x on 12 workers: 95% efficiency at 2, 82% at
4, 78% at 8, 71% at 12.

WHY BLAS IS PINNED TO ONE THREAD IN THE WORKERS. The matmuls here are small
(a ``(n_obs, n_quad) @ (n_quad, K)`` per monomial), so BLAS's own threads fight
the process pool for cores and lose: measured 1.83x with one BLAS thread
against 1.61x with sixteen, before any process parallelism at all. The pinning
happens by setting the threading environment for the CHILDREN ONLY, around pool
creation, and restoring it afterwards -- this module never changes the
threading behaviour of the caller's own process, and importing ``mhs`` never
changes it either. A library that sets ``OMP_NUM_THREADS`` at import time has
decided something that belongs to the program using it.

``threadpoolctl`` would do this more precisely and is deliberately NOT a
dependency: ``pyproject.toml`` records why (the sibling project's was missing
while the development environment happened to supply it, and a fresh install
broke on the first compressed solve and nowhere earlier). Environment variables
work because the children are SPAWNED, so they import numpy fresh and read
them.

MEMORY. Workers return their own column block and the parent writes it in, so
the peak is the output plus the blocks in flight. ``plan_chunks`` sizes the
chunk count so that in-flight total stays under a budget rather than scaling
with the worker count, and results are consumed as they complete rather than
gathered. The alternative -- shared memory, or a memmap each worker opens -- is
better for an output far larger than RAM, and ``out=`` already accepts a memmap
for that case; this module does not need to know which it was given.

THE CALL SITE MUST BE GUARDED, because the workers are spawned and therefore
re-import the module they were launched from. From a script that means

    if __name__ == "__main__":
        ...

and from a function, notebook or interactive session it is automatic. Without
it every child re-runs the script and spawns its own children; the parent sees
only ``BrokenProcessPool``, so ``by_source`` catches that and says this
instead. ``call`` must be picklable for the same reason: a module-level
function or a ``functools.partial`` of one, never a lambda or a closure.

    import functools, numpy as np, mhs, mhs.parallel as mp

    call = functools.partial(mhs.traction_matrix, obs, nrm, material=mat,
                             eps=0.1)
    T = mp.by_source(call, tris, workers=12)            # (n_obs, 3, n_dof, 3)

    # interaction_matrix derives its RECEIVERS from tris, so a chunk of
    # sources must be told the full receiver set -- otherwise each chunk
    # collocates on itself alone and the blocks do not even have the same
    # number of rows. `obs_tris` is that argument, and the collocation points
    # follow from it:
    call = functools.partial(mhs.interaction_matrix, material=mat, eps=0.1,
                             obs_tris=tris, receiver="strike", source="strike")
    K = mp.by_source(call, tris, workers=12, source_axis=1, tris_kw="tris")
"""
from __future__ import annotations

import concurrent.futures as _cf
import multiprocessing as _mp
import os
from contextlib import contextmanager

import numpy as np

from . import defaults

#: The threading variables every BLAS this package might meet reads. Set for
#: the children only; see the module docstring.
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextmanager
def pinned_blas(n_threads: int = 1):
    """Set the BLAS threading environment, then restore exactly what was there.

    Restores by DELETING variables that were absent rather than setting them to
    their old string, so a caller who had none ends with none -- otherwise this
    would silently pin the caller's process for the rest of its life, which is
    the failure the module docstring is about.
    """
    before = {k: os.environ.get(k) for k in THREAD_ENV}
    try:
        for k in THREAD_ENV:
            os.environ[k] = str(int(n_threads))
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def plan_chunks(n_src: int, workers: int, bytes_per_src: int,
                budget: int | None = None) -> list[tuple[int, int]]:
    """``[(lo, hi), ...]`` source ranges: at least one per worker, more if the
    blocks in flight would otherwise exceed ``budget``.

    More chunks than workers is the point. With exactly one chunk each, the
    blocks in flight are the whole output a second time; with enough chunks the
    peak is bounded by the budget instead, and a straggler cannot hold a core
    idle at the end.
    """
    n_src, workers = int(n_src), max(1, int(workers))
    if n_src <= 0:
        return []
    cap = int(defaults.PARALLEL_INFLIGHT_BYTES if budget is None else budget)
    per_chunk = max(1, cap // max(workers * max(bytes_per_src, 1), 1))
    n_chunks = max(workers, -(-n_src // per_chunk))
    n_chunks = min(n_chunks, n_src)
    edges = np.linspace(0, n_src, n_chunks + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _run_chunk(args):
    """Worker body: call the pickled partial on one slice of ``tris``."""
    call, tris, lo, hi, tris_kw = args
    sub = tris[lo:hi]
    block = call(**{tris_kw: sub}) if tris_kw else call(sub)
    return lo, hi, np.ascontiguousarray(block)


def by_source(call, tris, *, workers: int | None = None, out=None,
              source_axis: int = -2, tris_kw: str | None = None,
              blas_threads: int = 1, chunks: int | None = None):
    """Evaluate ``call`` over chunks of ``tris`` in parallel and stitch.

    ``call`` is anything picklable that takes one ``tris`` sub-array -- in
    practice ``functools.partial`` of an ``mhs`` entry point with the observers
    and material already bound. It must depend on the sources ONLY through that
    argument: an entry point that derives its observers from ``tris``, as
    ``interaction_matrix`` does, has to be given ``obs_pts`` explicitly or each
    chunk will collocate on itself alone. ``tris_kw`` names the keyword to pass
    the sub-array as, when it is not the first positional.

    ``source_axis`` is where the slip-DOF axis sits in the result: ``-2`` for
    every matrix whose shape ends ``(..., n_dof, 3)``, and ``1`` for
    ``interaction_matrix``, whose first two axes are both DOF axes.

    ``workers=1`` runs in this process with no pool, which is the path the
    gates compare against. The result is BITWISE the serial one when BLAS
    threading matches, because each source's block depends on no other source;
    with ``blas_threads`` changing the gemm shape it can differ in the last
    bits, which is why the gate pins both.
    """
    tris = np.ascontiguousarray(np.asarray(tris, float).reshape(-1, 3, 3))
    n_src = tris.shape[0]
    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)
    workers = max(1, min(int(workers), n_src))

    # One chunk first, to learn the block shape without assuming it.
    head = plan_chunks(n_src, workers, 1, budget=1) if n_src else []
    probe_hi = head[0][1] if head else n_src
    _, _, first = _run_chunk((call, tris, 0, probe_hi, tris_kw))
    axis = source_axis if source_axis >= 0 else first.ndim + source_axis
    per_src = int(first.size // max(first.shape[axis], 1)) * first.itemsize
    dof_per_src = first.shape[axis] // max(probe_hi, 1)

    shape = list(first.shape)
    shape[axis] = dof_per_src * n_src
    shape = tuple(shape)
    if out is None:
        buf = np.empty(shape, dtype=first.dtype)
    else:
        if out.shape != shape:
            raise ValueError(f"out has shape {out.shape}, expected {shape}")
        buf = out

    def write(lo, hi, block):
        sl = [slice(None)] * buf.ndim
        sl[axis] = slice(lo * dof_per_src, hi * dof_per_src)
        buf[tuple(sl)] = block

    write(0, probe_hi, first)
    del first
    if probe_hi >= n_src:
        return buf

    rest = plan_chunks(n_src - probe_hi, workers, per_src * dof_per_src,
                       budget=chunks and None)
    tasks = [(call, tris, probe_hi + a, probe_hi + b, tris_kw)
             for a, b in rest]
    if workers == 1:
        for t in tasks:
            write(*_run_chunk(t))
        return buf

    # Spawn, not fork: a forked child inherits a numpy/BLAS state that was
    # initialised before the environment was pinned, and on macOS a forked
    # process that touches certain frameworks is undefined behaviour.
    try:
        with pinned_blas(blas_threads):
            ctx = _mp.get_context("spawn")
            with _cf.ProcessPoolExecutor(max_workers=workers,
                                         mp_context=ctx) as pool:
                futures = [pool.submit(_run_chunk, t) for t in tasks]
                for fut in _cf.as_completed(futures):
                    write(*fut.result())    # consumed as they land, not gathered
    except _cf.process.BrokenProcessPool as exc:
        raise RuntimeError(
            "mhs.parallel.by_source: the worker pool died. The usual cause is "
            "a calling SCRIPT whose top level is not guarded, because the "
            "workers are SPAWNED and so re-import it: every child then re-runs "
            "the script and spawns its own children. Wrap the call site in\n"
            "    if __name__ == '__main__':\n"
            "or run it from a function, a notebook or an interactive session. "
            "The other cause is a `call` that is not picklable -- it must be a "
            "module-level function or a functools.partial of one, not a lambda "
            "or a closure. Use workers=1 to run in this process and get the "
            "real traceback.") from exc
    return buf
