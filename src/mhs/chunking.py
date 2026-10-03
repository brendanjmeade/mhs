"""Byte arithmetic for the matrix forms, and the refusal that carries it.

There is no ``disp_block`` here and there will not be one: a block is
``disp_matrix(obs[a:b], tris[c:d], ...)``, and slicing already says that. cutde
needs a block entry point because its blocks are a ragged batch submitted in one
GPU launch with an offset array; ``mhs`` has no launch to amortise, so a second
spelling of one thing would be a bug.

What this module does provide is the arithmetic, so a caller sizing a loop and
the refusal message explaining why a call was declined read from one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import defaults

#: Trailing shape per obs/source pair, by output kind. ``(3,)`` for a vector
#: readout and ``(3, 3)`` for a tensor one, times the 3 slip components.
PAIR_SHAPE = {
    "disp": (3, 3),
    "stress": (3, 3, 3),
    "strain": (3, 3, 3),
    "eigenstress": (3, 3, 3),
}


def pair_bytes(kind: str, dtype=np.float64, parts: bool = False) -> int:
    """Bytes per obs/source pair for one output ``kind``.

    ``parts=True`` returns two arrays of that shape, so it is twice the cost --
    the ceiling must be checked against what the call will actually allocate,
    not against the shape of one of its outputs.
    """
    if kind not in PAIR_SHAPE:
        raise KeyError(f"unknown output kind {kind!r}; "
                       f"expected one of {sorted(PAIR_SHAPE)}")
    n = int(np.prod(PAIR_SHAPE[kind])) * np.dtype(dtype).itemsize
    return 2 * n if parts else n


def default_ceiling() -> int:
    """``MATRIX_MAX_BYTES``, additionally capped at a fraction of total RAM.

    The fraction matters on a machine smaller than the absolute number: a limit
    above physical memory is not a limit.
    """
    cap = int(defaults.MATRIX_MAX_BYTES)
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return cap            # unknown RAM: the absolute number stands alone
    return min(cap, int(defaults.MATRIX_RAM_FRACTION * total))


@dataclass(frozen=True)
class ChunkPlan:
    """What a chunked assembly would cost. Allocates nothing."""

    n_obs: int
    n_src: int
    bytes_total: int
    bytes_per_chunk: int
    obs_per_chunk: int
    n_chunks: int


def chunk_plan(n_obs: int, n_src: int, kind: str = "stress",
               dtype=np.float64, parts: bool = False,
               ceiling: int | None = None) -> ChunkPlan:
    """Plan an obs-sliced assembly that fits under ``ceiling``.

    Slices the OBS axis because it is the leading one, so each chunk is a
    contiguous block of the full array and a caller can write it straight into a
    memmap at an offset.
    """
    n_obs, n_src = int(n_obs), int(n_src)
    if n_obs < 0 or n_src < 0:
        raise ValueError(f"negative sizes: n_obs={n_obs}, n_src={n_src}")
    per = pair_bytes(kind, dtype, parts)
    row = per * max(n_src, 1)                 # bytes for one obs point
    total = row * n_obs
    cap = default_ceiling() if ceiling is None else int(ceiling)
    per_chunk_obs = max(1, cap // max(row, 1))
    per_chunk_obs = min(per_chunk_obs, max(n_obs, 1))
    n_chunks = -(-n_obs // per_chunk_obs) if n_obs else 0
    return ChunkPlan(n_obs=n_obs, n_src=n_src, bytes_total=total,
                     bytes_per_chunk=row * per_chunk_obs,
                     obs_per_chunk=per_chunk_obs, n_chunks=n_chunks)


def _gib(n: int) -> str:
    """Binary units, labelled as such. 216 B x 10k x 10k is 21.6 GB == 20.1 GiB,
    and a refusal message that mislabels the one is a refusal nobody can check."""
    return f"{n / 1024**3:.2f} GiB"


def refuse_if_too_large(n_obs: int, n_src: int, kind: str,
                        dtype=np.float64, parts: bool = False,
                        ceiling: int | None = None) -> None:
    """Raise with the arithmetic if this call would allocate past the ceiling.

    The message is the point. A bare "too large" leaves the caller guessing; the
    numbers below tell them the per-pair cost, the largest obs count that fits at
    this source count, and the two escapes.
    """
    cap = default_ceiling() if ceiling is None else int(ceiling)
    plan = chunk_plan(n_obs, n_src, kind, dtype, parts, ceiling=cap)
    if plan.bytes_total <= cap:
        return
    raise MemoryError(
        f"{kind} matrix for {n_obs} obs x {n_src} sources would allocate "
        f"{_gib(plan.bytes_total)}"
        f"{' (parts=True returns two arrays)' if parts else ''}, over the "
        f"{_gib(cap)} ceiling.\n"
        f"  per obs/source pair: {pair_bytes(kind, dtype, parts)} B "
        f"({np.dtype(dtype).name})\n"
        f"  largest n_obs that fits at n_src={n_src}: {plan.obs_per_chunk}\n"
        f"  so this is {plan.n_chunks} chunks of {_gib(plan.bytes_per_chunk)}\n"
        f"Escapes: pass out= (a np.memmap, or a reused slab, or a float32 "
        f"buffer to halve it), or slice the inputs -- mhs.chunking.chunk_plan() "
        f"reports this arithmetic without allocating. The ceiling itself is "
        f"mhs.defaults.MATRIX_MAX_BYTES; it refuses rather than tries because a "
        f"request this size does not fail on most machines, it swaps.")
