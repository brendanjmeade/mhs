"""The on-fault interaction matrix for a network, assembled once and cached.

ONE EXPENSIVE STEP, THEN EVERYTHING ELSE IS CHEAP. Assembling ``K`` is the only
costly operation in this study; a whole quasi-static loading history is then a
sequence of single-column reads. So ``K`` is assembled once per geometry and
cached to disk, and friction realizations, load directions, eps sweeps and both
friction variants all reuse it.

``mhs.interaction_matrix`` is the right entry point and ``stress_matrix`` is the
wrong one. The interaction matrix resolves traction onto each receiver's own
(strike, dip, normal) frame inside the assembly loop and never materialises a
stress tensor: 48 bytes per element pair against 216, and at production size the
library's own memory ceiling refuses the stress form outright.

THE LAYOUT IS SOURCE-MAJOR, and that is a performance decision with a factor of
twenty in it. The quasi-static cascade's hot operation is "element j slipped, so
update the traction on every receiver" -- one SOURCE's column, every receiver.
``interaction_matrix`` returns ``(n_rec, n_src, n_a, n_b)``, in which that
column is strided across the whole array: at 20k elements, 20000 scattered
cache lines per slip event, milliseconds each, hours over a run. Transposed to
``(n_src, n_b, n_a, n_rec)`` the same read is one contiguous 960 kB block, about
50 microseconds. The sibling payload script made the same choice for the same
reason -- "stored SOURCE-MAJOR ... which is the slice the page reads on a click".

THE MEMORY CEILING IS RAISED DELIBERATELY, not worked around.
``mhs.defaults.MATRIX_MAX_BYTES`` is 8 GiB and the 20k tier needs 19.2 GB.
``mhs/defaults.py`` names it "the only one" a caller may legitimately rebind,
"a resource limit, not an acceptance criterion", so :func:`raise_ceiling` does
exactly that and says so on the way past. Every other number in ``mhs.defaults``
stays where it is.

WORKERS ARE PROCESSES, NOT THREADS. ``mhs`` is pure numpy with no numba
kernels, and threading its per-source loop measures 1.28x at two threads and
then gets worse, because the body is many small numpy calls holding the GIL.
``mhs.parallel.by_source`` spawns processes instead, which measures 8.5x on 12
workers -- so a caller must be under ``if __name__ == "__main__"`` and must pass
a module-level callable. The partial built here is picklable for that reason;
a lambda would raise ``BrokenProcessPool`` at the first worker.
"""
from __future__ import annotations

import functools
import hashlib
import json
import pathlib
import subprocess
import time

import numpy as np

import mhs
from mhs import defaults as mhs_defaults
from mhs.parallel import by_source

#: Receiver components, in this order. Strike and dip shear are what the
#: Coulomb criterion compares; "normal" is the library's accepted synonym for
#: the frame's third row on the receiver side, and it carries sigma_n.
RECEIVER = ("strike", "dip", "normal")

#: Source components. Tensile is omitted deliberately: a frictional fault slips
#: in its own plane, and an opening mode would need a different constitutive
#: statement than the Coulomb criterion this study applies.
SOURCE = ("strike", "dip")

N_A, N_B = len(RECEIVER), len(SOURCE)

#: Bytes per element pair in the stored matrix. The arithmetic the ceiling and
#: the cache both quote, in one place.
BYTES_PER_PAIR = N_A * N_B * 8


def required_bytes(n_elements: int) -> int:
    """Bytes for one source-major matrix at ``n_elements``."""
    return int(n_elements) ** 2 * BYTES_PER_PAIR


def raise_ceiling(n_elements: int, *, quiet: bool = False) -> int | None:
    """Rebind ``MATRIX_MAX_BYTES`` if this matrix needs more than the default.

    Returns the previous value when it changed, so a caller can restore it.
    This is the one number in ``mhs.defaults`` a caller may rebind, and the
    rebind is announced rather than silent: a resource limit that moves without
    saying so is how a swap storm becomes a mystery.
    """
    need = required_bytes(n_elements)
    have = int(mhs_defaults.MATRIX_MAX_BYTES)
    if need <= have:
        return None
    was = have
    mhs_defaults.MATRIX_MAX_BYTES = int(need * 1.05)
    if not quiet:
        print(f"  MATRIX_MAX_BYTES {was / 1024 ** 3:.1f} -> "
              f"{mhs_defaults.MATRIX_MAX_BYTES / 1024 ** 3:.1f} GiB for "
              f"{n_elements} elements ({need / 1024 ** 3:.1f} GiB). Sanctioned: "
              f"mhs/defaults.py calls this the one user-facing ceiling.")
    return was


def _git() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5,
                              cwd=pathlib.Path(__file__).resolve().parent
                              ).stdout.strip() or None
    except Exception:
        return None


def cache_key(tris: np.ndarray, mu: float, lam: float, eps: float) -> str:
    """A short hash of everything the matrix depends on.

    KEYED ON THE VERTICES THEMSELVES, not on the generator's parameters. An
    earlier version hashed ``net.meta``, which is wrong in a way that only a
    tripwire found: two networks with identical metadata and DIFFERENT
    triangles collide, so a mesh whose vertex order had been reversed got a
    cache hit and came back as the original matrix -- with its sign refusal
    silently skipped. Hashing the array is 1 ms at 13k elements and makes a
    stale cache a miss rather than a wrong answer, which is the whole point of
    having one.
    """
    tris = np.ascontiguousarray(np.asarray(tris, float))
    geom = hashlib.sha256(tris.tobytes()).hexdigest()
    payload = {"geometry_sha256": geom, "n": int(tris.shape[0]),
               "mu": mu, "lam": lam, "eps": eps,
               "receiver": list(RECEIVER), "source": list(SOURCE),
               "layout": "source_major_(n_src,n_b,n_a,n_rec)"}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def refuse_if_sign_wrong(K: np.ndarray) -> None:
    """Refuse a matrix whose self-interaction diagonal is not negative.

    Slip must RELIEVE the shear that drives it, so every diagonal entry of the
    strike-from-strike and dip-from-dip shear blocks is negative. This is the
    condition the whole quasi-static scheme rests on -- it is what makes the
    active-set submatrix negative definite and the cascade convergent -- and it
    is exactly what a frame flipped in both the kernel and its reference would
    leave looking plausible.

    ``tests/gates/mhs/verify_image_kernel`` clause [h] owns this as a physics
    claim about the library. What is checked here is that THIS network does not
    violate it, which is a different question: a network has mixed
    orientations, near-horizontal elements where the frame is a stated
    convention rather than a derivation, and a separation floor letting distinct
    faults sit 2.5 eps apart.
    """
    n = K.shape[0]
    idx = np.arange(n)
    bad = []
    for b, name in enumerate(SOURCE):
        a = RECEIVER.index(name)
        diag = K[idx, b, a, idx]
        if not np.all(diag < 0.0):
            worst = int(np.argmax(diag))
            bad.append(f"{name}: {int((diag >= 0).sum())} of {n} non-negative, "
                       f"worst element {worst} at {diag[worst]:.4e}")
    if bad:
        raise SystemExit(
            "self-interaction diagonal is not negative, so slip would not "
            "relieve the shear that drives it and the cascade cannot be "
            "trusted to converge:\n  " + "\n  ".join(bad))


def _assemble_raw(tris: np.ndarray, mu: float, lam: float, eps: float,
                  workers: int | None) -> np.ndarray:
    """``(n_rec, n_src, n_a, n_b)`` straight from the library."""
    mat = mhs.Material(mu=mu, lam=lam)
    call = functools.partial(mhs.interaction_matrix, material=mat, eps=eps,
                             receiver=RECEIVER, source=SOURCE, obs_tris=tris)
    if workers in (0, 1):
        return call(tris=tris)
    # source_axis=1 and tris_kw="tris" are interaction_matrix's contract;
    # obs_tris is bound above so every chunk sees the FULL receiver set.
    return by_source(call, tris, workers=workers, source_axis=1,
                     tris_kw="tris")


def to_source_major(K: np.ndarray) -> np.ndarray:
    """``(n_rec, n_src, n_a, n_b)`` -> contiguous ``(n_src, n_b, n_a, n_rec)``."""
    return np.ascontiguousarray(np.transpose(K, (1, 3, 2, 0)))


def assemble(net, *, mu: float = 30.0, lam: float = 30.0,
             eps: float | None = None, workers: int | None = None,
             cache_dir: pathlib.Path | None = None,
             mmap: bool = False, verbose: bool = True):
    """The source-major interaction matrix for ``net``, from cache or fresh.

    Returns ``(K, info)`` with ``K`` shaped ``(n_src, n_b, n_a, n_rec)``:
    ``K[j, b, a, i]`` is the traction on receiver ``i``'s own plane along
    direction ``a`` of its frame, from unit slip along direction ``b`` of
    source ``j``'s frame.

    ``mmap=True`` returns a read-only memory map instead of a resident array,
    which is what makes the 20k tier openable without 19 GB of RSS for a job
    that only reads columns.
    """
    eps = float(net.eps if eps is None else eps)
    n = net.n_elements
    key = cache_key(net.tris, mu, lam, eps)
    cache_dir = (pathlib.Path(__file__).resolve().parent / "cache"
                 if cache_dir is None else pathlib.Path(cache_dir))
    npy, side = cache_dir / f"K_{key}.npy", cache_dir / f"K_{key}.json"

    if npy.exists() and side.exists():
        info = json.loads(side.read_text())
        if verbose:
            print(f"  cache hit {npy.name}  ({info['bytes'] / 1024 ** 3:.2f} "
                  f"GiB, assembled in {info['assemble_s']:.1f} s)")
        K = np.load(npy, mmap_mode="r" if mmap else None)
        if K.shape != (n, N_B, N_A, n):
            raise RuntimeError(
                f"cached matrix {npy.name} has shape {K.shape}, expected "
                f"{(n, N_B, N_A, n)}. The key matched but the array did not, "
                f"so the cache is corrupt rather than stale -- delete it.")
        return K, info

    need = required_bytes(n)
    if verbose:
        print(f"  assembling {n} x {n} x {N_A} x {N_B} = "
              f"{need / 1024 ** 3:.2f} GiB (+ the same again for the "
              f"transpose)")
    was = raise_ceiling(n, quiet=not verbose)
    t0 = time.time()
    try:
        raw = _assemble_raw(net.tris, mu, lam, eps, workers)
    finally:
        if was is not None:
            mhs_defaults.MATRIX_MAX_BYTES = was
    K = to_source_major(raw)
    del raw
    dt = time.time() - t0
    refuse_if_sign_wrong(K)

    info = {"key": key, "n_elements": n, "bytes": int(K.nbytes),
            "assemble_s": dt, "us_per_pair": 1e6 * dt / (n * n),
            "mu_GPa": mu, "lam_GPa": lam, "eps_km": eps,
            "workers": workers, "receiver": list(RECEIVER),
            "source": list(SOURCE),
            "layout": "(n_src, n_b, n_a, n_rec), C-contiguous",
            "git": _git(), "net": net.meta}
    if verbose:
        print(f"  assembled in {dt:.1f} s = {info['us_per_pair']:.2f} "
              f"us/pair on {workers} workers")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy, K)
    side.write_text(json.dumps(info, indent=1, sort_keys=True, default=str))
    if verbose:
        print(f"  cached -> {npy.name}")
    return K, info


def project(K: np.ndarray, rec_dir: np.ndarray,
            src_dir: np.ndarray) -> np.ndarray:
    """Contract to the one-component operator the fixed-direction variant uses.

    ``rec_dir`` and ``src_dir`` are ``(n, 2)`` unit vectors in each element's
    own (strike, dip) plane. The result ``P[j, i]`` is the shear on receiver
    ``i`` ALONG ``rec_dir[i]`` from unit slip on source ``j`` along
    ``src_dir[j]`` -- still source-major, and one eighth the size, which is
    what makes the reference variant cheap to iterate on.
    """
    sh = K[:, :, :2, :]                      # drop the normal row
    return np.ascontiguousarray(
        np.einsum("jbai,ia,jb->ji", sh, rec_dir, src_dir))


def normal_block(K: np.ndarray) -> np.ndarray:
    """``(n_src, n_b, n_rec)``: normal traction from unit slip. Coupling only.

    The fixed-sigma_n variant never touches this; the fully coupled one needs
    it every step, which is why it is a view rather than a copy.
    """
    return K[:, :, RECEIVER.index("normal"), :]
