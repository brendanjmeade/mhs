#!/usr/bin/env python
"""Build the two data payloads the website draws, by hand, from `mhs`.

    python website/scripts/make_payload.py fault
    python website/scripts/make_payload.py volume [--spacing 1.5] [--workers 8]

WHY THIS LIVES HERE AND NOT IN THE PACKAGE. The sibling project keeps its
encoder in the Python package so a gate can cover it. `mhs` declares two runtime
dependencies and treats the omissions as design, so a web exporter inside it
would be exactly the scope that package refuses. The cost of keeping it out here
is that no gate covers the quantisation -- which is the one step that can
silently corrupt a picture into a plausible-looking lie -- so instead this script
DECODES WHAT IT WROTE and prints the measured round-trip error. Read that number;
it is the only check there is.

Both figures share ONE fault, built by `dipping_fault` below, so the clickable
interaction matrix on the home page and the volume on Examples are the same
object seen two ways.

The encoding (log10 into uint8, index 0 reserved for "no data") and the manifest
schema are the sibling site's, unchanged, because its loader is reused verbatim.
log10 rather than linear because the fields span several decades: measured there
at 2.4 % worst relative error against 99.7 % for linear uint8.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

import mhs
from mhs import tdcs
from mhs.parallel import pinned_blas

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "public" / "data"

SCHEMA = 1
NODATA = 0
LEVELS = 254                      # levels 1..255; 0 means "outside the body"

# ---------------------------------------------------------------- the model --
MU, LAM = 30.0, 30.0              # GPa; nu = 0.25
EPS = 0.5                         # km -- the mollification width
SLIP_M = 1.0                      # metres of right-lateral slip
LENGTH, WIDTH = 40.0, 20.0        # km along strike, km down dip
DIP_DEG = 60.0
N_STRIKE, N_DIP = 16, 8           # -> 256 triangles
FRICTION = 0.6                    # the apparent friction in Delta-CFS


def dipping_fault(length=LENGTH, width=WIDTH, dip_deg=DIP_DEG,
                  n_s=N_STRIKE, n_d=N_DIP, top=0.0) -> np.ndarray:
    """A surface-breaking rectangular fault as `(n, 3, 3)` triangles.

    Strike along +x, dipping toward +y, top edge at `z = top`. Each quad is cut
    on the same diagonal, so the two triangles of a quad share an edge and the
    mesh is conforming -- which matters because a collocation point sits at a
    shrunk centroid and a non-conforming mesh would put receivers in gaps.
    """
    dip = np.radians(dip_deg)
    s = np.linspace(-length / 2.0, length / 2.0, n_s + 1)
    d = np.linspace(0.0, width, n_d + 1)
    S, D = np.meshgrid(s, d, indexing="ij")
    V = np.stack([S, D * np.cos(dip), top - D * np.sin(dip)], axis=-1)
    tris = [t for i in range(n_s) for j in range(n_d)
            for t in ([V[i, j], V[i + 1, j], V[i + 1, j + 1]],
                      [V[i, j], V[i + 1, j + 1], V[i, j + 1]])]
    return np.ascontiguousarray(np.asarray(tris, float))


def cartesian_slip(tris: np.ndarray, magnitude_km: float) -> np.ndarray:
    """Uniform strike slip, as the Cartesian `(n, 3)` the matrices contract."""
    local = np.zeros((tris.shape[0], 3))
    local[:, 0] = magnitude_km          # (strike, dip, tensile)
    return tdcs.to_cartesian(tris, local)


def material() -> "mhs.Material":
    return mhs.Material(mu=MU, lam=LAM)


def _meta() -> dict:
    return {"mu_GPa": MU, "lam_GPa": LAM, "nu": LAM / (2.0 * (LAM + MU)),
            "eps_km": EPS, "slip_m": SLIP_M, "friction": FRICTION,
            "length_km": LENGTH, "width_km": WIDTH, "dip_deg": DIP_DEG,
            "n_triangles": N_STRIKE * N_DIP * 2, "units": "km, GPa"}


# ------------------------------------------------------------ the hero: CFS --
def build_fault(out: pathlib.Path) -> None:
    """The interaction matrix the home page clicks through.

    Three matrices are written, not one, so the page can show what Delta-CFS is
    made of: the shear traction resolved on the receiver's own strike direction,
    the normal traction (tension positive, so positive is unclamping), and their
    combination. Each is stored SOURCE-MAJOR -- `buf[j*n : (j+1)*n]` is every
    receiver's response to unit slip on source `j` -- which is the slice the page
    reads on a click.
    """
    tris = dipping_fault()
    mat, n = material(), N_STRIKE * N_DIP * 2
    print(f"fault: {n} triangles, "
          f"z {tris[..., 2].min():.2f} .. {tris[..., 2].max():.2f} km")

    t0 = time.perf_counter()
    kw = dict(eps=EPS, source="strike", obs_tris=tris)
    shear = mhs.interaction_matrix(tris, mat, receiver="strike", **kw)[:, :, 0, 0]
    normal = mhs.interaction_matrix(tris, mat, receiver="normal", **kw)[:, :, 0, 0]
    print(f"  two interaction matrices in {time.perf_counter() - t0:.1f} s")

    # Per metre of slip. The matrices are per unit (km) of slip, so scale to the
    # stated slip and to MPa for the colour bar.
    scale = (SLIP_M * 1e-3) * 1e3            # km of slip, GPa -> MPa
    shear, normal = shear * scale, normal * scale
    cfs = shear + FRICTION * normal

    # THE SIGN, anchored rather than assumed: a patch that slips must RELIEVE
    # its own shear stress, so every diagonal entry of the shear matrix is
    # negative. That one fact fixes what the colours mean on the page, and it is
    # checked here rather than reasoned about, because getting it backwards
    # would invert the caption and still look plausible.
    diag = np.diag(shear)
    if not (diag < 0).all():
        raise SystemExit(f"shear self-interaction is not negative "
                         f"(max {diag.max():.3e}); the sign convention moved "
                         f"and the page's colour legend is no longer valid")
    print(f"  diag(shear) in [{diag.min():.3f}, {diag.max():.3f}] MPa, all < 0")

    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, A in (("shear", shear), ("normal", normal), ("cfs", cfs)):
        buf = np.ascontiguousarray(A.T, dtype=np.float32).tobytes()
        (out / f"{name}.bin").write_bytes(buf)
        finite = np.isfinite(A)
        arrays[name] = {"bytes": len(buf), "n": n,
                        "order": "source-major: buf[j*n + i] is receiver i "
                                 "from unit slip on source j",
                        "min": float(A[finite].min()),
                        "max": float(A[finite].max()),
                        "sha256": hashlib.sha256(buf).hexdigest()[:16]}
        print(f"  {name}.bin  {len(buf) / 1024:.0f} kB  "
              f"[{A.min():+.3f}, {A.max():+.3f}] MPa")

    payload = {"schema": SCHEMA, "unit": "MPa", "meta": _meta(),
               "triangles": [[[round(float(c), 4) for c in v] for v in t]
                             for t in tris],
               "centroids": [[round(float(c), 4) for c in t.mean(axis=0)]
                             for t in tris],
               "arrays": arrays}
    (out / "fault.json").write_text(json.dumps(payload) + "\n")
    kb = (out / "fault.json").stat().st_size / 1024
    print(f"  fault.json {kb:.0f} kB")


# -------------------------------------------------------- the volume fields --
def _invariants(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """von Mises and maximum shear from a stack of `(n, 3, 3)` stresses."""
    tr = np.trace(sig, axis1=-2, axis2=-1) / 3.0
    dev = sig - tr[..., None, None] * np.eye(3)
    vm = np.sqrt(1.5 * np.einsum("...ij,...ij->...", dev, dev))
    w = np.linalg.eigvalsh(sig)              # ascending
    return vm, 0.5 * (w[..., 2] - w[..., 0])


def _chunk_half(args):
    """One observer chunk of the HALF-SPACE field: |u|, von Mises, max shear."""
    pts, tris, slip, eps, mu, lam = args
    mat = mhs.Material(mu=mu, lam=lam)
    u = np.einsum("oisk,sk->oi", mhs.disp_matrix(pts, tris, mat, eps=eps), slip)
    sig = np.einsum("oijsk,sk->oij",
                    mhs.stress_matrix(pts, tris, mat, eps=eps), slip)
    vm, ms = _invariants(sig)
    return np.linalg.norm(u, axis=1), vm, ms


def _chunk_full(args):
    """The same chunk in a FULL space -- the vendored engine, no image terms."""
    pts, tris, slip, eps, mu, lam = args
    from mhs.fullspace import api
    nu = lam / (2.0 * (lam + mu))
    u = np.zeros((pts.shape[0], 3))
    sig = np.zeros((pts.shape[0], 3, 3))
    for t, s in zip(tris, slip):
        u += api.displacement(pts, t, s, mu, nu, eps)
        sig += api.stress(pts, t, s, mu, nu, eps)
    vm, ms = _invariants(sig)
    return np.linalg.norm(u, axis=1), vm, ms


FIELDS = ("u_mag", "von_mises", "max_shear")


def _evaluate(which, pts, tris, slip, workers, chunk=1500):
    """Run `_chunk_half`/`_chunk_full` over observer chunks and stitch.

    Observer chunks, not source chunks: the matrix for the whole grid could
    never be allocated -- 123k observers against 256 triangles is 7 TB for the
    stress form -- so each chunk is built and contracted against the slip
    immediately, and only the three scalars per point come back.
    """
    fn = _chunk_half if which == "half" else _chunk_full
    edges = list(range(0, pts.shape[0], chunk)) + [pts.shape[0]]
    tasks = [(pts[a:b], tris, slip, EPS, MU, LAM)
             for a, b in zip(edges[:-1], edges[1:])]
    out = {f: np.empty(pts.shape[0]) for f in FIELDS}

    t0 = time.perf_counter()
    done = 0

    def store(i, res):
        a, b = edges[i], edges[i + 1]
        for f, v in zip(FIELDS, res):
            out[f][a:b] = v

    if workers <= 1:
        for i, t in enumerate(tasks):
            store(i, fn(t))
            done += t[0].shape[0]
            _progress(which, done, pts.shape[0], t0)
    else:
        # Spawn and pin BLAS in the children, for the reason mhs.parallel gives:
        # these matmuls are small, so BLAS's own threads fight the pool and lose.
        with pinned_blas(1):
            ctx = mp.get_context("spawn")
            with cf.ProcessPoolExecutor(max_workers=workers,
                                        mp_context=ctx) as pool:
                futs = {pool.submit(fn, t): i for i, t in enumerate(tasks)}
                for fut in cf.as_completed(futs):
                    i = futs[fut]
                    store(i, fut.result())
                    done += tasks[i][0].shape[0]
                    _progress(which, done, pts.shape[0], t0)
    print()
    dt = time.perf_counter() - t0
    pairs = pts.shape[0] * tris.shape[0]
    print(f"  {which}: {dt:.1f} s for {pairs / 1e6:.1f}M pairs "
          f"= {dt / pairs * 1e6:.2f} us/pair ({workers} workers)")
    return out


def _progress(which, done, total, t0):
    el = time.perf_counter() - t0
    eta = el / max(done, 1) * (total - done)
    sys.stdout.write(f"\r  {which}: {done}/{total} points, "
                     f"{el:.0f}s elapsed, {eta:.0f}s left   ")
    sys.stdout.flush()


# ------------------------------------------------------------- the encoding --
def encode_log_u8(v: np.ndarray, inside: np.ndarray) -> tuple[bytes, dict]:
    """Positive, many-decade field -> uint8 over log10. 0 means outside."""
    v = np.asarray(v, float)
    good = inside & np.isfinite(v) & (v > 0.0)
    if not good.any():
        raise ValueError("no positive finite samples to encode")
    lo, hi = float(np.log10(v[good].min())), float(np.log10(v[good].max()))
    if hi <= lo:
        hi = lo + 1e-12
    q = np.zeros(v.shape, np.uint8)
    t = (np.log10(np.where(good, v, 1.0)) - lo) / (hi - lo)
    q[good] = 1 + np.clip(np.round(t[good] * (LEVELS - 1)), 0,
                          LEVELS - 1).astype(np.uint8)
    return q.tobytes(), {"encoding": "log10_u8", "log_min": lo, "log_max": hi,
                         "min": float(v[good].min()),
                         "max": float(v[good].max())}


def decode_log_u8(buf: bytes, meta: dict) -> np.ndarray:
    q = np.frombuffer(buf, np.uint8).astype(float)
    t = (q - 1.0) / (LEVELS - 1)
    out = 10.0 ** (meta["log_min"] + t * (meta["log_max"] - meta["log_min"]))
    return np.where(q == NODATA, np.nan, out)


def encode_codes_u8(v: np.ndarray) -> tuple[bytes, dict]:
    """A categorical code, stored as itself. The reader binds it to NEAREST."""
    q = np.asarray(v, float).astype(np.uint8)
    return q.tobytes(), {"encoding": "codes_u8", "filter": "nearest",
                         "values": sorted(int(x) for x in np.unique(q))}


# ------------------------------------------------------------- the volume -----
def build_volume(out: pathlib.Path, spacing: float, workers: int,
                 half_only: bool = False) -> None:
    tris = dipping_fault()
    slip = cartesian_slip(tris, SLIP_M * 1e-3)

    # The box. It reaches ABOVE the free surface on purpose: those points are
    # outside the body and encode as "no data", so the viewer draws the free
    # surface as a real top boundary instead of as the edge of the sampled box.
    lo = np.array([-45.0, -40.0, -48.0])
    hi = np.array([45.0, 50.0, 6.0])
    dims = np.maximum(2, np.floor((hi - lo) / spacing).astype(int) + 1)
    gx, gy, gz = (lo[i] + spacing * np.arange(dims[i]) for i in range(3))
    # x fastest, matching the loader's (dims[2], dims[1], dims[0]) reshape.
    Z, Y, X = np.meshgrid(gz, gy, gx, indexing="ij")
    pts_all = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    n_all = pts_all.shape[0]

    inside = pts_all[:, 2] <= 0.0
    live = pts_all[inside]
    print(f"grid {tuple(int(d) for d in dims)} at {spacing} km = {n_all} points, "
          f"{live.shape[0]} inside the half space "
          f"({100 * (1 - live.shape[0] / n_all):.0f}% above z=0, not evaluated)")
    print(f"  {live.shape[0] * tris.shape[0] / 1e6:.1f}M observer-source pairs "
          f"per kernel")

    states = ["half"] if half_only else ["half", "full"]
    got = {}
    for which in states:
        res = _evaluate(which, live, tris, slip, workers)
        got[which] = {f: _scatter(res[f], inside, n_all) for f in FIELDS}

    out.mkdir(parents=True, exist_ok=True)
    grid = {"dims": [int(d) for d in dims],
            "origin": [float(x) for x in lo],
            "spacing": [spacing] * 3, "n_points": int(n_all)}
    man = {"schema": SCHEMA, "grid": grid, "nodata": NODATA, "levels": LEVELS,
           "source": {"volume_run": "make_payload.py", "sampled_from": None,
                      "git": _git()},
           "meta": _meta(), "states": {}, "arrays": {}}

    def put(name, buf, meta):
        (out / f"{name}.bin").write_bytes(buf)
        meta = dict(meta, bytes=len(buf),
                    sha256=hashlib.sha256(buf).hexdigest()[:16])
        man["arrays"][name] = meta
        return meta

    # The region mask: one code, shared by every state, because unlike the
    # sibling model there is no topography and no inclusion -- the body is
    # z <= 0 and nothing else varies.
    rbuf, rmeta = encode_codes_u8(np.where(inside, 1, 0))
    rkey = f"region.{hashlib.sha256(rbuf).hexdigest()[:16]}"
    put(rkey, rbuf, rmeta)

    checks: list[tuple[str, float, float]] = []
    for which in states:
        st = {"kind": "state", "fields": {}, "region": rkey}
        for f in FIELDS:
            buf, meta = encode_log_u8(got[which][f], inside)
            key = f"{which}.{f}"
            put(key, buf, meta)
            st["fields"][f] = key
            checks.append((key, *_roundtrip(got[which][f], inside, buf, meta)))
        man["states"][which] = st

    if not half_only:
        st = {"kind": "difference", "fields": {}, "region": rkey}
        for f in FIELDS:
            d = np.abs(got["half"][f] - got["full"][f])
            buf, meta = encode_log_u8(d, inside)
            key = f"diff_half_minus_full.{f}"
            put(key, buf, meta)
            st["fields"][f] = key
            checks.append((key, *_roundtrip(d, inside, buf, meta)))
        man["states"]["diff_half_minus_full"] = st
        # What the free surface is worth, in one number, for the caption.
        a, b = got["half"]["u_mag"][inside], got["full"]["u_mag"][inside]
        surf = (pts_all[:, 2] >= -1e-9) & inside
        man["meta"]["free_surface_ratio_median"] = float(
            np.median(a[a > 0] / b[a > 0]))
        man["meta"]["free_surface_ratio_at_surface"] = float(np.median(
            got["half"]["u_mag"][surf] / got["full"]["u_mag"][surf]))

    (out / "geometry.json").write_text(json.dumps(_geometry(tris, grid)) + "\n")
    man["geometry"] = "geometry.json"

    total = sum(a["bytes"] for a in man["arrays"].values())
    print(f"  {len(man['arrays'])} arrays, {total / 1024 / 1024:.1f} MB total")
    print("  round-trip, measured against the encoder's arithmetic bound:")
    bad = []
    for key, got_err, bound in sorted(checks, key=lambda c: -c[1]):
        a = man["arrays"][key]
        over = got_err > 1.15 * bound + 1e-12
        if over:
            bad.append(key)
        print(f"    {key:34s} {a['log_max'] - a['log_min']:5.2f} dec  "
              f"{got_err * 100:5.2f} % (bound {bound * 100:5.2f} %)"
              f"{'  <-- OVER BOUND' if over else ''}")
    if bad:
        raise SystemExit(f"quantisation exceeded its own bound on {bad}: the "
                         f"encoder is wrong, not the data. Do not publish this "
                         f"payload.")

    man["quantisation"] = {k: {"measured_rel_err": g, "bound_rel_err": b}
                           for k, g, b in checks}
    (out / "manifest.json").write_text(
        json.dumps(man, indent=1, sort_keys=True) + "\n")


def _scatter(v, inside, n_all):
    out = np.zeros(n_all)
    out[inside] = v
    return out


def _roundtrip(v, inside, buf, meta) -> tuple[float, float]:
    """Decode what was written; return (measured, expected) relative error.

    The expected bound is ARITHMETIC, not borrowed: 254 levels spread over
    `d` decades of log10 put the levels `d/253` apart, so rounding to the
    nearest moves log10 by at most half that and the relative error is
    `10**(d/506) - 1`. Quoting a fixed "2.4 %" from another dataset would
    measure that dataset's dynamic range instead of this encoder -- and these
    fields are wider: a difference array is near zero wherever the two states
    agree, so it spans nearly seven decades where a state spans three.
    """
    back = decode_log_u8(buf, meta)
    good = inside & np.isfinite(v) & (v > 0.0) & np.isfinite(back)
    if not good.any():
        return 0.0, 0.0
    worst = float(np.max(np.abs(back[good] - v[good]) / np.abs(v[good])))
    decades = meta["log_max"] - meta["log_min"]
    return worst, float(10.0 ** (decades / (2.0 * (LEVELS - 1))) - 1.0)


def _geometry(tris: np.ndarray, grid: dict) -> dict:
    """Outlines the viewer draws over the slices: the box, the fault, z = 0.

    The fault goes out as its four CORNERS, not as a bounding box. A dipping
    plane's bounding box is a solid volume, and drawing that instead of the
    plane would put a wireframe box where the picture needs a sheet -- it would
    read as a second body in the half space rather than as the source.
    """
    dip = np.radians(DIP_DEG)
    hl, w = LENGTH / 2.0, WIDTH
    corners = [[-hl, 0.0, 0.0], [hl, 0.0, 0.0],
               [hl, w * np.cos(dip), -w * np.sin(dip)],
               [-hl, w * np.cos(dip), -w * np.sin(dip)]]
    return {"box": {"origin": grid["origin"], "spacing": grid["spacing"],
                    "dims": grid["dims"]},
            "fault": {"outline": [[round(float(c), 4) for c in p]
                                  for p in corners],
                      "n_triangles": int(tris.shape[0]),
                      "trace": [corners[0], corners[1]]},
            "free_surface": {"z": 0.0}}


def _git() -> str | None:
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short",
                            "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


# -------------------------------------------------------------------- main ----
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fault", help="the clickable interaction matrix (seconds)")
    v = sub.add_parser("volume", help="the 3-D field (minutes to tens of)")
    v.add_argument("--spacing", type=float, default=1.5,
                   help="grid spacing in km (default 1.5); try 3 or 4 first")
    v.add_argument("--workers", type=int, default=8)
    v.add_argument("--half-only", action="store_true",
                   help="skip the full-space state, for a quick look")
    v.add_argument("--out", type=pathlib.Path, default=OUT)
    a = ap.parse_args(argv)

    if a.cmd == "fault":
        build_fault(OUT)
    else:
        build_volume(a.out, a.spacing, a.workers, a.half_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
