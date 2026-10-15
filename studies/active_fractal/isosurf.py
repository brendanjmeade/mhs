#!/usr/bin/env python
"""Opaque stress isosurfaces on the rough crack, both kernels, for the site.

    python isosurf.py field --state rough_state.npz --scene envelope --probe
    python isosurf.py field --state rough_state.npz --scene envelope --out env.npz
    python isosurf.py mesh  --field env.npz

WHY ISOSURFACES AND NOT MORE SLICES. A collocation scheme only ever asks a
kernel for traction at an element's own centroid; a slice asks for stress on a
plane; a level set asks for it EVERYWHERE AT ONCE, and then draws the one
place where the answer equals a number. That is the sharpest form of the
question this repository is about, and it is also the composition the slice
viewer cannot reach: three opaque planes through a box hide whatever is inside
them, including the crack itself.

TWO SCENES, BECAUSE ONE GRID CANNOT HOLD BOTH SCALES. The classical kernel's
spurious lines are thin: the field goes as A/d off a shared edge with
A ~= 2e-4 GPa km measured on this mesh, so a level of 0.05 GPa is crossed 4 m
from the edge. Resolving that over a 12 km crack is ~1e9 points, hundreds of
hours. So the ENVELOPE scene is the whole crack at 4 eps, where the two kernels
agree and the picture is the shape of the field; and the DETAIL scene is a
1.6 km slab at 0.4 eps, where they do not. Each scene states its own spacing in
eps and the levels it can and cannot resolve.

THE SAME TWO-STAGE SPLIT AS roughvol.py, for the same reason: a second spawn
pool raised after the first has closed deadlocks on this machine. `field`
evaluates and saves; `mesh` contours what was saved, so re-levelling a surface
costs seconds rather than hours.

STUDY-ONLY DEPENDENCIES, like roughvol's cutde: this script needs `vtk` for
contouring and decimation. `mhs` itself declares two runtime dependencies and
treats the omissions as design, so neither belongs inside the package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import roughvol as R                                         # noqa: E402

MU, LAM = R.MU, R.LAM
SCHEMA = R.SCHEMA
FRICTION = 0.6                   # as in website/scripts/make_payload.py

#: Scalars, in the order they are stored. All three come from ONE stress
#: tensor, so carrying all three costs the reduction and nothing else -- the
#: kernel evaluation is the whole cost and it is already paid.
SCALARS = ("von_mises", "diff_stress", "dcfs")

#: The sampled regions. `envelope` is axis aligned on the crack; `detail` is a
#: slab in the crack's OWN frame, which is 203k points against the 459k of the
#: axis-aligned box that would contain the same patch -- a 30 deg dip over
#: 1.6 km rises 0.8 km, and all of that would be empty half space.
SCENES = {
    "envelope": dict(kind="box", lo=(-7.0, -7.0, -24.5), hi=(7.0, 7.0, -15.5),
                     spacing=0.2),
    "detail": dict(kind="slab", size=(1.6, 1.6, 0.6), spacing=0.02),
}

#: Observer chunk. The stress form is 216 B per observer-source pair, so 1000
#: observers against 1044 sources is 225 MB in a worker and eight of those fit;
#: the README's 4000-observer plateau (36.3 us/pair against 43.3) would need
#: 900 MB each. The curve is nearly flat here and the memory is not.
CHUNK = 1000

#: The dCFS sign anchor, and why it is not simply "all of them".
#: Slip relieves the Coulomb stress on planes parallel to the slipping element
#: -- but only in the INTERIOR of a slipping patch. A patch loads its own rim,
#: and that is physics rather than a sign error: measured on this solution, the
#: elements with positive dCFS sit a median 0.50 km from locked ground against
#: 1.00 km for the negative ones. The fraction negative against clearance from
#: the nearest locked element is 72.3 % over the whole patch, 91.8 % beyond
#: 0.75 km, 98.4 % beyond 1.00 and 100.0 % beyond 1.25 on 117 elements -- where
#: a flipped slip direction would give 0.0 %. So the gate is taken past the
#: perimeter effect, where the statement is clean and the separation is total.
ANCHOR_CLEARANCE_KM = 1.25
ANCHOR_MIN_NEGATIVE = 0.95

#: Measured rate, observer-source pairs per second, all workers together, used
#: only to project a run before it starts. From the published volume:
#: 93,636 x 1044 / 4800 s, which covers its cutde pass too and so runs slow; the
#: mollified pass alone measured 3.87e4 here. Deliberately the conservative one,
#: because this number's whole job is to stop an overnight run being launched on
#: an optimistic estimate. A field readout parallelises at about 1.45x, NOT the
#: 8.5x the README quotes for matrix ASSEMBLY -- measured again here at 1.47x,
#: 38 us/pair in one process against 25.8 on eight.
PAIRS_PER_SEC = 2.0e4

_W: dict = {}


# --------------------------------------------------------------- geometry --
def mean_frame(tris: np.ndarray, mask=None) -> tuple[np.ndarray, np.ndarray]:
    """``(centroid, frame)`` of a mean plane through some or all of the crack.

    The frame's ROWS are (strike, dip, normal) and it is read from
    ``mhs.tdcs.slip_frame`` applied to one triangle carrying the area-weighted
    mean normal, rather than restated here: that convention -- tensile is the
    upward normal, strike is ``(-n_y, n_x, 0)``, dip is their cross product
    pointing up dip -- is stated once in ``tdcs`` and a second copy of it is a
    second thing to get wrong.

    TWO DIFFERENT MEAN PLANES USE THIS, and they are not the same plane. The
    dCFS receiver is the WHOLE crack's mean plane, because the receiver is a
    declared physical question and must be identical in both scenes for the two
    to be comparable. The detail slab is oriented on the LOCAL mean plane of the
    facets it contains, because that is a sampling box and the crack is rough:
    measured on this mesh, the surface wanders 0.47 km across a 2.4 km footprint
    about the global plane and only 0.18 km about the local one, so a 0.6 km
    slab on the global normal would sample mostly empty half space and clip the
    crack at both ends.
    """
    from mhs import tdcs
    use = tris if mask is None else tris[mask]
    n = np.cross(use[:, 1] - use[:, 0], use[:, 2] - use[:, 0])
    mean = n.sum(0)
    mean /= np.linalg.norm(mean)
    # One triangle with exactly that normal: rotate a reference triangle in the
    # xy plane onto it, which is what `rotate_to_normal` already does.
    import network
    ref = np.array([[[0.0, 0.0, -20.0], [1.0, 0.0, -20.0], [0.0, 1.0, -20.0]]])
    probe = network.rotate_to_normal(ref, mean)
    return use.reshape(-1, 3).mean(0), tdcs.slip_frame(probe)[0]


def worst_edge(tris: np.ndarray, delta: np.ndarray
               ) -> tuple[np.ndarray, float, np.ndarray]:
    """Midpoint of the interior edge whose slip mismatch is largest, it, and
    the edge's two endpoints.

    The detail scene has to be centred somewhere, and "where the picture looks
    good" is not a rule. This is the same edge census the page already quotes
    -- 1518 interior edges -- scored by the quantity the claim is about, the
    slip difference across the shared edge, which is what leaves a residual
    dislocation line on it.
    """
    import network
    verts, idx = network.weld_vertices(tris, 1e-9 * R.H_KM)
    seen: dict = {}
    for t, tri in enumerate(idx):
        for p, q in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((int(tri[p]), int(tri[q]))))
            seen.setdefault(key, []).append(t)
    best, best_mis = None, -1.0
    for key, owners in seen.items():
        if len(owners) != 2:
            continue
        mis = abs(float(delta[owners[0]] - delta[owners[1]]))
        if mis > best_mis:
            best, best_mis = key, mis
    assert best is not None
    ends = np.array([verts[best[0]], verts[best[1]]])
    return 0.5 * (ends[0] + ends[1]), best_mis, ends


def grid_of(scene: str, tris: np.ndarray, delta: np.ndarray) -> dict:
    """The sampled points and everything needed to put a contour back in world.

    Returns the grid in INDEX space -- origin, spacing, dims -- plus the affine
    that maps it to world, because ``vtkFlyingEdges3D`` contours an axis-aligned
    image and the slab is not one. Extract in local coordinates, then move the
    vertices with the same affine the samples came from.
    """
    s = SCENES[scene]
    h = float(s["spacing"])
    if s["kind"] == "box":
        lo, hi = np.array(s["lo"]), np.array(s["hi"])
        dims = np.maximum(2, np.floor((hi - lo) / h).astype(int) + 1)
        rot = np.eye(3)
        origin = lo
    else:
        size = np.array(s["size"])
        dims = np.maximum(2, np.floor(size / h).astype(int) + 1)
        centre, mis, _ = worst_edge(tris, delta)
        # The facets the slab can see, which is what it must be aligned with.
        near = np.linalg.norm(tris.mean(1) - centre, axis=1) < 0.5 * float(
            np.hypot(*np.asarray(s["size"])[:2]))
        _, frame = mean_frame(tris, near)
        rot = frame.T                      # columns are (strike, dip, normal)
        origin = centre - rot @ (0.5 * (dims - 1) * h)
        off = (tris[near].reshape(-1, 3) - centre) @ frame[2]
        if off.max() > 0.5 * s["size"][2] or off.min() < -0.5 * s["size"][2]:
            raise SystemExit(
                f"the crack leaves the slab: facet offsets "
                f"[{off.min():+.3f}, {off.max():+.3f}] km against a half "
                f"thickness of {0.5 * s['size'][2]:.3f} km")
        s = dict(s, centre=centre.tolist(), mismatch_km=mis,
                 n_facets=int(near.sum()),
                 offset_km=[float(off.min()), float(off.max())])
    ax = [h * np.arange(d) for d in dims]
    K, J, I = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")   # x fastest
    local = np.column_stack([I.ravel(), J.ravel(), K.ravel()])
    pts = local @ rot.T + origin
    if pts[:, 2].max() >= 0.0:
        raise SystemExit("the grid reaches the free surface; mhs refuses an "
                         "observer at z >= 0")
    return dict(pts=pts, dims=dims, spacing=h, origin=origin, rot=rot,
                scene=scene, meta=s)


# ---------------------------------------------------------------- scalars --
def receiver(tris: np.ndarray, slip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(n_hat, s_hat)`` of the dCFS receiver: the crack's own mean plane.

    A Coulomb stress change is not an invariant -- it is a question asked of one
    plane -- so the plane is declared here and reported in the manifest. The
    question this one asks is the natural one for a rough fault: would a patch
    PARALLEL to the crack's mean plane, slipping the way this crack slips, be
    brought closer to failure? So the normal is the mean normal and the shear
    direction is the mean slip direction projected into that plane.
    """
    _, frame = mean_frame(tris)
    n_hat = frame[2]
    s = slip.sum(0)
    s = s - (s @ n_hat) * n_hat
    return n_hat, s / np.linalg.norm(s)


def scalars(sig: np.ndarray, n_hat: np.ndarray, s_hat: np.ndarray) -> np.ndarray:
    """``(3, n)``: von Mises, sigma_1 - sigma_3, and dCFS on the receiver.

    SIGN CONVENTION, stated once. ``sigma . n`` is the traction on the plane
    with ``n`` as its outward normal and mhs does not orient it for you, so
    ``t @ n_hat`` is TENSION POSITIVE. dCFS is then
    ``tau_s + mu * sigma_n`` with that tension-positive normal, which is the
    form ``make_payload.build_fault`` already uses and means positive =
    unclamped or sheared toward failure. ``cmd_field`` anchors the sign on the
    crack's own faces rather than trusting this comment.

    NON-FINITE INPUT IS EXPECTED, from the classical kernel only. A grid point
    can land close enough to an element edge that the Nikkhoo-Walter log terms
    overflow, and the tensor arrives as inf or nan; ``eigvalsh`` then raises
    ``LinAlgError`` for the whole chunk rather than for the point. Those points
    are carried as nan and counted by the caller: the classical field really is
    unbounded there, and inventing a finite number for it is the one thing this
    page must not do.
    """
    out = np.full((3, sig.shape[0]), np.nan)
    ok = np.isfinite(sig).all(axis=(1, 2))
    if not ok.any():
        return out
    sg = sig[ok]
    w = np.linalg.eigvalsh(sg)
    t = sg @ n_hat
    out[0, ok] = R.invariants(sg)
    out[1, ok] = w[..., 2] - w[..., 0]
    out[2, ok] = t @ s_hat + FRICTION * (t @ n_hat)
    return out


# ---------------------------------------------------------- the two passes --
def _init(tris, slip, eps, n_hat, s_hat):
    import mhs
    _W.update(tris=tris, slip=slip, eps=float(eps), n_hat=n_hat, s_hat=s_hat,
              mat=mhs.Material(mu=MU, lam=LAM))


def _chunk(pts):
    import mhs
    from mhs.parallel import pinned_blas
    with pinned_blas(1):
        S = mhs.stress_matrix(pts, _W["tris"], _W["mat"], _W["eps"])
        sig = np.einsum("omnsk,sk->omn", S, _W["slip"])
        return scalars(sig, _W["n_hat"], _W["s_hat"])


def mollified(pts, tris, slip, eps, n_hat, s_hat, workers) -> np.ndarray:
    import concurrent.futures as cf
    import multiprocessing as mp
    npt = pts.shape[0]
    parts = [pts[i:i + CHUNK] for i in range(0, npt, CHUNK)]
    out = np.empty((3, npt))
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=workers,
                                mp_context=mp.get_context("spawn"),
                                initializer=_init,
                                initargs=(tris, slip, eps, n_hat, s_hat)) as ex:
        done = 0
        for i, got in enumerate(ex.map(_chunk, parts, chunksize=1)):
            out[:, i * CHUNK:i * CHUNK + got.shape[1]] = got
            done += got.shape[1]
            if i % 10 == 0 or done == npt:
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                print(f"    mollified {done:8d}/{npt}  {el / 60:6.1f} min  "
                      f"eta {(npt - done) / max(rate, 1e-9) / 60:6.1f} min  "
                      f"{el * 1e6 / max(done * tris.shape[0], 1):5.1f} us/pair",
                      flush=True)
    return out


def classical(pts, tris, slip, n_hat, s_hat, chunk=4000) -> np.ndarray:
    """The same three scalars from cutde, with no regularisation.

    Structurally ``roughvol.classical``, which carries the warning that matters:
    the slip must go through ``tdcs.from_cartesian`` because cutde wants it in
    its own local triad, and contracting a Cartesian slip against cutde's
    strain matrix silently compares two different bases.
    """
    import cutde.halfspace as hs
    import mhs
    from mhs import tdcs
    nu = mhs.Material(mu=MU, lam=LAM).nu_if_you_must
    loc = tdcs.from_cartesian(tris, slip)
    V6 = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    out = np.empty((3, pts.shape[0]))
    t0 = time.time()
    for i in range(0, pts.shape[0], chunk):
        p = pts[i:i + chunk]
        E = np.asarray(hs.strain_matrix(p.copy(), tris.copy(), nu))
        Es = np.einsum("ovsk,sk->ov", E, loc)
        e = np.zeros((p.shape[0], 3, 3))
        for k, (x, y) in enumerate(V6):
            e[:, x, y] = Es[:, k]
            e[:, y, x] = Es[:, k]
        tr = e[:, 0, 0] + e[:, 1, 1] + e[:, 2, 2]
        sg = 2.0 * MU * e
        for x in range(3):
            sg[:, x, x] += LAM * tr
        out[:, i:i + p.shape[0]] = scalars(sg, n_hat, s_hat)
    print(f"    classical {pts.shape[0]} points in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return out


# ------------------------------------------------------------------ stage 1 --
def cmd_field(a) -> int:
    z = np.load(a.state)
    tris, slip, eps = z["tris"], z["slip"], float(z["eps"])
    r, _ = R.build()
    if not np.array_equal(r.tris, tris):
        raise SystemExit("the state's mesh is not the one roughvol.build() "
                         "makes now; re-run `roughvol.py solve` first")
    g = grid_of(a.scene, tris, z["delta"])
    pts, dims = g["pts"], g["dims"]
    npt, nsrc = pts.shape[0], tris.shape[0]
    print(f"  scene {a.scene}: grid {[int(d) for d in dims]} at "
          f"{g['spacing']} km = {g['spacing'] / eps:.2f} eps, {npt} points "
          f"x {nsrc} sources = {npt * nsrc / 1e6:.1f} M pairs")
    print(f"  estimate {npt * nsrc / PAIRS_PER_SEC / 3600:.2f} h at the "
          f"measured {1e6 / PAIRS_PER_SEC:.0f} us/pair")
    if "centre" in g["meta"]:
        c, m = g["meta"]["centre"], g["meta"]["mismatch_km"]
        print(f"  slab centred on the worst interior edge at "
              f"{np.round(c, 3).tolist()}, slip mismatch {m * 1e3:.3f} m")

    n_hat, s_hat = receiver(tris, slip)
    print(f"  dCFS receiver: normal {np.round(n_hat, 4).tolist()}, "
          f"slip direction {np.round(s_hat, 4).tolist()}, friction {FRICTION}")

    if a.probe:
        return probe(pts, tris, slip, eps, n_hat, s_hat, npt, nsrc)

    # The sign anchor, before the hours rather than after: a patch unloads
    # itself, so dCFS on planes parallel to the crack just off its own faces
    # must be mostly NEGATIVE. This is the same kind of check that stops
    # `make_payload.build_fault` publishing a flipped CFS.
    anchor(tris, slip, eps, n_hat, s_hat, z["active"])

    def save(**extra):
        np.savez_compressed(
            a.out, dims=dims, origin=g["origin"], rot=g["rot"],
            spacing=g["spacing"], tris=tris, slip=slip, eps=eps,
            scene=a.scene, scalars=np.array(SCALARS), n_hat=n_hat,
            s_hat=s_hat, friction=FRICTION,
            centre=np.array(g["meta"].get("centre", [np.nan] * 3)), **extra)
        print(f"  saved {a.out}" + (" (mollified only, classical to come)"
                                    if "cutde" not in extra else ""))

    # The mollified pass is hours and the classical one is minutes, so the
    # hours are banked BEFORE the minutes are attempted and are reloaded if
    # they are already on disk for this exact grid. Written because the
    # classical pass once raised inside eigvalsh after a 2.2 h mollified pass
    # that had not been saved, and the hours were simply gone.
    done = None
    if pathlib.Path(a.out).exists():
        with np.load(a.out, allow_pickle=False) as z0:
            if (str(z0["scene"]) == a.scene and "mhs" in z0
                    and np.array_equal(z0["dims"], dims)
                    and float(z0["spacing"]) == float(g["spacing"])
                    and z0["mhs"].shape == (len(SCALARS), npt)):
                done = z0["mhs"]
                print(f"  reusing the mollified pass already in {a.out}")
    got = {"mollified": done if done is not None else
           mollified(pts, tris, slip, eps, n_hat, s_hat, a.workers)}
    if done is None:
        save(mhs=got["mollified"])
    got["classical"] = classical(pts, tris, slip, n_hat, s_hat)
    save(mhs=got["mollified"], cutde=got["classical"])
    for name, arr in got.items():
        bad = int((~np.isfinite(arr)).any(axis=0).sum())
        if bad:
            print(f"    {name}: {bad} of {npt} points ({100 * bad / npt:.3f} %)"
                  f" are non-finite and are carried as nan")
    for i, name in enumerate(SCALARS):
        m, c = got["mollified"][i], got["classical"][i]
        print(f"    {name:12s} mollified [{np.nanmin(m):+.4e}, "
              f"{np.nanmax(m):+.4e}]  classical [{np.nanmin(c):+.4e}, "
              f"{np.nanmax(c):+.4e}]")
    return 0


def probe(pts, tris, slip, eps, n_hat, s_hat, npt, nsrc) -> int:
    """One chunk, timed, before anything long is launched.

    The sibling repository launched a nine-hour grid on a one-hour estimate
    because the baseline had been measured with multi-threaded BLAS while the
    pool pins its children to one thread. A measurement costs a minute.
    """
    _init(tris, slip, eps, n_hat, s_hat)
    sub = pts[:CHUNK]
    t0 = time.time()
    _chunk(sub[:50])                                   # JIT, not measured
    warm = time.time() - t0
    t0 = time.time()
    _chunk(sub)
    el = time.time() - t0
    us = el * 1e6 / (sub.shape[0] * nsrc)
    serial = npt * nsrc * us / 1e6 / 3600
    print(f"  probe: {warm:.1f} s warming, then {sub.shape[0]} observers in "
          f"{el:.1f} s = {us:.1f} us/pair in ONE process")
    print(f"  -> {serial:.2f} h serial; at the measured 1.45x for a field "
          f"readout, {serial / 1.45:.2f} h on this machine")
    print(f"  -> the {1e6 / PAIRS_PER_SEC:.0f} us/pair baseline predicts "
          f"{npt * nsrc / PAIRS_PER_SEC / 3600:.2f} h")
    return 0


def anchor(tris, slip, eps, n_hat, s_hat, active) -> None:
    """Refuse to run if the Coulomb sign convention is inverted.

    THE TEST IS ON EACH ELEMENT'S OWN PLANE, not on the figure's receiver. What
    needs anchoring is the assembly in :func:`scalars` -- the traction
    contraction and the sign of the friction term -- and the one statement
    about it that cannot be argued is that a slipping patch relieves itself.
    The figure's receiver is a declared choice and is reported here rather than
    gated, because against a plane 14 deg from the local facets the sign is a
    fact about the roughness, not about the convention.

    See :data:`ANCHOR_CLEARANCE_KM` for why the gate is taken past the rim.
    """
    from mhs import tdcs
    from scipy.spatial import cKDTree
    import mhs
    idx = np.flatnonzero(active)
    own_n = tdcs.unit_normals(tris)
    own_s = slip / np.maximum(np.linalg.norm(slip, axis=1), 1e-30)[:, None]
    pts = tris[idx].mean(1) + 2.0 * eps * own_n[idx]
    mat = mhs.Material(mu=MU, lam=LAM)
    sig = np.einsum("omnsk,sk->omn", mhs.stress_matrix(pts, tris, mat, eps), slip)
    t = np.einsum("omn,on->om", sig, own_n[idx])
    d = (np.einsum("om,om->o", t, own_s[idx])
         + FRICTION * np.einsum("om,on->o", t, own_n[idx]))

    locked = tris[~active].mean(1)
    clear = (cKDTree(locked).query(tris[idx].mean(1))[0] if locked.size
             else np.full(idx.size, np.inf))
    deep = clear > ANCHOR_CLEARANCE_KM
    glob = scalars(sig, n_hat, s_hat)[2]
    print(f"  sign anchor, dCFS on each element's own plane 2 eps off its own "
          f"face:")
    print(f"    whole patch      {idx.size:4d} elements, "
          f"{(d < 0).mean():6.1%} negative, median {np.median(d):+.3e} GPa")
    print(f"    clearance > {ANCHOR_CLEARANCE_KM} km {int(deep.sum()):4d} "
          f"elements, {(d[deep] < 0).mean():6.1%} negative, median "
          f"{np.median(d[deep]):+.3e} GPa")
    print(f"    on the figure's receiver (reported, not gated): "
          f"{(glob < 0).mean():6.1%} negative, median {np.median(glob):+.3e} GPa")
    if deep.sum() < 20:
        raise SystemExit(f"only {int(deep.sum())} elements sit more than "
                         f"{ANCHOR_CLEARANCE_KM} km inside the slipping patch, "
                         f"so the anchor has nothing clean to stand on")
    if (d[deep] < 0).mean() < ANCHOR_MIN_NEGATIVE:
        raise SystemExit(
            f"inside a slipping patch, slip must RELIEVE the Coulomb stress on "
            f"planes parallel to the slipping element; it came back negative "
            f"on only {(d[deep] < 0).mean():.1%} of the {int(deep.sum())} "
            f"interior elements, against a {ANCHOR_MIN_NEGATIVE:.0%} floor. "
            f"The slip direction or the friction sign is inverted. Do not "
            f"publish a Coulomb figure past this.")


# ------------------------------------------------------------------ stage 2 --
#: Triangles per surface after decimation. The site's largest payload today is
#: 262 kB; an indexed float32 surface at this budget is about 1.4 MB, fetched
#: only when its own state is selected.
MAX_TRIANGLES = 60_000

#: Level rule, stated once: the percentiles of the MOLLIFIED scalar inside the
#: box. Both kernels are then contoured at the same number, which is the only
#: way the comparison means anything, and the number is a property of the field
#: rather than of what looked good.
LEVEL_PERCENTILES = (90.0, 99.0)

#: Art direction, carried in the GLB so Blender opens the same materials. Cool
#: for the broad level, warm for the tight one; red and blue for the signed
#: lobes, which is the convention a Coulomb figure is read with.
PALETTE = {
    "von_mises": [(0.24, 0.36, 0.58), (0.93, 0.66, 0.26)],
    "diff_stress": [(0.26, 0.42, 0.52), (0.90, 0.72, 0.36)],
    "dcfs": [(0.74, 0.24, 0.22), (0.22, 0.40, 0.72)],
}


def _image(values, dims, spacing):
    """A vtkImageData in INDEX space: the contour is transformed afterwards."""
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.util import numpy_support
    img = vtkImageData()
    img.SetDimensions(int(dims[0]), int(dims[1]), int(dims[2]))
    img.SetSpacing(spacing, spacing, spacing)
    img.SetOrigin(0.0, 0.0, 0.0)
    arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(values, np.float64),
                                     deep=True)
    arr.SetName("f")
    img.GetPointData().SetScalars(arr)
    return img


def _polydata_arrays(pd):
    """``(points, triangles, normals)`` from a vtkPolyData of triangles."""
    from vtkmodules.util import numpy_support
    pts = numpy_support.vtk_to_numpy(pd.GetPoints().GetData())
    polys = pd.GetPolys()
    off = numpy_support.vtk_to_numpy(polys.GetOffsetsArray())
    conn = numpy_support.vtk_to_numpy(polys.GetConnectivityArray())
    if off.size and not np.array_equal(np.diff(off), np.full(off.size - 1, 3)):
        raise SystemExit("the contour is not all triangles; decimation and the "
                         "GLB writer both assume it is")
    nrm = pd.GetPointData().GetNormals()
    return (pts.astype(np.float64), conn.reshape(-1, 3).astype(np.uint32),
            numpy_support.vtk_to_numpy(nrm).astype(np.float64) if nrm else None)


def contour(values, dims, spacing, level, max_tris=MAX_TRIANGLES):
    """One level set, decimated to budget, with the cost of that measured.

    NO SMOOTHING. A marching-cubes surface of a field this smooth needs none,
    and the detail scene exists to show structures a few cells across -- a
    smoothing pass is exactly what would quietly remove them and manufacture
    the agreement the page is testing. Decimation is measured against the
    undecimated contour for the same reason.
    """
    from vtkmodules.vtkFiltersCore import (vtkCleanPolyData, vtkFlyingEdges3D,
                                           vtkPolyDataNormals,
                                           vtkQuadricDecimation)
    from vtkmodules.vtkFiltersGeneral import vtkDistancePolyDataFilter
    fe = vtkFlyingEdges3D()
    fe.SetInputData(_image(values, dims, spacing))
    fe.SetValue(0, float(level))
    fe.ComputeNormalsOff()
    fe.ComputeGradientsOff()
    fe.Update()
    # CLEANING IS NOT COSMETIC. Where the level passes exactly through a grid
    # node, flying edges emits coincident vertices and zero-area triangles:
    # measured on an analytic ellipsoid, 84 of 1064 vertices came back with 18
    # coincident copies each and a ZERO normal, which in a lit scene is a black
    # speckle and in a mesh statistic is a lie.
    clean = vtkCleanPolyData()
    clean.SetInputConnection(fe.GetOutputPort())
    clean.PointMergingOn()
    clean.ConvertPolysToLinesOff()
    clean.ConvertLinesToPointsOff()
    clean.ConvertStripsToPolysOff()
    clean.Update()
    raw = clean.GetOutput()
    n_raw = raw.GetNumberOfPolys()
    if n_raw == 0:
        return None
    pd, dev = raw, 0.0
    if n_raw > max_tris:
        dec = vtkQuadricDecimation()
        dec.SetInputData(raw)
        dec.SetTargetReduction(1.0 - max_tris / n_raw)
        dec.VolumePreservationOn()
        dec.Update()
        pd = dec.GetOutput()
        dist = vtkDistancePolyDataFilter()
        dist.SetInputData(0, pd)
        dist.SetInputData(1, raw)
        dist.SignedDistanceOff()
        dist.Update()
        from vtkmodules.util import numpy_support
        dev = float(np.abs(numpy_support.vtk_to_numpy(
            dist.GetOutput().GetPointData().GetArray("Distance"))).max())
    nf = vtkPolyDataNormals()
    nf.SetInputData(pd)
    nf.SetFeatureAngle(45.0)
    nf.SplittingOn()
    nf.ConsistencyOn()
    nf.Update()
    out = nf.GetOutput()
    pts, tri, nrm = _polydata_arrays(out)
    raw_pts, _, _ = _polydata_arrays(raw)
    if nrm is None or np.abs(np.linalg.norm(nrm, axis=1) - 1.0).max() > 1e-4:
        raise SystemExit("the contour has degenerate normals, so cleaning did "
                         "not remove every zero-area triangle; a lit scene "
                         "would show them as black speckle")
    return dict(points=pts, triangles=tri, normals=nrm, raw_points=raw_pts,
                n_raw=int(n_raw), deviation_km=dev, polydata=out)


def surface_distance(a, b) -> tuple[float, float]:
    """``(median, max)`` distance from every vertex of ``a`` to the surface ``b``.

    This is the number the comparison rests on, and it is a DISTANCE rather than
    a difference of fields: two level sets of the same scalar at the same level
    are the same object or they are not, and how far apart they are is in
    kilometres. Point to SURFACE, not point to vertex, so a denser mesh does not
    read as a closer one.
    """
    from vtkmodules.util import numpy_support
    from vtkmodules.vtkFiltersGeneral import vtkDistancePolyDataFilter
    d = vtkDistancePolyDataFilter()
    d.SetInputData(0, a)
    d.SetInputData(1, b)
    d.SignedDistanceOff()
    d.Update()
    v = np.abs(numpy_support.vtk_to_numpy(
        d.GetOutput().GetPointData().GetArray("Distance")))
    return float(np.median(v)), float(v.max())


def surface_extent(pd) -> tuple[float, float]:
    """``(area_km2, n_pieces)`` of a contour.

    Area is the kernel discriminator the distance cannot be: two level sets at
    the SAME level enclose the same region or they do not, and the classical
    field's edge tail inflates its own, monotonically with how far into the tail
    the level sits. The piece count is reported because it describes what the
    viewer sees -- a tail level set of a rough crack is dozens of lobes, not one
    shell -- and expressly NOT as a discriminator: measured, the mollified
    surface carries MORE pieces than the classical one at the same level
    (38 against 34 on envelope von Mises p99), because fragmentation here is set
    by the roughness of the crack and not by the kernel.
    """
    from vtkmodules.vtkFiltersCore import (vtkCleanPolyData, vtkMassProperties,
                                            vtkPolyDataConnectivityFilter,
                                            vtkTriangleFilter)
    tf = vtkTriangleFilter()
    tf.SetInputData(pd)
    tf.Update()
    mp = vtkMassProperties()
    mp.SetInputData(tf.GetOutput())
    mp.Update()
    # Weld first. The shipped surface has been through vtkPolyDataNormals with
    # splitting on, which DUPLICATES vertices along every feature seam: that
    # disconnects the mesh topologically without disconnecting it in space, and
    # counting pieces on it counts seams instead (714 against the true 38 on
    # envelope von Mises p99). Area is unaffected -- the triangles are the same.
    weld = vtkCleanPolyData()
    weld.SetInputData(pd)
    weld.PointMergingOn()
    weld.SetTolerance(0.0)
    weld.Update()
    cf = vtkPolyDataConnectivityFilter()
    cf.SetInputData(weld.GetOutput())
    cf.SetExtractionModeToAllRegions()
    cf.ColorRegionsOn()
    cf.Update()
    return float(mp.GetSurfaceArea()), int(cf.GetNumberOfExtractedRegions())


def tube_radius(pts: np.ndarray, ends: np.ndarray, nbins: int = 10
                ) -> tuple[float, float]:
    """``(radius, closest)``: how far a surface stands off an edge, in km.

    The quantity to compare with ``r = A / L`` is the tube's inner radius along
    the MIDDLE of the edge, so the edge is cut into bins over its central half,
    the nearest vertex in each bin is taken, and the median of those is the
    radius. A plain minimum over every vertex is not that number and is biased
    low: it is the smallest standoff anywhere, which falls at the edge's
    endpoints where several edges meet and the field is larger than one line's.
    Both are returned, because the minimum is still the right thing to report
    for a surface that has no tube at all.

    For the classical kernel the radius tests the prediction. For the mollified
    kernel it is a control: a smooth sheet has no reason to stand off that line
    at any particular distance.
    """
    a, b = ends[0], ends[1]
    d = b - a
    tt = ((pts - a) @ d) / float(d @ d)
    dist = np.linalg.norm(pts - (a + np.clip(tt, 0.0, 1.0)[:, None] * d), axis=1)
    mid = (tt >= 0.25) & (tt <= 0.75)
    if mid.sum() < nbins:
        return float("nan"), float(dist.min())
    which = np.clip(((tt[mid] - 0.25) / 0.5 * nbins).astype(int), 0, nbins - 1)
    per = [dist[mid][which == k].min() for k in range(nbins)
           if (which == k).any()]
    return float(np.median(per)), float(dist.min())


def fill_nonfinite(values, dims) -> tuple[np.ndarray, int, float]:
    """Give an unbounded grid node the most extreme value next to it.

    The classical kernel returns inf or nan at a node that lands close enough
    to an element edge for the Nikkhoo-Walter log terms to overflow -- one node
    in 203,391 on the detail grid, and the field there is genuinely unbounded,
    not merely unknown. Marching cubes still has to put that node on a side of
    the level, so it is given the largest-magnitude FINITE value among its six
    face neighbours, sign kept. Because the field grows monotonically toward
    the edge, that is the side the node is really on for any level this page
    contours, and it invents no number: every value in the grid was computed
    somewhere in it. Returns the filled grid, how many nodes were filled, and
    the largest magnitude used, both of which are published.

    Neighbours are read from the ORIGINAL grid, never from the fill in
    progress. Filling in place cascades: the rim of a non-finite blob is filled
    first and its interior then finds freshly invented neighbours, so a whole
    unresolved neighbourhood is flood-filled from its edge and the refusal
    below never fires -- and the answer depends on the order the nodes come
    out of ``nonzero``. Caught by the selfcheck, which walls a node in.
    """
    v = np.asarray(values, float).reshape(int(dims[2]), int(dims[1]),
                                          int(dims[0])).copy()
    bad = ~np.isfinite(v)
    n = int(bad.sum())
    if n == 0:
        return v.ravel(), 0, 0.0
    used = 0.0
    orig = v.copy()           # read neighbours from HERE, never from the fill
    for z0, y0, x0 in zip(*np.nonzero(bad)):
        cand = []
        for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                           (0, 0, 1), (0, 0, -1)):
            zz, yy, xx = z0 + dz, y0 + dy, x0 + dx
            if (0 <= zz < v.shape[0] and 0 <= yy < v.shape[1]
                    and 0 <= xx < v.shape[2]
                    and np.isfinite(orig[zz, yy, xx])):
                cand.append(orig[zz, yy, xx])
        if not cand:
            raise SystemExit("a non-finite node has no finite face neighbour, "
                             "so the classical field is unresolved over a whole "
                             "neighbourhood and this grid cannot be contoured")
        pick = max(cand, key=abs)
        v[z0, y0, x0] = pick
        used = max(used, abs(pick))
    return v.ravel(), n, used


def level_error(raw_points, values, dims, spacing, level):
    """Worst departure of the UNDECIMATED contour from the level it claims.

    Trilinear interpolation of the sampled field at the contour's own vertices.
    Marching cubes interpolates linearly along cell edges, so this is not a
    tautology: it is the same number the renderer's smooth sampling would see,
    and it catches an index-order or spacing mistake at once.
    """
    from scipy.ndimage import map_coordinates
    grid = np.asarray(values).reshape(int(dims[2]), int(dims[1]), int(dims[0]))
    c = (raw_points / spacing).T[::-1]          # (z, y, x) fractional indices
    got = map_coordinates(grid, c, order=1, mode="nearest")
    scale = max(abs(float(level)), 1e-30)
    return float(np.abs(got - level).max() / scale)


# --------------------------------------------------------------------- glb --
def write_glb(path: pathlib.Path, parts: list[dict]) -> int:
    """Binary glTF, hand written, because nothing here can write one.

    Neither ``pygltflib`` nor ``trimesh`` is installed, and the container is a
    header plus two chunks -- the same trade the sibling repository made when it
    hand-rolled VTK ImageData rather than take a 100 MB wheel for one writer.
    One format then serves both consumers: the page loads these with three.js's
    own GLTFLoader, and Blender opens the identical file.

    NORMALS POINT ALONG -grad f, which is vtk's own contour convention and is
    measured rather than assumed (on an analytic ellipsoid, within 7.6 deg of
    it at every vertex, mean 0.999). For a stress envelope the enclosed region
    is the high-stress one, so that is the outward normal; the materials are
    double sided either way.

    COORDINATES ARE THE MODEL'S OWN: kilometres, z up, depth negative, no
    transform node. glTF declares y up, so an importer that honours that will
    lay the scene on its side and a reader has to rotate it; the alternative is
    baking a rotation into every vertex, which would make the numbers in the
    file disagree with the numbers in the manifest. The manifest says
    ``axis_up: z`` for that reason.
    """
    buf, views, accs, meshes, nodes, mats = bytearray(), [], [], [], [], []

    def put(arr, target, kind, comp):
        while len(buf) % 4:
            buf.append(0)
        off = len(buf)
        data = np.ascontiguousarray(arr)
        buf.extend(data.tobytes())
        views.append({"buffer": 0, "byteOffset": off,
                      "byteLength": len(buf) - off, "target": target})
        a = {"bufferView": len(views) - 1, "componentType": comp,
             "count": int(data.shape[0]) * (1 if data.ndim == 1 else 1),
             "type": kind}
        if kind == "VEC3" and comp == 5126:
            a["min"] = [float(v) for v in data.min(0)]
            a["max"] = [float(v) for v in data.max(0)]
        accs.append(a)
        return len(accs) - 1

    for part in parts:
        pos = np.asarray(part["points"], np.float32)
        attrs = {"POSITION": put(pos, 34962, "VEC3", 5126)}
        if part.get("normals") is not None:
            attrs["NORMAL"] = put(np.asarray(part["normals"], np.float32),
                                  34962, "VEC3", 5126)
        prim = {"attributes": attrs, "mode": 4, "material": len(mats)}
        if part.get("triangles") is not None:
            idx = np.asarray(part["triangles"], np.uint32).reshape(-1)
            a = put(idx, 34963, "SCALAR", 5125)
            accs[a]["count"] = int(idx.size)
            prim["indices"] = a
        r, g, b = part.get("color", (0.8, 0.8, 0.8))
        mats.append({"name": part["name"] + "_mat", "doubleSided": True,
                     "pbrMetallicRoughness": {
                         "baseColorFactor": [r, g, b, 1.0],
                         "metallicFactor": float(part.get("metallic", 0.0)),
                         "roughnessFactor": float(part.get("roughness", 0.45))}})
        meshes.append({"name": part["name"], "primitives": [prim]})
        nodes.append({"mesh": len(meshes) - 1, "name": part["name"]})

    js = {"asset": {"version": "2.0", "generator": "mhs isosurf.py"},
          "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}],
          "nodes": nodes, "meshes": meshes, "materials": mats,
          "accessors": accs, "bufferViews": views,
          "buffers": [{"byteLength": len(buf)}]}
    jb = bytearray(json.dumps(js, separators=(",", ":")).encode())
    while len(jb) % 4:
        jb.extend(b" ")
    while len(buf) % 4:
        buf.append(0)
    out = bytearray()
    out.extend(b"glTF")
    out.extend(struct.pack("<II", 2, 12 + 8 + len(jb) + 8 + len(buf)))
    out.extend(struct.pack("<I", len(jb)) + b"JSON" + jb)
    out.extend(struct.pack("<I", len(buf)) + b"BIN\x00" + buf)
    path.write_bytes(out)
    return len(out)


def read_glb(path: pathlib.Path) -> dict:
    """Parse back what was written, so the round trip is checked not assumed."""
    raw = path.read_bytes()
    if raw[:4] != b"glTF" or struct.unpack("<I", raw[4:8])[0] != 2:
        raise SystemExit(f"{path} is not a glTF 2.0 binary")
    total = struct.unpack("<I", raw[8:12])[0]
    if total != len(raw):
        raise SystemExit(f"{path} declares {total} bytes and is {len(raw)}")
    off, chunks = 12, {}
    while off < len(raw):
        ln, tag = struct.unpack("<I", raw[off:off + 4])[0], raw[off + 4:off + 8]
        chunks[tag.rstrip(b"\x00")] = raw[off + 8:off + 8 + ln]
        off += 8 + ln
    js = json.loads(chunks[b"JSON"])
    bin_ = chunks[b"BIN"]
    if js["buffers"][0]["byteLength"] > len(bin_):
        raise SystemExit(f"{path}'s buffer is shorter than it declares")
    out = []
    for mesh in js["meshes"]:
        prim = mesh["primitives"][0]
        got = {"name": mesh["name"]}
        for key, acc_i in prim["attributes"].items():
            a = js["accessors"][acc_i]
            v = js["bufferViews"][a["bufferView"]]
            got[key] = np.frombuffer(bin_, np.float32, a["count"] * 3,
                                     v["byteOffset"]).reshape(-1, 3)
        if "indices" in prim:
            a = js["accessors"][prim["indices"]]
            v = js["bufferViews"][a["bufferView"]]
            got["INDICES"] = np.frombuffer(bin_, np.uint32, a["count"],
                                           v["byteOffset"]).reshape(-1, 3)
        out.append(got)
    return {"meshes": out, "json": js}


def crack_part(tris, name, color=(0.16, 0.17, 0.19)):
    """The crack itself, flat shaded: non-indexed, one normal per facet."""
    pts = tris.reshape(-1, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return dict(name=name, points=pts, normals=np.repeat(n, 3, axis=0),
                triangles=None, color=color, roughness=0.85)


def cmd_mesh(a) -> int:
    z = np.load(a.field, allow_pickle=False)
    dims, spacing = z["dims"], float(z["spacing"])
    origin, rot = z["origin"], z["rot"]
    scene = str(z["scene"])
    tris, eps = z["tris"], float(z["eps"])
    out = pathlib.Path(a.out)
    (out / "iso").mkdir(parents=True, exist_ok=True)

    surfaces, checks, measured = [], [], {}
    # The edge the detail slab is centred on, in world coordinates: the tube
    # prediction below is about this line, so the measurement must be too.
    worst_ends = (worst_edge(tris, np.linalg.norm(z["slip"], axis=1))[2]
                  if scene == "detail" else None)
    for si, name in enumerate(SCALARS):
        mhs_v, nf_m, _ = fill_nonfinite(z["mhs"][si], dims)
        cut_v, nf_c, nf_mag = fill_nonfinite(z["cutde"][si], dims)
        if nf_m or nf_c:
            print(f"  {name}: filled {nf_m} mollified and {nf_c} classical "
                  f"unbounded nodes, largest magnitude used "
                  f"{nf_mag:.4e} GPa")
        if name == "dcfs":
            top = float(np.percentile(np.abs(mhs_v), LEVEL_PERCENTILES[1]))
            levels = [("+", +top), ("-", -top)]
        else:
            levels = [(f"p{int(p)}", float(np.percentile(mhs_v, p)))
                      for p in LEVEL_PERCENTILES]
        for li, (tag, level) in enumerate(levels):
            both = {}
            for kern, vals in (("mollified", mhs_v), ("classical", cut_v)):
                c = contour(vals, dims, spacing, level)
                key = f"{scene}.{kern}.{name}.{tag}"
                if c is None:
                    print(f"  {key:44s} EMPTY at {level:+.4e} GPa")
                    continue
                both[kern] = c
                err = level_error(c["raw_points"], vals, dims, spacing, level)
                world = c["points"] @ rot.T + origin
                nrm = c["normals"] @ rot.T if c["normals"] is not None else None
                fn = f"iso/{key}.glb"
                nb = write_glb(out / fn, [dict(
                    name=key, points=world, normals=nrm,
                    triangles=c["triangles"], color=PALETTE[name][li % 2],
                    roughness=0.45)])
                sha = hashlib.sha256((out / fn).read_bytes()).hexdigest()[:16]
                area, pieces = surface_extent(c["polydata"])
                surfaces.append({
                    "scene": scene, "kernel": kern, "scalar": name,
                    "level_tag": tag, "level_GPa": level,
                    "level_MPa": level * 1e3, "file": fn,
                    "triangles": int(c["triangles"].shape[0]),
                    "vertices": int(c["points"].shape[0]),
                    "triangles_before_decimation": c["n_raw"],
                    "decimation_deviation_km": c["deviation_km"],
                    "level_rel_error": err, "bytes": nb, "sha256": sha,
                    "area_km2": area, "pieces": pieces,
                    "unbounded_nodes_filled": nf_c if kern == "classical"
                    else nf_m})
                checks.append((key, err, c["deviation_km"], spacing))
                if scene == "detail":
                    measured[(kern, name, tag)] = tube_radius(
                        world, worst_ends)
                print(f"  {key:44s} {level:+.4e} GPa  "
                      f"{c['n_raw']:7d} -> {c['triangles'].shape[0]:6d} tri  "
                      f"level err {err * 100:5.2f} %  "
                      f"decim {c['deviation_km'] * 1e3:6.1f} m  "
                      f"{area:8.2f} km2 in {pieces:4d}  "
                      f"{nb / 1024:7.0f} kB")
            # How far apart the two kernels put the SAME level: the comparison
            # this page exists for, in metres rather than in adjectives.
            if len(both) == 2:
                med, mx = surface_distance(both["classical"]["polydata"],
                                           both["mollified"]["polydata"])
                for srec in surfaces[-2:]:
                    srec["distance_to_other_kernel_km"] = {"median": med,
                                                           "max": mx}
                pair = {r["kernel"]: r for r in surfaces[-2:]}
                ratio = (pair["classical"]["area_km2"]
                         / pair["mollified"]["area_km2"]
                         if pair["mollified"]["area_km2"] > 0 else float("nan"))
                for srec in surfaces[-2:]:
                    srec["classical_area_excess"] = ratio - 1.0
                print(f"  {'':44s} classical to mollified: median "
                      f"{med * 1e3:7.1f} m, max {mx * 1e3:8.1f} m  "
                      f"({med / spacing:.3f} and {mx / spacing:.2f} cells), "
                      f"area {100 * (ratio - 1.0):+.1f} %")

    if scene == "detail":
        # What the detail scene can and cannot resolve, said in cells before
        # anyone looks at it. A is the 1/d coefficient of the classical field
        # off a shared edge, measured on this mesh at one edge whose slip
        # mismatch is 0.0231 m (2.0e-4 GPa km) and scaled linearly to the worst
        # edge in this slab; r = A / L is where that field reaches the level.
        delta = np.linalg.norm(z["slip"], axis=1)
        _, mis, ends = worst_edge(tris, delta)
        A = 2.0e-4 * mis / 0.0231e-3
        print(f"  predicted classical tube radius, worst edge in the slab "
              f"(mismatch {mis * 1e3:.3f} m):")
        world_ends = ends
        for srec in surfaces:
            if srec["kernel"] != "classical":
                continue
            r = A / abs(srec["level_GPa"])
            srec["predicted_tube_radius_km"] = r
            srec["predicted_tube_radius_cells"] = r / spacing
            print(f"    {srec['scalar']:12s} {srec['level_tag']:3s} "
                  f"{srec['level_MPa']:+8.3f} MPa -> {r * 1e3:6.1f} m = "
                  f"{r / spacing:5.2f} cells"
                  f"{'' if r / spacing >= 2.5 else '   <-- UNDER 2.5 CELLS'}")

        # And then MEASURED off the surface that was extracted, because a tube
        # that does not match the prediction is a bug and not a discovery. The
        # classical level set should come within about r of the worst edge; the
        # mollified one is the control, a smooth sheet with no reason to stand
        # off that line at any particular distance.
        print("  measured tube radius on that edge, against the prediction:")
        for srec in surfaces:
            key = (srec["scalar"], srec["level_tag"])
            got = measured.get((srec["kernel"],) + key)
            if got is None:
                continue
            srec["tube_radius_km"], srec["closest_approach_km"] = got
            if srec["kernel"] == "classical":
                r = srec["predicted_tube_radius_km"]
                srec["tube_radius_ratio"] = got[0] / r
                moll = measured[("mollified",) + key]
                print(f"    {srec['scalar']:12s} {srec['level_tag']:3s} "
                      f"classical {got[0] * 1e3:6.1f} m against "
                      f"{r * 1e3:6.1f} m predicted  ratio "
                      f"{got[0] / r:4.2f}   mollified {moll[0] * 1e3:6.1f} m")

        # The prediction's CONTENT is r proportional to 1/L; its prefactor was
        # measured at a different edge and scaled by slip mismatch, so the
        # absolute radius is the weak half of it and the scaling is the strong
        # half. Tested separately, because a prefactor 25 % out with the right
        # law means something quite different from the wrong law.
        print("  and the 1/L law the prediction is really about:")
        for name in SCALARS:
            tags = [r for r in surfaces if r["kernel"] == "classical"
                    and r["scalar"] == name and "tube_radius_km" in r]
            if len(tags) != 2 or not np.isfinite(
                    [t["tube_radius_km"] for t in tags]).all():
                continue
            lo, hi = sorted(tags, key=lambda r: abs(r["level_GPa"]))
            if lo["tube_radius_km"] <= 0:
                continue
            want = abs(lo["level_GPa"]) / abs(hi["level_GPa"])
            if abs(want - 1.0) < 1e-9:
                continue        # the +L and -L lobes: same magnitude, no law
            got = hi["tube_radius_km"] / lo["tube_radius_km"]
            hi["tube_law_ratio"], hi["tube_law_expected"] = got, want
            print(f"    {name:12s} r(high)/r(low) {got:5.3f} against "
                  f"{want:5.3f} from the levels alone, "
                  f"{100 * abs(got / want - 1):4.1f} % apart")

    # The crack: the whole mesh for the envelope, the facets the slab can see
    # for the detail. Re-exported as GLB rather than re-read from
    # rough_geometry.json because this page has exactly one loader; both come
    # from roughvol.build(), and the check below refuses a drift between them.
    if scene == "detail":
        # Clipped in the SLAB'S OWN FRAME, to its footprint plus a tenth. A
        # sphere around the centre would reach well past the sampled box, and
        # the camera frames whatever is in the scene -- the surfaces would end
        # up small inside a crack that is mostly not being contoured.
        local = (tris.mean(1) - origin) @ rot
        half = 0.5 * (dims - 1) * spacing
        keep = tris[np.all(np.abs(local[:, :2] - half[:2]) < 1.1 * half[:2],
                           axis=1)]
    else:
        keep = tris
    cf = f"iso/crack_{scene}.glb"
    nb = write_glb(out / cf, [crack_part(keep, f"crack_{scene}")])
    print(f"  {cf:44s} {keep.shape[0]:6d} facets  {nb / 1024:7.0f} kB")
    # The same crack is published twice, in two formats for two viewers. That
    # is a drift waiting to happen, so it is checked rather than trusted.
    rg = out / "rough_geometry.json"
    if rg.exists() and scene == "envelope":
        g = json.loads(rg.read_text())["fault"]
        want = np.array(g["vertices"])[np.array(g["triangles"])]
        got = read_glb(out / cf)["meshes"][0]["POSITION"].reshape(-1, 3, 3)
        drift = float(np.abs(got - want).max())
        print(f"  crack GLB against rough_geometry.json: "
              f"max |diff| = {drift * 1e5:.2f} cm")
        if drift > 1e-4:                       # the JSON rounds to 1e-4 km
            raise SystemExit("the crack in the GLB is not the crack the "
                             "roughness page draws")

    man_path = out / "iso_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {
        "schema": SCHEMA, "axis_up": "z", "units": "km, GPa",
        "surfaces": [], "scenes": {}}
    man["surfaces"] = [s for s in man["surfaces"] if s["scene"] != scene]
    man["surfaces"] += surfaces
    man["scenes"][scene] = {
        "grid": {"dims": [int(d) for d in dims], "spacing": spacing,
                 "origin": [float(v) for v in origin],
                 "rotation": [[float(v) for v in row] for row in rot]},
        "spacing_over_eps": spacing / eps,
        "crack": cf, "n_facets": int(keep.shape[0]),
        "max_triangles": MAX_TRIANGLES,
        "level_percentiles": list(LEVEL_PERCENTILES)}
    # Only the slab has a centre; the envelope's is NaN, and json.dumps writes
    # that as a bare NaN, which is valid Python and NOT valid JSON -- the page
    # would have failed in JSON.parse with the canvas simply never appearing.
    # allow_nan=False below turns any future one into an exception here.
    if np.isfinite(z["centre"]).all():
        man["scenes"][scene]["centre"] = [float(v) for v in z["centre"]]
    man["receiver"] = {
        "normal": [float(v) for v in z["n_hat"]],
        "slip_direction": [float(v) for v in z["s_hat"]],
        "friction": float(z["friction"]),
        "sign": "normal is tension positive; positive dCFS is toward failure"}
    man["eps_km"] = eps
    man_path.write_text(
        json.dumps(man, indent=1, sort_keys=True, allow_nan=False) + "\n")

    worst = max((c[1] for c in checks), default=0.0)
    print(f"  {len(surfaces)} surfaces, "
          f"{sum(s['bytes'] for s in surfaces) / 1024 ** 2:.2f} MiB; worst "
          f"level error {worst * 100:.2f} %")
    if worst > 0.02:
        raise SystemExit(f"a contour is {worst:.1%} off the level it claims; "
                         f"the grid order or the spacing is wrong")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("field", help="evaluate both kernels on a scene's grid")
    f.add_argument("--state", required=True)
    f.add_argument("--scene", choices=sorted(SCENES), required=True)
    f.add_argument("--out", default=None)
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--probe", action="store_true",
                   help="time one chunk and print the projection, then stop")
    m = sub.add_parser("mesh", help="contour a saved field -> GLB + manifest")
    m.add_argument("--field", required=True)
    m.add_argument("--out", default=str(pathlib.Path(__file__).resolve()
                                        .parents[2] / "website" / "public"
                                        / "data"))
    a = ap.parse_args(argv)
    if a.cmd == "field":
        a.out = a.out or f"iso_{a.scene}.npz"
        return cmd_field(a)
    return cmd_mesh(a)


if __name__ == "__main__":
    raise SystemExit(main())
