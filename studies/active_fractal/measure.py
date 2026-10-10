"""Box-counting dimension of a fault set, and the nulls that give it meaning.

WHAT THIS MEASURES, AND WHAT IT CANNOT. Box counting on a FINITE set saturates
at both ends: for boxes smaller than the sample spacing every sample owns a box
and the slope goes to 0, and for boxes approaching the domain every box is
occupied and the slope goes to 0 again. In between, a set of finite-size planar
patches is NOT a clean fractal at any construction:

    r below the smallest patch   a box is occupied iff a plane crosses it, so
                                 N(r) ~ A_total / r^2 and the slope is 2
    r above the patch sizes      only the arrangement of patches is visible, so
                                 the slope approaches the dimension of the
                                 centers (3 for Poisson, alpha for a Levy dust)

So a single number fitted across that crossover is a function of the window.
That is a property of fault networks rather than of this code, and it is why
every result here carries its window and its LOCAL slopes.

NEVER QUOTE AN ABSOLUTE DIMENSION FROM THIS MODULE. Quote a DIFFERENCE against
a null measured in the same window with the same boxes. Measured on sets whose
dimension is known in closed form, in one window:

    flat sheet        want 2.000   got 1.887   err -0.113
    Cantor dust       want 1.893   got 1.783   err -0.110
    the DIFFERENCE    want 0.107   got 0.104   err -0.003

The bias is systematic, it is about -0.11 over a one-decade window, and it
very nearly CANCELS in a difference. So a difference is good to ~0.01 where an
absolute is wrong by 0.11.

R^2 DOES NOT DETECT THIS. Every fit above has R^2 >= 0.999, including the one
off by -0.16. A near-perfect straight line with the wrong slope is the normal
case here, so R^2 must not be used as a quality gate; the local slopes and the
window are what to read.

AND THE REASON THE NULLS ARE NOT OPTIONAL. A random subset of a set that is
exactly 2-dimensional measures lower-dimensional purely because it has fewer
members, with nothing mechanical involved. Measured on a flat sheet of 13122
triangles, same window throughout:

    full sheet          D = 1.887
    random 50 % subset  D = 1.886   shift -0.001
    random 20 % subset  D = 1.852   shift -0.036
    random  5 % subset  D = 1.528   shift -0.359
    random  1 % subset  D = 0.876   shift -1.011

A 1 % active fraction loses a FULL DIMENSION to cardinality alone. So
"D_active < D_geom" is not evidence of anything until it is compared with a
subset of the same size, which is what :func:`null_random` and
:func:`null_by_score` are for.

AREA-WEIGHTED, NOT CENTROID. Elements are not the set; the surfaces are. A
centroid count would measure the discretization: a long fault carries more
centroids than a short one for reasons that have nothing to do with how much
surface is there. Each triangle is instead covered by a deterministic
barycentric lattice whose spacing is a fixed fraction of the smallest box, so
the count converges to the area the set actually occupies and is independent of
how the surface was cut into triangles.

THE LATTICE IS DETERMINISTIC, not random. A random area sample would make the
dimension a function of the draw, and two sets could then differ by sampling
noise alone -- which is precisely the artefact the nulls exist to exclude.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

#: Lattice spacing as a fraction of the smallest box edge. 1/4 puts at least
#: ~16 samples in any box a triangle crosses, so a box is missed only if the
#: triangle clips a corner of it.
SAMPLE_SPACING_OVER_RMIN = 0.25

#: Fewest boxes a radius must occupy to enter a fit. Below this the count is
#: in its large-r saturation and the slope it contributes is a measurement of
#: the domain, not of the set.
MIN_BOXES_IN_FIT = 8

#: Fewest distinct radii a window must contain for a slope to be reported.
MIN_RADII_IN_FIT = 4

#: Grid phases averaged per radius. One fixed lattice is NOT enough and this is
#: not a refinement: a single origin makes the count depend on whether the box
#: edge happens to be commensurate with structure in the set, and the error is
#: not noise but a systematic alias. Measured on sets of known dimension, one
#: origin against eight:
#:
#:     flat sheet,  D = 2 exactly      1 origin 1.896   8 origins 1.998
#:     Cantor dust, D = 1.893          1 origin 1.802   8 origins 1.878
#:
#: and with one origin the dust's local slopes include NEGATIVE values -- the
#: count rising with box size, which cannot happen and is pure aliasing.
N_ORIGINS = 8

#: Seed for the phase offsets. Fixed, so the estimator stays deterministic:
#: a dimension that moved between runs could not be compared with a null.
ORIGIN_SEED = 20261010

#: The window rule, and it is a rule rather than a choice because picking the
#: window that straightens the line is how a crossover becomes a dimension.
#:
#: r_lo = 2 x the finest FEATURE -- the largest fault, or a cascade's finest
#: cell. Not the element size: until a box is bigger than a fault, that fault
#: reads locally 2-dimensional. At 1x the finest feature the count is still in
#: its plateau (one box per fault) and the measured dimension is wrong by
#: -0.121 where 2x gives -0.017 on the same network.
#:
#: r_hi = extent/6 -- fewer than six boxes per axis and the count is measuring
#: the domain.
WINDOW_RLO_OVER_FEATURE = 2.0
WINDOW_RHI_OVER_EXTENT = 1.0 / 6.0

#: Multiplicative bias of the estimator, measured at the window rule above on
#: five sets whose dimension is known in closed form (a sliver, a flat sheet,
#: three Cantor dusts) spanning D = 1.0 to 2.377:
#:
#:     D_true   D_raw    ratio
#:      1.000   0.941    0.941
#:      1.500   1.445    0.963
#:      1.893   1.809    0.956
#:      2.000   1.874    0.937
#:      2.377   2.236    0.941
#:
#: One factor fits all five to within 0.027, against raw errors up to 0.141.
#: End to end on cascade NETWORKS of real oriented faults, where the arrangement
#: dimension is known exactly, the calibrated error is -0.017 over a 0.83-decade
#: window and -0.082 over 0.73 decades.
#:
#: SO: an absolute dimension from this module is good to about 0.02-0.08, and
#: only when the window is wide. A DIFFERENCE in a matched window is good to
#: ~0.01, because the bias cancels. Prefer the difference, always.
CALIBRATION_K = 0.9462


def window_for(feature_km: float, extent_km: float) -> dict:
    """The window rule as keyword arguments for :func:`dimension`.

    ``feature_km`` is the finest feature the set actually has -- the largest
    fault length, or a cascade's finest cell. Raises rather than returning a
    window narrower than half a decade, because a slope fitted over less than
    that was wrong by up to 0.7 in testing and no calibration repairs it.
    """
    r_lo = WINDOW_RLO_OVER_FEATURE * feature_km
    r_hi = WINDOW_RHI_OVER_EXTENT * extent_km
    if r_hi <= r_lo or math.log10(r_hi / r_lo) < 0.5:
        raise ValueError(
            f"window [{r_lo:g}, {r_hi:g}] km spans "
            f"{math.log10(max(r_hi / r_lo, 1.0000001)):.2f} decades, under the "
            f"0.5 needed for a slope. The domain must be at least "
            f"{6 * WINDOW_RLO_OVER_FEATURE * 10 ** 0.5:.0f}x the finest "
            f"feature ({feature_km:g} km), i.e. "
            f"{6 * WINDOW_RLO_OVER_FEATURE * 10 ** 0.5 * feature_km:.1f} km, "
            f"and is {extent_km:g} km.")
    return dict(r_lo=r_lo, r_hi=r_hi)


def calibrated(fit: "DimFit") -> float:
    """``fit.dimension`` with the multiplicative bias removed."""
    return fit.dimension / CALIBRATION_K


@dataclasses.dataclass(frozen=True)
class DimFit:
    """A dimension, and everything needed to judge whether to believe it."""

    dimension: float            # -slope of log N on log r
    stderr: float               # standard error of that slope
    r2: float                   # coefficient of determination of the fit
    window: tuple[float, float]  # (r_lo, r_hi) actually fitted, km
    radii: np.ndarray           # every radius probed, km
    counts: np.ndarray          # N(r) at each radius
    local: np.ndarray           # local slope between consecutive radii
    n_points: int               # lattice samples that went in

    def __str__(self) -> str:
        return (f"D = {self.dimension:.3f} +- {self.stderr:.3f} "
                f"(R2 {self.r2:.4f}, r {self.window[0]:.3g}-"
                f"{self.window[1]:.3g} km, "
                f"{math.log10(self.window[1] / self.window[0]):.2f} decades)")


def barycentric_lattice(k: int) -> np.ndarray:
    """``((k+1)(k+2)/2, 3)`` barycentric coordinates on a ``k``-subdivided tri.

    ``k = 1`` is the three vertices. Spacing along an edge is ``1/k``, so a
    triangle of longest edge ``L`` is sampled at ``L/k``.
    """
    k = max(1, int(k))
    out = [(i / k, j / k, 1.0 - i / k - j / k)
           for i in range(k + 1) for j in range(k + 1 - i)]
    return np.asarray(out, float)


def area_samples(tris: np.ndarray, spacing: float) -> np.ndarray:
    """Points covering ``tris`` at roughly ``spacing``, area-weighted.

    Each triangle gets a lattice fine enough that its own longest edge is
    sampled at ``spacing``, so the areal density of points is the same on every
    triangle regardless of its size. Triangles are grouped by their lattice
    order so the work is a handful of vectorised einsums rather than a loop
    over elements.
    """
    tris = np.asarray(tris, float).reshape(-1, 3, 3)
    e = np.stack([tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0],
                  tris[:, 2] - tris[:, 1]], axis=1)
    longest = np.linalg.norm(e, axis=2).max(axis=1)
    k = np.maximum(1, np.ceil(longest / max(spacing, 1e-12)).astype(np.int64))
    out = []
    for kk in np.unique(k):
        sel = tris[k == kk]
        bary = barycentric_lattice(int(kk))
        out.append(np.einsum("bv,nvc->nbc", bary, sel).reshape(-1, 3))
    return np.concatenate(out, axis=0)


def box_count(pts: np.ndarray, r: float, origin: np.ndarray) -> int:
    """Occupied boxes of edge ``r`` on a lattice anchored at ``origin``.

    Cells are keyed by a single int64 so the count is one ``np.unique``. The
    key is built from a per-axis extent rather than a fixed multiplier, which
    is what keeps it collision-free for any ``r``.
    """
    idx = np.floor((pts - origin) / r).astype(np.int64)
    idx -= idx.min(axis=0)
    n = idx.max(axis=0) + 1
    key = (idx[:, 0] * n[1] + idx[:, 1]) * n[2] + idx[:, 2]
    return int(np.unique(key).size)


def box_count_phased(pts: np.ndarray, r: float, origin: np.ndarray,
                     n_origins: int = N_ORIGINS) -> float:
    """``box_count`` averaged over grid phases, as the GEOMETRIC mean.

    Geometric rather than arithmetic because the quantity is fitted in log
    space: averaging the counts and then taking the log is not the same as
    averaging the logs, and it is the second one the slope is built from.
    """
    phase = np.random.default_rng(ORIGIN_SEED).random((n_origins, 3))
    logs = [math.log(box_count(pts, r, origin - p * r)) for p in phase]
    return math.exp(sum(logs) / len(logs))


def dimension(tris: np.ndarray, *, r_lo: float, r_hi: float,
              n_radii: int = 12, origin: np.ndarray | None = None,
              spacing: float | None = None) -> DimFit:
    """Area-weighted box-counting dimension of ``tris`` over ``[r_lo, r_hi]``.

    The window is an ARGUMENT, never chosen from the data: picking the window
    that gives the cleanest line is how a crossover becomes a dimension. Local
    slopes come back alongside so a window can be judged after the fact and the
    same one applied to every set being compared.
    """
    tris = np.asarray(tris, float).reshape(-1, 3, 3)
    if tris.size == 0:
        raise ValueError("no triangles to measure")
    if not (r_hi > r_lo > 0.0):
        raise ValueError(f"need 0 < r_lo < r_hi, got {r_lo}, {r_hi}")
    if spacing is None:
        spacing = SAMPLE_SPACING_OVER_RMIN * r_lo
    pts = area_samples(tris, spacing)
    if origin is None:
        origin = pts.min(axis=0)
    radii = np.geomspace(r_lo, r_hi, n_radii)
    counts = np.array([box_count_phased(pts, float(r), origin)
                       for r in radii], dtype=float)

    local = np.full(radii.size, np.nan)
    lr, lc = np.log(radii), np.log(counts)
    local[1:] = -(lc[1:] - lc[:-1]) / (lr[1:] - lr[:-1])

    ok = counts >= MIN_BOXES_IN_FIT
    if int(ok.sum()) < MIN_RADII_IN_FIT:
        raise ValueError(
            f"only {int(ok.sum())} of {n_radii} radii occupy at least "
            f"{MIN_BOXES_IN_FIT} boxes, so there is no fit to report. The "
            f"window [{r_lo:g}, {r_hi:g}] km is above the set's extent or the "
            f"set is too small.")
    x, y = lr[ok], lc[ok]
    n = x.size
    sx, sy = x.sum(), y.sum()
    sxx = float((x * x).sum())
    sxy = float((x * y).sum())
    den = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / den
    icept = (sy - slope * sx) / n
    resid = y - (slope * x + icept)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    s_err = (math.sqrt(ss_res / (n - 2) * n / den) if n > 2 else float("nan"))
    return DimFit(dimension=-slope, stderr=s_err, r2=r2,
                  window=(float(math.exp(x.min())), float(math.exp(x.max()))),
                  radii=radii, counts=counts, local=local,
                  n_points=int(pts.shape[0]))


# ------------------------------------------------------------------ nulls ----
def null_random(tris: np.ndarray, n_keep: int, *, seed: int, n_real: int,
                **kw) -> list[DimFit]:
    """``n_real`` realizations of a uniformly random subset of ``n_keep``.

    The question this answers is narrow and worth stating: is the active set
    more clustered than a set of the same size drawn without regard to
    mechanics? It does NOT control for the loading geometry -- that is
    :func:`null_by_score`.
    """
    tris = np.asarray(tris, float).reshape(-1, 3, 3)
    rng = np.random.default_rng(seed)
    n = tris.shape[0]
    if not (0 < n_keep <= n):
        raise ValueError(f"n_keep must lie in (0, {n}], got {n_keep}")
    return [dimension(tris[rng.choice(n, n_keep, replace=False)], **kw)
            for _ in range(n_real)]


def null_by_score(tris: np.ndarray, score: np.ndarray,
                  n_keep: int, **kw) -> DimFit:
    """The ``n_keep`` highest-``score`` elements, as one subset.

    With ``score`` the INITIAL resolved Coulomb stress -- computed with the
    interaction matrix set to zero -- this is the set that would fail if
    elements did not talk to each other. It isolates orientation and loading
    from elastic interaction, which makes it the null the headline claim has to
    beat, and the direct test of whether geometry alone manufactures asperities.
    """
    tris = np.asarray(tris, float).reshape(-1, 3, 3)
    score = np.asarray(score, float).ravel()
    if score.size != tris.shape[0]:
        raise ValueError(f"score has {score.size} entries for "
                         f"{tris.shape[0]} triangles")
    idx = np.argsort(-score)[:n_keep]
    return dimension(tris[idx], **kw)


# ------------------------------------------------------- calibration sets ----
# Sets whose dimension is known by construction, so the estimator can be shown
# to recover it BEFORE it is pointed at a fault network. The headline claim
# rests entirely on this estimator, so "it looked reasonable on the data" is
# not a standard it can be held to.
def plane_tris(extent: float = 64.0, h: float = 1.0) -> np.ndarray:
    """A flat square sheet. D = 2 exactly."""
    n = max(1, int(round(extent / h)))
    g = np.linspace(-extent / 2.0, extent / 2.0, n + 1)
    X, Y = np.meshgrid(g, g, indexing="ij")
    V = np.stack([X, Y, np.full_like(X, -10.0)], axis=-1)
    return np.asarray([t for i in range(n) for j in range(n)
                       for t in ([V[i, j], V[i + 1, j], V[i + 1, j + 1]],
                                 [V[i, j], V[i + 1, j + 1], V[i, j + 1]])],
                      float)


def line_tris(extent: float = 64.0, h: float = 1.0,
              width: float = 0.02) -> np.ndarray:
    """A sliver: one long strip of negligible width. D = 1 over r >> width."""
    n = max(1, int(round(extent / h)))
    g = np.linspace(-extent / 2.0, extent / 2.0, n + 1)
    out = []
    for i in range(n):
        a, b = g[i], g[i + 1]
        p = [np.array([a, 0.0, -10.0]), np.array([b, 0.0, -10.0]),
             np.array([b, width, -10.0]), np.array([a, width, -10.0])]
        out += [[p[0], p[1], p[2]], [p[0], p[2], p[3]]]
    return np.asarray(out, float)


def cantor_dust_tris(level: int = 4, extent: float = 81.0, keep: int = 2,
                     base: int = 3, patch: float = 0.25) -> np.ndarray:
    """A 3-D Cantor dust of ``D = 3 log(keep)/log(base)``, as tiny patches.

    Each surviving cell at the finest level carries one small flat square, so
    the set is a dust of patches rather than of points -- which is exactly the
    situation a fault network is in, and therefore the honest calibration. The
    plateau to look for is between ``patch`` and the coarsest cell.
    """
    rng = np.random.default_rng(0)
    cells = [np.zeros(3)]
    size = extent
    for _ in range(level):
        size /= base
        nxt = []
        for c in cells:
            offs = np.stack(np.meshgrid(*[np.arange(base)] * 3, indexing="ij"),
                            axis=-1).reshape(-1, 3)
            pick = rng.choice(len(offs), keep ** 3 if keep < base else len(offs),
                              replace=False)
            for o in offs[pick]:
                nxt.append(c + o * size)
        cells = nxt
    out = []
    for c in cells:
        x, y, z = c + size / 2.0
        p = [np.array([x - patch, y - patch, -10.0 - z]),
             np.array([x + patch, y - patch, -10.0 - z]),
             np.array([x + patch, y + patch, -10.0 - z]),
             np.array([x - patch, y + patch, -10.0 - z])]
        out += [[p[0], p[1], p[2]], [p[0], p[2], p[3]]]
    return np.asarray(out, float)
