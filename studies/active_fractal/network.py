"""Synthetic fault networks, as the bare ``(n, 3, 3)`` triangles mhs takes.

A network is many disconnected planar rectangular patches. mhs needs no mesh
object, no connectivity and no watertightness -- it takes vertices -- so a
network here is a concatenation of per-patch grids plus an integer saying which
patch each triangle came from.

WHAT SETS THE GEOMETRIC DIMENSION. Fault lengths follow a power law,
``N(>l) ~ l^-a``, and that alone does NOT make a network fractal: the
ARRANGEMENT of the faults does. Three arrangements are available and only one
of them has a dimension known in closed form.

    centers="cascade"   THE DEFAULT, and the only one that can be validated. A
                        multiplicative cascade: subdivide a cube into base^3
                        cells, keep keep^3 of them, recurse. The surviving set
                        has dimension ``3 log(keep) / log(base)`` EXACTLY, so
                        sweeping (keep, base) sweeps a KNOWN D_geom and the
                        whole measuring pipeline can be checked end to end.
                        Measured, with real oriented faults placed in the
                        cells and the estimator's calibration applied:

                            D known   D measured   error   levels
                              1.893       1.911    +0.018     4
                              1.500       1.544    +0.044     3
                              2.377       2.323    -0.055     2
                              2.584       2.486    -0.099     2

                        The error is a function of how many cascade levels fit
                        inside the measuring window, not of the estimator.

    centers="uniform"   Poisson centers. The union of surfaces then shows a
                        CROSSOVER, not a fractal: below the smallest fault a
                        box is occupied iff a plane crosses it, so
                        N(r) ~ A_total / r^2 and the slope is 2; above the mean
                        center spacing the patches fill the box and the slope
                        goes to 3. One line fitted across that is a function of
                        the window. Kept so the crossover can be shown rather
                        than asserted.

    centers="levy"      Kept, but NOT as a dimension knob, because the obvious
                        claim about it is false here. For a true Levy flight
                        the range has dimension alpha; this construction draws
                        TRUNCATED steps and WRAPS at the box faces, and both
                        destroy that result. Measured, the dimension moves the
                        WRONG WAY: alpha = 1.8, 2.2, 2.6 gives calibrated
                        D = 1.72, 1.64, 1.53. Large alpha means short steps
                        means a more confined walk, and wrapping randomises the
                        long ones. It clusters, and the clustering is monotone
                        in alpha, but alpha is not a dimension.

THE MEASURING WINDOW IS BOUNDED BELOW BY THE LARGEST FAULT, not the smallest
element. Until a box is bigger than a fault, that fault reads locally
2-dimensional, so clean scaling needs ``r > l_max``, and with at least six
boxes per axis above, ``box >~ 60 * l_max``. That is a constraint on the
GENERATOR, which is why it is stated here.

ELEMENTS ARE PRUNED, FAULTS ARE NOT REJECTED, and the difference is the
independent variable. Two elements closer than a few eps are nearly the same
element: their columns in the interaction matrix nearly coincide and the
active-set submatrix the complementarity solve factorises stops being negative
definite. The obvious guard -- reject a candidate fault that comes too close to
an existing one -- is SIZE DEPENDENT, because a long fault sweeps more area and
collides more often. Measured, at a nominal a = 1.5 whose own drawn pool gives
1.504:

    fault rejection, floor 10 eps     realized a = 2.59
    fault rejection, floor  5 eps     realized a = 2.52
    element pruning,  floor  5 eps    realized a = 1.37, 52 % pruned
    element pruning,  floor 2.5 eps   realized a = 1.54,  4.6 % pruned
    element pruning,  floor 1.2 eps   realized a = 1.55,  0.7 % pruned

So rejection moves the exponent by a full unit and is unusable. Pruning at
2.5 eps keeps it, and the floor is stated in eps rather than in h because
clearance in units of eps is what governs error everywhere in this project.
1.2 eps prunes less still, and is not taken: it is tighter than the worst
on-element collocation clearance the library itself runs at (1.26), and that
one is the designed case rather than two distinct elements.

Intersecting faults therefore truncate each other, which is what faults do.

THE NOMINAL EXPONENT IS AN INPUT; THE REALIZED ONE IS MEASURED. ``meta`` carries
the bounded-Pareto MLE of the lengths actually placed, and that is the number to
quote and to sweep. A nominal value reported as though it were realized is the
same mistake as quoting an unconverged supremum.

Units are km throughout, and every vertex is below z = 0 -- the library refuses
a source above the free surface.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

#: Separation floor between elements, in units of eps. See the module docstring
#: for the measurement that chose it: 5 eps prunes 52 % and moves the length
#: exponent, 1.2 eps is tighter than the library's own worst on-element
#: collocation clearance.
MIN_SEP_OVER_EPS = 2.5

#: Fewest elements a pruned fault may retain and still be kept. Below this it
#: is not a fault, it is a fragment, and its "length" would no longer describe
#: the patch that went into the matrix.
MIN_ELEMENTS_PER_FAULT = 2

#: Perpendicular clearance a collocation point must keep from ANOTHER element's
#: surface, in units of eps. MEASURED, after a first guess of 3 eps taken from
#: the library's remark that the eigenstress dominates "within ~3 eps" -- which
#: is about the eigenstress, not about conditioning, and is four times stricter
#: than conditioning needs. At N = 4000, s/h = 0.7:
#:
#:     clear   n kept   pruned   B mean   min eig sym(-P)   negative
#:      0        3572    10.7 %   0.594        -11.4            2
#:      0.25     3242    19.0 %   0.532         +9.9            0
#:      0.5      2982    25.5 %   0.479        +10.1            0
#:      0.75     2786    30.4 %   0.440        +39.9            0
#:      1.0      2621    34.5 %   0.409        +45.7            0
#:      3.0      1871    53.2 %   0.274        +65.4            0
#:
#: 0.75 is the knee. Below it the eigenvalue margin collapses to ~10 against a
#: spectrum whose median is 92, so another realization could cross zero; above
#: it the branching ratio falls with no conditioning bought. Holding it at 0.75
#: keeps sym(-P) at +38 to +50 from 1292 to 10479 elements, so the floor works
#: at production scale and not only on the size it was tuned at.
MIN_FACE_CLEARANCE_OVER_EPS = 0.75


@dataclasses.dataclass(frozen=True)
class Network:
    """A fault network: triangles, which fault each belongs to, and the inputs.

    ``tris`` is ``(n, 3, 3)`` as mhs wants it. ``fault_id`` is ``(n,)`` so a
    per-element result can be grouped back onto faults, which is what the
    avalanche and connectivity readouts need.
    """

    tris: np.ndarray            # (n, 3, 3) vertices, km, all z < 0
    fault_id: np.ndarray        # (n,) index into the per-fault arrays
    length: np.ndarray          # (n_faults,) km along strike
    width: np.ndarray           # (n_faults,) km down dip
    azimuth: np.ndarray         # (n_faults,) deg, strike direction from +x
    dip: np.ndarray             # (n_faults,) deg from horizontal
    center: np.ndarray          # (n_faults, 3) km
    h: float                    # target element size, km
    eps: float                  # mollification width, km
    meta: dict

    @property
    def n_elements(self) -> int:
        return int(self.tris.shape[0])

    @property
    def n_faults(self) -> int:
        return int(self.length.shape[0])

    def areas(self) -> np.ndarray:
        """``(n,)`` triangle areas, km^2 -- the weight the estimator uses."""
        e1 = self.tris[:, 1] - self.tris[:, 0]
        e2 = self.tris[:, 2] - self.tris[:, 0]
        return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def pareto_lengths(rng, n: int, a: float, lmin: float,
                   lmax: float) -> np.ndarray:
    """``n`` lengths from a bounded Pareto with CUMULATIVE exponent ``a``.

    ``N(>l) ~ l^-a`` on ``[lmin, lmax]``, by inverse CDF. The exponent is the
    cumulative one because that is what the fault-length literature quotes; the
    density exponent is ``a + 1``.
    """
    if not (lmax > lmin > 0.0):
        raise ValueError(f"need 0 < lmin < lmax, got {lmin}, {lmax}")
    if a <= 0.0:
        raise ValueError(f"cumulative exponent a must be positive, got {a}")
    u = rng.random(n)
    lo, hi = lmin ** -a, lmax ** -a
    return (lo - u * (lo - hi)) ** (-1.0 / a)


def pareto_mle(lengths: np.ndarray, lmin: float, lmax: float) -> float:
    """Bounded-Pareto MLE for the cumulative exponent of ``lengths``.

    The score is ``n/a + n ln lmin - sum ln L + n r^a ln r / (1 - r^a)`` with
    ``r = lmin/lmax``, solved by bisection because it is monotone in ``a``.
    Verified to recover 1.0015, 1.5004 and 2.4961 from 200k draws at nominal
    1.0, 1.5 and 2.5 -- worth stating, because an earlier sign error on the
    truncation term returned 1.64 for a nominal 1.5 and looked plausible.
    """
    L = np.asarray(lengths, float)
    n = L.size
    if n < 2:
        return float("nan")
    r = lmin / lmax
    log_r = math.log(r)
    s = float(np.log(L).sum())

    def score(a: float) -> float:
        ra = r ** a
        return n / a + n * math.log(lmin) - s + n * ra * log_r / (1.0 - ra)

    lo, hi = 1e-4, 50.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def expected_elements_per_fault(a: float, lmin: float, lmax: float,
                                aspect: float, h: float) -> float:
    """``E[2 L^2 / (aspect h^2)]`` under the bounded Pareto, in closed form.

    The budget arithmetic, and it is load-bearing rather than decorative: it is
    what shows that placing faults LARGEST FIRST is unusable. At a = 1.5 over
    [2, 40] km with h = 1 km the mean is 42 elements per fault, so a 20k budget
    buys ~480 faults in random order -- but the largest single fault is 1600
    elements, so largest-first spends the whole budget on four of them.
    """
    r = lmin / lmax
    if abs(a - 2.0) < 1e-9:
        m2 = a * lmin ** a * math.log(lmax / lmin)
    else:
        m2 = a * lmin ** a * (lmax ** (2.0 - a) - lmin ** (2.0 - a)) / (2.0 - a)
    return 2.0 * (m2 / (1.0 - r ** a)) / (aspect * h * h)


def isotropic_normals(rng, n: int) -> np.ndarray:
    """``(n, 3)`` unit normals uniform on the UPPER hemisphere.

    A normalised Gaussian is exactly uniform on the sphere, which drawing dip
    and azimuth from independent uniforms is not. The hemisphere flip is the
    axial convention ``mhs.tdcs.unit_normals`` also uses: a plane's normal and
    its negation are the same plane.

    WHY THIS EXISTS SEPARATELY FROM ``build_network``'s dip/azimuth draw, with
    the numbers, because the obvious story about that draw is wrong. Fabric
    eigenvalues of the orientation tensor over 200k draws:

        dip U[30,90], azimuth U[0,360)   S = 0.3535 0.3528 0.2937   off by 0.020
        dip U[0,90],  azimuth U[0,360)   S = 0.4993 0.2514 0.2493   off by 0.166
        this function                    S = 0.3344 0.3338 0.3318   off by 0.001

    Uniform-in-dip UNDER-weights steep dips relative to isotropic, not over:
    isotropy needs ``P(dip) ~ sin(dip)``, which rises with dip. And over the
    truncated range the weighting barely matters -- the deficient axis is the
    vertical at 0.2933, exactly ``(3/pi) * int cos^2`` for uniform dip on
    [30, 90], while SIN-weighting the same range gives 0.25, i.e. further from
    isotropic. So the fault network's anisotropy is almost entirely the
    TRUNCATED DIP RANGE, which is a modelling choice worth keeping, and this
    function is for the cloud, where no orientation family may be privileged
    before the mechanics runs.
    """
    g = rng.normal(size=(int(n), 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return np.where(g[:, 2:3] < 0.0, -g, g)


def subparallel_normals(rng, n: int, mean_normal, spread_deg: float
                        ) -> np.ndarray:
    """``(n, 3)`` unit normals scattered about ``mean_normal``.

    THE ISOTROPIC CLOUD CANNOT CASCADE, AND THIS IS WHY IT MATTERS. The
    triggering budget counts only the positive part of the transfer, so
    neighbours at differing orientations relax each other about as often as
    they trigger, and the budget cancels against itself: the fault network
    measured ``B_intra = 0.526`` among CO-ORIENTED elements of one fault
    against ``B_inter = 0.0005-0.011`` between faults, and an isotropic cloud
    is all "inter". Packing harder does not fix that -- it is a property of the
    orientations, not of the distance -- so the remaining lever is the spread.

    This is also the configuration the question is actually about. A real rock
    holds far more fractures than participate, and the ones that matter are the
    kinematically coherent subset. Drawing every element near one orientation
    removes the loading's ability to do the selecting -- they are all similarly
    favoured -- so what is left to distinguish them is POSITION and the elastic
    interaction, which is the thing being measured.

    ``spread_deg`` is the standard deviation of the polar angle away from
    ``mean_normal``, with azimuth uniform about it: 0 is a perfectly parallel
    set, and large values approach :func:`isotropic_normals` without ever
    being it, because a half-normal has no cut-off. Orientation stays
    INDEPENDENT OF POSITION, so the no-interaction null is still spatially
    unstructured by construction and the study's central comparison survives
    unchanged.
    """
    m = np.asarray(mean_normal, float)
    m = m / np.linalg.norm(m)
    a = np.cross(m, [0.0, 0.0, 1.0] if abs(m[2]) < 0.9 else [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(m, a)
    th = np.abs(rng.normal(0.0, math.radians(spread_deg), int(n)))
    ph = rng.uniform(0.0, 2.0 * math.pi, int(n))
    v = (np.cos(th)[:, None] * m
         + np.sin(th)[:, None] * (np.cos(ph)[:, None] * a
                                  + np.sin(ph)[:, None] * b))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(v[:, 2:3] < 0.0, -v, v)


def fabric(normals: np.ndarray) -> np.ndarray:
    """Eigenvalues ``S1 >= S2 >= S3`` of the orientation tensor ``<n (x) n>``.

    They sum to 1. Isotropic is ``(1/3, 1/3, 1/3)``; a single orientation
    drives ``S1 -> 1``. Sign-invariant in ``n``, so the hemisphere flip above
    does not affect it.
    """
    nrm = np.asarray(normals, float).reshape(-1, 3)
    T = np.einsum("ij,ik->jk", nrm, nrm) / nrm.shape[0]
    return np.sort(np.linalg.eigvalsh(T))[::-1]


def oriented_triangle(centers: np.ndarray, normals: np.ndarray,
                      size: float) -> np.ndarray:
    """``(n, 3, 3)`` equilateral triangles of edge ``size``, one per normal.

    VERTEX ORDER PUTS THE MESH NORMAL ALONG ``+normals``, which is the whole
    reason this is a function and not three lines inline. The assembly uses
    ``mhs.tdcs``'s UPWARD frame for the basis projection and the mesh's own
    vertex-order normal for the eigenstress; when those two disagree the
    eigenstress -- dominant within a few eps -- is subtracted with the wrong
    sign and the self-interaction diagonal comes out POSITIVE. That is the bug
    ``kernel.refuse_if_sign_wrong`` caught on ``rect_patch``, and the ordering
    below is chosen so it cannot recur: the vertices run counter-clockwise seen
    from ``+normal``, which makes ``cross(v1-v0, v2-v0)`` parallel to it with
    coefficient ``+2.598 * R^2``.

    ``mhs.fullspace.frame.equilateral`` builds the same shape but always flat
    in ``z = 0`` with normal ``+z`` and takes no orientation, so composing it
    with a rotation would cost more than building in the plane directly.
    """
    c = np.asarray(centers, float).reshape(-1, 3)
    n = np.asarray(normals, float).reshape(-1, 3)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    # Any vector not parallel to n gives an in-plane axis; pick per element so
    # the choice is never near-degenerate.
    alt = np.where(np.abs(n[:, 0:1]) < 0.9,
                   np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    e1 = np.cross(n, alt)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(n, e1)
    R = float(size) / math.sqrt(3.0)          # circumradius of edge `size`
    angles = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
    verts = [c + R * (math.cos(t) * e1 + math.sin(t) * e2) for t in angles]
    return np.ascontiguousarray(np.stack(verts, axis=1))


def cascade_dimension(keep: int, base: int) -> float:
    """``3 log(keep)/log(base)`` -- the cascade's dimension, in closed form."""
    if not (1 <= keep < base):
        raise ValueError(
            f"need 1 <= keep < base for a dimension below 3; got keep={keep}, "
            f"base={base}. keep == base keeps every cell and is space-filling.")
    return 3.0 * math.log(keep) / math.log(base)


def cascade_cells(rng, keep: int, base: int, level: int,
                  extent: float) -> tuple[np.ndarray, float]:
    """Surviving cell corners of a multiplicative cascade, and the cell size.

    ``keep**3`` of every ``base**3`` subcells survive at each of ``level``
    levels, so exactly ``keep**(3*level)`` cells come back and the dimension is
    :func:`cascade_dimension`. A FIXED count per cell rather than an
    independent probability per cell, deliberately: the fixed count makes the
    dimension exact for every realization instead of exact only in expectation,
    and an experiment that compares a measured dimension against a known one
    should not have to average over the construction first.
    """
    cells = [np.zeros(3)]
    size = float(extent)
    offs = np.stack(np.meshgrid(*[np.arange(base)] * 3, indexing="ij"),
                    axis=-1).reshape(-1, 3)
    for _ in range(max(1, int(level))):
        size /= base
        nxt = []
        for c in cells:
            for o in offs[rng.choice(len(offs), keep ** 3, replace=False)]:
                nxt.append(c + o * size)
        cells = nxt
    return np.asarray(cells, float), size


def levy_dust(rng, n: int, alpha: float, box: np.ndarray,
              step_min: float, step_max: float) -> np.ndarray:
    """``(n, 3)`` points on a Levy dust of fractal dimension ``alpha``.

    An isotropic random walk whose step lengths are power-law with cumulative
    exponent ``alpha``; the visited set has dimension ``alpha`` for
    ``0 < alpha < 3``. Steps WRAP at the box faces rather than reflecting,
    because reflection piles density against the walls and that density is
    exactly what a dimension estimate reads.
    """
    if not (0.0 < alpha < 3.0):
        raise ValueError(
            f"alpha must lie in (0, 3) for a dust of dimension alpha in 3-D; "
            f"got {alpha}. alpha >= 3 is space-filling and the construction "
            f"stops being informative.")
    lo, hi = box[0], box[1]
    span = hi - lo
    pts = np.empty((n, 3))
    p = lo + rng.random(3) * span
    steps = pareto_lengths(rng, n, alpha, step_min, step_max)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for k in range(n):
        pts[k] = p
        p = lo + np.mod(p + steps[k] * dirs[k] - lo, span)
    return pts


def rect_patch(center: np.ndarray, azimuth_deg: float, dip_deg: float,
               length: float, width: float, h: float) -> np.ndarray:
    """One planar rectangular fault as ``(m, 3, 3)`` triangles.

    ``azimuth_deg`` rotates the strike direction from +x in the horizontal
    plane; ``dip_deg`` is from horizontal, so 90 is vertical. Both triangles of
    a cell are cut on the same diagonal, which keeps the patch conforming -- a
    collocation point sits at a shrunk centroid, and a non-conforming patch
    would put receivers in gaps.

    The element count comes from ``h``, so every fault in a network is
    discretized at the SAME element size. That is what makes one element's area
    comparable to another's, which both the area-weighted dimension estimate and
    a single scalar eps depend on.
    """
    th = math.radians(azimuth_deg)
    dl = math.radians(dip_deg)
    strike = np.array([math.cos(th), math.sin(th), 0.0])
    # Down-dip is strike rotated +90 degrees about z, then tilted down by the
    # dip. THE +90 IS NOT COSMETIC. With -90 the vertex order below yields a
    # mesh normal whose z is -cos(dip), i.e. pointing DOWN, while
    # ``mhs.tdcs``'s frame always orients its normal UP. The assembly uses the
    # frame for the basis projection and the MESH normal for the eigenstress
    # (``assemble._unit_normal``, deliberately un-flipped, because the mesh
    # normal is what says which way the slip jump goes), so the two disagreed
    # and the eigenstress -- the dominant term within a few eps of the element
    # -- was subtracted with the wrong sign. The self-interaction diagonal then
    # came out at +48 instead of -50 on every element of the network, which is
    # exactly what ``kernel.refuse_if_sign_wrong`` is there to catch, and did.
    # With +90 the cross product below gives z = +cos(dip) >= 0, matching the
    # library's own ``dipping_fault`` builder.
    down = np.array([-math.sin(th) * math.cos(dl),
                     math.cos(th) * math.cos(dl),
                     -math.sin(dl)])
    n_s = max(1, int(round(length / h)))
    n_d = max(1, int(round(width / h)))
    u = np.linspace(-length / 2.0, length / 2.0, n_s + 1)
    v = np.linspace(-width / 2.0, width / 2.0, n_d + 1)
    V = (np.asarray(center, float)[None, None, :]
         + u[:, None, None] * strike[None, None, :]
         + v[None, :, None] * down[None, None, :])
    tris = [t for i in range(n_s) for j in range(n_d)
            for t in ([V[i, j], V[i + 1, j], V[i + 1, j + 1]],
                      [V[i, j], V[i + 1, j + 1], V[i, j + 1]])]
    return np.asarray(tris, float)


class _Hash:
    """Uniform spatial hash for the separation floor, position and FACES.

    Cell size equals the largest floor, so a candidate inspects its own cell
    and the 26 neighbours. Written here rather than reaching for a KD-tree
    because mhs declares numpy and numba and nothing else, and a study should
    not widen that.

    TWO FLOORS, AND THE SECOND ONE WAS FOUND BY DIAGNOSIS RATHER THAN GUESSED.
    A centroid floor alone let the interaction matrix go indefinite on a dense
    cloud: ``sym(-P)`` had exactly ONE negative eigenvalue out of 2826
    (-6.7 against a median of +92.7), and its eigenvector put 99.4 % of its
    weight on a single PAIR of elements.

    The first guess was that the pair must be near-COPLANAR and nearly
    coincident, so an orientation floor was added -- and it did not help,
    because the guess was backwards. The offending pair was nearly
    PERPENDICULAR (|n.n'| = 0.0099) at 0.295 h separation: one element's
    collocation point was sitting essentially ON the other's face, where a
    mollified element's traction is largest and where the relative asymmetry
    of the projected operator reached 1.13 against 0.008 for a fault network.

    Coplanar side-by-side elements are FINE -- that is precisely what a fault's
    own discretization is, and it yields no negative eigenvalues. So the floor
    that matters is the one this project rests on everywhere else: a
    collocation point's CLEARANCE FROM ANOTHER ELEMENT'S SURFACE, measured in
    eps. A candidate is rejected when its centroid lies within
    ``MIN_FACE_CLEARANCE_OVER_EPS * eps`` of a neighbour's plane AND inside
    that neighbour's footprint, or vice versa.
    """

    __slots__ = ("cell", "face_clear", "reach", "grid", "_g")

    def __init__(self, cell: float, *, face_clear: float = 0.0,
                 reach: float = 0.0):
        self.cell = float(cell)
        self.face_clear = float(face_clear)
        self.reach = float(reach)
        self._g = max(self.cell, self.reach + self.face_clear)
        self.grid: dict[tuple[int, int, int], list[tuple]] = {}

    def _key(self, p):
        c = self._g
        return (int(p[0] // c), int(p[1] // c), int(p[2] // c))

    def _on_face(self, c, q, nq, tq=None) -> bool:
        """Is point ``c`` inside the eps-band of the element at ``q``/``nq``?

        ``reach`` is the CIRCUMSCRIBED disc, so it is a cheap conservative
        bound and nothing more: its area is ``pi (h/sqrt 3)^2 = 1.047 h^2``
        against the triangle's own ``0.433 h^2``, 2.42 times too much. Given
        ``tq`` the disc only pre-filters and the triangle itself decides.
        Measured on a production cloud, that recovered 45.7 % and 44.3 % of the
        pruning at 3000 and 8000 elements -- 28.5 % falling to 15.5 % -- so
        nearly half of what was being discarded was the approximation rather
        than a real interpenetration.

        The containment test asks that ``c`` fall on the same side of all three
        edges, which is winding independent: a triangle wound against ``nq``
        flips all three signs together, and this study has already paid once
        for assuming a winding (``growth.kink_candidates``).
        """
        dv = c - q
        perp = abs(float(np.dot(dv, nq)))
        if perp >= self.face_clear:
            return False
        inplane = math.sqrt(max(float(np.dot(dv, dv)) - perp * perp, 0.0))
        if inplane >= self.reach:
            return False                 # outside the disc is outside both
        if tq is None:
            return True
        p = c - float(np.dot(c - tq[0], nq)) * nq
        s0 = float(np.dot(np.cross(tq[1] - tq[0], p - tq[0]), nq))
        s1 = float(np.dot(np.cross(tq[2] - tq[1], p - tq[1]), nq))
        s2 = float(np.dot(np.cross(tq[0] - tq[2], p - tq[2]), nq))
        return (s0 >= 0.0 and s1 >= 0.0 and s2 >= 0.0) or \
               (s0 <= 0.0 and s1 <= 0.0 and s2 <= 0.0)

    def keep_mask(self, pts: np.ndarray,
                  normals: np.ndarray | None = None,
                  tris: np.ndarray | None = None) -> np.ndarray:
        """Which of ``pts`` clear everything already held AND each other.

        Greedy in the given order and it ADDS as it goes, so the result is
        deterministic for a given seed and a kept point never has to be undone.
        ``normals`` enables the face-clearance floor; without it only the
        distance floor applies, which is what the fault-network generator wants
        -- its elements are coplanar neighbours by construction and that is the
        configuration the face rule deliberately permits. ``tris`` narrows that
        floor from each element's circumscribed disc to the element, which is
        worth about half the pruning; see :meth:`_on_face`.

        THE ORDER BIASES WHICH ELEMENT OF A CLOSE PAIR SURVIVES, which matters
        for what this generator is used to measure. The exclusion region is
        orientation dependent, so greedy admission by index leaves survivors
        whose mutual orientations are tilted toward large inter-normal angles
        -- face-on stacked pairs, the most strongly coupled ones, are the ones
        preferentially dropped. Any interaction effect measured on a pruned
        cloud is therefore an UNDERESTIMATE, and the remedy is to prune less
        rather than to reweight afterwards.
        """
        c2 = self.cell * self.cell
        use_f = (normals is not None and self.face_clear > 0.0
                 and self.reach > 0.0)
        ok = np.ones(len(pts), bool)
        for i, p in enumerate(pts):
            ni = normals[i] if normals is not None else None
            ti = tris[i] if tris is not None else None
            i0, j0, k0 = self._key(p)
            clash = False
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        for q, nq, tq in self.grid.get((i0 + di, j0 + dj,
                                                        k0 + dk), ()):
                            if float(np.sum((p - q) ** 2)) < c2:
                                clash = True
                                break
                            if use_f and (self._on_face(p, q, nq, tq)
                                          or self._on_face(q, p, ni, ti)):
                                clash = True
                                break
                        if clash:
                            break
                    if clash:
                        break
                if clash:
                    break
            if clash:
                ok[i] = False
            else:
                self.grid.setdefault((i0, j0, k0), []).append((p, ni, ti))
        return ok


def disc_mesh(a: float, h: float, *, depth: float = -20.0,
              normal=(0.0, 0.0, 1.0)) -> np.ndarray:
    """A TILING triangulation of a disc of radius ``a``: the crack primitive.

    Equilateral lattice, up- and down-pointing triangles sharing edges, keeping
    those whose centroid falls inside ``a``. Elements TILE rather than merely
    sit side by side, and that is the point: the coplanarity sweep measured a
    perforated sheet of disconnected patches to be a far stiffer object than a
    continuous crack -- compliance comes from contiguity, coherence from
    coplanarity, and a crack needs both.

    WHY A DISC. It is one of the few 3-D crack geometries with a closed-form
    solution, so it is the validation target rather than a production
    geometry. Under uniform shear with no friction the slip is elliptical,
    ``delta(r) = delta_0 sqrt(1 - r^2/a^2)`` with
    ``delta_0 = 8(1-nu)/(pi(2-nu)) d_tau a / mu`` (Eshelby 1957) -- the
    familiar ``24/(7 pi)`` being its value at ``nu = 1/4`` and nothing more
    general. ``growth.penny`` states that coefficient, the energy and the
    front anisotropy in one place; the mollified solve reproduces the slip
    one to 0.6-2 % depending on ``eps``.

    ``depth`` places the crack deep enough that the free surface is weak; the
    half-space kernel has no full-space mode, so "infinite medium" is
    approximated rather than exact and the depth is reported.
    """
    dy = h * math.sqrt(3.0) / 2.0
    nx = int(math.ceil(a / h)) + 2
    ny = int(math.ceil(a / dy)) + 2
    tri = []
    for j in range(-ny, ny + 1):
        y0 = j * dy
        off = 0.5 * h * (j & 1)
        for i in range(-nx, nx + 1):
            x0 = i * h + off
            up = [(x0, y0), (x0 + h, y0), (x0 + 0.5 * h, y0 + dy)]
            dn = [(x0 + h, y0), (x0 + 1.5 * h, y0 + dy),
                  (x0 + 0.5 * h, y0 + dy)]
            for t in (up, dn):
                c = np.mean(t, axis=0)
                if c[0] ** 2 + c[1] ** 2 <= a * a:
                    tri.append(t)
    out = np.zeros((len(tri), 3, 3))
    out[:, :, :2] = np.asarray(tri)
    out[:, :, 2] = depth

    n_hat = np.asarray(normal, float)
    n_hat /= np.linalg.norm(n_hat)
    if abs(float(n_hat @ np.array([0.0, 0.0, 1.0])) - 1.0) > 1e-12:
        # Rotate the flat disc onto the requested normal about the axis
        # perpendicular to both, which is the minimal rotation.
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(z, n_hat)
        c = float(z @ n_hat)
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]],
                       [-v[1], v[0], 0.0]])
        R = np.eye(3) + vx + vx @ vx / (1.0 + c)
        mid = out.mean(axis=(0, 1))
        out = (out - mid) @ R.T + mid
    return out


def build_aligned(*, seed: int, n_elements: int = 5000, h: float = 0.5,
                  n_planes: int = 20, scatter_over_h: float = 0.0,
                  tilt_deg: float = 0.0, spacing_over_h: float = 1.5,
                  eps_over_h: float = 0.1, box_xy: float = 12.0,
                  z_top: float = -2.0, z_bottom: float = -14.0,
                  vertical: bool = False,
                  band_dip_deg: float = 85.0) -> Network:
    """Equal-size elements on ``n_planes`` NEAR-coplanar bands.

    THE ONE EXPERIMENT THAT SEPARATES COPLANARITY FROM SIZE. Every element is
    the same equilateral triangle, so the size hierarchy that made the
    power-law network's soft modes -- which turned out to be nothing but crack
    compliance of its biggest fault -- is absent by construction. What varies
    is how nearly coplanar the elements on a band are:

        scatter_over_h = 0    a perfect plane; the band IS a fault
        scatter_over_h ~ 1    elements displaced perpendicular by an element
        scatter_over_h >> 1   a cloud, with the bands no longer meaningful

    WHY PERPENDICULAR SCATTER IS THE RIGHT KNOB. The stress-transfer lobes are
    the reason: two identically-oriented patches PROMOTE each other when the
    offset is in-plane (+0.001 to +0.041 measured) and SHADOW each other when
    it is off-plane (-0.023 one element away). The shadow subtends far more
    solid angle than the promoting lobe, which is why a random arrangement is
    net shadowing -- ``measure.promoting_share`` reads 0.42, i.e. 58 % of the
    coupling weight suppresses rather than promotes. Displacing elements
    perpendicular to their band moves them out of each other's promoting lobe
    and into the shadow WITHOUT changing their orientations, their size, their
    number or their density, which is what makes this a controlled test rather
    than a comparison of two different networks.

    ``tilt_deg`` adds orientation scatter about each element's own band normal,
    for the complementary sweep: coplanar positions with mis-aligned faces.
    """
    rng = np.random.default_rng(seed)
    eps = eps_over_h * h
    per = max(1, int(round(n_elements / max(1, n_planes))))

    # Band geometry. ``vertical`` makes every band a VERTICAL plane centred at
    # mid-depth, which exists because the isotropic version has two defects at
    # low band counts: a steeply inclined band of this extent pokes through the
    # free surface and is clipped, biasing the geometry by orientation, and
    # randomly placed bands INTERSECT. Measured on 20 isotropic bands in a
    # 12 km box, 28 % of elements sat within one element width of a DIFFERENT
    # band at a median |n.n'| of 0.526 -- the near-perpendicular close contact
    # the face-clearance floor exists to prune, and dropping that floor to keep
    # element counts comparable made sym(-P) indefinite at every scatter.
    # A single vertical band has no cross-band contact at all.
    half = 0.5 * box_xy
    if vertical:
        # NEARLY vertical, never EXACTLY. mhs.tdcs flips normals to the upper
        # hemisphere, so at n_z = 0 the choice is decided by float noise: a
        # band built with n_z exactly 0 came back with 183 of 580 mesh normals
        # ANTI-ALIGNED with their own vertex order, and the self-interaction
        # diagonal went positive on exactly those -- `refuse_if_sign_wrong`
        # caught it. ``build_network`` escapes this only because its dip is
        # uniform on [30, 90] and exactly 90 has measure zero. A few degrees
        # off vertical is physically the same fault and numerically unambiguous.
        az = rng.uniform(0.0, 2.0 * np.pi, n_planes)
        t = math.radians(90.0 - band_dip_deg)
        bn = np.column_stack([np.cos(az) * math.cos(t),
                              np.sin(az) * math.cos(t),
                              np.full(n_planes, math.sin(t))])
        bc = np.column_stack([rng.uniform(-half, half, n_planes),
                              rng.uniform(-half, half, n_planes),
                              np.full(n_planes, 0.5 * (z_top + z_bottom))])
    else:
        bn = isotropic_normals(rng, n_planes)
        bc = np.column_stack([rng.uniform(-half, half, n_planes),
                              rng.uniform(-half, half, n_planes),
                              rng.uniform(z_bottom, z_top, n_planes)])

    cen, nrm, fid = [], [], []
    s = spacing_over_h * h
    side = math.sqrt(per) * s                 # in-plane extent of one band
    for b in range(n_planes):
        n_hat = bn[b]
        # An orthonormal in-plane frame. The seed vector is chosen away from
        # n_hat so the cross product is well conditioned.
        seed_v = np.array([0.0, 0.0, 1.0])
        if abs(float(n_hat @ seed_v)) > 0.9:
            seed_v = np.array([1.0, 0.0, 0.0])
        u = np.cross(n_hat, seed_v)
        u /= np.linalg.norm(u)
        v = np.cross(n_hat, u)

        # A REGULAR IN-PLANE LATTICE, not a Poisson scatter. A fault's own
        # discretization is a TILING: elements sit side by side and do not
        # overlap. Poisson in-plane positions put some pairs 0.25 h apart --
        # two 0.5 km triangles 0.125 km apart interpenetrate -- and that made
        # sym(-P) indefinite at exactly the coplanar end this experiment is
        # about, which would have been read as "coplanar arrangements are
        # unstable" when it was only "my point process overlaps them".
        side_n = int(math.ceil(math.sqrt(per)))
        gi, gj = np.meshgrid(np.arange(side_n), np.arange(side_n),
                             indexing="ij")
        a = (gi.ravel()[:per] - 0.5 * (side_n - 1)) * s
        c = (gj.ravel()[:per] - 0.5 * (side_n - 1)) * s
        off = rng.normal(0.0, scatter_over_h * h, per)      # PERPENDICULAR
        pts = bc[b] + a[:, None] * u + c[:, None] * v + off[:, None] * n_hat
        cen.append(pts)

        if tilt_deg > 0.0:
            # Small random rotation of each face away from the band normal.
            d = rng.normal(0.0, math.radians(tilt_deg), (per, 3))
            nn = n_hat + d
            nn /= np.linalg.norm(nn, axis=1, keepdims=True)
        else:
            nn = np.repeat(n_hat[None, :], per, axis=0)
        nrm.append(nn)
        fid.append(np.full(per, b))

    cen = np.concatenate(cen)
    nrm = np.concatenate(nrm)
    fid = np.concatenate(fid)
    nrm[nrm[:, 2] < 0.0] *= -1.0               # upper hemisphere, as elsewhere

    keep = cen[:, 2] < z_top - h                # stay below the free surface
    cen, nrm, fid = cen[keep], nrm[keep], fid[keep]

    tris = oriented_triangle(cen, nrm, h)
    # Centroids and normals, as the cloud builder does: the face-clearance
    # floor needs the normals, and it targets NEARLY PERPENDICULAR contacts --
    # coplanar side-by-side neighbours are exactly what a fault's own
    # discretization is and are not pruned. Pruning is reported per
    # configuration because it must not drift across the scatter sweep, or the
    # comparison would be between different element counts rather than
    # different arrangements.
    # DISTANCE FLOOR ONLY, no face clearance -- the same choice
    # ``build_network`` makes, and for the same reason. The face floor was
    # diagnosed on a NEARLY PERPENDICULAR pair (|n.n'| = 0.0099) whose
    # collocation point sat on the other's face; coplanar side-by-side
    # neighbours are what a fault's own discretization IS and are not the
    # pathology. Applying it here would prune 42.7 % of the perfectly coplanar
    # configuration against 5.9 % of the scattered one, so the sweep would
    # compare different element counts rather than different arrangements --
    # measured, which is why this is not a judgement call. Positive
    # definiteness is therefore verified per configuration instead of assumed.
    hsh = _Hash(MIN_SEP_OVER_EPS * eps)
    keep = hsh.keep_mask(cen)
    pruned = 1.0 - keep.mean()
    tris, fid = np.ascontiguousarray(tris[keep]), fid[keep]

    n = tris.shape[0]
    used = np.unique(fid)
    remap = np.zeros(int(fid.max()) + 1, np.int64)
    remap[used] = np.arange(used.size)
    fid = remap[fid]
    meta = {"kind": "aligned", "n_planes": int(used.size),
            "vertical": bool(vertical),
            "band_dip_deg": float(band_dip_deg),
            "scatter_over_h": float(scatter_over_h),
            "tilt_deg": float(tilt_deg), "box_xy_km": float(box_xy),
            "spacing_over_h": float(spacing_over_h),
            "pruned_fraction": float(pruned), "h_km": float(h),
            "elements_per_plane": float(n / max(1, used.size)),
            "cascade_cell_km": None, "d_geom_known": None}
    return Network(tris=tris, fault_id=fid,
                   length=np.full(used.size, side),
                   width=np.full(used.size, side),
                   azimuth=np.zeros(used.size), dip=np.zeros(used.size),
                   center=bc[used], h=h, eps=eps, meta=meta)


def build_cloud(*, seed: int, n_elements: int = 20000, h: float = 0.5,
                spacing_over_h: float = 1.5, eps_over_h: float = 0.1,
                centers: str = "uniform", cascade: tuple[int, int, int] = (3, 4, 3),
                z_top: float = -2.0, spread_deg: float | None = None,
                mean_normal=(0.0, 0.0, 1.0)) -> Network:
    """A DENSE CLOUD of independently-oriented single-triangle slip patches.

    The element IS the fracture here: one equilateral triangle of edge ``h``,
    its own isotropic orientation, no fault hierarchy and therefore NO
    INHERITED COHERENCE. Any alignment among the elements that end up slipping
    has to be produced by the elastic interaction, which is the entire point --
    the fault-network generator above imposes within-fault coherence by
    construction, and measured only 0.1-2.0 % of its coupling budget crossing
    between faults, so nothing there could organize.

    SPACING IS THE CRITICAL PARAMETER AND IT IS CUBIC. Measured on two single
    triangles with random relative orientations, the coupling is
    ``|K_ij|/|K_ii| ~ 0.10 (h/r)^3`` with the coefficient stable from r/h = 1.5
    up. Summing over a cloud at spacing ``s`` and keeping the triggering half
    gives a branching ratio ``B ~ 0.6 (h/s)^3 ln(L/s)``:

        s = 1.2 h   B ~ 1.0  (critical)      s = 2.0 h   B ~ 0.21
        s = 1.5 h   B ~ 0.54                 s = 3.0 h   B ~ 0.10

    A factor of two in spacing is a factor of EIGHT in coupling, so the
    production value is calibrated by measuring B rather than chosen. The box
    size follows from ``n_elements`` and the spacing, not the other way round.

    ``spacing_over_h`` IS A DENSITY, NOT A NEIGHBOUR DISTANCE, and the two
    differ by a factor that matters here. For a Poisson cloud the mean
    nearest-neighbour distance is ``0.554 * s``, so ``spacing_over_h = 1.5``
    puts the typical nearest neighbour at ``0.86 h`` -- CLOSER than an element's
    own edge, where the coupling is far above what ``s`` alone would suggest.
    The realized nearest-neighbour distribution is therefore measured and
    reported in ``meta``, and the calibration reads B rather than ``s``.

    ``centers="uniform"`` is the control: a homogeneous background, so any
    localization is unambiguously mechanical, and -- because orientation is
    drawn independently of position -- the no-interaction null is spatially
    unstructured by construction. ``centers="cascade"`` puts the same elements
    on a multiplicative cascade and asks whether mechanics amplifies, ignores
    or overrides pre-existing geometric clustering; it needs the isotropic cube
    the existing guard enforces, so the element count is fixed by the cascade
    at ``keep**(3*level)``.
    """
    rng = np.random.default_rng(seed)
    if centers not in ("uniform", "cascade"):
        raise ValueError(f"centers must be 'uniform' or 'cascade' for a cloud, "
                         f"got {centers!r}")
    eps = eps_over_h * h
    s = spacing_over_h * h
    reach = h / math.sqrt(3.0)          # circumradius: the element's half-extent

    if centers == "cascade":
        keep, base, level = cascade
        n_elements = keep ** (3 * level)
        box = s * float(base ** level)
    else:
        box = s * float(n_elements) ** (1.0 / 3.0)
    z_bottom = z_top - box
    # Centres are clipped into [z_bottom + reach, z_top - reach] below, so no
    # vertex can pass z_top < 0 whatever h is. What CAN go wrong is a band too
    # thin to hold one element, and that is what this refuses. (An earlier
    # version tested `z_top - reach >= 0`, which is the wrong inequality
    # entirely and never fired; the clipping was doing the work.)
    if box <= 2.0 * reach:
        raise ValueError(
            f"the depth band is {box:g} km and one element of edge {h:g} km "
            f"spans {2 * reach:g} km, so no element fits. Raise n_elements or "
            f"spacing_over_h, or lower h.")

    half = box / 2.0
    if centers == "cascade":
        corners, cell = cascade_cells(rng, *cascade, box)
        cen = corners + cell / 2.0
        cen[:, 0] -= half
        cen[:, 1] -= half
        cen[:, 2] = z_top - cen[:, 2]
    else:
        cell = None
        cen = np.column_stack([
            rng.uniform(-half, half, n_elements),
            rng.uniform(-half, half, n_elements),
            rng.uniform(z_bottom, z_top, n_elements)])
    # Keep every vertex strictly below the free surface and inside the band.
    cen[:, 2] = np.clip(cen[:, 2], z_bottom + reach, z_top - reach)

    nrm = (isotropic_normals(rng, cen.shape[0]) if spread_deg is None else
           subparallel_normals(rng, cen.shape[0], mean_normal, spread_deg))
    tris = oriented_triangle(cen, nrm, h)

    # The separation floor still applies, and at s = 1.5 h with eps = 0.1 h it
    # is 0.25 h -- six times below the spacing -- so pruning should be ~zero.
    # It is measured rather than assumed.
    hsh = _Hash(MIN_SEP_OVER_EPS * eps,
                face_clear=MIN_FACE_CLEARANCE_OVER_EPS * eps, reach=reach)
    keep_mask = hsh.keep_mask(tris.mean(axis=1), nrm, tris)
    n_pruned = int((~keep_mask).sum())
    tris = np.ascontiguousarray(tris[keep_mask])
    nrm = nrm[keep_mask]
    cen = cen[keep_mask]
    n = tris.shape[0]
    if not np.all(tris[:, :, 2] < 0.0):
        raise RuntimeError(
            "a vertex landed at or above z = 0, which mhs refuses; the depth "
            "band is clipped per element above, so this is a bug.")

    # Per-"fault" arrays degenerate to per-element: one triangle each. They are
    # filled with the element's own geometry rather than left stale, because
    # downstream code reads them.
    dip = np.degrees(np.arccos(np.clip(np.abs(nrm[:, 2]), 0.0, 1.0)))
    azi = np.degrees(np.arctan2(nrm[:, 0], -nrm[:, 1])) % 360.0

    cent = tris.mean(axis=1)
    d2 = ((cent[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2) if n <= 6000 \
        else None
    if d2 is not None:
        np.fill_diagonal(d2, np.inf)
        d_nn = np.sqrt(d2.min(axis=1))
        nn = {"mean": float(d_nn.mean()), "min": float(d_nn.min()),
              "median": float(np.median(d_nn))}
    else:
        nn = None

    area_tot = float(0.5 * np.linalg.norm(
        np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]),
        axis=1).sum())
    S = fabric(nrm)
    meta = {
        "kind": "cloud", "seed": seed, "h_km": h, "eps_km": eps,
        "eps_over_h": eps_over_h, "spacing_over_h": spacing_over_h,
        "spacing_km": s, "box_xy_km": box, "z_top_km": z_top,
        "z_bottom_km": z_bottom, "centers": centers,
        "cascade": list(cascade) if centers == "cascade" else None,
        "cascade_cell_km": cell,
        "d_geom_known": (cascade_dimension(cascade[0], cascade[1])
                         if centers == "cascade" else 3.0),
        "n_elements": n, "n_faults": n, "pruned": n_pruned,
        "pruned_fraction": n_pruned / max(1, n + n_pruned),
        "min_sep_over_eps": MIN_SEP_OVER_EPS, "min_sep_km": MIN_SEP_OVER_EPS * eps,
        "face_clearance_over_eps": MIN_FACE_CLEARANCE_OVER_EPS,
        "face_clearance_km": MIN_FACE_CLEARANCE_OVER_EPS * eps,
        "total_area_km2": area_tot,
        "P32_per_km": area_tot / (box ** 3),
        "nearest_neighbour_km": nn,
        "fabric_S": [float(v) for v in S],
        "fabric_anisotropy": float(abs(S[0] - 1.0 / 3.0)),
        "window_r_lo_km": 2.0 * h, "window_r_hi_km": box / 6.0,
        "units": "km",
    }
    return Network(tris=tris, fault_id=np.arange(n, dtype=np.int64),
                   length=np.full(n, h), width=np.full(n, h),
                   azimuth=azi, dip=dip, center=cen, h=h, eps=eps, meta=meta)


def build_network(*, seed: int, n_elements_target: int, a: float = 1.5,
                  l_min: float = 2.0, l_max: float = 40.0,
                  aspect: float = 2.0, h: float = 1.0,
                  eps_over_h: float = 0.1,
                  box_xy: float = 100.0, z_top: float = -2.0,
                  z_bottom: float = -42.0,
                  dip_range: tuple[float, float] = (30.0, 90.0),
                  centers: str = "cascade", alpha: float = 2.2,
                  cascade: tuple[int, int, int] = (2, 3, 4)) -> Network:
    """A fault network with roughly ``n_elements_target`` elements.

    Faults are taken in POOL ORDER -- which is random with respect to size, and
    deliberately not sorted (see :func:`expected_elements_per_fault`) -- meshed
    at the common ``h``, and pruned element-by-element against the separation
    floor. The loop stops when the budget is reached, so the count overshoots by
    at most the last fault's size.

    The orientation distribution is a deliberate INPUT, not a detail: a fault's
    chance of failing depends on how its plane sits in the regional load, so
    ``dip_range`` alone can concentrate slip. That is what the no-interaction
    null is for, and why it is measured before any cascade runs.
    """
    rng = np.random.default_rng(seed)
    if centers not in ("cascade", "levy", "uniform"):
        raise ValueError(f"centers must be 'cascade', 'levy' or 'uniform', got {centers!r}")
    if z_top >= 0.0:
        raise ValueError(
            f"z_top must be below the free surface; mhs refuses a source at "
            f"z > 0. Got {z_top}.")
    eps = eps_over_h * h

    half = box_xy / 2.0
    box = np.array([[-half, -half, z_bottom], [half, half, z_top]])

    cell_size = None
    if centers == "cascade":
        keep, base, level = cascade
        depth = z_top - z_bottom
        if abs(depth - box_xy) > 1e-9 * box_xy:
            raise ValueError(
                f"a cascade needs an ISOTROPIC domain: the box is "
                f"{box_xy:g} km across and {depth:g} km deep. Squashing one "
                f"axis changes what cubic boxes measure, so the dimension "
                f"would no longer be 3 log(keep)/log(base) and the one "
                f"construction here with a known answer would stop having "
                f"one. Set z_bottom = z_top - box_xy.")
        corners, cell_size = cascade_cells(rng, keep, base, level, box_xy)
        # cell CORNERS in [0, box_xy)^3 -> cell CENTRES in the real domain
        cen = corners + cell_size / 2.0
        cen[:, 0] += -box_xy / 2.0
        cen[:, 1] += -box_xy / 2.0
        cen[:, 2] = z_top - cen[:, 2]
        pool = cen.shape[0]
        # A fault must fit its cell, or the arrangement stops being what sets
        # the dimension: l_max above the cell size would put the window's lower
        # bound above the cascade's own finest level.
        l_max = min(l_max, cell_size)
        l_min = min(l_min, 0.5 * l_max)
        lengths = pareto_lengths(rng, pool, a, l_min, l_max)
        per_fault = expected_elements_per_fault(a, l_min, l_max, aspect, h)
    else:
        # Size the pool from the budget arithmetic rather than a magic number,
        # with a factor of 4 for pruning and for faults too tall for the band.
        per_fault = expected_elements_per_fault(a, l_min, l_max, aspect, h)
        pool = max(256, int(4 * n_elements_target / max(1.0, per_fault)))
        lengths = pareto_lengths(rng, pool, a, l_min, l_max)
    dips = rng.uniform(dip_range[0], dip_range[1], pool)
    azis = rng.uniform(0.0, 360.0, pool)
    if centers == "levy":
        cen = levy_dust(rng, pool, alpha, box, step_min=h,
                        step_max=float(box_xy))
    elif centers == "uniform":
        cen = box[0] + rng.random((pool, 3)) * (box[1] - box[0])

    hsh = _Hash(MIN_SEP_OVER_EPS * eps)
    tris_all, ids, keep, placed = [], [], [], []
    n_elem = 0
    n_pruned = 0
    n_seen = 0
    n_too_tall = 0
    for k in range(pool):
        if n_elem >= n_elements_target:
            break
        n_seen += 1
        L = float(lengths[k])
        W = L / aspect
        dip = float(dips[k])
        half_v = 0.5 * W * math.sin(math.radians(dip))
        zc_hi, zc_lo = z_top - half_v, z_bottom + half_v
        if zc_hi <= zc_lo:
            n_too_tall += 1
            continue
        c = cen[k].copy()
        if centers == "cascade":
            # A cascade centre may NOT be clamped: moving it would move the
            # arrangement whose dimension is the known quantity. A fault
            # that would breach the band is dropped instead.
            if not (zc_lo <= c[2] <= zc_hi):
                n_too_tall += 1
                continue
        else:
            c[2] = min(max(c[2], zc_lo), zc_hi)
        patch = rect_patch(c, azis[k], dip, L, W, h)
        mask = hsh.keep_mask(patch.mean(axis=1))
        if int(mask.sum()) < MIN_ELEMENTS_PER_FAULT:
            continue
        n_pruned += int((~mask).sum())
        patch = patch[mask]
        tris_all.append(patch)
        ids.append(np.full(patch.shape[0], len(keep), dtype=np.int64))
        keep.append(k)
        placed.append(c)          # the CLAMPED center, which is what was built
        n_elem += patch.shape[0]

    if not tris_all:
        raise RuntimeError(
            f"no fault could be placed: the depth band "
            f"[{z_bottom}, {z_top}] km cannot hold a fault of width "
            f"{l_min / aspect:g} km at dip {dip_range[1]:g} deg "
            f"({n_too_tall} of {n_seen} candidates were too tall). Widen the "
            f"band, lower l_min, or lower the maximum dip.")

    tris = np.ascontiguousarray(np.concatenate(tris_all, axis=0))
    if not np.all(tris[:, :, 2] < 0.0):
        raise RuntimeError(
            "a vertex landed at or above z = 0, which mhs refuses. The depth "
            "band is enforced per fault above, so this is a placement bug "
            "rather than a parameter choice.")
    keep = np.asarray(keep, dtype=np.int64)
    lk = lengths[keep]
    meta = {
        "seed": seed, "a_nominal": a, "a_realized": pareto_mle(lk, l_min, l_max),
        "l_min_km": l_min, "l_max_km": l_max, "aspect": aspect,
        "h_km": h, "eps_km": eps, "eps_over_h": eps_over_h,
        "box_xy_km": box_xy, "z_top_km": z_top, "z_bottom_km": z_bottom,
        "dip_range_deg": list(dip_range), "centers": centers,
        "alpha": alpha if centers == "levy" else None,
        "cascade": list(cascade) if centers == "cascade" else None,
        "cascade_cell_km": cell_size,
        "d_geom_known": (cascade_dimension(cascade[0], cascade[1])
                         if centers == "cascade" else None),
        "window_r_lo_km": l_max, "window_r_hi_km": box_xy / 6.0,
        "n_faults": int(keep.size), "n_elements": int(tris.shape[0]),
        "n_elements_target": n_elements_target,
        "n_candidates_seen": n_seen, "n_too_tall": n_too_tall,
        "expected_elements_per_fault": per_fault,
        "pruned_fraction": n_pruned / max(1, n_pruned + int(tris.shape[0])),
        "min_sep_over_eps": MIN_SEP_OVER_EPS,
        "min_sep_km": MIN_SEP_OVER_EPS * eps,
        "total_area_km2": float(
            0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0],
                                          tris[:, 2] - tris[:, 0]),
                                 axis=1).sum()),
        "units": "km",
    }
    return Network(tris=tris, fault_id=np.concatenate(ids), length=lk,
                   width=lk / aspect, azimuth=azis[keep], dip=dips[keep],
                   center=np.asarray(placed, float), h=h, eps=eps, meta=meta)
