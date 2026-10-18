"""The folded image of a source whose smeared slip reaches above z = 0.

THE PROBLEM. The half-space kernel is Mindlin's, with the image "lifted" to
the smeared source (:mod:`mhs.kernels.image`): right to O(eps^2) for a BURIED
source. A source within ~3 eps of the free surface has part of its smeared
slip above z = 0, and continued there the lifted formula puts Mindlin's image
point and Boussinesq branch line INSIDE the body: O(1/eps) within eps of the
trace. On a surface-breaking fault with continuous quadratic slip that error
is not local -- it moved the whole fault's moment by 10-20 % and left the
collocation form's Coulomb solve divergent -- while a P0 centroid, 2.5 eps
down, feels it only as O(eps/depth).

THE FIX. The regularised half-space problem must take as its body force the
smeared slip FOLDED back into the body: the part above z = 0 replaced by its
mirror image. That is mass-preserving (a constant slip stays constant up to
the trace, where CLIPPING would halve it), it is the even extension, which is
exact in antiplane strain, and it adds no kink, because the exact slip of a
surface-breaking crack has zero gradient at the surface. The folded source is
the element plus its mirror element with reflected slip, restricted to the
body, and an identity does the rest: the Kelvin part of the half-space
response to the folded density plus the reflected-Kelvin part of its image
equals the FULL-SPACE field of element + mirror element over all of space --
both clq closed forms. What is left is the Boussinesq correction: by symmetry
the pair leaves zero shear traction on z = 0 but a normal traction ``s`` (it
imposes a roller surface, and for dip slip the kink between fault and mirror
is a line of vertical Burgers vector along the trace), so the half-space
field is

    K_fold = clq(e) + clq(mirror e) + Boussinesq[s],   s = sigma_zz of the pair on z = 0,

with ``s`` TWICE the element's own elastic sigma_zz at the surface (the mirror
contributes the same zz by symmetry), evaluated from the closed forms. Zero
in antiplane strain; O(1) for dip slip.

THE BOUSSINESQ TERM is a surface integral of ``s`` against the classical
point-load stress (and displacement) for the body z <= 0, done per receiver on
a polar grid centred under it, with the load subtracted at the receiver's
foot point so that only a smooth integrand is quadratured:

    Sigma_c(x) = s(x_h) DISK(R1, zeta) + int_{r<R1} Sigma_B(x - x') [s(x') - s(x_h)] dS',

DISK being Love's closed-form on-axis field of a uniform load over a disk of
radius R1 at depth zeta, and R1 far enough out that ``s`` is negligible
beyond it. The angular rule follows the source: spacing eps/4 where the
circle crosses the disk holding the source's sharp surface features (its
smeared crease), 2 eps on the flanks, coarse elsewhere. Every number is in
:mod:`mhs.defaults`.

WHO IS FOLDED, AND WHERE. Sources with a vertex within FOLD_SOURCE_DEPTH_EPS
of the surface, at receivers within FOLD_BAND_DEPTH_EPS of it and within
FOLD_REACH_EPS of the source's crease along the surface; everywhere else the
lift stays, and :func:`image` is what replaces ``image_total`` there. For a
buried element the construction and the lift agree to O(eps^2/depth^2) (the
seam, gated), and restricting the fold to that band changed nothing on the
vertical reference (defaults). The eigenstress of a folded source carries the
mirror blob's weight too, with the mirror's normal and slip sign, which
:func:`mhs.kernels.assemble._eigen_terms` assembles.

SIGNS, measured rather than argued. The mirror triangle keeps the vertex
order, so its normal is ``-Q n`` and the reflected slip ``Q b`` is fed to the
closed forms as ``-Q b``: the slip columns carry ``MIRROR_SLIP_SIGN``. The
point-load kernel takes the offset receiver MINUS load; reversed, sigma_xz and
sigma_yz flip while the other four components do not, and the construction
then matched Mindlin's image in four components and missed by O(1) in the
shears -- which is what the seam gate is for.
"""
from __future__ import annotations

import numpy as np

from .. import defaults as _d
from ..fullspace import kernels as _fs_kernels
from ..fullspace.shape import n_nodes

#: Reflection in the free surface (the body is z <= 0).
REFLECT = np.array([1.0, 1.0, -1.0])
#: The mirror element's slip columns, in its own (vertex-order) convention:
#: the reflected slip Q b across a surface whose stored normal is -Q n.
MIRROR_SLIP_SIGN = np.array([-1.0, -1.0, 1.0])
_ZHAT = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------ who, where ----
def straddles(tri, eps: float) -> bool:
    """True when the source's shallowest vertex is within the fold depth."""
    tri = np.asarray(tri, float)
    return bool(float(-tri[:, 2].max()) < _d.FOLD_SOURCE_DEPTH_EPS * float(eps))


def mirror(tri) -> np.ndarray:
    """The reflected triangle, vertex order kept (normal -Q n)."""
    return np.asarray(tri, float) * REFLECT


def crease(tri, eps: float):
    """``(centre (2,), radius)`` of the surface disk holding the source's sharp
    features: its vertices within the fold depth, projected, plus the margin
    and their depth (a barely buried edge still radiates a 1/rho crease)."""
    tri = np.asarray(tri, float)
    eps = float(eps)
    depth = -tri[:, 2]
    top = tri[depth < _d.FOLD_SOURCE_DEPTH_EPS * eps]
    if top.shape[0] == 0:                       # not a straddling source
        top = tri[depth == depth.min()]
    centre = top[:, :2].mean(axis=0)
    spread = float(np.linalg.norm(top[:, :2] - centre, axis=1).max())
    return centre, spread + _d.FOLD_CREASE_MARGIN_EPS * eps + float((-top[:, 2]).max())


def band(obs, tri, eps: float) -> np.ndarray:
    """Indices of the receivers that take the folded image of this source."""
    obs = np.asarray(obs, float)
    eps = float(eps)
    centre, _ = crease(tri, eps)
    depth = -obs[:, 2]
    reach = np.hypot(obs[:, 0] - centre[0], obs[:, 1] - centre[1])
    return np.flatnonzero((depth < _d.FOLD_BAND_DEPTH_EPS * eps)
                          & (reach < _d.FOLD_REACH_EPS * eps))


def _nu(mu: float, lam: float) -> float:
    return 0.5 * lam / (lam + mu)


def _blocks(obs, tri, mu, lam, eps, want, order):
    return _fs_kernels.nodal_influence(np.asarray(obs, float), np.asarray(tri, float),
                                       order, float(mu), _nu(mu, lam), float(eps),
                                       want=tuple(want))


# --------------------------------------------- the Boussinesq point load ----
def point_stress(dx, dy, zeta, nu: float) -> np.ndarray:
    """``(..., 3, 3)`` stress at depth ``zeta >= 0`` and horizontal offset
    ``(dx, dy)`` = receiver - load, of a unit compressive normal point load on
    the surface of the body z <= 0 (total force 1, pushing into the body).

    Boussinesq's solution reflected into this frame. ``(1 - zeta/rho)/r^2`` is
    written ``1/(rho (rho + zeta))`` so the axis is regular; the surface
    limit is sigma_zz = sigma_rz = 0 off the load and
    sigma_r = -sigma_theta = (1 - 2 nu) / (2 pi r^2).
    """
    dx = np.asarray(dx, float)
    dy = np.asarray(dy, float)
    zeta = np.broadcast_to(np.asarray(zeta, float), dx.shape)
    r2 = dx * dx + dy * dy
    rho = np.sqrt(r2 + zeta * zeta)
    rho3 = rho ** 3
    rho5 = rho3 * rho * rho
    a = 1.0 / (rho * (rho + zeta))
    c = 1.0 - 2.0 * nu
    s_th = c * (-a + zeta / rho3)                           # sigma_theta * 2 pi
    s_r_th = 2.0 * c * a - 3.0 * zeta * r2 / rho5 - c * zeta / rho3
    r = np.sqrt(r2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ex = np.where(r > 0, dx / r, 0.0)
        ey = np.where(r > 0, dy / r, 0.0)
    pref = 1.0 / (2.0 * np.pi)
    S = np.empty(dx.shape + (3, 3))
    S[..., 0, 0] = pref * (s_th + s_r_th * ex * ex)
    S[..., 1, 1] = pref * (s_th + s_r_th * ey * ey)
    S[..., 0, 1] = S[..., 1, 0] = pref * s_r_th * ex * ey
    t_rz = pref * 3.0 * r * zeta * zeta / rho5
    S[..., 0, 2] = S[..., 2, 0] = t_rz * ex
    S[..., 1, 2] = S[..., 2, 1] = t_rz * ey
    S[..., 2, 2] = -pref * 3.0 * zeta ** 3 / rho5
    return S


def point_disp(dx, dy, zeta, mu: float, nu: float) -> np.ndarray:
    """``(..., 3)`` displacement of the same load; ``u_z < 0`` under it."""
    dx = np.asarray(dx, float)
    dy = np.asarray(dy, float)
    zeta = np.broadcast_to(np.asarray(zeta, float), dx.shape)
    r2 = dx * dx + dy * dy
    rho = np.sqrt(r2 + zeta * zeta)
    pref = 1.0 / (4.0 * np.pi * mu)
    ur_over_r = pref * (zeta / rho ** 3 - (1.0 - 2.0 * nu) / (rho * (rho + zeta)))
    u = np.empty(dx.shape + (3,))
    u[..., 0] = ur_over_r * dx
    u[..., 1] = ur_over_r * dy
    u[..., 2] = -pref * (2.0 * (1.0 - nu) / rho + zeta * zeta / rho ** 3)
    return u


def disk_stress(a: float, zeta, nu: float) -> np.ndarray:
    """``(..., 3, 3)`` on-axis stress at depth ``zeta`` of unit pressure over
    the disk ``r < a`` (Love); the whole plane as ``a -> inf``:
    ``diag(-(1 + 2 nu)/2, -(1 + 2 nu)/2, -1)`` at every depth."""
    zeta = np.asarray(zeta, float)
    rho = np.sqrt(a * a + zeta * zeta)
    szz = -(1.0 - zeta ** 3 / rho ** 3)
    srr = -0.5 * ((1.0 + 2.0 * nu) - 2.0 * (1.0 + nu) * zeta / rho
                  + zeta ** 3 / rho ** 3)
    S = np.zeros(zeta.shape + (3, 3))
    S[..., 0, 0] = srr
    S[..., 1, 1] = srr
    S[..., 2, 2] = szz
    return S


def disk_disp(a: float, zeta, mu: float, nu: float) -> np.ndarray:
    """``(..., 3)`` on-axis displacement of the same disk load: the point
    kernel integrated over it, ``u_z = -(1/2mu)[2(1-nu)(rho_a - zeta) + zeta
    - zeta^2/rho_a]``; ``(1 - nu) a / mu`` into the body at the surface."""
    zeta = np.asarray(zeta, float)
    rho = np.sqrt(a * a + zeta * zeta)
    u = np.zeros(zeta.shape + (3,))
    u[..., 2] = -(0.5 / mu) * (2.0 * (1.0 - nu) * (rho - zeta) + zeta
                               - zeta * zeta / rho)
    return u


# ------------------------------------------------------ the surface load ----
def surface_load(pts_h, tri, mu: float, lam: float, eps: float,
                 order: int) -> np.ndarray:
    """``s`` at horizontal surface points ``(n, 2)``: the elastic sigma_zz of
    element + mirror on z = 0, which is twice the element's own there, as
    ``(n, K, 3)`` per node and Cartesian slip component."""
    pts_h = np.asarray(pts_h, float)
    n = pts_h.shape[0]
    obs = np.column_stack([pts_h, np.zeros(n)])
    blk = _blocks(obs, tri, mu, lam, eps, ("H", "E"), order)
    H = np.asarray(blk["H"])                                 # (n, K, 3, 3, slip)
    phi = np.asarray(blk["E"])                               # (n, K)
    nrm = _unit_normal(tri)
    # (H - E)_zz per slip m: E_zz[m] = phi (lam n_m + 2 mu n_z delta_mz)
    e_zz = lam * nrm + 2.0 * mu * nrm[2] * _ZHAT
    return 2.0 * (H[:, :, 2, 2, :] - phi[:, :, None] * e_zz[None, None, :])


def _unit_normal(tri) -> np.ndarray:
    v = np.asarray(tri, float)
    n = np.cross(v[1] - v[0], v[2] - v[0])
    return n / np.linalg.norm(n)


# ----------------------------------------------------- the surface grid ----
def _gl(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def _composite(breaks, n):
    u, wu = _gl(n)
    a = np.asarray(breaks[:-1], float)
    b = np.asarray(breaks[1:], float)
    return ((a[:, None] + (b - a)[:, None] * u[None, :]).ravel(),
            ((b - a)[:, None] * wu[None, :]).ravel())


def _halfwidth(ra, rb, rho_s, radius):
    """Angular half-width of the part of the circles ``r in [ra, rb]`` (about
    the receiver) inside the disk of ``radius`` about the crease centre."""
    if rb < rho_s - radius or ra > rho_s + radius:
        return 0.0
    if rho_s <= radius:
        return np.pi
    return float(np.arcsin(min(1.0, radius / rho_s)))


def _angular(theta_c, w_fine, w_med, rb, eps):
    """Composite Gauss on one circle: fine sector, medium flanks, coarse rest."""
    n = _d.FOLD_N_ANGULAR
    if w_fine >= np.pi:
        m = max(8, int(np.ceil(2 * np.pi * rb / (n * _d.FOLD_FINE_SPACING_EPS * eps))))
        return _composite(np.linspace(0, 2 * np.pi, m + 1), n)
    if w_med >= np.pi:
        m = max(8, int(np.ceil(2 * np.pi * rb / (n * _d.FOLD_MEDIUM_SPACING_EPS * eps))))
        return _composite(np.linspace(0, 2 * np.pi, m + 1), n)
    pieces = []
    if w_fine > 0:
        m = max(1, int(np.ceil(2 * w_fine * rb / (n * _d.FOLD_FINE_SPACING_EPS * eps))))
        pieces.append(np.linspace(theta_c - w_fine, theta_c + w_fine, m + 1))
    if w_med > w_fine:
        m = max(1, int(np.ceil((w_med - w_fine) * rb / (n * _d.FOLD_MEDIUM_SPACING_EPS * eps))))
        pieces.insert(0, np.linspace(theta_c - w_med, theta_c - w_fine, m + 1))
        pieces.append(np.linspace(theta_c + w_fine, theta_c + w_med, m + 1))
    start = theta_c + w_med if w_med > 0 else 0.0
    pieces.append(np.linspace(start, start + 2 * np.pi - 2 * w_med,
                              max(1, _d.FOLD_N_COARSE // n) + 1))
    nodes, weights = [], []
    for br in pieces:
        if br.size >= 2 and br[-1] > br[0]:
            t, w = _composite(br, n)
            nodes.append(t)
            weights.append(w)
    return np.concatenate(nodes), np.concatenate(weights)


def polar_grid(xh, zeta: float, centre, r_f: float, eps: float, diam: float):
    """Surface quadrature about the receiver's foot point ``xh``: points
    ``(n, 2)``, weights ``(n,)`` carrying the area measure, and ``R1``."""
    xh = np.asarray(xh, float)
    centre = np.asarray(centre, float)
    rho_s = float(np.hypot(*(centre - xh)))
    theta_c = float(np.arctan2(centre[1] - xh[1], centre[0] - xh[0]))
    r1_target = rho_s + 4 * r_f + _d.FOLD_R1_MARGIN_DIAM * diam
    geo = [0.0]
    if 0 < zeta < 0.5 * eps:
        geo += [z for z in (0.5 * zeta, zeta, 2 * zeta, 4 * zeta) if z < 0.5 * eps]
    r = 0.5 * eps
    while True:
        geo.append(r)
        if r >= r1_target:
            break
        r *= 2.0
    R1 = geo[-1]
    lo_m, hi_m = rho_s - 4 * r_f, rho_s + 4 * r_f
    lo_f, hi_f = rho_s - r_f, rho_s + r_f
    breaks = [g for g in geo if g < lo_m or g > hi_m]
    ann = []
    for a0, b0, width in ((max(0.0, lo_m), max(0.0, lo_f), _d.FOLD_MEDIUM_SUBPANEL_EPS),
                          (max(0.0, lo_f), min(R1, hi_f), _d.FOLD_FINE_SUBPANEL_EPS),
                          (min(R1, hi_f), min(R1, hi_m), _d.FOLD_MEDIUM_SUBPANEL_EPS)):
        if b0 > a0:
            ann += list(np.linspace(a0, b0, max(1, int(np.ceil((b0 - a0) / (width * eps)))) + 1))
    breaks = np.unique(np.concatenate([np.asarray(breaks, float), np.asarray(ann, float)]))
    breaks = breaks[breaks <= R1 + 1e-12 * R1]
    pts, wts = [], []
    for ra, rb in zip(breaks[:-1], breaks[1:]):
        if rb - ra < 1e-9 * eps:
            continue
        rq, wr = _composite([ra, rb], _d.FOLD_N_RADIAL)
        tq, wt = _angular(theta_c, _halfwidth(ra, rb, rho_s, r_f),
                          _halfwidth(ra, rb, rho_s, 4 * r_f), rb, eps)
        R, T = np.meshgrid(rq, tq, indexing="ij")
        pts.append(np.stack([xh[0] + R * np.cos(T), xh[1] + R * np.sin(T)],
                            axis=-1).reshape(-1, 2))
        wts.append((wr[:, None] * wt[None, :] * R).ravel())
    return np.concatenate(pts), np.concatenate(wts), R1


# --------------------------------------------------------- the fold image ----
def boussinesq(obs, tri, mu: float, lam: float, eps: float, order: int,
               want=("H",)) -> dict:
    """The Boussinesq correction of ONE source at every ``obs``: ``U`` as
    ``(N, K, 3, 3)`` [obs, node, i, slip] and ``H`` as ``(N, K, 3, 3, 3)``
    [obs, node, a, b, slip], the vendored engine's layouts."""
    obs = np.asarray(obs, float)
    tri = np.asarray(tri, float)
    mu, lam, eps = float(mu), float(lam), float(eps)
    nu = _nu(mu, lam)
    centre, r_f = crease(tri, eps)
    diam = max(float(np.linalg.norm(tri[i] - tri[j]))
               for i in range(3) for j in range(3))
    K = n_nodes(order)
    s_h = surface_load(obs[:, :2], tri, mu, lam, eps, order)       # (N, K, 3)
    out = {}
    if "U" in want:
        out["U"] = np.empty((obs.shape[0], K, 3, 3))
    if "H" in want:
        out["H"] = np.empty((obs.shape[0], K, 3, 3, 3))
    for o in range(obs.shape[0]):
        xh = obs[o, :2]
        zeta = max(0.0, float(-obs[o, 2]))
        pts, wts, R1 = polar_grid(xh, zeta, centre, r_f, eps, diam)
        s = surface_load(pts, tri, mu, lam, eps, order) - s_h[o][None]
        dx, dy = xh[0] - pts[:, 0], xh[1] - pts[:, 1]              # receiver - load
        if "H" in want:
            SB = point_stress(dx, dy, zeta, nu)
            out["H"][o] = (np.einsum("n,nab,nkm->kabm", wts, SB, s)
                           + s_h[o][:, None, None, :] * disk_stress(R1, zeta, nu)[None, :, :, None])
        if "U" in want:
            UB = point_disp(dx, dy, zeta, mu, nu)
            out["U"][o] = (np.einsum("n,ni,nkm->kim", wts, UB, s)
                           + s_h[o][:, None, :] * disk_disp(R1, zeta, mu, nu)[None, :, None])
    return out


def image(obs, tri, mu: float, lam: float, eps: float, order: int,
          want=("H",)) -> dict:
    """The folded image of ONE source at every ``obs``: the mirror element's
    full-space field (slip columns signed) plus the Boussinesq correction.
    Same keys and layouts as ``image_total``, which it replaces in the band."""
    blk = _blocks(obs, mirror(tri), mu, lam, eps, tuple(want), order)
    corr = boussinesq(obs, tri, mu, lam, eps, order, want)
    out = {}
    for k in want:
        out[k] = np.asarray(blk[k]) * MIRROR_SLIP_SIGN + corr[k]
    return out
