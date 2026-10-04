"""
mindlin_kernels.py
==================

Pointwise mollified Mindlin (1936) half-space Green's function for a
buried point force, derived via the Grodskii-Neuber-Papkovich potential
approach following Apostol (2016, 2017).

Status: Stage 1 complete. All 6 validation tests in
test_mindlin_pointsource.py pass. PDE residual O(eps^2), BC residual
O(eps^2), eps -> 0 convergence at order 2, surface formulas match
Apostol Eqs (31) and (37) to machine precision.

Construction
------------
Apostol (2016) gives explicit closed-form Grodskii potentials b, β such
that the Mindlin half-space displacement is

    u = b - (1 / (4(1-nu))) * grad(r . b + beta)

where r is the observation position vector. We use his Eqs (26), (28),
(35) and verify symbolically (see derive_mindlin.py Phase D) that
T_3k(x_3=0) = 0 to machine precision in the unmollified limit.

Mollification
-------------
- Substitute r_1 -> R_{1,eps} = sqrt(r_1^2 + eps^2) and
  r_2 -> R_{2,eps} = sqrt(r_2^2 + eps^2) in b, β.
- The displacement u from the mollified b, β satisfies the regularized
  Cauchy-Navier equation EXCEPT for the missing Cortez blob term in the
  direct Kelvin part. We add the blob term explicitly:

    G^{eps,HS}_{ij} = u^{Apostol-mollified}_i(F_j)
                    + eps^2 / (8 pi mu R_{1,eps}^3) * delta_{ij}

  This makes the direct part satisfy L G^direct = -delta phi_eps EXACTLY,
  while the half-space correction is mollified to O(eps^2) in both PDE
  residual and boundary traction.

Properties (verified by verify_mindlin_pde_residual.py):

  PDE residual in z<0:
    L_ij G^{eps,HS}_{jk} + delta_ik phi_eps(R_1) = O(eps^2)
    (exact at the source for the direct part with the blob; correction
     terms contribute O(eps^2) since Laplacian of 1/R_eps^n is O(eps^2)
     rather than zero).

  Free-surface traction at z=0:
    T_3k(x_3=0) = 0 exactly in eps=0 limit; O(eps^2) for eps > 0.

  Coordinate convention:
    Half-space z <= 0, free surface z = 0, source at source[2] < 0.
    Force ordering: F = (F_East, F_North, F_Up). Matches cutde, mh.

Reference
---------
Apostol, B. F. (2016), "Elastic Equilibrium of the Half-Space Revisited.
  Mindlin and Boussinesq Problems", J. Elast. 125, 139-148.
"""

import numpy as np
import sympy as sp


# ============================================================
# Build sympy expressions once at module load
# ============================================================

# Symbols
_x, _y, _z, _z0 = sp.symbols("x y z z0", real=True)
_mu_s, _nu_s, _eps_s = sp.symbols("mu nu eps", positive=True, real=True)

# Mollified distances
_r1 = sp.sqrt(_x**2 + _y**2 + (_z - _z0) ** 2 + _eps_s**2)
_r2 = sp.sqrt(_x**2 + _y**2 + (_z + _z0) ** 2 + _eps_s**2)
_abs_z0 = -_z0
_abs_zpz0 = -(_z + _z0)
_Q = _r2 + _abs_zpz0

_K = 1 / (4 * sp.pi * _mu_s)


def _b_beta_vertical():
    """Apostol Eq (26), (28): b and β for F = (0, 0, 1) at (0, 0, z_0)."""
    bx = sp.Integer(0)
    by = sp.Integer(0)
    bz = _K * (1 / _r1 + (3 - 4 * _nu_s) / _r2 + 2 * _z0 * (_z + _z0) / _r2**3)
    I_func = -sp.log(_r2 - (_z + _z0))
    beta = _K * (
        _abs_z0 / _r1
        + (3 - 4 * _nu_s) * _abs_z0 / _r2
        + 4 * (1 - _nu_s) * (1 - 2 * _nu_s) * I_func
    )
    return bx, by, bz, beta


def _b_beta_horizontal_x():
    """Apostol Eq (35): b and β for F = (1, 0, 0) at (0, 0, z_0)."""
    bx = _K * (1 / _r1 + 1 / _r2)
    by = sp.Integer(0)
    bz = 2 * _K * (_abs_z0 / _r2**2 - (1 - 2 * _nu_s) / _Q) * _x / _r2
    beta = 2 * _K * (1 - 2 * _nu_s) * (_abs_z0 / _r2 - (1 - 2 * _nu_s)) * _x / _Q
    return bx, by, bz, beta


def _b_beta_horizontal_y():
    """Apostol Eq (35) with x <-> y."""
    bx = sp.Integer(0)
    by = _K * (1 / _r1 + 1 / _r2)
    bz = 2 * _K * (_abs_z0 / _r2**2 - (1 - 2 * _nu_s) / _Q) * _y / _r2
    beta = 2 * _K * (1 - 2 * _nu_s) * (_abs_z0 / _r2 - (1 - 2 * _nu_s)) * _y / _Q
    return bx, by, bz, beta


def _u_from_b_beta(bx, by, bz, beta):
    """u_i = b_i - (1/(4(1-nu))) * d/dx_i (x*bx + y*by + z*bz + β)."""
    rb_plus_beta = _x * bx + _y * by + _z * bz + beta
    factor = sp.Rational(1) / (4 * (1 - _nu_s))
    u = [bx, by, bz]
    grad = [sp.diff(rb_plus_beta, ax) for ax in (_x, _y, _z)]
    return [u[i] - factor * grad[i] for i in range(3)]


# Compute all 3*3 = 9 components of G^{eps,HS}: u_i from F_j
# Column j=0 -> F_x, j=1 -> F_y, j=2 -> F_z
_G_symbolic = sp.zeros(3, 3)
for _j, _f in enumerate([_b_beta_horizontal_x, _b_beta_horizontal_y, _b_beta_vertical]):
    _bx, _by, _bz, _beta = _f()
    _u = _u_from_b_beta(_bx, _by, _bz, _beta)
    for _i in range(3):
        _G_symbolic[_i, _j] = _u[_i]


# Add Cortez blob to the direct part (i == j), WITH the Kelvin prefactor
# C1 = 1/(16 pi mu (1-nu)):  + C1 * 2(1-nu) eps^2 / R_{1,eps}^3
#                            = eps^2 / (8 pi mu R_{1,eps}^3).
# This is the blob term that makes the direct Kelvin G satisfy
# L G = -delta phi_eps (matches kelvin_d2G in mollified_elastic_kernels).
# Apostol's u^direct already accounts for the (3-4nu)/r_1 and
# d_i d_j/r_1^3 pieces; only the blob term is missing.
_blob_term = _eps_s**2 / (8 * sp.pi * _mu_s * _r1**3)
for _i in range(3):
    _G_symbolic[_i, _i] += _blob_term


# ============================================================
# Correction-only kernel (Apostol b_corr, β_corr — only r2 dependence)
# ============================================================
#
# Apostol's b and β each split as direct (depends only on r1) + correction
# (depends only on r2, plus log term). The correction part has NO
# near-singular behavior when obs is near or on the source: r2 is the
# distance to the IMAGE (which is in the upper half-space z>0), so r2
# remains large for all in-half-space obs/source pairs.
#
# We build the correction-only Apostol kernel symbolically here and
# lambdify it. The full Mindlin kernel can then be assembled as
#
#     G_mindlin_clean = G_kelvin_standard(obs - source) + G_correction(obs, source)
#
# which mirrors the structure of the existing analytical_kelvin_*
# machinery and enables exact hybrid analytical + smooth-quadrature
# integration over triangles, valid even ON the source surface.

# -- Apostol b_corr, β_corr for each force direction ---
# Vertical:
_bx_v_corr = sp.Integer(0)
_by_v_corr = sp.Integer(0)
_bz_v_corr = _K * ((3 - 4 * _nu_s) / _r2 + 2 * _z0 * (_z + _z0) / _r2**3)
_I_v = -sp.log(_r2 - (_z + _z0))
_beta_v_corr = _K * (
    (3 - 4 * _nu_s) * _abs_z0 / _r2
    + 4 * (1 - _nu_s) * (1 - 2 * _nu_s) * _I_v
)
# Horizontal-x (Apostol Eq 35, all terms which only involve r2):
_bx_hx_corr = _K * (1 / _r2)          # (the bx direct K/r1 is excluded)
_by_hx_corr = sp.Integer(0)
_bz_hx_corr = 2 * _K * (_abs_z0 / _r2**2 - (1 - 2 * _nu_s) / _Q) * _x / _r2
_beta_hx_corr = (
    2 * _K * (1 - 2 * _nu_s) * (_abs_z0 / _r2 - (1 - 2 * _nu_s)) * _x / _Q
)
# Horizontal-y (x <-> y swap):
_bx_hy_corr = sp.Integer(0)
_by_hy_corr = _K * (1 / _r2)
_bz_hy_corr = 2 * _K * (_abs_z0 / _r2**2 - (1 - 2 * _nu_s) / _Q) * _y / _r2
_beta_hy_corr = (
    2 * _K * (1 - 2 * _nu_s) * (_abs_z0 / _r2 - (1 - 2 * _nu_s)) * _y / _Q
)

# Lambda-args tuple (used by lambdify for all sympy expressions in this module)
_lam_args = (_x, _y, _z, _z0, _mu_s, _nu_s, _eps_s)

# Compute u_apostol from these (correction-only) b, β.
# Reuse the same formula structure as before: u = b - grad(r·b + β) / (4(1-nu))
_G_corr_symbolic = sp.zeros(3, 3)
for _jcol, (_bx_c, _by_c, _bz_c, _beta_c) in enumerate([
    (_bx_hx_corr, _by_hx_corr, _bz_hx_corr, _beta_hx_corr),
    (_bx_hy_corr, _by_hy_corr, _bz_hy_corr, _beta_hy_corr),
    (_bx_v_corr,  _by_v_corr,  _bz_v_corr,  _beta_v_corr),
]):
    _rb_plus_beta = _x * _bx_c + _y * _by_c + _z * _bz_c + _beta_c
    _factor = sp.Rational(1) / (4 * (1 - _nu_s))
    _u_c = [_bx_c, _by_c, _bz_c]
    _grad = [sp.diff(_rb_plus_beta, _ax) for _ax in (_x, _y, _z)]
    for _i in range(3):
        _G_corr_symbolic[_i, _jcol] = _u_c[_i] - _factor * _grad[_i]

# Lambdify G_correction and its derivatives. These integrands are SMOOTH
# for any obs and source in z <= 0 (no near-singular behavior).
_G_corr_lambdified = sp.lambdify(_lam_args, _G_corr_symbolic, modules="numpy", cse=True)

# d G_corr / d source_m derivatives (each lambdified)
_dGc_dx_sym = sp.zeros(3, 3)
_dGc_dy_sym = sp.zeros(3, 3)
_dGc_dz0_sym = sp.zeros(3, 3)
for _i in range(3):
    for _j in range(3):
        g = _G_corr_symbolic[_i, _j]
        _dGc_dx_sym[_i, _j] = sp.diff(g, _x)
        _dGc_dy_sym[_i, _j] = sp.diff(g, _y)
        _dGc_dz0_sym[_i, _j] = sp.diff(g, _z0)
_dGc_dx_lam = sp.lambdify(_lam_args, _dGc_dx_sym, modules="numpy", cse=True)
_dGc_dy_lam = sp.lambdify(_lam_args, _dGc_dy_sym, modules="numpy", cse=True)
_dGc_dz0_lam = sp.lambdify(_lam_args, _dGc_dz0_sym, modules="numpy", cse=True)

# Mixed d^2 G_corr / (d obs_p * d source_m): 8 distinct second derivatives
_d2Gc_xx_sym  = sp.zeros(3, 3)
_d2Gc_yy_sym  = sp.zeros(3, 3)
_d2Gc_xy_sym  = sp.zeros(3, 3)
_d2Gc_xz_sym  = sp.zeros(3, 3)
_d2Gc_yz_sym  = sp.zeros(3, 3)
_d2Gc_xz0_sym = sp.zeros(3, 3)
_d2Gc_yz0_sym = sp.zeros(3, 3)
_d2Gc_zz0_sym = sp.zeros(3, 3)
for _i in range(3):
    for _j in range(3):
        g = _G_corr_symbolic[_i, _j]
        _d2Gc_xx_sym[_i, _j]  = sp.diff(g, _x, 2)
        _d2Gc_yy_sym[_i, _j]  = sp.diff(g, _y, 2)
        _d2Gc_xy_sym[_i, _j]  = sp.diff(g, _x, _y)
        _d2Gc_xz_sym[_i, _j]  = sp.diff(g, _x, _z)
        _d2Gc_yz_sym[_i, _j]  = sp.diff(g, _y, _z)
        _d2Gc_xz0_sym[_i, _j] = sp.diff(g, _x, _z0)
        _d2Gc_yz0_sym[_i, _j] = sp.diff(g, _y, _z0)
        _d2Gc_zz0_sym[_i, _j] = sp.diff(g, _z, _z0)
_d2Gc_xx_lam  = sp.lambdify(_lam_args, _d2Gc_xx_sym,  modules="numpy", cse=True)
_d2Gc_yy_lam  = sp.lambdify(_lam_args, _d2Gc_yy_sym,  modules="numpy", cse=True)
_d2Gc_xy_lam  = sp.lambdify(_lam_args, _d2Gc_xy_sym,  modules="numpy", cse=True)
_d2Gc_xz_lam  = sp.lambdify(_lam_args, _d2Gc_xz_sym,  modules="numpy", cse=True)
_d2Gc_yz_lam  = sp.lambdify(_lam_args, _d2Gc_yz_sym,  modules="numpy", cse=True)
_d2Gc_xz0_lam = sp.lambdify(_lam_args, _d2Gc_xz0_sym, modules="numpy", cse=True)
_d2Gc_yz0_lam = sp.lambdify(_lam_args, _d2Gc_yz0_sym, modules="numpy", cse=True)
_d2Gc_zz0_lam = sp.lambdify(_lam_args, _d2Gc_zz0_sym, modules="numpy", cse=True)


# Lambdify into a fast NumPy callable. Variables: (x, y, z, z0, mu, nu, eps).
# The lambdified function takes scalar inputs and returns a 3x3 matrix.
_G_lambdified = sp.lambdify(
    (_x, _y, _z, _z0, _mu_s, _nu_s, _eps_s),
    _G_symbolic,
    modules="numpy",
)


# Derivatives of G with respect to SOURCE coordinates (needed for the DD /
# slip-discontinuity kernel via Somigliana). Note:
#   x = obs_x - source_x  =>  d/d(source_x) = -d/dx
#   y = obs_y - source_y  =>  d/d(source_y) = -d/dy
#   z0 = source_z         =>  d/d(source_z) = +d/dz0
# We compute the full (3,3,3) tensor dG[i,j,m] = dG_ij / d(source_m).
_dG_dx = sp.zeros(3, 3)
_dG_dy = sp.zeros(3, 3)
_dG_dz0 = sp.zeros(3, 3)
for _i in range(3):
    for _j in range(3):
        _dG_dx[_i, _j] = sp.diff(_G_symbolic[_i, _j], _x)
        _dG_dy[_i, _j] = sp.diff(_G_symbolic[_i, _j], _y)
        _dG_dz0[_i, _j] = sp.diff(_G_symbolic[_i, _j], _z0)

# Lambdify each component matrix; we'll combine them at runtime to form
# the (3,3,3) source-derivative tensor.
_dG_dx_lam = sp.lambdify(
    (_x, _y, _z, _z0, _mu_s, _nu_s, _eps_s), _dG_dx, modules="numpy"
)
_dG_dy_lam = sp.lambdify(
    (_x, _y, _z, _z0, _mu_s, _nu_s, _eps_s), _dG_dy, modules="numpy"
)
_dG_dz0_lam = sp.lambdify(
    (_x, _y, _z, _z0, _mu_s, _nu_s, _eps_s), _dG_dz0, modules="numpy"
)


# Second derivatives of G needed for strain/stress evaluation.
# We need d^2 G_ij / (d obs_p * d source_m). In sympy variables:
#   obs_x derivative -> d/dx
#   obs_y derivative -> d/dy
#   obs_z derivative -> d/dz
#   source_x derivative -> -d/dx
#   source_y derivative -> -d/dy
#   source_z derivative -> +d/dz0
# Mixed second derivatives reduce to 8 distinct SymPy quantities:
_d2G_xx   = sp.zeros(3, 3)  # d^2 / dx^2
_d2G_yy   = sp.zeros(3, 3)  # d^2 / dy^2
_d2G_xy   = sp.zeros(3, 3)  # d^2 / (dx dy)
_d2G_xz   = sp.zeros(3, 3)  # d^2 / (dx dz)
_d2G_yz   = sp.zeros(3, 3)  # d^2 / (dy dz)
_d2G_xz0  = sp.zeros(3, 3)  # d^2 / (dx dz0)
_d2G_yz0  = sp.zeros(3, 3)  # d^2 / (dy dz0)
_d2G_zz0  = sp.zeros(3, 3)  # d^2 / (dz dz0)
for _i in range(3):
    for _j in range(3):
        g = _G_symbolic[_i, _j]
        _d2G_xx[_i, _j]  = sp.diff(g, _x, 2)
        _d2G_yy[_i, _j]  = sp.diff(g, _y, 2)
        _d2G_xy[_i, _j]  = sp.diff(g, _x, _y)
        _d2G_xz[_i, _j]  = sp.diff(g, _x, _z)
        _d2G_yz[_i, _j]  = sp.diff(g, _y, _z)
        _d2G_xz0[_i, _j] = sp.diff(g, _x, _z0)
        _d2G_yz0[_i, _j] = sp.diff(g, _y, _z0)
        _d2G_zz0[_i, _j] = sp.diff(g, _z, _z0)

_d2G_xx_lam  = sp.lambdify(_lam_args, _d2G_xx,  modules="numpy", cse=True)
_d2G_yy_lam  = sp.lambdify(_lam_args, _d2G_yy,  modules="numpy", cse=True)
_d2G_xy_lam  = sp.lambdify(_lam_args, _d2G_xy,  modules="numpy", cse=True)
_d2G_xz_lam  = sp.lambdify(_lam_args, _d2G_xz,  modules="numpy", cse=True)
_d2G_yz_lam  = sp.lambdify(_lam_args, _d2G_yz,  modules="numpy", cse=True)
_d2G_xz0_lam = sp.lambdify(_lam_args, _d2G_xz0, modules="numpy", cse=True)
_d2G_yz0_lam = sp.lambdify(_lam_args, _d2G_yz0, modules="numpy", cse=True)
_d2G_zz0_lam = sp.lambdify(_lam_args, _d2G_zz0, modules="numpy", cse=True)


# ============================================================
# Public API
# ============================================================


def mindlin_G(obs, source, mu, nu, eps):
    """
    Mollified Mindlin half-space displacement Green's tensor.

    G^{eps,HS}_{ij}(obs, source) such that u_i(obs) = G[i, j] * F_j for
    a point force F at source.

    Free surface at z = 0, material at z <= 0. Source must be buried
    (source[2] < 0). Coordinate system: (x, y, z) = (East, North, Up).

    Parameters
    ----------
    obs    : (3,) observation point, obs[2] <= 0
    source : (3,) source point, source[2] < 0
    mu     : shear modulus
    nu     : Poisson's ratio
    eps    : mollification parameter (>= 0)

    Returns
    -------
    G : (3, 3) numpy array
    """
    obs = np.asarray(obs, dtype=float)
    source = np.asarray(source, dtype=float)
    if source[2] >= 0:
        raise ValueError(f"Source must satisfy source[2] < 0; got {source[2]}")
    if obs[2] > 0:
        raise ValueError(f"Observation must satisfy obs[2] <= 0; got {obs[2]}")

    # Translation invariance in (x, y): use relative coordinates
    x = obs[0] - source[0]
    y = obs[1] - source[1]
    z = obs[2]
    z0 = source[2]
    # Result is a python list of lists or numpy array
    G = _G_lambdified(x, y, z, z0, mu, nu, eps)
    return np.asarray(G, dtype=float)


# ============================================================
# Public API for the correction-only kernel functions
# (defined symbolically above as Apostol's b/β terms that depend
# only on the image distance r2 — smooth everywhere in z<=0)
# ============================================================


def mindlin_G_correction(obs, source, mu, nu, eps):
    """Half-space correction part of G (Apostol r2-only construction).

    G_correction = u_apostol(b_correction[r2->r2e], beta_correction[r2->r2e]).
    Depends ONLY on the image distance r2, so it is smooth everywhere in
    z<=0 (no near-singular behavior at obs = source).

    Used for the hybrid analytical+quadrature triangle integration: the
    Mindlin half-space displacement field is

        G^{eps,HS}_clean = G_kelvin_standard(obs - source) + G_correction.

    This is a slightly different mollification of Mindlin than what
    mindlin_G (the Apostol-mollified version) returns, but both have the
    same O(eps^2) convergence to singular Mindlin and the same surface
    BC O(eps^2) residual."""
    obs = np.asarray(obs, dtype=float)
    source = np.asarray(source, dtype=float)
    x = obs[0] - source[0]
    y = obs[1] - source[1]
    z = obs[2]
    z0 = source[2]
    G = _G_corr_lambdified(x, y, z, z0, mu, nu, eps)
    return np.asarray(G, dtype=float)


def mindlin_DG_source_correction(obs, source, mu, nu, eps):
    """d G_correction / d source_m. Returns (3, 3, 3) tensor."""
    obs = np.asarray(obs, dtype=float)
    source = np.asarray(source, dtype=float)
    x = obs[0] - source[0]
    y = obs[1] - source[1]
    z = obs[2]
    z0 = source[2]
    args = (x, y, z, z0, mu, nu, eps)
    DG = np.zeros((3, 3, 3))
    DG[:, :, 0] = -np.asarray(_dGc_dx_lam(*args), dtype=float)
    DG[:, :, 1] = -np.asarray(_dGc_dy_lam(*args), dtype=float)
    DG[:, :, 2] = +np.asarray(_dGc_dz0_lam(*args), dtype=float)
    return DG


def mindlin_DDG_obs_source_correction(obs, source, mu, nu, eps):
    """d^2 G_correction / (d obs_p * d source_m). Returns (3, 3, 3, 3) tensor."""
    obs = np.asarray(obs, dtype=float)
    source = np.asarray(source, dtype=float)
    x = obs[0] - source[0]
    y = obs[1] - source[1]
    z = obs[2]
    z0 = source[2]
    args = (x, y, z, z0, mu, nu, eps)
    dxx  = np.asarray(_d2Gc_xx_lam(*args),  dtype=float)
    dyy  = np.asarray(_d2Gc_yy_lam(*args),  dtype=float)
    dxy  = np.asarray(_d2Gc_xy_lam(*args),  dtype=float)
    dxz  = np.asarray(_d2Gc_xz_lam(*args),  dtype=float)
    dyz  = np.asarray(_d2Gc_yz_lam(*args),  dtype=float)
    dxz0 = np.asarray(_d2Gc_xz0_lam(*args), dtype=float)
    dyz0 = np.asarray(_d2Gc_yz0_lam(*args), dtype=float)
    dzz0 = np.asarray(_d2Gc_zz0_lam(*args), dtype=float)
    DDG = np.zeros((3, 3, 3, 3))
    DDG[:, :, 0, 0] = -dxx
    DDG[:, :, 0, 1] = -dxy
    DDG[:, :, 0, 2] = +dxz0
    DDG[:, :, 1, 0] = -dxy
    DDG[:, :, 1, 1] = -dyy
    DDG[:, :, 1, 2] = +dyz0
    DDG[:, :, 2, 0] = -dxz
    DDG[:, :, 2, 1] = -dyz
    DDG[:, :, 2, 2] = +dzz0
    return DDG


def mindlin_DG_source(obs, source, mu, nu, eps):
    """
    First derivative of G with respect to SOURCE coordinates (analytical
    via SymPy-derived expressions).

    Returns (3, 3, 3) tensor DG[i, j, m] = d G[i, j] / d source_m.

    Used by the slip-discontinuity (DD) triangle integration:
      u_i(obs) = integral_T [C_kljm n_l (dG_ij/d source_m) Δu_k] dA(source)
    """
    obs = np.asarray(obs, dtype=float)
    source = np.asarray(source, dtype=float)
    if source[2] >= 0:
        raise ValueError(f"Source must satisfy source[2] < 0; got {source[2]}")
    x = obs[0] - source[0]
    y = obs[1] - source[1]
    z = obs[2]
    z0 = source[2]
    # Note coordinate identities:
    #   d/d(source_x) = -d/dx   (since x = obs_x - source_x)
    #   d/d(source_y) = -d/dy
    #   d/d(source_z) = +d/dz0
    DG = np.zeros((3, 3, 3))
    DG[:, :, 0] = -np.asarray(_dG_dx_lam(x, y, z, z0, mu, nu, eps), dtype=float)
    DG[:, :, 1] = -np.asarray(_dG_dy_lam(x, y, z, z0, mu, nu, eps), dtype=float)
    DG[:, :, 2] = +np.asarray(_dG_dz0_lam(x, y, z, z0, mu, nu, eps), dtype=float)
    return DG


def mindlin_DDG_obs_source(obs, source, mu, nu, eps):
    """
    Mixed second derivative tensor d^2 G_ij / (d obs_p * d source_m) of
    the mollified Mindlin Green's function, analytical via SymPy.

    Returns (3, 3, 3, 3) tensor DDG[i, j, p, m] where:
        DDG[i, j, p, m] = d^2 G_ij(obs, source) / (d obs_p * d source_m)

    Used for strain/stress evaluation at observation points:
        d u_i / d obs_p = integral_T [C_kljm n_l DDG[i, j, p, m] Δu_k] dA
    """
    obs = np.asarray(obs, dtype=float)
    source = np.asarray(source, dtype=float)
    if source[2] >= 0:
        raise ValueError(f"Source must satisfy source[2] < 0; got {source[2]}")
    x = obs[0] - source[0]
    y = obs[1] - source[1]
    z = obs[2]
    z0 = source[2]
    args = (x, y, z, z0, mu, nu, eps)

    # Evaluate the 8 distinct sympy second derivatives
    dxx  = np.asarray(_d2G_xx_lam(*args),  dtype=float)
    dyy  = np.asarray(_d2G_yy_lam(*args),  dtype=float)
    dxy  = np.asarray(_d2G_xy_lam(*args),  dtype=float)
    dxz  = np.asarray(_d2G_xz_lam(*args),  dtype=float)
    dyz  = np.asarray(_d2G_yz_lam(*args),  dtype=float)
    dxz0 = np.asarray(_d2G_xz0_lam(*args), dtype=float)
    dyz0 = np.asarray(_d2G_yz0_lam(*args), dtype=float)
    dzz0 = np.asarray(_d2G_zz0_lam(*args), dtype=float)

    DDG = np.zeros((3, 3, 3, 3))
    # Sign table (d obs_p / d source_m -> sympy chain):
    #   obs_x = +d/dx,  obs_y = +d/dy,  obs_z = +d/dz
    #   source_x = -d/dx, source_y = -d/dy, source_z = +d/dz0
    # So DDG[..., p, m]:
    #   (p=0 x, m=0 x): -dxx
    #   (p=0 x, m=1 y): -dxy
    #   (p=0 x, m=2 z): +dxz0
    #   (p=1 y, m=0 x): -dxy           (Schwarz: dxy = dyx)
    #   (p=1 y, m=1 y): -dyy
    #   (p=1 y, m=2 z): +dyz0
    #   (p=2 z, m=0 x): -dxz           (Schwarz: dxz = dzx)
    #   (p=2 z, m=1 y): -dyz
    #   (p=2 z, m=2 z): +dzz0
    DDG[:, :, 0, 0] = -dxx
    DDG[:, :, 0, 1] = -dxy
    DDG[:, :, 0, 2] = +dxz0
    DDG[:, :, 1, 0] = -dxy
    DDG[:, :, 1, 1] = -dyy
    DDG[:, :, 1, 2] = +dyz0
    DDG[:, :, 2, 0] = -dxz
    DDG[:, :, 2, 1] = -dyz
    DDG[:, :, 2, 2] = +dzz0
    return DDG


def mindlin_DG(obs, source, mu, nu, eps, h=1e-4):
    """First derivative tensor d/d(obs_m) of mindlin_G via central FD."""
    DG = np.zeros((3, 3, 3))
    obs = np.asarray(obs, dtype=float)
    for m in range(3):
        e = np.zeros(3); e[m] = h
        DG[:, :, m] = (
            mindlin_G(obs + e, source, mu, nu, eps)
            - mindlin_G(obs - e, source, mu, nu, eps)
        ) / (2 * h)
    return DG


def mindlin_D2G(obs, source, mu, nu, eps, h=1e-3):
    """Second derivative tensor d^2/d(obs_m) d(obs_n) of mindlin_G via FD."""
    D2G = np.zeros((3, 3, 3, 3))
    obs = np.asarray(obs, dtype=float)
    for m in range(3):
        e_m = np.zeros(3); e_m[m] = h
        for n in range(3):
            e_n = np.zeros(3); e_n[n] = h
            if m == n:
                D2G[:, :, m, n] = (
                    mindlin_G(obs + e_m, source, mu, nu, eps)
                    - 2 * mindlin_G(obs, source, mu, nu, eps)
                    + mindlin_G(obs - e_m, source, mu, nu, eps)
                ) / (h * h)
            else:
                D2G[:, :, m, n] = (
                    mindlin_G(obs + e_m + e_n, source, mu, nu, eps)
                    - mindlin_G(obs + e_m - e_n, source, mu, nu, eps)
                    - mindlin_G(obs - e_m + e_n, source, mu, nu, eps)
                    + mindlin_G(obs - e_m - e_n, source, mu, nu, eps)
                ) / (4 * h * h)
    return D2G


def _main():
    """Quick sanity check."""
    mu, nu, eps = 1.0, 0.25, 0.3
    source = np.array([0.0, 0.0, -2.0])
    print(f"Mollified Mindlin G^(eps,HS) sample")
    print(f"  source = {source}, mu = {mu}, nu = {nu}, eps = {eps}\n")
    for obs in [
        np.array([1.0, 0.5, -0.5]),
        np.array([0.0, 0.0, 0.0]),
        np.array([3.0, 2.0, -1.5]),
    ]:
        G = mindlin_G(obs, source, mu, nu, eps)
        print(f"  obs = {obs}")
        for i in range(3):
            print("    " + "  ".join(f"{G[i,j]:+.4e}" for j in range(3)))
        print()


if __name__ == "__main__":
    _main()
