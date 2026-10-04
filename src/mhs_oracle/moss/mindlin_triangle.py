"""
mindlin_triangle.py
===================

Integration of the mollified Mindlin half-space Green's function over an
arbitrary flat triangle, via Gauss-Legendre quadrature on a reference
triangle.

Stage 2 of the Mollified Mindlin plan: numerical-quadrature implementation.
A future iteration may replace with analytical (closed-form) integration
using the moment-hierarchy machinery in analytical_kernels.py.

The displacement at observation point obs due to a uniform slip Δu
(Cartesian vector, units of displacement) on a flat triangular surface
T with outward normal n is:

    u_i(obs) = integral_T [ C_kljm n_l (dG_ij/d source_m) Δu_k ] dA(source)

where the slip index k and the normal index l share the FIRST index pair of
the isotropic stiffness tensor (traction-operator form):
    C_kljm = λ δ_kl δ_jm + μ (δ_kj δ_lm + δ_km δ_lj)

This module computes the kernel
    U[i, k] = integral_T [ C_kljm n_l (dG_ij/d source_m) ] dA(source)
so that u_i = U[i, k] * Δu_k.

References
----------
Same as mindlin_kernels.py (Apostol 2016, Mindlin 1936).
"""

import numpy as np

# RELATIVE imports, which is the one edit this vendored file carries.
# Upstream spells these absolutely ("from moss_kernel.X import ...") inside a
# try/except ImportError whose fallback is a FLAT "from X import ...". That
# fallback is a live hazard: if the package prefix is ever wrong the except
# branch fires, the flat import SUCCEEDS whenever that directory happens to be
# on sys.path, and which copy you get depends on path order -- silently. A
# relative import cannot do that; it resolves within this package or raises.
# No arithmetic is touched. The cost is that this file is no longer runnable as
# a standalone script, which a vendored library module has no need to be.
from .analytical_kernels import (
    analytical_dd_displacement,
    analytical_stress_kernel,
)
from .mindlin_kernels import (
    mindlin_DDG_obs_source,
    mindlin_DDG_obs_source_correction,
    mindlin_DG_source,
    mindlin_DG_source_correction,
)
from .mollified_elastic_kernels import triangle_quadrature


def _stiffness_contract(dG_dsrc, normal, mu, nu):
    """
    Compute the (3, 3) tensor
        T[i, k] = sum_{j, l, m} C_kljm n_l (dG_ij/d source_m)

    (slip index k pairs with the normal in the FIRST index pair of C),
    where C_kljm = lambda δ_kl δ_jm + mu (δ_kj δ_lm + δ_km δ_lj).

    Parameters
    ----------
    dG_dsrc : (3, 3, 3) tensor, dG_dsrc[i, j, m] = dG_ij / d source_m
    normal  : (3,) outward unit normal at the source point
    mu, nu  : elastic parameters

    Returns
    -------
    T : (3, 3) tensor.
    """
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    n = normal
    # Slip index k pairs with the normal in the FIRST index pair of C
    # (traction-operator form, as in tdhs_triangle._stiffness_contract):
    # T[i, k] = mu  * sum_l n_l dG[i, k, l]
    #         + mu  * sum_l n_l dG[i, l, k]
    #         + lam * n_k sum_m dG[i, m, m]
    # (lam/mu placement fixed 2026-09-17; the earlier form had them swapped
    # on the first and third terms -- identical at nu = 1/4 only.)
    T = np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            T[i, k] = (
                mu * sum(n[ll] * dG_dsrc[i, k, ll] for ll in range(3))
                + mu * sum(n[ll] * dG_dsrc[i, ll, k] for ll in range(3))
                + lam * n[k] * sum(dG_dsrc[i, m, m] for m in range(3))
            )
    return T


def mindlin_dd_kernel_pointwise(obs, source, normal, mu, nu, eps):
    """
    Pointwise DD displacement kernel T[i, k](obs, source, normal) such
    that u_i(obs) per unit slip in direction k on a surface element at
    source with outward unit normal n is T[i, k] * dA.

    Used as the integrand for triangle integration.
    """
    dG_dsrc = mindlin_DG_source(obs, source, mu, nu, eps)
    return _stiffness_contract(dG_dsrc, normal, mu, nu)


def integrate_mindlin_dd_kernel(obs, v1, v2, v3, normal, mu, nu, eps, n_quad=8):
    """
    Integrate the mollified Mindlin DD displacement kernel over a flat
    triangle (v1, v2, v3) with outward unit normal n.

    Returns U[i, k] such that the displacement at obs from uniform slip
    Δu_k (Cartesian) on the triangle is:
        u_i(obs) = U[i, k] * Δu_k

    Parameters
    ----------
    obs    : (3,) observation point, obs[2] <= 0
    v1, v2, v3 : (3,) triangle vertices
    normal : (3,) outward unit normal at the triangle (used everywhere
             on the flat triangle)
    mu     : shear modulus
    nu     : Poisson's ratio
    eps    : mollification parameter
    n_quad : Gauss-Legendre quadrature order per direction (total points
             on the triangle = n_quad^2)

    Returns
    -------
    U : (3, 3) kernel array.
    """
    xi1, xi2, wts = triangle_quadrature(n_quad)
    # 2 * triangle area (the factor for the affine-map Jacobian to 2x reference area)
    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    U = np.zeros((3, 3))
    for q in range(len(wts)):
        # Physical coordinates of quadrature point on the source triangle
        y = (1.0 - xi1[q] - xi2[q]) * v1 + xi1[q] * v2 + xi2[q] * v3
        T_pt = mindlin_dd_kernel_pointwise(obs, y, normal, mu, nu, eps)
        U += wts[q] * area2 * T_pt
    return U


def mindlin_dd_displacement(obs, v1, v2, v3, normal, slip, mu, nu, eps, n_quad=8):
    """
    Convenience: compute displacement at obs from a uniform Cartesian
    slip on triangle (v1, v2, v3) with outward unit normal n.

    Parameters
    ----------
    obs       : (3,) observation point
    v1, v2, v3 : (3,) triangle vertices
    normal    : (3,) outward unit normal
    slip      : (3,) Cartesian displacement-discontinuity vector
    mu, nu    : elastic parameters
    eps       : mollification parameter
    n_quad    : quadrature order

    Returns
    -------
    u : (3,) displacement vector at obs
    """
    slip = np.asarray(slip, dtype=float)
    U = integrate_mindlin_dd_kernel(obs, v1, v2, v3, normal, mu, nu, eps, n_quad)
    return U @ slip


# ============================================================
# Strain / stress integration
# ============================================================
#
# To compute strain and stress at obs, we differentiate the displacement
# w.r.t. obs coordinates. For a slip discontinuity Δu_k on a surface
# with normal n_l at source y, the displacement at obs is
#
#   u_i(obs) = ∫_T C_kljm n_l (∂G_ij(obs, y)/∂y_m) Δu_k dA(y).
#
# Differentiating w.r.t. obs_p:
#
#   ∂u_i/∂obs_p = ∫_T C_kljm n_l (∂^2 G_ij/(∂obs_p ∂y_m)) Δu_k dA(y).
#
# Strain ε_pq = (1/2)(∂u_p/∂obs_q + ∂u_q/∂obs_p) (symmetric part).
# Stress σ_pq = λ δ_pq ε_kk + 2μ ε_pq (isotropic Hooke's law).
#
# The integrand for the displacement gradient is built using the
# (3,3,3,3) tensor DDG[i, j, p, m] = ∂^2 G_ij/(∂obs_p ∂y_m).


def _stiffness_contract_DDG(DDG, normal, mu, nu):
    """
    Compute the (3, 3, 3) tensor
        S[i, p, k] = sum_{j, l, m} C_kljm n_l DDG[i, j, p, m]

    such that the displacement-gradient kernel from a slip Δu at the
    given source/normal is grad_u[i, p] = S[i, p, k] * Δu_k (same
    first-pair slip/normal pairing as _stiffness_contract).

    C_kljm = λ δ_kl δ_jm + μ (δ_kj δ_lm + δ_km δ_lj).
    (lam/mu placement fixed 2026-09-17, was swapped; identical at nu = 1/4.)
    """
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    n = normal
    S = np.zeros((3, 3, 3))
    for i in range(3):
        for p in range(3):
            for k in range(3):
                # μ δ_kj δ_lm term:  μ sum_l n_l DDG[i, k, p, l]
                term_mu1 = mu * sum(n[ll] * DDG[i, k, p, ll] for ll in range(3))
                # μ δ_km δ_lj term:  μ sum_l n_l DDG[i, l, p, k]
                term_mu2 = mu * sum(n[ll] * DDG[i, ll, p, k] for ll in range(3))
                # λ δ_kl δ_jm term:  λ n_k sum_m DDG[i, m, p, m]
                term_lam = lam * n[k] * sum(DDG[i, m, p, m] for m in range(3))
                S[i, p, k] = term_mu1 + term_mu2 + term_lam
    return S


def mindlin_dd_grad_kernel_pointwise(obs, source, normal, mu, nu, eps):
    """Pointwise displacement-gradient kernel S[i, p, k] from a unit slip
    in direction k on a surface element at source with normal n."""
    DDG = mindlin_DDG_obs_source(obs, source, mu, nu, eps)
    return _stiffness_contract_DDG(DDG, normal, mu, nu)


def integrate_mindlin_dd_grad_kernel(obs, v1, v2, v3, normal, mu, nu, eps, n_quad=8):
    """
    Integrate the displacement-gradient kernel over the source triangle.

    Returns G_du[i, p, k] such that
        (∂u_i/∂obs_p)(obs) = G_du[i, p, k] * Δu_k.

    Parameters identical to integrate_mindlin_dd_kernel.
    """
    xi1, xi2, wts = triangle_quadrature(n_quad)
    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    G_du = np.zeros((3, 3, 3))
    for q in range(len(wts)):
        y = (1.0 - xi1[q] - xi2[q]) * v1 + xi1[q] * v2 + xi2[q] * v3
        S = mindlin_dd_grad_kernel_pointwise(obs, y, normal, mu, nu, eps)
        G_du += wts[q] * area2 * S
    return G_du


# ============================================================
# Hybrid analytical integration (RECOMMENDED for on-fault / near-fault)
# ============================================================
#
# Pure Gauss-Legendre quadrature on the displacement-gradient kernel
# becomes unreliable when the observation point is on or very close to
# the source triangle, because the Kelvin part of the kernel has
# features at scale eps that the quadrature cannot resolve cheaply.
#
# Strategy: decompose the Mindlin Green's function as
#
#     G^{eps,HS} = G^{eps,Kelvin}(full space, with blob)  + G^{HS-correction}
#
# - The Kelvin part is integrated ANALYTICALLY using the existing
#   moment-hierarchy machinery in mollified_kernel/analytical_kernels.py
#   (which handles obs anywhere, including on the triangle, via the
#   Van Oosterom solid angle + edge antiderivatives).
# - The half-space correction is SMOOTH in z<=0 (no near-singular
#   behavior even for on-fault obs), so a small Gauss-Legendre rule
#   resolves it accurately.
#
# The resulting hybrid quadrature is accurate everywhere, including
# on the source triangle.


def integrate_mindlin_dd_kernel_analytical(obs, v1, v2, v3, normal, mu, nu, eps, n_quad=4):
    """
    Hybrid analytical+quadrature integration of the Mindlin DD
    displacement kernel.

    Returns U[i, k] such that u_i(obs) = U[i, k] * Δu_k (Cartesian slip).

    The Kelvin part is analytical (exact); the smooth half-space
    correction is integrated with Gauss-Legendre quadrature (n_quad
    points per side, total n_quad^2 points -- typically n_quad=4 is
    enough since the correction integrand is smooth).
    """
    # Direct full-space Kelvin part: ANALYTICAL via existing machinery
    U_direct = analytical_dd_displacement(obs, v1, v2, v3, normal, mu, nu, eps)

    # Half-space correction part: smooth-integrand Gauss-Legendre
    xi1, xi2, wts = triangle_quadrature(n_quad)
    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    U_corr = np.zeros((3, 3))
    for q in range(len(wts)):
        y = (1.0 - xi1[q] - xi2[q]) * v1 + xi1[q] * v2 + xi2[q] * v3
        DG_corr = mindlin_DG_source_correction(obs, y, mu, nu, eps)
        T_pt = _stiffness_contract(DG_corr, normal, mu, nu)
        U_corr += wts[q] * area2 * T_pt

    return U_direct + U_corr


def integrate_mindlin_stress_kernel_analytical(obs, v1, v2, v3, normal, mu, nu, eps, n_quad=4):
    """
    Hybrid analytical+quadrature integration of the Mindlin DD stress
    kernel.

    Returns H[m, n, k] such that sigma_mn(obs) = H[m, n, k] * Δu_k
    (Cartesian slip). Accurate for obs anywhere, including on the source
    triangle.

    The Kelvin part is analytical (exact); the correction integrand is
    smooth and integrated with Gauss-Legendre quadrature.
    """
    # Direct full-space Kelvin stress: ANALYTICAL
    H_direct = analytical_stress_kernel(obs, v1, v2, v3, normal, mu, nu, eps)

    # Half-space correction stress: numerical quadrature on the
    # correction-only mixed-derivative kernel
    xi1, xi2, wts = triangle_quadrature(n_quad)
    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)

    # Accumulate G_du_corr[i, p, k]: contribution to du_i/dobs_p from slip k
    G_du_corr = np.zeros((3, 3, 3))
    for q in range(len(wts)):
        y = (1.0 - xi1[q] - xi2[q]) * v1 + xi1[q] * v2 + xi2[q] * v3
        DDG_corr = mindlin_DDG_obs_source_correction(obs, y, mu, nu, eps)
        S_pt = _stiffness_contract_DDG(DDG_corr, normal, mu, nu)
        G_du_corr += wts[q] * area2 * S_pt

    # Build correction stress from displacement gradient:
    #   eps_pq = 1/2 (grad_u_pq + grad_u_qp)
    #   sig_pq = lambda delta_pq tr(eps) + 2 mu eps_pq
    # In tensor form contracted with slip k:
    H_corr = np.zeros((3, 3, 3))
    for k in range(3):
        grad_u = G_du_corr[:, :, k]  # (3, 3)
        eps_tensor = 0.5 * (grad_u + grad_u.T)
        sig_corr = lam * np.trace(eps_tensor) * np.eye(3) + 2.0 * mu * eps_tensor
        H_corr[:, :, k] = sig_corr

    return H_direct + H_corr


def mindlin_dd_field_analytical(obs, v1, v2, v3, normal, slip, mu, nu, eps, n_quad=4):
    """
    Hybrid analytical+quadrature evaluation of u, strain, stress at obs
    from a uniform Cartesian slip on triangle (v1, v2, v3).

    Returns dict with keys 'disp', 'grad_u', 'strain', 'stress'.

    Accurate everywhere including on the source triangle. Uses the
    decomposition G = G_Kelvin + G_correction, where the Kelvin part
    is integrated analytically and the smooth correction is integrated
    by Gauss-Legendre with n_quad points per side.
    """
    slip = np.asarray(slip, dtype=float)

    # ---- Displacement ----
    U = integrate_mindlin_dd_kernel_analytical(obs, v1, v2, v3, normal, mu, nu, eps, n_quad)
    u = U @ slip

    # ---- Stress (and hence strain via Hooke inverse) ----
    H = integrate_mindlin_stress_kernel_analytical(obs, v1, v2, v3, normal, mu, nu, eps, n_quad)
    stress = np.einsum("mnk,k->mn", H, slip)  # sigma_mn = H_mn,k * slip_k

    # Strain from stress via isotropic Hooke inverse:
    # sigma = lam tr(eps) I + 2 mu eps
    # tr(sigma) = (3 lam + 2 mu) tr(eps), so tr(eps) = tr(sigma) / (3 lam + 2 mu)
    # eps = (sigma - lam tr(eps) I) / (2 mu)
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    tr_eps = np.trace(stress) / (3.0 * lam + 2.0 * mu)
    strain = (stress - lam * tr_eps * np.eye(3)) / (2.0 * mu)

    # grad_u is not uniquely defined from strain alone (it's missing the
    # antisymmetric rotation part). Provide it via the displacement-gradient
    # integration result for completeness:
    xi1, xi2, wts = triangle_quadrature(n_quad)
    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    # Need full DDG (not just correction) to assemble grad_u analytically.
    # For now: include both parts.
    G_du_total = np.zeros((3, 3, 3))
    for q in range(len(wts)):
        y = (1.0 - xi1[q] - xi2[q]) * v1 + xi1[q] * v2 + xi2[q] * v3
        DDG_full = mindlin_DDG_obs_source(obs, y, mu, nu, eps)
        S_pt = _stiffness_contract_DDG(DDG_full, normal, mu, nu)
        G_du_total += wts[q] * area2 * S_pt
    # This is pure-quadrature for grad_u; combined with analytical strain/
    # stress above. For "exact" grad_u we would need to also integrate the
    # Kelvin DDG analytically (currently only DG and D2G are exposed).
    grad_u = G_du_total @ slip

    return {"disp": u, "grad_u": grad_u, "strain": strain, "stress": stress}


def mindlin_dd_strain_stress(obs, v1, v2, v3, normal, slip, mu, nu, eps, n_quad=8):
    """
    Compute displacement gradient, strain, and stress at obs from a
    uniform Cartesian slip on triangle (v1, v2, v3) with outward unit
    normal n.

    Returns dict with keys:
        'grad_u' : (3, 3) displacement gradient, grad_u[i, p] = du_i/dobs_p
        'strain' : (3, 3) symmetric strain tensor
        'stress' : (3, 3) symmetric stress tensor (isotropic Hooke's law)
    """
    slip = np.asarray(slip, dtype=float)
    G_du = integrate_mindlin_dd_grad_kernel(obs, v1, v2, v3, normal, mu, nu, eps, n_quad)
    # grad_u[i, p] = G_du[i, p, k] * slip[k]
    grad_u = G_du @ slip   # (3, 3, 3) @ (3,) -> (3, 3)
    strain = 0.5 * (grad_u + grad_u.T)
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    tr_eps = np.trace(strain)
    stress = lam * tr_eps * np.eye(3) + 2.0 * mu * strain
    return {"grad_u": grad_u, "strain": strain, "stress": stress}


def _main():
    """Quick sanity check: compute u from a Cartesian slip on a buried triangle."""
    mu, nu, eps = 1.0, 0.25, 0.3
    n_quad = 8
    # Small horizontal triangle at depth 2, slip in x direction
    v1 = np.array([-0.5, -0.5, -2.0])
    v2 = np.array([0.5, -0.5, -2.0])
    v3 = np.array([0.0, 0.5, -2.0])
    normal = np.array([0.0, 0.0, 1.0])  # outward = up
    slip = np.array([1.0, 0.0, 0.0])  # x-direction slip

    for obs in [
        np.array([2.0, 0.0, 0.0]),    # surface, offset 2 in x
        np.array([0.0, 2.0, 0.0]),    # surface, offset 2 in y
        np.array([0.0, 0.0, -1.0]),   # right above triangle
    ]:
        u = mindlin_dd_displacement(obs, v1, v2, v3, normal, slip, mu, nu, eps, n_quad)
        print(f"obs = {obs}: u = {u}")


if __name__ == "__main__":
    _main()
