"""Single source of truth for numeric constants and thresholds in ``clq``.

Put new numeric constants here, not inline (msd convention).
"""
from __future__ import annotations

# --- far-field hybrid producer ---------------------------------------------
# The divergence-theorem closed form loses digits roughly like (R/L)^4-5 for
# quadratic-weighted moments in the triangle's plane (~(R/L)^3-3.5 broadside),
# with R = sqrt(|obs - centroid|^2 + eps^2) and L the longest edge.  Measured
# worst U/H relative error, P0-P2, eps = 0.05 L, equilateral + test triangle:
# ~2e-9 at R = 10 L, 4e-8 at 20 L, 2e-4 at 100 L, 0.6 at 500 L, ~9 at 1000 L.
# Smaller eps is worse (2-20x at eps = 0.01 L).  Thin triangles are much
# worse: at h/L = 0.02 it is already 1e-5 at 9.5 L.  So far_field="analytic"
# is not usable beyond ~20 L (README "Far field").  Beyond D_STAR * L the
# hybrid producer fills the per-node weighted moment tables by Gauss
# quadrature of the (smooth) integrand instead, exact to ~2e-14.  D_STAR is
# relative to the longest edge only, so it does not protect thin triangles
# inside 10 L.  Checked by verify/verify_far_field.py.
D_STAR = 10.0
FAR_GAUSS_N = 12          # points per direction of the collapsed product rule (144 pts)
FAR_GAUSS_N_DISTANT = 8   # beyond D_STAR_DISTANT * L (64 pts)
D_STAR_DISTANT = 40.0

# --- edge primitives ---------------------------------------------------------
# Same-sign edge parameters with min(|u_a|, |u_b|) >= SERIES_U_OVER_RHO * rho use
# the large-|u| binomial series for int u^k / R^m du (no cancellation); the
# series terms decay like (rho/u_min)^(2j), so with ratio 2 the SERIES_TERMS
# truncation error is below 2^(-2*SERIES_TERMS) ~ 1e-19.
SERIES_U_OVER_RHO = 2.0
SERIES_TERMS = 32
# Edges with max(|u_a|, |u_b|) <= SERIES_RHO_OVER_U_INV * rho (both endpoints
# close to the foot of the perpendicular, rho >> |u|) use the small-|u| series
# instead of the closed form, whose u^k reduction loses (rho/u)^2 per level.
SMALL_U_OVER_RHO = 0.5

# --- geometry guards ---------------------------------------------------------
DEGENERATE_AREA_REL = 1e-14   # area / L^2 below this -> degenerate triangle
ON_PLANE_REL = 1e-12          # |z| / L below this counts as on the plane (eps = 0 guard)

# --- oracle quadrature (verify/, examples) -----------------------------------
ORACLE_GAUSS_N = 40
