"""The elastic material, as ``(mu, lam)`` and never as ``nu``.

Stated once, here, because it is the convention most easily lost: **coefficients
come from ``(mu, lam)``, never through a ``1/(1 - 2 nu)`` intermediate.** Three
reasons, in increasing order of how much they bite:

1. ``1/(1 - 2 nu)`` diverges as ``nu -> 1/2``, so an incompressible or
   nearly-incompressible material is a division by nearly zero in the one
   quantity every kernel multiplies by.
2. In ``(mu, lam)`` the combinations the kernels actually need are all finite
   there -- ``1 - 2 nu`` becomes ``mu / (lam + mu)``, which goes to **zero**
   rather than to ``0/0``. The Mindlin image terms that carry that factor
   therefore switch off cleanly instead of degrading, and the measured
   degradation of the naive mollification at ``nu = 0.49`` (bulk PDE residual
   9.1e-3 against 7.0e-4 at ``nu = 0.25``) should go with it.
3. A ``lam``/``mu`` swap is invisible at ``nu = 1/4`` (where ``lam == mu``),
   which is how a swapped kernel shipped for months in the reference tree, and
   how the Mindlin triangle code there kept the two swapped until 2026-09-17.
   Carrying ``(mu, lam)`` makes the two distinguishable at every call site, and
   the gates run at ``nu != 1/4`` besides.

This is a deliberate divergence from ``cutde``, whose public functions take
``nu`` alone and reconstruct ``lam = 2 mu nu / (1 - 2 nu)`` internally. ``mhs``
never forms that quotient; the one place ``nu`` appears in this package is
inside the gate that calls cutde, converting one way to talk to it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """Isotropic linear elastic material: shear modulus and Lame's first.

    Frozen, because a material is a value and two solves that differ by one are
    two different problems. The derived combinations are properties rather than
    stored fields so there is one statement of each.
    """

    mu: float
    lam: float

    def __post_init__(self) -> None:
        if not (self.mu > 0.0):
            raise ValueError(f"mu must be positive, got {self.mu!r}")
        # lam > -2 mu / 3 is the thermodynamic admissibility bound (positive
        # bulk modulus). Negative lam is admissible and physical (auxetic), so
        # this refuses only what is unphysical.
        if not (self.lam > -2.0 * self.mu / 3.0):
            raise ValueError(
                f"lam must exceed -2 mu / 3 = {-2.0 * self.mu / 3.0!r} for a "
                f"positive bulk modulus, got {self.lam!r}")

    # ---- derived, all finite as lam -> infinity (nu -> 1/2) ----------------
    @property
    def bulk(self) -> float:
        """``K = lam + 2 mu / 3``."""
        return self.lam + 2.0 * self.mu / 3.0

    @property
    def nu_if_you_must(self) -> float:
        """Poisson's ratio, for talking to code that insists on it.

        Named to be awkward on purpose: nothing inside ``mhs`` reads this. It
        exists so the cutde gate has one place to do the conversion, rather than
        each call site inventing its own.
        """
        return 0.5 * self.lam / (self.lam + self.mu)

    @classmethod
    def from_mu_nu(cls, mu: float, nu: float) -> "Material":
        """Build from ``(mu, nu)``, for reading someone else's parameters.

        This is the ONLY place in the package that forms ``1/(1 - 2 nu)``, and it
        is a boundary adapter, not a computation: everything downstream carries
        ``(mu, lam)``. Refuses ``nu >= 1/2`` rather than returning an infinity.
        """
        if not (-1.0 < nu < 0.5):
            raise ValueError(
                f"nu must lie in (-1, 1/2) to convert to lam; got {nu!r}. "
                f"An incompressible material has no finite lam -- construct "
                f"Material(mu, lam) with a large lam directly.")
        return cls(mu=float(mu), lam=2.0 * float(mu) * nu / (1.0 - 2.0 * nu))
