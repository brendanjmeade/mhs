"""Regional load, frictional strength, and the Coulomb yield function.

SIGN CONVENTIONS, stated once because every one of them can be wrong silently.

* ``mhs`` returns traction components in each element's own
  (strike, dip, tensile) frame, with ``tensile`` the UPWARD normal. So the
  third component is positive in TENSION.
* Rock mechanics wants compression positive, so the normal stress here is
  ``sigma_n = -t_tensile``, and the effective normal stress subtracts pore
  pressure: ``sigma_eff = sigma_n - p``.
* The yield function is ``F = |tau| - mu * sigma_eff``. ``F <= 0`` everywhere is
  mechanical equilibrium; ``F > 0`` is a violation that has to be relaxed.
* Slip is restricted to the fault plane. The tensile source component is not
  assembled at all, because an opening mode needs a different constitutive
  statement than Coulomb friction.

WHY THE DEFAULT STRENGTH IS DEPTH-INDEPENDENT, and this is the most consequential
choice in the module. Real effective normal stress grows with depth, roughly
0.0167 GPa/km once hydrostatic pore pressure is removed. Over a 40 km box that
is a factor of twenty in strength between the top and the bottom -- so the first
elements to fail would be the shallow ones, the active set would be a horizontal
slab, and its measured dimension would be reporting the DEPTH GRADIENT rather
than any mechanical localization. The headline question would answer itself for
a reason having nothing to do with geometry.

So ``sigma_n_mode="uniform"`` is the default: one effective normal stress for
every element, which leaves orientation and elastic interaction as the only
things that can select the active set. That is the experiment.

FUTURE WORK, AND IT IS IMPORTANT RATHER THAN OPTIONAL. ``sigma_n_mode=
"lithostatic"`` exists and runs, but it has not been studied, and the depth
gradient is not a detail to be left out permanently -- it is the dominant
control on where real seismicity sits, it sets the brittle-ductile transition
that bounds any real fault network from below, and a network experiment that
never confronts it is answering a deliberately simplified question. What makes
it a second experiment rather than a flag to flip is that depth and strength
become confounded: any measured ``D_active`` would mix mechanical localization
with a one-dimensional gradient, so separating them needs its own null -- the
natural one being the same network loaded with the gradient but with the
elastic interaction switched off, which isolates what depth alone does. Until
that null is built and measured, a lithostatic run should be read as a
demonstration and not as a result.

Note the orientation dependence of sigma_n is NOT removed by this: the regional
load still resolves a different normal stress onto every differently-oriented
plane, and that part is geometry, so it stays.

THERE ARE TWO FRICTION COEFFICIENTS, AND THE EXPERIMENT HAS NO MECHANISM
WITHOUT THEM. With a single coefficient -- pure Coulomb, no weakening -- the
quasi-static solution is a CONTINUOUS function of the load: an element that
reaches ``F = 0`` begins to slip at zero rate, its slip grows smoothly as the
load rises, and nothing ever jumps. There are then no events, no cascades and
no avalanche statistics to collect, which the first version of the solver
demonstrated by advancing the load to the first failure and then relaxing zero
slip forever.

A finite event needs the stress to fall BELOW the threshold that triggered it.
So an element has a STATIC strength ``mu_s * sigma_eff`` that it must reach to
fail, and once it has failed it can only support a DYNAMIC strength
``mu_d * sigma_eff``, with ``mu_d = drop_ratio * mu_s``. Triggering drops the
element from one to the other, which forces a finite slip increment of
``(mu_s - mu_d) * sigma_eff / |K_ii|``, and that increment is what loads the
neighbours and can trigger them in turn.

``drop_ratio`` is therefore the single most important physical parameter here:
at 1.0 there are no avalanches by construction, and the smaller it is the
larger the cascades. It is swept, not fixed.

Units are km and GPa, the project's convention. 1 MPa = 1e-3 GPa.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from mhs import tdcs

#: Lithostatic gradient, GPa/km: 2700 kg/m^3 * 9.81 m/s^2 = 26.5 MPa/km.
RHO_G = 0.0265

#: Hydrostatic pore-pressure gradient, GPa/km: 1000 kg/m^3 * 9.81 m/s^2.
RHO_W_G = 0.0098

#: Ratio of dynamic to static friction. 0.9 is a 10 % strength drop, which is
#: a modest and commonly used value; 1.0 disables events entirely and is
#: refused, because a run that produced no avalanche would look like a result.
DROP_RATIO = 0.9

#: Default uniform effective normal stress, GPa. 0.15 GPa = 150 MPa is the
#: lithostatic-minus-hydrostatic value at about 9 km, so it stands for a
#: mid-crustal fault without imposing the gradient. See the module docstring.
SIGMA_EFF_UNIFORM = 0.15

#: Unit deviatoric load tensors. Each has |S| scaled so that lambda is read in
#: GPa as the maximum shear stress the regional field carries, which keeps the
#: load parameter interpretable rather than being an arbitrary multiplier.
LOAD_TENSORS = {
    # Horizontal pure shear: maximum compression at 45 deg to x. The classic
    # strike-slip regime, and the one where a vertical fault at 30 deg to the
    # compression axis is optimally oriented.
    "strike_slip": np.array([[0.0, 1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]]),
    # Horizontal compression with the vertical as the intermediate axis
    # removed: thrust regime, optimal dip near 30 deg.
    "thrust": np.array([[-1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0]]),
    # Horizontal extension: normal-faulting regime, optimal dip near 60 deg.
    "normal": np.array([[1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0]]),
}


def resolve_tensor(tris: np.ndarray,
                   sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a stress tensor onto every element's own frame.

    Returns ``(tau, sigma_n)`` with ``tau`` shaped ``(n, 2)`` -- the shear
    along (strike, dip) -- and ``sigma_n`` shaped ``(n,)``, COMPRESSION
    POSITIVE.

    This is the one helper neither repository had: ``mhs`` resolves tractions
    inside ``interaction_matrix`` and never exposes a tensor-to-scalars step,
    and the sibling project's only ``resolved_shear`` is a one-liner in a figure
    module with no normal component. Written once, here, and used by both
    friction variants so the convention cannot be restated differently twice.
    """
    frame = tdcs.slip_frame(tris)            # rows [strike, dip, tensile]
    n_hat = frame[:, 2, :]
    t = np.einsum("ij,sj->si", np.asarray(sigma, float), n_hat)
    tau = np.stack([np.einsum("si,si->s", t, frame[:, 0, :]),
                    np.einsum("si,si->s", t, frame[:, 1, :])], axis=1)
    return tau, -np.einsum("si,si->s", t, n_hat)


@dataclasses.dataclass(frozen=True)
class Friction:
    """Per-element strength and the resolved regional load.

    ``tau_load`` and ``sigma_n_load`` are per unit ``lambda``, so the state at
    load ``lam`` with slip ``s`` is::

        tau   = lam * tau_load   + (interaction from s)
        sig_n = sigma_eff_0 + lam * sigma_n_load + (interaction from s)

    ``slip_dir`` is the unit direction of ``tau_load`` in each element's
    (strike, dip) plane -- the fixed slip direction the reference variant uses,
    taken from the LOAD alone so that it is a property of the geometry and the
    tectonics rather than of the solution.
    """

    mu: np.ndarray               # (n,) STATIC friction coefficient
    mu_dynamic: np.ndarray       # (n,) what a failed element can still support
    sigma_eff_0: np.ndarray      # (n,) effective normal stress at lam = 0, GPa
    tau_load: np.ndarray         # (n, 2) shear per unit lambda
    sigma_n_load: np.ndarray     # (n,) normal stress per unit lambda, compr +ve
    slip_dir: np.ndarray         # (n, 2) unit vector in (strike, dip)
    tau_load_mag: np.ndarray     # (n,) |tau_load|, the loading rate on F
    meta: dict

    def strength(self, sigma_eff: np.ndarray | None = None) -> np.ndarray:
        """STATIC strength: the level an element must reach to fail."""
        return self.mu * (self.sigma_eff_0 if sigma_eff is None else sigma_eff)

    def strength_dynamic(self,
                         sigma_eff: np.ndarray | None = None) -> np.ndarray:
        """DYNAMIC strength: the level a failed element relaxes down to."""
        return self.mu_dynamic * (self.sigma_eff_0 if sigma_eff is None
                                  else sigma_eff)

    def first_failure_lambda(self) -> np.ndarray:
        """``lambda`` at which each element fails with NO interaction at all.

        The ``K = 0`` state, which is the no-interaction null's score: the set
        that would fail if elements did not talk to each other. Elements whose
        loading does not drive them to failure come back as ``inf``.
        """
        denom = self.tau_load_mag - self.mu * self.sigma_n_load
        out = np.full(denom.shape, np.inf)
        ok = denom > 0.0
        out[ok] = self.strength()[ok] / denom[ok]
        return out


def build_friction(net, *, regime: str = "strike_slip",
                   mu_mode: str = "uniform", mu_mean: float = 0.6,
                   mu_spread: float = 0.0, drop_ratio: float = DROP_RATIO,
                   sigma_n_mode: str = "uniform",
                   sigma_eff: float = SIGMA_EFF_UNIFORM,
                   pore_ratio: float = RHO_W_G / RHO_G,
                   seed: int = 0) -> Friction:
    """Strength and resolved load for a network.

    ``mu_mode``:
      ``uniform``    one coefficient everywhere -- the control.
      ``per_fault``  constant within a fault, varying between them. The
                     physical choice: friction is a property of a fault's
                     gouge, not of a numerical element.
      ``per_element`` i.i.d. per element, which is the harshest heterogeneity
                     and the one most likely to manufacture asperities on its
                     own; useful precisely because it should NOT be mistaken
                     for a geometric effect.

    ``sigma_n_mode``: ``uniform`` (default, see the module docstring) or
    ``lithostatic``, which uses ``RHO_G`` and ``pore_ratio`` with each element's
    own centroid depth.
    """
    tris = net.tris
    n = tris.shape[0]
    rng = np.random.default_rng(seed)

    if not (0.0 < drop_ratio < 1.0):
        raise ValueError(
            f"drop_ratio must lie strictly in (0, 1); got {drop_ratio}. At 1.0"
            f" there is no strength drop, so quasi-static slip is continuous in"
            f" the load and the run would record events of zero size forever"
            f" instead of avalanches.")
    if regime not in LOAD_TENSORS:
        raise ValueError(f"regime must be one of {sorted(LOAD_TENSORS)}, "
                         f"got {regime!r}")
    tau_load, sigma_n_load = resolve_tensor(tris, LOAD_TENSORS[regime])

    if mu_mode == "uniform":
        mu = np.full(n, float(mu_mean))
    elif mu_mode == "per_fault":
        per = np.clip(rng.normal(mu_mean, mu_spread * mu_mean, net.n_faults),
                      0.05, 2.0)
        mu = per[net.fault_id]
    elif mu_mode == "per_element":
        mu = np.clip(rng.normal(mu_mean, mu_spread * mu_mean, n), 0.05, 2.0)
    else:
        raise ValueError(f"mu_mode must be 'uniform', 'per_fault' or "
                         f"'per_element', got {mu_mode!r}")

    if sigma_n_mode == "uniform":
        sig0 = np.full(n, float(sigma_eff))
    elif sigma_n_mode == "lithostatic":
        depth = np.abs(tris.mean(axis=1)[:, 2])
        sig0 = RHO_G * depth * (1.0 - float(pore_ratio))
    else:
        raise ValueError(f"sigma_n_mode must be 'uniform' or 'lithostatic', "
                         f"got {sigma_n_mode!r}")
    if not np.all(sig0 > 0.0):
        raise ValueError(
            "effective normal stress must be positive everywhere: a "
            "non-positive value means the fault is already open and Coulomb "
            "friction does not describe it.")

    mag = np.linalg.norm(tau_load, axis=1)
    safe = np.where(mag > 0.0, mag, 1.0)
    slip_dir = tau_load / safe[:, None]
    # An element the regional load puts in pure normal traction has no preferred
    # in-plane direction. It cannot fail by this criterion either, so the
    # direction is arbitrary; it is set to strike and recorded.
    flat = mag <= 0.0
    if np.any(flat):
        slip_dir[flat] = np.array([1.0, 0.0])

    meta = {"regime": regime, "mu_mode": mu_mode, "mu_mean": mu_mean,
            "mu_spread": mu_spread, "drop_ratio": drop_ratio,
            "sigma_n_mode": sigma_n_mode,
            "sigma_eff_GPa": sigma_eff, "pore_ratio": pore_ratio,
            "seed": seed, "n_elements": n,
            "mu_range": [float(mu.min()), float(mu.max())],
            "sigma_eff_range_GPa": [float(sig0.min()), float(sig0.max())],
            "tau_load_mag_range": [float(mag.min()), float(mag.max())],
            "n_unloadable": int(flat.sum()),
            "units": "km, GPa"}
    return Friction(mu=mu, mu_dynamic=mu * float(drop_ratio),
                    sigma_eff_0=sig0, tau_load=tau_load,
                    sigma_n_load=sigma_n_load, slip_dir=slip_dir,
                    tau_load_mag=mag, meta=meta)
