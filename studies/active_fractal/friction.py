"""Regional load, frictional strength, and the Coulomb yield function.

WHAT LAMBDA IS. The regional stress is a FULL tensor that grows along a fixed
path:

    sigma(lambda) = sigma_c I  +  lambda (D + k I)

with ``sigma_c I`` the confining part and ``lambda (D + k I)`` the tectonic
part. ``D`` is a fixed UNIT deviatoric tensor giving the SHAPE of the tectonic
stress -- its orientation and its faulting style, one of ``LOAD_TENSORS``. Its
eigenvalues are (+1, 0, -1), so its maximum shear stress is exactly 1, and
therefore

    LAMBDA IS THE MAXIMUM SHEAR STRESS OF THE TECTONIC FIELD, IN GPa,

equivalently half the differential stress ``(sigma_1 - sigma_3)/2``. A run at
``lambda = 0.077`` has a tectonic field whose maximum shear stress is 77 MPa.
Only the MAGNITUDE grows during a run; the shape ``D`` never changes, which is
what lets one scalar parameterise the whole loading history.

LAMBDA IS NOT TIME. It has units of stress, there is no rate anywhere in this
problem, and every state reached is a static equilibrium. The viewer's slider
says "load" for exactly that reason.

LAMBDA ONLY MEANS SOMETHING RELATIVE TO sigma_c. The problem is invariant under
scaling ``lambda`` and ``sigma_c`` together: double both and the same elements
fail, with doubled slip. The physical parameter is the RATIO
``lambda / sigma_c``, so quoting lambda alone says nothing. At
``sigma_c = 0.15`` GPa a run at ``lambda = 0.23`` GPa has tectonic shear
exceeding the confining stress by half, which is far outside any crustal state
and is what the compressive bound below exists to catch.

``k = mean_stress_ratio`` IS THE LOADING PATH, and it is the whole difference
between a purely deviatoric load and a full one. ``D`` alone is traceless, so
it raises the mean stress not at all: the most tension-favoured planes shed
normal stress at exactly the rate their shear grows, and past
``lambda ~ sigma_c`` their EFFECTIVE NORMAL STRESS GOES NEGATIVE. Coulomb
friction does not describe that -- the fault opens -- and worse, a negative
``sigma_eff`` makes the strength ``mu sigma_eff`` negative, so such an element
satisfies the yield criterion trivially however little shear it carries, and
joins the active set for no mechanical reason at all.

    k = 0      mean stress fixed; sigma_H up and sigma_h down equally. Purely
               deviatoric, and tension appears at lambda ~ sigma_c.
    k = 2/3    only sigma_H grows, with sigma_h and sigma_v held -- the ordinary
               picture of tectonic loading at fixed depth. The mean stress then
               rises at 2/3 the rate of the maximum shear, which is where the
               value comes from rather than from fitting anything.
    k >= 1     the minimum principal stress never decreases, so nothing ever
               goes tensile and lambda may legitimately run to infinity.

An isotropic tensor resolves the SAME normal stress onto every plane and no
shear onto any, so ``k`` enters only through ``sigma_n_load``, which is stored
already shifted by it. Every downstream formula -- the yield function, the
cascade's ``d``, ``first_failure_lambda`` -- is then unchanged by construction.

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
quasi-static solution is a CONTINUOUS function of the load. Friction that does
not weaken cannot go unstable: a patch unloads itself at stiffness
``k = |K_ii|`` and instability needs ``k < |dtau/ddelta|``, which is 0 here, so
the inequality fails at every patch size and stress. Past failure the slip
grows from ZERO SLIP at a FINITE rate ``d/|K_ii|`` -- it is the slip AMOUNT at
onset that vanishes, and hence the event size, not the rate -- and nothing
ever jumps. There are then no events, no cascades and
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

``drop_ratio`` therefore decides whether events exist AT ALL -- at 1.0 there
are none by construction and the value is refused -- and it sets the moment
each carries and how many there are. It does NOT set how large they are in
elements: the drop cancels from the triggering ratio, because a failing
element slips ``Delta_tau/|K_ii|`` while the gap its neighbour must re-close
after relocking is itself proportional to ``Delta_tau``. Measured, a 25x
change in the drop moved mean avalanche size by under 2 %. An earlier version
of this note called it "the single most important physical parameter", which
``kernel.branching_ratio``'s own docstring contradicts.

NONE OF WHICH THE STATIC PROBLEM NEEDS. At one load the single-coefficient
problem is an LCP with a unique solution -- see ``static.py`` -- and the two
static solutions bracket the whole stick-slip history exactly.

Units are km and GPa, the project's convention. 1 MPa = 1e-3 GPa.
"""
from __future__ import annotations

import dataclasses
import math

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


def rotate_about_vertical(sigma: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a stress tensor counter-clockwise about the vertical axis.

    The regional stress has an ORIENTATION as well as a shape and a size, and
    it is the one parameter a fault network cannot be re-meshed to absorb:
    rotating the stress relative to a fixed network asks which fault
    populations a given tectonic regime selects, which is what
    critically-stressed-fracture analysis is used for in practice.

    PERIOD 180 DEGREES, not 360. A stress tensor is symmetric, so sigma and
    its rotation by 180 deg are identical; and for a pure-shear deviator the
    rotation by 90 deg negates it, interchanging sigma_1 and sigma_3. Sweeping
    [0, 180) therefore covers every distinct orientation exactly once.

    Only the vertical axis is swept. Tilting the principal frame out of
    horizontal changes the FAULTING REGIME -- strike-slip to thrust or normal
    -- which ``LOAD_TENSORS`` already offers as separate named states rather
    than as a continuum.
    """
    t = math.radians(float(deg))
    c, s_ = math.cos(t), math.sin(t)
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.asarray(sigma, float) @ R.T


def resolve_tensor(tris: np.ndarray, sigma: np.ndarray, *,
                   planes: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a stress tensor onto every element's own frame.

    Returns ``(tau, sigma_n)`` with ``tau`` shaped ``(n, 2)`` -- the shear
    along (strike, dip) -- and ``sigma_n`` shaped ``(n,)``, COMPRESSION
    POSITIVE.

    This is the one helper neither repository had: ``mhs`` resolves tractions
    inside ``interaction_matrix`` and never exposes a tensor-to-scalars step,
    and the sibling project's only ``resolved_shear`` is a one-liner in a figure
    module with no normal component. Written once, here, and used by both
    friction variants so the convention cannot be restated differently twice.

    ``planes`` ``(n, 3)`` upward unit normals resolve onto those planes
    instead of the elements' -- the continuous layout's nodes, whose frame
    is the same recipe on the node normal (``tdcs.frame_from_normals``).
    """
    frame = (tdcs.slip_frame(tris) if planes is None
             else tdcs.frame_from_normals(planes))   # rows [strike, dip, tensile]
    n_hat = frame[:, 2, :]
    t = np.einsum("ij,sj->si", np.asarray(sigma, float), n_hat)
    tau = np.stack([np.einsum("si,si->s", t, frame[:, 0, :]),
                    np.einsum("si,si->s", t, frame[:, 1, :])], axis=1)
    return tau, -np.einsum("si,si->s", t, n_hat)


def slip_cartesian(tris: np.ndarray, slip_dir: np.ndarray,
                   slip: np.ndarray, *,
                   planes: np.ndarray | None = None) -> np.ndarray:
    """``(n, 3)`` slip as a Cartesian vector, from ``(strike, dip)`` parts.

    The inverse of what :func:`resolve_tensor` does to a traction, and needed
    wherever a solved slip has to leave the per-element frame -- evaluating
    the field it radiates, for instance. ``planes`` as in
    :func:`resolve_tensor`: the nodes' frames rather than the elements'.
    """
    frame = (tdcs.slip_frame(np.asarray(tris, float).reshape(-1, 3, 3))
             if planes is None else tdcs.frame_from_normals(planes))
    d = np.asarray(slip, float)[:, None]
    sd = np.asarray(slip_dir, float)
    return d * (sd[:, 0:1] * frame[:, 0, :] + sd[:, 1:2] * frame[:, 1, :])


def induced_normal_stress(tris: np.ndarray, slip_xyz: np.ndarray,
                          points: np.ndarray, normal, material,
                          eps: float) -> np.ndarray:
    """Slip-induced ``sigma_n`` at ``points``, on planes of given ``normal``.

    COMPRESSION POSITIVE, matching :func:`resolve_tensor` rather than the
    kernel's tension-positive traction row, and built from
    ``mhs.stress_matrix`` so it cannot inherit the convention of whatever is
    being tested with it. ``slip_xyz`` is an explicit Cartesian slip vector
    per element -- see :func:`slip_cartesian` -- so no frame convention
    enters here either.

    THIS EXISTS SO HANDEDNESS IS NEVER ENCODED. Left and right stepping are a
    chirality, and mapping a rotated geometry onto the classical map view by
    hand is the kind of sign bookkeeping that has gone wrong twice in this
    work. EXTENSIONAL against CONTRACTIONAL is frame independent -- it is the
    sign of this quantity in the step BETWEEN two cracks -- so a caller labels
    a configuration by MEASURING it, and the label cannot drift.

    WHAT IT SETTLED. Which stepping sense is extensional DEPENDS ON THE
    OFFSET: the step's own sigma_n changes sign as the offset grows, crossing
    zero near dz/a = 0.27 at a centre separation of 2.2 a. So the reversal in
    the linkage asymmetry is largely a RELABELLING rather than a change of
    physics, and Segall & Pollard's relation -- the extensional step unclamps
    and facilitates -- holds on both sides of it wherever the effect is
    appreciable, meaning every offset whose asymmetry exceeds 5 %.

    IT IS NOT QUITE EXACT, which is recorded rather than smoothed over. The
    two zero crossings do not coincide: the step's sigma_n turns over near
    dz/a = 0.27 and the asymmetry near 0.34, leaving a narrow window where the
    CONTRACTIONAL step is nominally favoured, by 0.2 to 2.6 %. Whether that
    window is real is UNRESOLVED -- its width, 0.07 in dz/a, is comparable to
    the uncertainty in the asymmetry's own crossing, which shifts by about
    0.05 when the measuring arc is widened.
    """
    import mhs                                                # noqa: PLC0415
    tri = np.asarray(tris, float).reshape(-1, 3, 3)
    nh = np.asarray(normal, float)
    nh = nh / np.linalg.norm(nh)
    S = mhs.stress_matrix(np.asarray(points, float).reshape(-1, 3), tri,
                          material, eps)
    sig = np.einsum("omnsk,sk->omn", S, np.asarray(slip_xyz, float))
    del S
    return -np.einsum("om,m->o", np.einsum("omn,n->om", sig, nh), nh)


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

    def strength_at(self, lam: float) -> np.ndarray:
        """STATIC strength at load ``lam``: ``mu (sigma_eff_0 + lam sigma_n_load)``,
        the normal-stress part of the load folded in. It is the threshold the
        scalar LCP's ``lam d + q <= g`` rearranges to, and the disc radius of
        ``static.solve_cone``; negative where the load has opened the element,
        which ``Friction.lambda_tensile`` bounds."""
        return self.mu * (self.sigma_eff_0 + float(lam) * self.sigma_n_load)

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

        THAT ``inf`` IS A PHYSICAL STATEMENT, NOT A GUARD. The load clamps as
        well as shears, so strength grows with ``lambda`` too, and an element
        with ``|tau| <= mu sigma_n`` per unit load never fails AT ANY LOAD.
        The locked set is therefore fixed by orientation and ``mu`` alone and
        does not shrink as the experiment runs: see ``locked_fraction``.
        """
        denom = self.tau_load_mag - self.mu * self.sigma_n_load
        out = np.full(denom.shape, np.inf)
        ok = denom > 0.0
        out[ok] = self.strength()[ok] / denom[ok]
        return out

    def locked_fraction(self) -> float:
        """Share of elements that CANNOT fail at any ``lambda``.

        The complement is the asymptotic active set of the no-interaction
        problem. Measured at ``mu = 0.6`` on the published power-law network it
        is 0.102, so 89.8 % is the ceiling rather than 100 %.
        """
        return float(np.mean(self.tau_load_mag
                             - self.mu * self.sigma_n_load <= 0.0))

    def lambda_tensile(self) -> np.ndarray:
        """``lambda`` at which each element's EFFECTIVE NORMAL STRESS hits zero.

        ``inf`` where the loading path never unclamps the plane. Beyond this
        value Coulomb friction is the wrong constitutive law -- the fault
        opens -- and the yield function actively misbehaves, because a negative
        ``sigma_eff`` gives a negative strength that any shear exceeds.
        """
        out = np.full(self.sigma_n_load.shape, np.inf)
        shed = self.sigma_n_load < 0.0
        out[shed] = self.sigma_eff_0[shed] / -self.sigma_n_load[shed]
        return out

    def max_lambda_compressive(self) -> float:
        """The largest ``lambda`` at which every element is still in
        compression -- the hard upper bound on a defensible run.

        ``inf`` when ``mean_stress_ratio >= 1``, because the loading path then
        never reduces the minimum principal stress.
        """
        return float(self.lambda_tensile().min())


def build_friction(net, *, regime: str = "strike_slip",
                   mu_mode: str = "uniform", mu_mean: float = 0.6,
                   mu_spread: float = 0.0, drop_ratio: float = DROP_RATIO,
                   sigma_n_mode: str = "uniform",
                   sigma_eff: float = SIGMA_EFF_UNIFORM,
                   pore_ratio: float = RHO_W_G / RHO_G,
                   mean_stress_ratio: float = 0.0,
                   azimuth_deg: float = 0.0,
                   seed: int = 0, nodes=None) -> Friction:
    """Strength and resolved load for a network.

    ``nodes`` -- an ``mhs.SharedNodes`` table of ``net.tris`` -- builds the
    same thing AT THE NODES of the continuous layout: the load resolved on
    each node's own plane, the strength from its depth, ``per_fault``
    friction from the group it was welded in. Every array is then one per
    node rather than one per element, and nothing downstream changes, since
    the LCP never asks what an unknown is.

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

    ``regime`` is a name from ``LOAD_TENSORS`` or an explicit symmetric
    ``(3, 3)`` unit deviator in the same tension-positive convention, for a
    study that sweeps the SHAPE of the regional stress and not only its
    orientation (``taiwan.py``'s family between thrust and strike-slip). A
    name resolves to its tensor and nothing else changes, so the two spellings
    of one state give the same Friction bit for bit; ``meta["regime"]`` keeps
    the name, or ``"tensor"``, with the tensor beside it either way.
    """
    tris = net.tris
    planes = None if nodes is None else nodes.normals
    points = tris.mean(axis=1) if nodes is None else nodes.coords
    fault_id = net.fault_id if nodes is None else nodes.group
    n = points.shape[0]
    rng = np.random.default_rng(seed)

    if not (0.0 < drop_ratio < 1.0):
        raise ValueError(
            f"drop_ratio must lie strictly in (0, 1); got {drop_ratio}. At 1.0"
            f" there is no strength drop, so quasi-static slip is continuous in"
            f" the load and the run would record events of zero size forever"
            f" instead of avalanches.")
    if isinstance(regime, str):
        if regime not in LOAD_TENSORS:
            raise ValueError(f"regime must be one of {sorted(LOAD_TENSORS)}, "
                             f"got {regime!r}")
        tensor, regime_name = LOAD_TENSORS[regime], regime
    else:
        tensor = np.asarray(regime, float)
        if tensor.shape != (3, 3) or not np.allclose(tensor, tensor.T, atol=1e-12):
            raise ValueError("an explicit regime must be a symmetric (3, 3) "
                             f"stress tensor; got shape {tensor.shape}")
        tensor = 0.5 * (tensor + tensor.T)      # a rotation's 1e-17 asymmetry
        regime_name = "tensor"
    if mean_stress_ratio < 0.0:
        raise ValueError(
            f"mean_stress_ratio must be >= 0; got {mean_stress_ratio}. A "
            f"negative value would make the loading path REDUCE the mean "
            f"stress, driving every plane toward tension at once.")
    tau_load, sigma_n_load = resolve_tensor(
        tris, rotate_about_vertical(tensor, azimuth_deg), planes=planes)
    # The isotropic part of the loading path. It resolves the same normal
    # stress onto every plane and no shear onto any, so folding it in here
    # leaves every downstream formula untouched. See the module docstring.
    sigma_n_load = sigma_n_load + float(mean_stress_ratio)

    if mu_mode == "uniform":
        mu = np.full(n, float(mu_mean))
    elif mu_mode == "per_fault":
        per = np.clip(rng.normal(mu_mean, mu_spread * mu_mean, net.n_faults),
                      0.05, 2.0)
        mu = per[fault_id]
    elif mu_mode == "per_element":
        mu = np.clip(rng.normal(mu_mean, mu_spread * mu_mean, n), 0.05, 2.0)
    else:
        raise ValueError(f"mu_mode must be 'uniform', 'per_fault' or "
                         f"'per_element', got {mu_mode!r}")

    if sigma_n_mode == "uniform":
        sig0 = np.full(n, float(sigma_eff))
    elif sigma_n_mode == "lithostatic":
        depth = np.abs(points[:, 2])
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

    meta = {"regime": regime_name, "tensor": tensor.tolist(),
            "mu_mode": mu_mode, "mu_mean": mu_mean,
            "mu_spread": mu_spread, "drop_ratio": drop_ratio,
            "sigma_n_mode": sigma_n_mode,
            "sigma_eff_GPa": sigma_eff, "pore_ratio": pore_ratio,
            "seed": seed, "n_elements": n,
            "unknowns": "elements" if nodes is None else "shared nodes",
            "mu_range": [float(mu.min()), float(mu.max())],
            "sigma_eff_range_GPa": [float(sig0.min()), float(sig0.max())],
            "tau_load_mag_range": [float(mag.min()), float(mag.max())],
            "n_unloadable": int(flat.sum()),
            "mean_stress_ratio": float(mean_stress_ratio),
            "azimuth_deg": float(azimuth_deg),
            "units": "km, GPa"}
    fric = Friction(mu=mu, mu_dynamic=mu * float(drop_ratio),
                    sigma_eff_0=sig0, tau_load=tau_load,
                    sigma_n_load=sigma_n_load, slip_dir=slip_dir,
                    tau_load_mag=mag, meta=meta)
    # Recorded rather than enforced here: the bound belongs to a RUN, which is
    # what chooses lambda_max, and build_friction does not know one.
    meta["locked_fraction"] = fric.locked_fraction()
    meta["max_lambda_compressive"] = fric.max_lambda_compressive()
    return fric
