"""The network's soft modes: what it would localize onto, and how softly.

WHY THE SPECTRUM OF THE ELASTIC OPERATOR IS THE LOCALIZATION QUESTION.
``-P`` maps a slip distribution to the shear-stress drop it induces, so
``v^T (-P) v`` is the STIFFNESS of slip pattern ``v`` -- how much stress that
pattern relieves per unit slip. The eigenvectors of ``sym(-P)`` with the
SMALLEST eigenvalues are therefore the network's most compliant collective
slip patterns: the shapes it can deform in while relieving least stress.

THIS IS AN ELASTIC-INTERACTION EFFECT, NOT A GEOMETRIC ONE, and the
distinction matters because an earlier note here had it backwards. With no
interaction ``P`` is diagonal, every eigenvalue equals ``|P_ii|``, and there
are NO soft modes at all -- every slip pattern is equally stiff and there is
nothing to localize onto. The entire low-lying spectrum exists because
elements talk to each other.

The structure is specific. ``-P`` has positive diagonal ``|P_ii|`` and
off-diagonal ``-P_ij``, so a PROMOTING pair (``P_ij > 0``, j's slip pushes i
toward failure) contributes a NEGATIVE off-diagonal. A matrix with positive
diagonal and negative off-diagonals is cooperative: its lowest eigenvector has
same-sign components, meaning those elements slip TOGETHER and their mutual
promotion partly cancels each one's self-stiffness. So the soft modes are the
mutually-promoting subsets -- which, from the two-element map, means the
near-coplanar collinear ones. An incipient fault.

WHY THIS PREDICTS LOCALIZATION BEFORE ANY FRICTION IS CHOSEN. Rudnicki & Rice
(1975) put localization onset at loss of ellipticity of the incremental
operator, and the standard consequence is that the problem then has no unique
solution -- localization IS loss of uniqueness. Our discrete analogue is the
positive definiteness of ``sym(-P)``, which is exactly what makes the LCP
uniquely solvable. Add slip weakening at rate ``W`` and the tangent operator
becomes ``sym(-P) - W``; a SINGLE patch goes unstable when ``W > |P_ii|``, but
a COLLECTIVE mode needs only ``W > lambda_min``. Measured here, lambda_min is
2.4 against diagonal entries of tens of GPa/km, so the network destabilises
collectively at a weakening rate roughly an order of magnitude below anything
that would destabilise one patch -- and the eigenvector says which structures
go first.

So the friction coefficient and the load decide WHETHER the threshold is
reached. The elastic spectrum decides WHAT localizes when it is.
"""
from __future__ import annotations

import dataclasses

import numpy as np

#: Modes kept. The interesting physics is the bottom of the spectrum; 64 is
#: enough to see whether the soft end is a few isolated structures or a
#: continuum, without carrying a 3 GiB eigenvector block around.
N_MODES = 64


@dataclasses.dataclass
class Mode:
    """One soft mode, with the measurements that answer "how localized?"."""

    index: int
    eigenvalue: float           # stiffness of this slip pattern
    n_effective: float          # 1/sum(v^4): participating elements
    sign_coherence: float       # |sum v| / sum|v|; 1 = all slipping together
    n_faults_90: int            # faults carrying 90 % of the mode's weight
    planarity: float
    linearity: float
    extent_km: float            # rms spread of the participating elements
    top_fault: int
    top_fault_share: float


def soft_modes(P: np.ndarray, n_modes: int = N_MODES, *, verbose: bool = False):
    """``(eigenvalues, eigenvectors)`` for the ``n_modes`` softest patterns.

    The symmetric part is the right object even though ``P`` is markedly
    non-symmetric: ``v^T (-P) v == v^T sym(-P) v`` for every ``v``, so the
    quadratic form -- the stiffness -- sees only the symmetric part, and it is
    the quadratic form that positive definiteness and the uniqueness theorem
    are about.

    The full decomposition is taken rather than an iterative solver because
    ``mhs`` has no scipy, and because at this size the dense factorization is
    minutes rather than the hours an unpreconditioned Lanczos would need to
    resolve a clustered low end.
    """
    S = -0.5 * (P + P.T)
    if verbose:
        print(f"    eigendecomposition of sym(-P), {S.shape[0]} x {S.shape[0]}",
              flush=True)
    w, v = np.linalg.eigh(S)                 # ascending
    return w[:n_modes].copy(), np.ascontiguousarray(v[:, :n_modes])


def measure(w: np.ndarray, V: np.ndarray, net, *, share: float = 0.9):
    """Per-mode statistics. ``V`` is ``(n_elements, n_modes)``, unit columns.

    ``n_effective = 1 / sum(v^4)`` is the inverse participation ratio in its
    standard form: for a mode spread evenly over ``m`` elements it returns
    ``m``, and for one concentrated on a single element it returns 1. It is
    the direct answer to "a small number of structures, or everything at
    once", and it needs no null because its limits are exact.

    ``sign_coherence`` separates a mode in which elements slip TOGETHER from
    one in which they slip against each other. A cooperative, localizing mode
    has it near 1; an oscillatory mode that merely happens to be soft does
    not, and would be a very different physical statement.
    """
    cent = net.tris.mean(axis=1)
    out = []
    for k in range(V.shape[1]):
        v = V[:, k]
        p4 = float((v ** 4).sum())
        n_eff = 1.0 / p4 if p4 > 0 else float("nan")
        a = np.abs(v)
        sign_coh = float(abs(v.sum()) / a.sum()) if a.sum() > 0 else 0.0

        # Which faults carry the mode. Weight by v^2, the energy share.
        wgt = v ** 2
        per_fault = np.bincount(net.fault_id, weights=wgt,
                                minlength=net.n_faults)
        order = np.argsort(-per_fault)
        csum = np.cumsum(per_fault[order])
        n90 = int(np.searchsorted(csum, share * csum[-1]) + 1)

        # Shape of the participating set: the elements holding 90 % of weight.
        sel = np.zeros(v.size, bool)
        eo = np.argsort(-wgt)
        ec = np.cumsum(wgt[eo])
        sel[eo[:int(np.searchsorted(ec, share * ec[-1]) + 1)]] = True
        pts = cent[sel]
        if pts.shape[0] >= 3:
            c = pts - pts.mean(axis=0)
            ev = np.linalg.eigvalsh(c.T @ c / pts.shape[0])[::-1]
            pla = float((ev[1] - ev[2]) / ev[0]) if ev[0] > 0 else 0.0
            lin = float((ev[0] - ev[1]) / ev[0]) if ev[0] > 0 else 0.0
            extent = float(np.sqrt(ev.sum()))
        else:
            pla = lin = extent = 0.0

        out.append(Mode(index=k, eigenvalue=float(w[k]), n_effective=n_eff,
                        sign_coherence=sign_coh, n_faults_90=n90,
                        planarity=pla, linearity=lin, extent_km=extent,
                        top_fault=int(order[0]),
                        top_fault_share=float(per_fault[order[0]]
                                              / per_fault.sum())))
    return out


def weakening_threshold(w: np.ndarray, diag: np.ndarray) -> dict:
    """Where collective instability beats single-patch instability.

    A patch with slip-weakening rate ``W`` goes unstable alone when
    ``W > |P_ii|``. A collective mode goes unstable when ``W`` exceeds that
    mode's eigenvalue. The RATIO of the two is how much earlier the network
    localizes than any element does on its own, and it is the one number that
    says whether collective localization is a real effect here or a rounding
    error.
    """
    lam = float(w[0])
    d = np.abs(diag)
    return {"lambda_min": lam,
            "diag_median": float(np.median(d)),
            "diag_min": float(d.min()),
            "collective_advantage_median": float(np.median(d) / lam),
            "collective_advantage_min": float(d.min() / lam),
            "note": "a collective mode destabilises at a weakening rate this "
                    "many times smaller than the softest single patch needs"}
