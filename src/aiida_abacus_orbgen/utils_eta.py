"""The eta metric (paper eqs 12-13), free of any AiiDA import so it can be unit-tested.

    eta(A,B) = min_w sqrt( sum_nk f~ (e_A - e_B + w)^2 / sum_nk f~ ),  f~ = sqrt(f_A f_B)

with f the *smeared* occupations, so the geometric mean of the two band structures is
what weights the sum.  ``eta10`` is the same thing with the Fermi level lifted by 10 eV,
which is only meaningful if the eigenvalues actually reach that far above eps_F -- the
``band_span_above_fermi`` helper exists so a caller can refuse to quote it otherwise.
"""
from __future__ import annotations

import re

import numpy as np

KCAL_MEV = 43.364


def parse_istate_info(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ABACUS ``istate.info`` -> (E[nk, nb], occ[nk, nb], kpts[nk, 3])."""
    # two spellings exist in the wild:
    #   running_scf.log :  " 1/360 kpoint (Cartesian) = 0.0000 0.0000 0.0000 (231 pws)"
    #   istate.info     :  "BAND  Energy(ev)  Occupation   Kpoint = 1        (0 0 0)"
    kpat = re.compile(r"^\s*(\d+)/(\d+)\s+kpoint\s*\(Cartesian\)\s*=\s*"
                      r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)")
    kpat_info = re.compile(r"Kpoint\s*=\s*\d+\s*\(([^)]*)\)")
    bpat = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+([0-9.eE+\-]+)\s*$")
    blocks, kpts, cur = [], [], None
    for line in text.splitlines():
        m = kpat.match(line)
        if m:
            cur = {}
            blocks.append(cur)
            kpts.append(tuple(float(m.group(i)) for i in (3, 4, 5)))
            continue
        m = kpat_info.search(line)
        if m:                                  # istate.info spelling
            cur = {}
            blocks.append(cur)
            kpts.append(tuple(float(v) for v in m.group(1).split()))
            continue
        m = bpat.match(line)
        if m and cur is not None:
            cur[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    if not blocks:
        raise ValueError("no k-point blocks in istate.info")
    nb = max(max(b) for b in blocks)
    E = np.full((len(blocks), nb), np.nan)
    O = np.full((len(blocks), nb), np.nan)
    for ik, blk in enumerate(blocks):
        for ib, (e, o) in blk.items():
            E[ik, ib - 1] = e
            O[ik, ib - 1] = o
    return E, O, np.array(kpts)


def model_occupation(E: np.ndarray, eps_f: float, sigma_ev: float) -> np.ndarray:
    """``0.5 erfc((E - eps_F)/sigma)`` -- used for the lifted-Fermi weights."""
    from scipy.special import erfc
    return 0.5 * erfc((E - eps_f) / sigma_ev)


def eta(E_a, O_a, E_b, O_b) -> tuple[float, float, np.ndarray]:
    """(eta in eV, optimal shift w in eV, shifted residual matrix in eV)."""
    nb = min(E_a.shape[1], E_b.shape[1])
    nk = min(E_a.shape[0], E_b.shape[0])
    ea, eb = E_a[:nk, :nb], E_b[:nk, :nb]
    f = np.sqrt(np.clip(O_a[:nk, :nb], 0, None) * np.clip(O_b[:nk, :nb], 0, None))
    d = ea - eb
    good = np.isfinite(d) & np.isfinite(f)
    if not np.any(f[good] > 0):
        return float("nan"), 0.0, d
    w = -np.sum((d * f)[good]) / np.sum(f[good])
    res = d + w
    value = float(np.sqrt(np.sum((res ** 2 * f)[good]) / np.sum(f[good])))
    return value, float(w), res


def band_span_above_fermi(E: np.ndarray, eps_f: float) -> float:
    """How far the spectrum reaches above eps_F -- eta10 needs >= lift + 3 sigma."""
    finite = E[np.isfinite(E)]
    return float(finite.max() - eps_f) if finite.size else 0.0


def eta10_is_valid(E: np.ndarray, eps_f: float, lift_ev: float, sigma_ev: float) -> bool:
    return band_span_above_fermi(E, eps_f) >= lift_ev + 3.0 * sigma_ev
