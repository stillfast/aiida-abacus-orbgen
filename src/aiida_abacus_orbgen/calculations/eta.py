"""``orbgen.eta`` -- band-structure error between two SCF outputs, on demand.

``istate.info`` is the one ABACUS output the AiiDA plugin does *not* retrieve by default
(see ``abacus.orbgen.base``, which adds it to ``additional_retrieve_list``).  This
calculation takes two such files -- typically the LCAO run with the contracted basis and
the plane-wave reference at the same geometry -- and returns the paper's eta.
"""
from __future__ import annotations

from aiida.engine import calcfunction
from aiida.orm import Dict, SinglefileData

from ..utils_eta import eta as _eta
from ..utils_eta import (band_span_above_fermi, eta10_is_valid, model_occupation,
                         parse_istate_info)


@calcfunction
def orbgen_eta(istate_lcao: SinglefileData, istate_pw: SinglefileData,
               settings: Dict) -> Dict:
    """eta (and eta_10 when the spectrum reaches far enough) between two SCF outputs."""
    s = settings.get_dict()
    lift = float(s.get("lift_eV", 10.0))
    sigma_ev = float(s.get("sigma_eV", 0.2041))          # 0.015 Ry
    eps_f = s.get("eps_f_eV")

    E_a, O_a, _ = parse_istate_info(istate_lcao.get_content(mode="r"))
    E_b, O_b, _ = parse_istate_info(istate_pw.get_content(mode="r"))

    value, w, _ = _eta(E_a, O_a, E_b, O_b)
    result = {
        "eta_meV": value * 1000.0,
        "shift_meV": w * 1000.0,
        "nkpts": int(min(E_a.shape[0], E_b.shape[0])),
        "nbands": int(min(E_a.shape[1], E_b.shape[1])),
    }

    if eps_f is not None:
        eps_f = float(eps_f)
        f_a = model_occupation(E_a, eps_f + lift, sigma_ev)
        f_b = model_occupation(E_b, eps_f + lift, sigma_ev)
        valid = eta10_is_valid(E_b, eps_f, lift, sigma_ev)
        value10, _, _ = _eta(E_a, f_a, E_b, f_b)
        result.update({
            "eta10_meV": value10 * 1000.0, "eta10_valid": bool(valid),
            "lift_eV": lift,
            "span_above_fermi_eV": band_span_above_fermi(E_b, eps_f),
        })
        if not valid:
            result["eta10_note"] = (
                "the spectrum stops {:.2f} eV above eps_F, less than the {:g} eV lift, so "
                "eta_10 degenerates to an unweighted RMS over all bands".format(
                    result["span_above_fermi_eV"], lift))
    return Dict(dict=result)
