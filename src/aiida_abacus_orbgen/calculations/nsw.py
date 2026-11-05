"""``orbgen.nsw`` -- generate the NSW (numerical spherical wave) primitive basis.

The primitive basis is fully determined by the pseudopotential plus three numbers:

    (r_cut [a.u.], ecutjy [Ry], l_max)

It is produced by truncated spherical Bessel functions, one set per angular momentum,
whose count per ``l`` follows from ``ecutjy`` and ``r_cut``.  That is a pure Python /
Numpy operation (no ABACUS run), so this step is a ``calcfunction`` and its output is
cached by AiiDA whenever the same triple is requested again -- which matters a lot for
``abacus.orbgen.converge``, where many grid points share the same primitive.

Outputs
-------
nsw_orbital : SinglefileData
    ``{elem}_{xc}_{r_cut}au_{ecutjy}Ry_{nzeta}.orb`` -- the primitive basis.
nsw_info : Dict
    ``{filename, r_cut, ecutjy, l_maxmax, nzeta, nbf_per_atom, nbes_per_l}``.
"""
from __future__ import annotations

import json
import os
import tempfile

from aiida.engine import calcfunction
from aiida.orm import Dict, SinglefileData

from ..utils import decode_config


@calcfunction
def orbgen_nsw(siab_config: Dict, settings: Dict) -> dict:
    """Generate the NSW primitive orbital for one ``(r_cut, ecutjy, l_max)``.

    Parameters
    ----------
    siab_config
        The SIAB/orbgen configuration dict (element, pseudopotential, ``ecutjy``,
        ``ecutwfc``, ``geoms``, ...).  Overrides from ``settings`` are applied on top.
    settings
        ``r_cut``, ``ecutjy``, ``l_max``, ``dr`` (optional), ``xc`` (optional).
        This is the part ``abacus.orbgen.converge`` scans.

    Returns
    -------
    dict with keys ``nsw_orbital`` (SinglefileData) and ``nsw_info`` (Dict).
    """
    from aiida_orbgen.interfaces.nsw import generate_nsw

    cfg = decode_config(siab_config.get_dict())
    s = settings.get_dict()

    r_cut = float(s["r_cut"])
    ecutjy = float(s["ecutjy"])
    l_max = int(s["l_max"])
    dr = float(s.get("dr", 0.01))

    # the primitive carries r_cut / ecutjy inside its name, so override them in the
    # SIAB config rather than only in the filename
    cfg = json.loads(json.dumps(cfg))          # deep copy, keep the node immutable
    cfg["bessel_nao_rcut"] = [r_cut]
    cfg["ecutjy"] = ecutjy
    if "geoms" in cfg and cfg["geoms"]:
        cfg["geoms"][0]["lmaxmax"] = l_max

    workdir = tempfile.mkdtemp(prefix="orbgen_nsw_")
    out_dir = os.path.join(workdir, "primitive_jy")
    path = generate_nsw(cfg, output_dir=out_dir, lmaxmax=l_max, dr=dr)

    filename = os.path.basename(path)
    orbital = SinglefileData(file=path, filename=filename)

    nbf = _count_basis_functions(filename)

    # The spillage fit locates each reference geometry by a folder name that SIAB
    # derives from (element, prototype, bond length, r_cut).  Publish those names so the
    # caller can key the SCF outputs with exactly them -- a mismatch here makes the fit
    # silently skip geometries.
    from aiida_orbgen.interfaces.nsw import folder_rcut
    from aiida_orbgen.interfaces.stru import dft_folder_name

    # SIAB renders r_cut with str(), so 12.0 becomes "12.0au"; its own pipeline
    # normalises through folder_rcut() first and therefore writes "12au".  Match that or
    # the fit will look for folders that do not exist.
    canonical_rcut = folder_rcut(r_cut)

    folders = []
    for geom in cfg.get("geoms", []):
        pertmags = geom.get("pertmags") or []
        if not isinstance(pertmags, list):
            pertmags = []
        for pert in pertmags:
            elem = cfg.get("element") or cfg.get("elem")
            folders.append(dft_folder_name(elem, geom.get("proto", "dimer"),
                                           pert, canonical_rcut))

    info = {
        "filename": filename,
        "r_cut": r_cut,
        "ecutjy": ecutjy,
        "l_maxmax": l_max,
        "nbf_per_atom": nbf,
        "dft_folders": folders,
        "workdir": workdir,                    # provenance only, scratch space
    }
    return {"nsw_orbital": orbital, "nsw_info": Dict(info)}


def _count_basis_functions(filename: str) -> int:
    """``U_gga_12au_125Ry_41s41p40d40f39g.orb`` -> 995.

    Reads the ``nX`` groups out of the name, so it does not need to parse the file.
    """
    import re

    core = filename.rsplit(".", 1)[0]
    groups = re.findall(r"(\d+)([spdfgh])", core)
    return sum(int(n) * (2 * "spdfgh".index(l) + 1) for n, l in groups)
