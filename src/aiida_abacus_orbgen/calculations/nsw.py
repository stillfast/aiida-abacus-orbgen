"""``orbgen.nsw`` -- let SIAB generate the primitive basis **and** the ABACUS input deck.

The whole point of this step is that nothing here re-derives what SIAB already knows.
``generate_all`` writes, for one ``(r_cut, ecutjy, l_max)``:

    {out}/primitive_jy/{elem}_{xc}_{r_cut}au_{ecutjy}Ry_{nzeta}.orb   the NSW basis
    {out}/{dft_folder}/INPUT                                        SIAB's parameters
    {out}/{dft_folder}/STRU                                         SIAB's geometry

and this calculation ships all three into the AiiDA graph:

    nsw_orbital  SinglefileData                the primitive basis
    dft_inputs   namespace of SinglefileData   SIAB's INPUT, one per reference geometry
    structures   namespace of StructureData    SIAB's STRU, parsed back
    nsw_info     Dict                          folder names, nzeta, nbf/atom

Why bother taking SIAB's INPUT verbatim: hand-writing the parameters (the previous
version of this plugin did) silently drops things SIAB sets -- ``bessel_nao_rcut``,
and the per-geometry ``nbands`` (the monomer needs
``__iop_spill_guess_atomic_nbands__`` while the dimers use ``geoms[0].nbands``).  Those
mismatches only show up much later, inside the spillage fit.
"""
from __future__ import annotations

import json
import os
import re
import tempfile

from aiida.engine import calcfunction
from aiida.orm import Dict, SinglefileData, StructureData

from ..utils import decode_config

#: keys the AiiDA plugin writes itself; keeping them would fight the plugin
AIIIDA_MANAGED = {"suffix", "pseudo_dir", "orbital_dir", "stru_file", "kpoint_file",
                  "wannier_card"}


def parse_input_text(text: str) -> dict:
    """ABACUS ``INPUT`` -> dict, exactly as written (values keep their spelling)."""
    out = {}
    for raw in text.splitlines():
        line = raw.split("//")[0].split("#")[0].strip()
        if not line or line.upper().startswith("INPUT_PARAMETERS"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts[0], parts[1].strip()
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        out[key] = value
    return out


def structure_from_coords(coords, cell_a: float) -> StructureData:
    """SIAB writes ``Cartesian_angstrom_center_xyz`` in a cubic cell of side ``cell_a``."""
    stru = StructureData(cell=[[cell_a, 0.0, 0.0], [0.0, cell_a, 0.0], [0.0, 0.0, cell_a]])
    for xyz in coords:
        stru.append_atom(symbols="U", position=tuple(xyz))
    return stru


@calcfunction
def orbgen_nsw(siab_config: Dict, settings: Dict) -> dict:
    """Generate the NSW primitive basis plus the ABACUS input deck, via SIAB."""
    from aiida_orbgen.interfaces.pipeline import generate_all

    cfg = decode_config(siab_config.get_dict())
    s = settings.get_dict()

    r_cut = float(s["r_cut"])
    ecutjy = float(s["ecutjy"])
    l_max = int(s["l_max"])
    dr = float(s.get("dr", 0.01))

    cfg = json.loads(json.dumps(cfg))                 # deep copy, keep the node immutable
    cfg["bessel_nao_rcut"] = [int(r_cut) if float(r_cut).is_integer() else r_cut]
    cfg["ecutjy"] = ecutjy
    if cfg.get("geoms"):
        cfg["geoms"][0]["lmaxmax"] = l_max

    workdir = tempfile.mkdtemp(prefix="orbgen_nsw_")
    result = generate_all(cfg, output_root=workdir, lmaxmax=l_max, dr=dr)

    orbital = SinglefileData(file=result["nsw"], filename=os.path.basename(result["nsw"]))

    dft_inputs, structures, folders, params = {}, {}, [], {}
    for i, job in enumerate(result["dft"]):
        folder = os.path.basename(job["folder"])
        key = re.sub(r"[^0-9a-zA-Z_]", "_", folder)     # valid AiiDA link label
        with open(job["input"]) as handle:
            text = handle.read()
        dft_inputs[key] = SinglefileData(file=job["input"], filename="INPUT")
        parsed = parse_input_text(text)
        params[folder] = {k: v for k, v in parsed.items() if k not in AIIIDA_MANAGED}

        celldm = float(cfg["geoms"][0].get("celldm", 35.0))
        from aiida_orbgen.interfaces.stru import generate_atom_coords
        # Use the prototype generate_all reports for *this* folder.  Taking
        # geoms[0].proto instead makes the monomer a dimer: two U atoms land on top of
        # each other and ABACUS dies with
        #   "two atoms are too close! type:U atom 1 / atom 2, distance = 0 Bohr"
        coords = generate_atom_coords(job.get("proto") or cfg["geoms"][0].get("proto", "dimer"),
                                      float(job.get("pert", 0.0)), celldm)
        structures[key] = structure_from_coords(coords, celldm * 0.529177210903)
        folders.append(folder)

    info = {
        "element": cfg.get("element") or cfg.get("elem"),
        "filename": orbital.filename,
        "r_cut": r_cut,
        "ecutjy": ecutjy,
        "l_maxmax": l_max,
        "nbf_per_atom": _count_basis_functions(orbital.filename),
        "dft_folders": folders,
        "folder_map": {re.sub(r"[^0-9a-zA-Z_]", "_", f): f for f in folders},
        "parameters": params,          # what SIAB says each SCF should use
        "workdir": workdir,            # provenance only
    }
    return {"nsw_orbital": orbital, "dft_inputs": dft_inputs,
            "structures": structures, "nsw_info": Dict(info)}


def _count_basis_functions(filename: str) -> int:
    """``U_gga_12au_125Ry_41s41p40d40f39g.orb`` -> 995."""
    core = filename.rsplit(".", 1)[0]
    groups = re.findall(r"(\d+)([spdfgh])", core)
    return sum(int(n) * (2 * "spdfgh".index(l) + 1) for n, l in groups)
