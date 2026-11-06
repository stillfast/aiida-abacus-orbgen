"""``orbgen.nao`` -- contract the NSW primitive into the final NAO by spillage minimisation.

Given the SCF results obtained *with the primitive basis* (overlap / Hamiltonian /
kinetic matrices and the wavefunction coefficients), SIAB minimises the generalised
spillage

    S = sum_n <psi_n| (1 - P) |psi_n> ,      P = projector onto the contracted set

over the contraction coefficients.  The reference states are the primitive-basis
eigenstates, which is why this step needs the *retrieved* SCF outputs and not just the
energies.

The fit itself is the ``orbgen`` CLI of ABACUS-CSW-NAO.  It is a Python entry point, not
an external code, so this is a ``calcfunction`` as well.

Inputs
------
nsw_orbital : SinglefileData
    Primitive orbital produced by :func:`..calculations.nsw.orbgen_nsw`.
scf_outputs : dict[str, FolderData]
    One retrieved ``OUT.ABACUS`` per reference geometry, keyed by folder name.
orbgen_config : Dict
    The ``orbgen.json`` describing ``orbitals`` (the ``nzeta`` schemes), ``geoms`` and
    the DFT parameters.  ``abacus_command`` is neutralised internally so the fit can
    never start ABACUS on the machine running the daemon.

Outputs
-------
nao_orbitals : dict of SinglefileData, ``nao_info`` : Dict
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile

from aiida.engine import calcfunction
from aiida.orm import Dict, SinglefileData

from ..utils import decode_config

# Anything that looks like a real ABACUS launch is replaced by a no-op.  The fit only
# reads the numbers from OUT.ABACUS; it must never trigger a calculation, and in
# particular never on the daemon host.
_NOOP_ABACUS = "echo ORBGEN-NAO-SKIPPED-ABACUS"


@calcfunction
def orbgen_nao(nsw_orbital: SinglefileData, orbgen_config: Dict, folder_names: Dict,
               **scf_outputs) -> dict:
    """Run the spillage fit and return the contracted orbitals.

    ``scf_outputs`` is a dynamic namespace rather than a plain dict: AiiDA serialises a
    dict argument into a ``Dict`` node, which (a) cannot hold ``FolderData`` values and
    (b) rejects the dots in names like ``U-dimer-2.10-12au``.  The namespace keys are
    therefore the sanitised spellings and ``folder_names`` maps them back.
    """
    mapping = decode_config(folder_names.get_dict())
    outputs = {mapping.get(key, key): node for key, node in scf_outputs.items()}
    cfg = json.loads(json.dumps(decode_config(orbgen_config.get_dict())))
    cfg["abacus_command"] = _NOOP_ABACUS
    # the fit reads the primitive from `primitive_jy/` next to the config
    cfg.setdefault("primitive_dir", "primitive_jy")

    workdir = tempfile.mkdtemp(prefix="orbgen_nao_")
    prim_dir = os.path.join(workdir, "primitive_jy")
    os.makedirs(prim_dir, exist_ok=True)
    # SinglefileData has no get_file() in AiiDA 2.x -- use get_content(mode="rb")
    with open(os.path.join(prim_dir, nsw_orbital.filename), "wb") as handle:
        handle.write(nsw_orbital.get_content(mode="rb"))

    # materialise one folder per geometry with its retrieved OUT.ABACUS
    for folder, retrieved in outputs.items():
        dest = os.path.join(workdir, folder)
        os.makedirs(dest, exist_ok=True)
        _copy_tree(retrieved, dest)

    cfg_path = os.path.join(workdir, "orbgen.fitonly.json")
    with open(cfg_path, "w") as handle:
        json.dump(cfg, handle, indent=4)

    run_dir = os.path.join(workdir, "run")
    log_path = os.path.join(workdir, "fit.log")
    proc = subprocess.run(
        ["orbgen", "-i", cfg_path, "-o", run_dir, "-l", log_path],
        cwd=workdir, capture_output=True, text=True, timeout=6 * 3600,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-30:])
        raise RuntimeError(f"orbgen fit failed (rc={proc.returncode}):\n{tail}")

    orbitals, spillages, filenames = {}, [], {}
    for name in sorted(os.listdir(run_dir)):
        if not name.endswith(".orb"):
            continue
        node = SinglefileData(file=os.path.join(run_dir, name), filename=name)
        # A calcfunction that returns a dict of nodes turns each key into an output
        # link label, and those only allow alphanumerics and underscores -- a key of
        # "U_gga_..._2s1p1d1f.orb" is rejected with
        #   invalid link label `nao_orbitals__..._2s1p1d1f.orb`
        key = re.sub(r"[^0-9a-zA-Z_]", "_", name)
        orbitals[key] = node
        filenames[key] = name
    with open(log_path) as handle:
        for line in handle:
            if "orbital optimization ends with spillage" in line:
                spillages.append(float(line.rsplit("=", 1)[-1]))

    info = {"orbitals": sorted(filenames.values()), "filenames": filenames,
            "spillages": spillages}
    return {"nao_orbitals": orbitals, "nao_info": Dict(info)}


def _copy_tree(folder, dest: str) -> None:
    """Lay a retrieved folder out the way the spillage fit expects.

    The ABACUS-AiiDA plugin stores the output directory under a *renamed* folder
    (``OUT.aiida`` by default), so the contents have to be copied into ``OUT.ABACUS``
    one by one -- copying the top level naively would hand the fit a directory entry
    where it expects a file.
    """
    target = os.path.join(dest, "OUT.ABACUS")
    os.makedirs(target, exist_ok=True)

    top = folder.list_object_names()
    sub = next((n for n in ("OUT.aiida", "OUT.ABACUS") if n in top), None)
    if sub is None:
        raise ValueError(f"retrieved folder has no OUT.* directory (has {top})")

    def _extract(relpath: str, destino: str) -> None:
        # FolderData.open() returns an Iterator[BinaryIO]; there is no copyfile() on the
        # File objects (AttributeError: 'File' object has no attribute 'copyfile')
        with folder.open(relpath, mode="rb") as src, open(destino, "wb") as dst:
            dst.write(src.read())

    for name in folder.list_object_names(sub):
        _extract(f"{sub}/{name}", os.path.join(target, name))

    # the fit wants INPUT/STRU at the geometry level (they live inside OUT.* for ABACUS)
    for name in ("INPUT", "STRU"):
        if name in folder.list_object_names(sub):
            _extract(f"{sub}/{name}", os.path.join(dest, name))
