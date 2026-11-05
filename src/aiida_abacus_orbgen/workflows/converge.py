"""``abacus.orbgen.converge`` -- scan (r_cut, ecutjy, l_max), compare with PW, pick the best.

Motivation
----------
The three NSW parameters trade cost against completeness and the only honest way to
choose them is to measure the error they cause:

  1. run **one plane-wave reference** per geometry (the "truth");
  2. run :class:`OrbgenBaseWorkChain` for every point of ``r_cut x ecutjy x l_max``;
  3. score each candidate by its total-energy error against that reference and keep the
     **cheapest** one inside ``tolerance_mev``.

Cost model
----------
The SCF cost is dominated by the number of contracted functions, with ``r_cut`` as a
secondary term (it fixes the sparsity of the overlap and the Bessel count)::

    cost = nbasis_per_atom * natoms * (1 + r_cut / r_cut_reference) ** cost_exponent

Any monotone-in-size model gives the same ordering for realistic grids; the two knobs
are exposed so the ranking can be retuned without touching the selection logic.

Reference modes
---------------
``pw``  (default) one ``abacus.abacus`` run with ``basis_type: pw`` -- the paper's
        criterion, absolute completeness.
``nsw`` reuse the largest candidate.  Cheaper but it only measures ``ecutjy``
        convergence against another NAO, which is exactly the trap that makes a bad
        ``r_cut`` look fine.

Outputs
-------
table : Dict   every candidate with error, cost, pass/fail
best : Dict    the selected parameter set
"""
from __future__ import annotations

import re

from aiida import orm
from aiida.engine import WorkChain, ToContext
from aiida.plugins import CalculationFactory

from .base import OrbgenBaseWorkChain

KCAL_TO_MEV = 43.364


def energy_from_retrieved(node) -> float | None:
    """Total energy out of an ABACUS retrieved folder.

    Reads ``!FINAL_ETOT_IS`` from ``running_scf.log``.  The folder is called
    ``OUT.ABACUS`` in a plain ABACUS run and ``OUT.aiida`` when the AiiDA plugin
    renames it, so both are tried; this avoids depending on the plugin's
    ``output_parameters`` key names.
    """
    if "retrieved" not in node.outputs:
        return None
    folder = node.outputs.retrieved
    for sub in ("OUT.ABACUS", "OUT.aiida"):
        name = f"{sub}/running_scf.log"
        if name not in folder.list_object_names(sub) if sub in folder.list_object_names() else False:
            continue
        try:
            text = folder.get_object_content(name)
        except (OSError, ValueError):
            continue
        hits = re.findall(r"!FINAL_ETOT_IS\s+(-?[0-9.]+)", text)
        if hits:
            return float(hits[-1])
    return None


class OrbgenConvergeWorkChain(WorkChain):
    """Grid scan over (r_cut, ecutjy, l_max) against a plane-wave reference."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # ---- what the base workchain needs (explicit: a Dict cannot hold nodes)
        spec.input("siab_config", valid_type=orm.Dict)
        spec.input("orbgen_config", valid_type=orm.Dict)
        spec.input_namespace("structures", valid_type=orm.StructureData, dynamic=True)
        spec.input("upf", valid_type=orm.UpfData)
        spec.input("abacus_code", valid_type=orm.InstalledCode)
        spec.input("scf_parameters", valid_type=orm.Dict)
        spec.input("kpoints", valid_type=orm.KpointsData, required=False)
        spec.input("clean_workdir", valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input("scf_options", valid_type=orm.Dict, required=False,
                   help="metadata.options for every CalcJob (carries the SLURM partition)")
        spec.input("scf_settings", valid_type=orm.Dict, required=False)

        # ---- the grid
        spec.input("r_cuts", valid_type=orm.List, help="r_cut in a.u., e.g. [9,10,11,12]")
        spec.input("ecutjys", valid_type=orm.List, help="ecutjy in Ry, e.g. [100,125,150]")
        spec.input("l_maxes", valid_type=orm.List, help="l_max, e.g. [3,4]")

        # ---- selection policy
        spec.input("reference_mode", valid_type=orm.Str, default=lambda: orm.Str("pw"))
        spec.input("pw_parameters", valid_type=orm.Dict, required=False,
                   help="PW reference parameters; defaults to scf_parameters with "
                        "basis_type: pw and ks_solver: dav")
        spec.input("tolerance_mev", valid_type=orm.Float,
                   default=lambda: orm.Float(1 * KCAL_TO_MEV),
                   help="accept when |dE| <= this per atom (default 1 kcal/mol)")
        spec.input("cost_exponent", valid_type=orm.Float, default=lambda: orm.Float(1.0))
        spec.input("r_cut_reference", valid_type=orm.Float, default=lambda: orm.Float(10.0))

        spec.outline(
            cls.outline_grid,
            cls.run_reference,
            cls.run_candidates,
            cls.select,
        )

        spec.output("table", valid_type=orm.Dict)
        spec.output("best", valid_type=orm.Dict)

        spec.exit_code(320, "ERROR_NO_CANDIDATE",
                       message="no (r_cut, ecutjy, l_max) met tolerance_mev; widen the "
                               "grid or relax the tolerance")
        spec.exit_code(321, "ERROR_NO_REFERENCE",
                       message="the plane-wave reference produced no total energy")

    # ------------------------------------------------------------------ step 0
    def outline_grid(self):
        self.ctx.grid = [
            {"r_cut": float(r), "ecutjy": float(e), "l_max": int(l)}
            for l in self.inputs.l_maxes.get_list()
            for e in self.inputs.ecutjys.get_list()
            for r in self.inputs.r_cuts.get_list()
        ]
        if not self.ctx.grid:
            raise ValueError("empty (r_cut, ecutjy, l_max) grid")
        self.ctx.natoms = len(next(iter(self.inputs.structures.values())).sites)

    # ------------------------------------------------------------------ step 1
    def run_reference(self):
        if self.inputs.reference_mode.value == "nsw":
            self.report("reference_mode = nsw: the largest candidate is the reference, "
                        "so this only measures ecutjy convergence")
            return
        params = self.inputs.scf_parameters.get_dict()
        if "pw_parameters" in self.inputs:
            params = self.inputs.pw_parameters.get_dict()
        params = dict(params)
        params.setdefault("basis_type", "pw")
        params.setdefault("ks_solver", "dav")

        AbacusCalculation = CalculationFactory("abacus.abacus")
        element = self.inputs.siab_config.get_dict()["elem"]
        running = {}
        for name, structure in self.inputs.structures.items():
            inputs = {
                "code": self.inputs.abacus_code,
                "structure": structure,
                "parameters": orm.Dict(dict={"input": params}),
                "pseudos": {element: self.inputs.upf},
                "metadata": {"label": f"pwref-{name}"},
            }
            if "kpoints" in self.inputs:
                inputs["kpoints"] = self.inputs.kpoints
            if "scf_options" in self.inputs:
                inputs["metadata"]["options"] = self.inputs.scf_options.get_dict()
            running[f"pw_{name}"] = self.submit(AbacusCalculation, **inputs)
        self.report(f"submitted {len(running)} plane-wave reference SCF(s)")
        return ToContext(**running)

    # ------------------------------------------------------------------ step 2
    def run_candidates(self):
        running = {}
        for i, triple in enumerate(self.ctx.grid):
            inputs = {
                "siab_config": self.inputs.siab_config,
                "settings": orm.Dict(dict=triple),
                "orbgen_config": self.inputs.orbgen_config,
                "structures": self.inputs.structures,
                "upf": self.inputs.upf,
                "abacus_code": self.inputs.abacus_code,
                "scf_parameters": self.inputs.scf_parameters,
                "clean_workdir": self.inputs.clean_workdir,
            }
            if "kpoints" in self.inputs:
                inputs["kpoints"] = self.inputs.kpoints
            if "scf_options" in self.inputs:
                inputs["scf_options"] = self.inputs.scf_options
            if "scf_settings" in self.inputs:
                inputs["scf_settings"] = self.inputs.scf_settings
            running[f"cand_{i:03d}"] = self.submit(OrbgenBaseWorkChain, **inputs)
        self.report(f"submitted {len(running)} OrbgenBaseWorkChain candidates")
        return ToContext(**running)

    # ------------------------------------------------------------------ step 3
    def select(self):
        natoms = self.ctx.natoms
        e_ref = self._reference_energies()
        if not e_ref:
            return self.exit_codes.ERROR_NO_REFERENCE

        tol = self.inputs.tolerance_mev.value
        rc_ref = self.inputs.r_cut_reference.value
        exponent = self.inputs.cost_exponent.value

        table, best = [], None
        for i, triple in enumerate(self.ctx.grid):
            node = self.ctx.get(f"cand_{i:03d}")
            if node is None or not node.is_finished_ok:
                continue
            res = node.outputs.result.get_dict()
            errors = {}
            for name, e in res["scf_energies"].items():
                if name in e_ref and e is not None:
                    errors[name] = (e - e_ref[name]) / natoms * 1000.0     # meV/atom
            if not errors:
                continue
            mean_err = sum(errors.values()) / len(errors)
            cost = res["nbf_per_atom"] * natoms * (1.0 + triple["r_cut"] / rc_ref) ** exponent
            row = {**triple,
                   "nbasis_per_atom": res["nbf_per_atom"],
                   "mean_error_mev": mean_err,
                   "mean_error_kcal": mean_err / KCAL_TO_MEV,
                   "errors_mev": errors,
                   "spillages": res.get("spillages"),
                   "cost": cost,
                   "passes": abs(mean_err) <= tol}
            table.append(row)
            if row["passes"] and (best is None or row["cost"] < best["cost"]):
                best = row

        if best is None:
            self.report(f"no candidate within {tol:.2f} meV/atom")
            self.out("table", orm.Dict(dict={"tolerance_mev": tol, "rows": table}))
            return self.exit_codes.ERROR_NO_CANDIDATE

        self.report(f"selected r_cut={best['r_cut']} ecutjy={best['ecutjy']} "
                    f"l_max={best['l_max']}  dE = {best['mean_error_mev']:.2f} meV/atom  "
                    f"({best['nbasis_per_atom']} functions/atom)")
        self.out("table", orm.Dict(dict={"tolerance_mev": tol, "rows": table}))
        self.out("best", orm.Dict(dict=best))

    # ----------------------------------------------------------------- helpers
    def _reference_energies(self) -> dict:
        e_ref = {}
        for key, node in self.ctx.items():
            if key.startswith("pw_"):
                e = energy_from_retrieved(node)
                if e is not None:
                    e_ref[key[len("pw_"):]] = e
        if e_ref:
            return e_ref
        # reference_mode == nsw: the largest candidate becomes the reference
        candidates = [(i, self.ctx.get(f"cand_{i:03d}")) for i in range(len(self.ctx.grid))]
        candidates = [(i, n) for i, n in candidates if n is not None and n.is_finished_ok]
        if not candidates:
            return {}
        i, biggest = max(candidates,
                         key=lambda t: t[1].outputs.result.get_dict()["nbf_per_atom"])
        self.report(f"NSW reference = candidate {i} ({biggest.uuid})")
        out = {}
        for name, node in biggest.ctx.items():
            if name.startswith("scf_"):
                e = energy_from_retrieved(node)
                if e is not None:
                    out[name[len("scf_"):]] = e
        return out
