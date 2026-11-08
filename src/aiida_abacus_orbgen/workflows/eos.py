"""``abacus.orbgen.eos`` -- equation of state for one orbital, then V0 / B0 / B0'.

This is step 3 of the campaign: steps 1 and 2 fix ``ecutwfc`` and ``(r_cut, ecutjy)``,
and this one answers "does the resulting NAO reproduce the structure and the elastic
response?".  It deliberately does **not** generate anything: it takes an orbital (the
output of ``abacus.orbgen.base`` or any registered one) and runs a fixed-lattice SCF at
several cell volumes.

    V(E) scan -> 3rd-order Birch-Murnaghan -> V0, B0, B0', E0

The fit is the same closed form used in ``5-nao_eos/eos_benchmark.py`` so the numbers are
directly comparable with the fcc benchmark.
"""
from __future__ import annotations

import numpy as np
from aiida import orm
from aiida.engine import WorkChain, ToContext
from aiida.plugins import CalculationFactory

EV_A3_TO_GPA = 160.2176634


def eos_birch_murnaghan(V, E0, V0, B0, BP):
    """3rd-order Birch-Murnaghan energy, ``B0`` in eV/A^3."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + 9.0 * V0 * B0 / 16.0 * ((eta - 1.0) ** 3 * BP
                                       + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta))


def fit_birch_murnaghan(volumes, energies) -> dict:
    from scipy.optimize import least_squares

    V, E = np.asarray(volumes, float), np.asarray(energies, float)
    i = int(np.argmin(E))
    result = least_squares(
        lambda p: eos_birch_murnaghan(V, *p) - E,
        [E[i], V[i], 0.5, 4.0],
        bounds=([-np.inf, V[i] * 0.5, 1e-3, 0.1], [np.inf, V[i] * 2.0, 50.0, 20.0]),
    )
    E0, V0, B0, BP = result.x
    rms = float(np.sqrt(np.mean((eos_birch_murnaghan(V, *result.x) - E) ** 2)))
    return {"V0": float(V0), "B0_GPa": float(B0 * EV_A3_TO_GPA), "BP": float(BP),
            "E0": float(E0), "rms_meV": rms * 1000.0, "npoints": int(len(V))}


class OrbgenEosWorkChain(WorkChain):
    """Volume scan + Birch-Murnaghan fit for one orbital."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("structure", valid_type=orm.StructureData)
        spec.input("upf", valid_type=orm.UpfData)
        spec.input("orbital", valid_type=orm.SinglefileData,
                   help="the contracted .orb to test (from abacus.orbgen.base)")
        spec.input("abacus_code", valid_type=orm.InstalledCode)
        spec.input("scf_parameters", valid_type=orm.Dict)
        spec.input("kpoints_distance", valid_type=orm.Float,
                   default=lambda: orm.Float(0.25))
        spec.input("scales", valid_type=orm.List,
                   help="linear volume scale factors, e.g. [0.98,0.99,1.0,1.01,1.02]")
        # Required, not optional: without it the CalcJobs go out with
        # queue_name=None and 1 MPI rank, sbatch fails with "No partition specified",
        # and AiiDA only discovers it after five failed submission attempts -- leaving
        # the workchain stuck in Pausing.  Failing at build time is much cheaper.
        spec.input("scf_options", valid_type=orm.Dict, required=True,
                   validator=cls._validate_scf_options)
        spec.input("scf_settings", valid_type=orm.Dict, required=False,
                   help="ABACUS `settings` Dict.  Always forwarded, even when empty: an "
                        "absent `settings` port and an empty Dict hash differently, so "
                        "omitting it silently prevents any cache hit against a campaign "
                        "that supplied settings={}.")

        spec.outline(cls.run_scan, cls.fit)

        spec.output("eos", valid_type=orm.Dict)
        spec.exit_code(330, "ERROR_MISSING_POINTS",
                       message="fewer than four volume points produced an energy")

    @staticmethod
    def _validate_scf_options(value, _port=None):
        """Reject a scheduling Dict that cannot produce a runnable sbatch script."""
        data = value.get_dict() if hasattr(value, "get_dict") else dict(value)
        if not data.get("queue_name"):
            raise ValueError("scf_options needs 'queue_name' (the SLURM partition); "
                             "without it sbatch fails with 'No partition specified' and "
                             "the job only surfaces as five failed submissions")
        resources = data.get("resources") or {}
        if resources.get("tot_num_mpiprocs", resources.get("num_mpiprocs_per_machine", 1)) < 2:
            raise ValueError("scf_options needs more than one MPI rank "
                             f"(got {resources!r}); an ABACUS run on 1 rank is not useful")

    # ------------------------------------------------------------------ step 1
    def run_scan(self):
        from aiida_abacus.data.orbital import AtomicOrbitalData
        import io
        import re

        AbacusCalculation = CalculationFactory("abacus.abacus")
        orbital = AtomicOrbitalData.get_or_create(
            io.BytesIO(self.inputs.upf.get_content(mode="rb")),
            io.BytesIO(self.inputs.orbital.get_content(mode="rb")),
            filename=self.inputs.upf.filename,
            filename_orbital=self.inputs.orbital.filename,
        )
        if not orbital.is_stored:
            orbital.store()

        element = re.match(r"([A-Z][a-z]?)", self.inputs.upf.filename).group(1)
        base = self.inputs.structure
        cell = np.array(base.cell)

        kpoints = orm.KpointsData()
        # AiiDA 2.x spells it `set_kpoints_mesh_from_density`, the argument is a k-point
        # distance in 1/Angstrom, and it refuses to run unless the cell is set first
        # ("Cannot define a mesh from a density without having defined a cell").
        kpoints.set_cell(base.cell)
        kpoints.set_kpoints_mesh_from_density(self.inputs.kpoints_distance.value)

        running = {}
        for scale in self.inputs.scales.get_list():
            # `scale` is a volume factor, so the linear factor is its cube root
            atoms = base.get_ase()
            atoms.set_cell(cell * float(scale) ** (1 / 3), scale_atoms=True)
            scaled = orm.StructureData(ase=atoms)

            inputs = {
                "code": self.inputs.abacus_code,
                "structure": scaled,
                "parameters": self.inputs.scf_parameters,
                "pseudos": {element: orbital},
                "kpoints": kpoints,
                # always present, even empty -- see the note on the input port
                "settings": orm.Dict(dict=(self.inputs.scf_settings.get_dict()
                                           if "scf_settings" in self.inputs else {})),
                "metadata": {"label": f"eos-scale-{scale:.4f}"},
            }
            if "scf_options" in self.inputs:
                inputs["metadata"]["options"] = self.inputs.scf_options.get_dict()
            running[f"scf_{str(scale).replace('.', '_')}"] = self.submit(AbacusCalculation,
                                                                        **inputs)
        self.report(f"submitted {len(running)} volume points")
        return ToContext(**running)

    # ------------------------------------------------------------------ step 2
    def fit(self):
        import re

        volumes, energies, points = [], [], {}
        for key, node in list(self.ctx.items()):
            if not key.startswith("scf_"):
                continue
            scale = float(key[len("scf_"):].replace("_", "."))
            if "retrieved" not in node.outputs:
                continue
            text = None
            for sub in ("OUT.aiida", "OUT.ABACUS"):
                try:
                    text = node.outputs.retrieved.get_object_content(f"{sub}/running_scf.log")
                    break
                except Exception:                                  # noqa: BLE001
                    continue
            if text is None:
                continue
            hits = re.findall(r"!FINAL_ETOT_IS\s+(-?[0-9.]+)", text)
            if not hits:
                continue
            energy = float(hits[-1])
            volumes.append(self.inputs.structure.get_cell_volume() * scale)
            energies.append(energy)
            points[f"{scale:.4f}"] = energy

        if len(volumes) < 4:
            self.report(f"only {len(volumes)} usable points")
            return self.exit_codes.ERROR_MISSING_POINTS

        fitted = fit_birch_murnaghan(volumes, energies)
        out = {"V0_per_atom": fitted["V0"] / len(self.inputs.structure.sites),
               "B0_GPa": fitted["B0_GPa"], "BP": fitted["BP"], "E0": fitted["E0"],
               "rms_meV": fitted["rms_meV"], "npoints": fitted["npoints"],
               "scales": self.inputs.scales.get_list(), "energies": points,
               "orbital": self.inputs.orbital.filename}
        # a WorkChain cannot return an unstored Data node; storing first is the price of
        # returning a summary from the workflow itself
        summary = orm.Dict(dict=out)
        summary.store()
        self.out("eos", summary)
        self.report(f"V0/atom = {out['V0_per_atom']:.4f} A^3, B0 = {out['B0_GPa']:.1f} GPa, "
                    f"B0' = {out['BP']:.2f} (rms {out['rms_meV']:.2f} meV)")
