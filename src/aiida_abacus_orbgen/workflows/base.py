"""``abacus.orbgen.base`` -- one (r_cut, ecutjy, l_max) triple, end to end.

    orbgen.nsw  ->  abacus.abacus (one SCF per geometry, with the primitive basis)
                ->  orbgen.nao

This is the unit of work that ``abacus.orbgen.converge`` repeats over a parameter grid,
so it deliberately does *not* know anything about convergence or plane-wave references.

Inputs
------
siab_config : Dict
    SIAB/orbgen configuration (element, pseudopotential, ``ecutjy``, ``ecutwfc``,
    ``geoms``, ``spill_guess``, ``__iop_spill_guess_atomic_nbands__``, ...).
settings : Dict
    ``r_cut``, ``ecutjy``, ``l_max`` -- the parameters being varied, plus optional ``dr``.
orbgen_config : Dict
    The ``orbgen.json`` used for the fit: the ``orbitals`` block (the ``nzeta`` ladder)
    and the ``geoms``.
structures : namespace of StructureData
    One entry per reference geometry.  The ``port`` namespace is accepted too.
upf : UpfData
    Pseudopotential; combined with each generated orbital into ``AtomicOrbitalData``.
abacus_code : InstalledCode
    The ``abacus.abacus`` code.
scf_parameters : Dict
    ``ecutwfc``, ``nbands``, ``ks_solver``, ``smearing_*``, ``scf_thr`` ... -- the
    ``input`` block handed to the CalcJob.
kpoints : KpointsData, optional
    Omitted for Gamma-only (molecule / dimer) references.
clean_workdir : Bool
    Remove the remote scratch of the SCF jobs at the end.

Outputs
-------
nsw_orbital : SinglefileData      the primitive basis
nao_orbitals : dict of SinglefileData   the contracted orbitals
result : Dict                     parameter set, spillages, SCF energies
"""
from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, ToContext
from aiida.plugins import CalculationFactory

from ..calculations.nsw import orbgen_nsw
from ..calculations.nao import orbgen_nao


class OrbgenBaseWorkChain(WorkChain):
    """NSW -> ABACUS SCF -> NAO for a single parameter triple."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("siab_config", valid_type=orm.Dict)
        spec.input("settings", valid_type=orm.Dict)
        spec.input("orbgen_config", valid_type=orm.Dict)
        spec.input_namespace("structures", valid_type=orm.StructureData, dynamic=True)
        spec.input("upf", valid_type=orm.UpfData)
        spec.input("abacus_code", valid_type=orm.InstalledCode)
        spec.input("scf_parameters", valid_type=orm.Dict)
        spec.input("kpoints", valid_type=orm.KpointsData, required=False)
        spec.input("clean_workdir", valid_type=orm.Bool, default=lambda: orm.Bool(False))

        spec.outline(
            cls.generate_nsw,
            cls.run_scf,
            cls.generate_nao,
            cls.collect,
        )

        spec.output("nsw_orbital", valid_type=orm.SinglefileData)
        spec.output_namespace("nao_orbitals", valid_type=orm.SinglefileData, dynamic=True)
        spec.output("result", valid_type=orm.Dict)

        spec.exit_code(
            310, "ERROR_SCF_MISSING_OUTPUT",
            message="SCF calculation {name} produced no retrieved output; the spillage "
                    "fit needs OUT.ABACUS of every reference geometry",
        )

    # ------------------------------------------------------------------ step 1
    def generate_nsw(self):
        """Produce the primitive basis (cached whenever the triple repeats)."""
        out = orbgen_nsw(self.inputs.siab_config, self.inputs.settings)
        self.ctx.nsw = out["nsw_orbital"]
        self.ctx.nsw_info = out["nsw_info"]
        self.report(
            f"NSW basis: {self.ctx.nsw_info.get_dict()['filename']} "
            f"({self.ctx.nsw_info.get_dict()['nbf_per_atom']} functions/atom)"
        )

    # ------------------------------------------------------------------ step 2
    def run_scf(self):
        """Run one ABACUS SCF per reference geometry with the *primitive* basis."""
        from aiida_abacus.data.orbital import AtomicOrbitalData

        AbacusCalculation = CalculationFactory("abacus.abacus")
        orbital = AtomicOrbitalData.get_or_create(
            self.inputs.upf.get_file().filepath,
            self.ctx.nsw.get_file().filepath,
            filename_orbital=self.ctx.nsw.filename,
        )
        if not orbital.is_stored:
            orbital.store()

        element = self.inputs.siab_config.get_dict()["elem"]
        params = orm.Dict(dict={"input": self.inputs.scf_parameters.get_dict()})

        running = {}
        for name, structure in self.inputs.structures.items():
            inputs = {
                "code": self.inputs.abacus_code,
                "structure": structure,
                "parameters": params,
                "pseudos": {element: orbital},
                "metadata": {"label": f"scf-{name}"},
            }
            if "kpoints" in self.inputs:
                inputs["kpoints"] = self.inputs.kpoints
            running[f"scf_{name}"] = self.submit(AbacusCalculation, **inputs)
        self.report(f"submitted {len(running)} SCF(s) with the NSW primitive basis")
        return ToContext(**running)

    # ------------------------------------------------------------------ step 3
    def generate_nao(self):
        """Contract the primitive into the final NAO by minimising the spillage."""
        retrieved = {}
        for key, node in self.ctx.items():
            if not key.startswith("scf_"):
                continue
            name = key[len("scf_"):]
            if "retrieved" not in node.outputs:
                # a failed SCF is not something the fit can work around
                return self.exit_codes.ERROR_SCF_MISSING_OUTPUT.format(name=name)
            retrieved[name] = node.outputs.retrieved
        if not retrieved:
            return self.exit_codes.ERROR_SCF_MISSING_OUTPUT.format(name="<none>")

        out = orbgen_nao(self.ctx.nsw, retrieved, self.inputs.orbgen_config)
        self.ctx.nao = out["nao_orbitals"]
        self.ctx.nao_info = out["nao_info"]

    # ------------------------------------------------------------------ step 4
    def collect(self):
        s = self.inputs.settings.get_dict()
        info = self.ctx.nao_info.get_dict()
        energies = {}
        for key, node in self.ctx.items():
            if key.startswith("scf_") and "output_parameters" in node.outputs:
                energies[key[len("scf_"):]] = node.outputs.output_parameters.get_dict()

        result = {
            "r_cut": s["r_cut"],
            "ecutjy": s["ecutjy"],
            "l_max": s["l_max"],
            "nsw_filename": self.ctx.nsw_info.get_dict()["filename"],
            "nbf_per_atom": self.ctx.nsw_info.get_dict()["nbf_per_atom"],
            "nao_orbitals": info["orbitals"],
            "spillages": info["spillages"],
            "scf_energies": energies,
        }

        self.out("nsw_orbital", self.ctx.nsw)
        self.out("result", orm.Dict(dict=result))
        for name, node in self.ctx.nao.items():
            self.out(f"nao_orbitals.{name.replace('.', '_')}", node)

        if self.inputs.clean_workdir.value:
            from aiida.tools import delete_nodes

            pks = [n.pk for k, n in self.ctx.items()
                   if k.startswith("scf_") and "remote_folder" in n.outputs]
            if pks:
                delete_nodes(pks, dry_run=False)
