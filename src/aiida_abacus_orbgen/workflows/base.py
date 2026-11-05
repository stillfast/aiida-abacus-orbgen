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

import re

from aiida import orm
from aiida.engine import WorkChain, ToContext
from aiida.plugins import CalculationFactory

from ..calculations.nsw import orbgen_nsw
from ..utils import encode_config
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
        spec.input("folder_names", valid_type=orm.Dict, required=False,
                   help="{namespace_key: SIAB folder name}.  AiiDA link labels allow only "
                        "alphanumerics and underscores, but SIAB names its reference "
                        "geometries `U-dimer-2.10-12au`, so the two spellings have to be "
                        "related explicitly.")
        spec.input("upf", valid_type=orm.UpfData)
        spec.input("abacus_code", valid_type=orm.InstalledCode)
        spec.input("scf_parameters", valid_type=orm.Dict)
        spec.input("kpoints", valid_type=orm.KpointsData, required=False)
        spec.input("scf_options", valid_type=orm.Dict, required=False,
                   help="metadata.options for the SCF CalcJobs (queue_name carries the "
                        "SLURM partition; without it sbatch fails with 'No partition "
                        "specified').  A WorkChain builder has no metadata.options of "
                        "its own, so it has to be an explicit input.")
        spec.input("scf_settings", valid_type=orm.Dict, required=False,
                   help="ABACUS `settings` Dict for the SCF.  The matrix files the "
                        "spillage fit needs are merged in automatically -- the plugin's "
                        "default retrieve list has only INPUT/kpoints/warning.log/running_scf.log "
                        "and the AiiDA repository will otherwise be useless to orbgen.nao.")
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
        import io

        orbital = AtomicOrbitalData.get_or_create(
            io.BytesIO(self.inputs.upf.get_content(mode="rb")),
            io.BytesIO(self.ctx.nsw.get_content(mode="rb")),
            filename=self.inputs.upf.filename,
            filename_orbital=self.ctx.nsw.filename,
        )
        if not orbital.is_stored:
            orbital.store()

        cfg = self.inputs.siab_config.get_dict()
        element = cfg.get("element") or cfg.get("elem")
        params = orm.Dict(dict={"input": self.inputs.scf_parameters.get_dict()})

        settings = dict(self.inputs.scf_settings.get_dict()) \
            if "scf_settings" in self.inputs else {}
        gamma_only = bool(self.inputs.scf_parameters.get_dict().get("gamma_only", 0))
        needed = ["data-0-H", "data-0-S", "data-0-T", "istate.info",
                  "WFC_NAO_GAMMA1.txt" if gamma_only else "WFC_NAO_K1.txt"]
        settings["additional_retrieve_list"] = sorted(
            set(settings.get("additional_retrieve_list", [])) | set(needed))

        mapping = self._folder_mapping()
        expected = set(self.ctx.nsw_info.get_dict().get("dft_folders") or [])
        given = set(mapping.values())
        if expected and expected != given:
            raise ValueError(
                "the `structures` keys must be the SIAB reference-geometry folder names, "
                f"otherwise the spillage fit cannot find them.\n  expected: {sorted(expected)}"
                f"\n  given:    {sorted(given)}")

        running = {}
        for name, structure in self.inputs.structures.items():
            inputs = {
                "code": self.inputs.abacus_code,
                "structure": structure,
                "parameters": params,
                "pseudos": {element: orbital},
                "settings": orm.Dict(dict=settings),
                "metadata": {"label": f"scf-{name}"},
            }
            if "scf_options" in self.inputs:
                inputs["metadata"]["options"] = self.inputs.scf_options.get_dict()
            if "kpoints" in self.inputs:
                inputs["kpoints"] = self.inputs.kpoints
            running[f"scf_{name}"] = self.submit(AbacusCalculation, **inputs)
        self.report(f"submitted {len(running)} SCF(s) with the NSW primitive basis")
        return ToContext(**running)

    # ------------------------------------------------------------------ step 3
    def _folder_mapping(self) -> dict:
        """{namespace key: SIAB folder name}, recomputed wherever it is needed.

        It has to be a method rather than a local variable: `run_scf` and `generate_nao`
        are separate steps of the workchain, so a local computed in the first one is not
        visible in the second (this bit with `NameError: name 'mapping' is not defined`).
        """
        if "folder_names" in self.inputs:
            return self.inputs.folder_names.get_dict()
        return {k: k for k in self.inputs.structures.keys()}

    def generate_nao(self):
        """Contract the primitive into the final NAO by minimising the spillage."""
        mapping = self._folder_mapping()
        retrieved = {}
        for key, node in self.ctx.items():
            if not key.startswith("scf_"):
                continue
            name = key[len("scf_"):]
            name = mapping.get(name, name)            # back to the SIAB folder name
            if "retrieved" not in node.outputs:
                # a failed SCF is not something the fit can work around
                return self.exit_codes.ERROR_SCF_MISSING_OUTPUT.format(name=name)
            retrieved[name] = node.outputs.retrieved
        if not retrieved:
            return self.exit_codes.ERROR_SCF_MISSING_OUTPUT.format(name="<none>")

        # keys must be valid link labels, so pass the sanitised spelling plus a map
        sanitised = {}
        inverse = {}
        for folder, node in retrieved.items():
            key = re.sub(r"[^0-9a-zA-Z_]", "_", folder)
            sanitised[key] = node
            inverse[key] = folder
        out = orbgen_nao(self.ctx.nsw, self.inputs.orbgen_config,
                         orm.Dict(dict=inverse), **sanitised)
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
        # geometry folder names ("U-dimer-2.10-12au") are dict keys here, and AiiDA Dict
        # rejects dots in keys -- encode them, decode_config() reverses it
        self.out("result", orm.Dict(dict=encode_config(result)))
        for name, node in self.ctx.nao.items():
            self.out(f"nao_orbitals.{name.replace('.', '_')}", node)

        if self.inputs.clean_workdir.value:
            from aiida.tools import delete_nodes

            pks = [n.pk for k, n in self.ctx.items()
                   if k.startswith("scf_") and "remote_folder" in n.outputs]
            if pks:
                delete_nodes(pks, dry_run=False)
