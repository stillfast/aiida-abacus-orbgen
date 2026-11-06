"""``abacus.orbgen.base`` -- one (r_cut, ecutjy, l_max) triple, end to end.

    orbgen.nsw  ->  abacus.abacus x N geometries  ->  orbgen.nao

Nothing in this chain re-states a parameter that SIAB already decided:

* ``orbgen.nsw`` runs SIAB's ``generate_all`` and returns the primitive orbital, the
  generated ``INPUT`` for every reference geometry, the matching ``StructureData`` and a
  manifest;
* ``run_scf`` submits one CalcJob per geometry using **SIAB's own parameters** (the
  previous version hand-wrote them, which silently lost ``bessel_nao_rcut`` and the
  per-geometry ``nbands``: the monomer needs ``__iop_spill_guess_atomic_nbands__``
  bands, the dimers ``geoms[0].nbands``);
* ``orbgen.nao`` fits the contraction and returns the final orbitals.

Inputs
------
siab_config : Dict      the SIAB / orbgen configuration
settings : Dict         r_cut, ecutjy, l_max, optional dr
orbgen_config : Dict    the ``orbitals`` (nzeta ladder) block handed to the fit
upf : UpfData           only used to type the pseudopotential for the CalcJob
abacus_code : InstalledCode
scf_options : Dict      metadata.options for the CalcJobs (queue_name is the SLURM
                        partition; a WorkChain builder has no metadata.options itself)
clean_workdir : Bool
"""
from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, ToContext
from aiida.plugins import CalculationFactory

from ..calculations.nsw import orbgen_nsw
from ..calculations.nao import orbgen_nao
from ..utils import encode_config

#: written by the AiiDA plugin, so SIAB's spelling must not be forwarded
PLUGIN_MANAGED = {"suffix", "pseudo_dir", "orbital_dir", "stru_file", "kpoint_file",
                  "wannier_card"}


class OrbgenBaseWorkChain(WorkChain):
    """NSW -> ABACUS SCF -> NAO for a single parameter triple."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("siab_config", valid_type=orm.Dict)
        spec.input("settings", valid_type=orm.Dict)
        spec.input("orbgen_config", valid_type=orm.Dict)
        spec.input("upf", valid_type=orm.UpfData)
        spec.input("abacus_code", valid_type=orm.InstalledCode)
        spec.input("scf_options", valid_type=orm.Dict, required=False,
                   help="metadata.options for the SCF CalcJobs (carries queue_name)")
        spec.input("clean_workdir", valid_type=orm.Bool, default=lambda: orm.Bool(False))

        spec.outline(cls.generate_nsw, cls.run_scf, cls.generate_nao, cls.collect)

        spec.output("nsw_orbital", valid_type=orm.SinglefileData)
        spec.output_namespace("nao_orbitals", valid_type=orm.SinglefileData, dynamic=True)
        spec.output("result", valid_type=orm.Dict)

        spec.exit_code(310, "ERROR_SCF_MISSING_OUTPUT",
                       message="SCF {name} produced no retrieved output")
        spec.exit_code(311, "ERROR_SCF_FAILED",
                       message="SCF {name} exited with {status}; the spillage fit needs "
                               "every reference geometry")

    # ------------------------------------------------------------------ step 1
    def generate_nsw(self):
        out = orbgen_nsw(self.inputs.siab_config, self.inputs.settings)
        self.ctx.nsw = out["nsw_orbital"]
        self.ctx.dft_inputs = out["dft_inputs"]
        self.ctx.structures = out["structures"]
        self.ctx.nsw_info = out["nsw_info"]
        info = self.ctx.nsw_info.get_dict()
        self.report(f"NSW basis {info['filename']} ({info['nbf_per_atom']} functions/atom), "
                    f"{len(info['dft_folders'])} reference geometries from SIAB")

    # ------------------------------------------------------------------ step 2
    def run_scf(self):
        from aiida_abacus.data.orbital import AtomicOrbitalData
        import io

        AbacusCalculation = CalculationFactory("abacus.abacus")
        orbital = AtomicOrbitalData.get_or_create(
            io.BytesIO(self.inputs.upf.get_content(mode="rb")),
            io.BytesIO(self.ctx.nsw.get_content(mode="rb")),
            filename=self.inputs.upf.filename,
            filename_orbital=self.ctx.nsw.filename,
        )
        if not orbital.is_stored:
            orbital.store()

        # `abacus.abacus` declares kpoints as required even for gamma_only runs, and the
        # SIAB deck sets gamma_only, so a 1x1x1 mesh is the matching choice
        kpoints = orm.KpointsData()
        kpoints.set_kpoints_mesh([1, 1, 1])

        info = self.ctx.nsw_info.get_dict()
        element = info["element"]
        running = {}
        for key, folder in info["folder_map"].items():
            params = dict(info["parameters"][folder])
            for managed in PLUGIN_MANAGED:
                params.pop(managed, None)

            inputs = {
                "code": self.inputs.abacus_code,
                "structure": self.ctx.structures[key],
                "parameters": orm.Dict(dict={"input": params}),
                "pseudos": {element: orbital},
                "kpoints": kpoints,
                "settings": orm.Dict(dict={"additional_retrieve_list": [
                    "data-0-H", "data-0-S", "data-0-T", "istate.info",
                    "WFC_NAO_GAMMA1.txt" if params.get("gamma_only") else "WFC_NAO_K1.txt",
                ]}),
                "metadata": {"label": f"scf-{key}"},
            }
            if "scf_options" in self.inputs:
                inputs["metadata"]["options"] = self.inputs.scf_options.get_dict()
            running[f"scf_{key}"] = self.submit(AbacusCalculation, **inputs)
        self.report(f"submitted {len(running)} SCF(s) with SIAB's own parameters")
        return ToContext(**running)

    # ------------------------------------------------------------------ step 3
    def generate_nao(self):
        info = self.ctx.nsw_info.get_dict()
        retrieved, inverse = {}, {}
        for key, node in list(self.ctx.items()):
            if not key.startswith("scf_"):
                continue
            name = key[len("scf_"):]
            if "retrieved" not in node.outputs:
                return self.exit_codes.ERROR_SCF_MISSING_OUTPUT.format(name=name)
            folder = info["folder_map"][name]
            retrieved[name] = node.outputs.retrieved
            inverse[name] = folder
        if not retrieved:
            return self.exit_codes.ERROR_SCF_MISSING_OUTPUT.format(name="<none>")

        out = orbgen_nao(self.ctx.nsw, self.inputs.orbgen_config,
                         orm.Dict(dict=inverse), **retrieved)
        self.ctx.nao = out["nao_orbitals"]
        self.ctx.nao_info = out["nao_info"]

    # ------------------------------------------------------------------ step 4
    def collect(self):
        s = self.inputs.settings.get_dict()
        info = self.ctx.nsw_info.get_dict()
        nao = self.ctx.nao_info.get_dict()
        energies = {}
        for key, node in list(self.ctx.items()):
            if not key.startswith("scf_"):
                continue
            try:
                energies[key[len("scf_"):]] = node.outputs.output_parameters.get_dict()
            except Exception:                                       # noqa: BLE001
                energies[key[len("scf_"):]] = None

        result = {
            "r_cut": s["r_cut"], "ecutjy": s["ecutjy"], "l_max": s["l_max"],
            "nsw_filename": info["filename"], "nbf_per_atom": info["nbf_per_atom"],
            "dft_folders": info["dft_folders"],
            "nao_orbitals": nao["orbitals"], "spillages": nao["spillages"],
            "scf_energies": energies,
        }
        self.out("nsw_orbital", self.ctx.nsw)
        # a WorkChain cannot return an unstored Data node ("tried returning an unstored
        # `Data` node ... use a calcfunction"), so the summary must be stored first
        summary = orm.Dict(dict=encode_config(result))
        summary.store()
        self.out("result", summary)
        for name, node in self.ctx.nao.items():
            # `name` is already a valid link label (orbgen.nao sanitises the keys)
            self.out(f"nao_orbitals.{name}", node)
