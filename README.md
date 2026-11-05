# aiida-abacus-orbgen

AiiDA plugin for generating ABACUS numerical atomic orbitals (NAO) from a truncated
spherical-wave (NSW) primitive basis, in the layout of `aiida-quantumespresso`:
`calculations/` for the leaves, `workflows/` for the orchestration, entry points in
`pyproject.toml`.

## Why two "calculations" are `calcfunction`s and not `CalcJob`s

`aiida-quantumespresso` wraps external executables (`pw.x`, `cp.x`) in `CalcJob`s. Only
one of the three steps here drives an executable:

| step | what it does | process type | why |
| --- | --- | --- | --- |
| `orbgen.nsw` | truncated spherical Bessel functions → primitive `.orb` | `calcfunction` | pure Numpy/SIAB, nothing to schedule |
| `abacus.abacus` | SCF with that primitive basis | `CalcJob` (from `aiida-abacus`) | real executable on the cluster |
| `orbgen.nao` | spillage minimisation → contracted `.orb` | `calcfunction` | pure SIAB; the numbers it needs are the *retrieved* SCF outputs |

Both calcfunctions are cached by AiiDA, which is what makes `abacus.orbgen.converge`
affordable: many grid points share the same primitive (same `r_cut`/`ecutjy`/`l_max`),
and each primitive is only ever generated once.

`orbgen.nao` **neutralises `abacus_command`** before invoking the fit, so the spillage
optimiser can never launch ABACUS on the host running the daemon.

## Entry points

```
aiida.calculations
  orbgen.nsw              aiida_abacus_orbgen.calculations.nsw:orbgen_nsw
  orbgen.nao              aiida_abacus_orbgen.calculations.nao:orbgen_nao
aiida.workflows
  abacus.orbgen.base      aiida_abacus_orbgen.workflows.base:OrbgenBaseWorkChain
  abacus.orbgen.converge  aiida_abacus_orbgen.workflows.converge:OrbgenConvergeWorkChain
```

## `abacus.orbgen.base`

One `(r_cut, ecutjy, l_max)` triple, end to end:

```
orbgen.nsw  ->  abacus.abacus x N geometries (with the primitive basis)  ->  orbgen.nao
```

Outputs `nsw_orbital`, `nao_orbitals` (namespace) and `result` (parameter set,
spillages, SCF energies).

## `abacus.orbgen.converge`

Scans `r_cut x ecutjy x l_max`, runs a **plane-wave reference** once per geometry, and
returns the *cheapest* candidate whose total-energy error is inside `tolerance_mev`
(default 1 kcal/mol = 43.36 meV/atom).

```python
from aiida.orm import Dict, List, Float, load_code
from aiida.plugins import WorkflowFactory

Orbgen = WorkflowFactory("abacus.orbgen.converge")
builder = Orbgen.get_builder()
builder.r_cuts = List([9, 10, 11, 12])
builder.ecutjys = List([100, 125, 150])
builder.l_maxes = List([4])                 # l_max = 4, as chosen for U
builder.tolerance_mev = Float(43.364)
builder.siab_config = Dict(siab_json)
builder.orbgen_config = Dict(orbgen_json)
builder.structures = {"dimer-2.40": stru1, "dimer-2.80": stru2}
builder.upf = upf
builder.abacus_code = load_code("abacus_lts@yeesuan")
builder.scf_parameters = Dict(dict(ecutwfc=200, nbands=60, scf_thr=1e-6))
```

`reference_mode = "pw"` (default) is the honest criterion. `"nsw"` reuses the largest
candidate as the reference — cheaper, but it then measures `ecutjy` convergence *against
another NAO*, which is exactly how a too-small `r_cut` can pass unnoticed.

Cost ranking (exposed as `cost_exponent` and `r_cut_reference`):

```
cost = nbasis_per_atom * natoms * (1 + r_cut / r_cut_reference) ** cost_exponent
```

## Install

```bash
cd aiida-abacus-orbgen
pip install -e .
verdi plugin list aiida.calculations | grep orbgen
```

## Status

The package imports and both process specs build (verified with
`PYTHONPATH=src python -c "import aiida_abacus_orbgen"`). **No AiiDA job has been
submitted** — the workflows are ready to run but untested end to end against a real
cluster profile.
