"""Smoke tests: the package imports and the two process specs build.

These do not need a database or a profile -- they only exercise AiiDA's process spec
construction, which is where most plugin mistakes surface.
"""
import pytest

pytest.importorskip("aiida")


def test_import():
    import aiida_abacus_orbgen as pkg

    assert pkg.__version__


def test_calculations_are_calc_functions():
    from aiida_abacus_orbgen.calculations import orbgen_nao, orbgen_nsw

    assert orbgen_nsw.is_process_function
    assert orbgen_nao.is_process_function


def test_workchain_specs_build():
    from aiida_abacus_orbgen.workflows import OrbgenBaseWorkChain, OrbgenConvergeWorkChain

    base = OrbgenBaseWorkChain.spec()
    for name in ("siab_config", "settings", "orbgen_config", "structures", "upf",
                 "abacus_code", "scf_parameters"):
        assert name in base.inputs, name
    for name in ("nsw_orbital", "nao_orbitals", "result"):
        assert name in base.outputs, name

    conv = OrbgenConvergeWorkChain.spec()
    for name in ("r_cuts", "ecutjys", "l_maxes", "tolerance_mev", "reference_mode"):
        assert name in conv.inputs, name
    assert "ERROR_NO_CANDIDATE" in conv.exit_codes
