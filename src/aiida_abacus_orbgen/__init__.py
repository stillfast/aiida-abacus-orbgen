"""AiiDA plugin for ABACUS numerical atomic orbital generation.

Pipeline
--------
``orbgen.nsw`` (calculation)
    Generate the **NSW** (numerical spherical wave) primitive basis for one
    ``(r_cut, ecutjy, l_max)`` triple.  Pure SIAB/Numpy, so a ``calcfunction``.

``abacus.abacus`` (calculation, from aiida-abacus)
    The SCF that projects the plane-wave states onto that primitive basis.
    One per reference geometry.

``orbgen.nao`` (calculation)
    Minimise the generalised spillage to contract the primitive into the final
    **NAO** orbitals.  Again pure SIAB, so a ``calcfunction``.

Workflows
---------
``abacus.orbgen.base``
    The complete chain for one parameter set: ``nsw -> [abacus scf] x geometries -> nao``.

``abacus.orbgen.converge``
    Batch over a grid of ``(r_cut, ecutjy, l_max)``, compare every candidate with a
    plane-wave reference and pick the cheapest parameter set that meets the tolerance.
"""
__version__ = "0.1.0"
