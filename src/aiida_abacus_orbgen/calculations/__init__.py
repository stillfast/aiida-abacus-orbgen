"""The two orbital-generation calculations (``orbgen.nsw`` and ``orbgen.nao``)."""
from .nsw import orbgen_nsw
from .nao import orbgen_nao

__all__ = ["orbgen_nsw", "orbgen_nao"]
