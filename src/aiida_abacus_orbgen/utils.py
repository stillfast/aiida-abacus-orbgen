"""Small helpers shared by the two calculations.

The SIAB / orbgen configuration is a plain JSON dict, but AiiDA's ``Dict`` refuses keys
containing a dot (``aiida.common.exceptions.ValidationError: key for attributes or extras
cannot contain the character '.'``).  Real configs do contain them -- ``"torch.lr":
0.001`` is an optimiser hyper-parameter that SIAB itself reads -- so the keys are
encoded with a sentinel before they go into a ``Dict`` and decoded on the way out.
Dropping the key instead would silently change the fit.
"""
from __future__ import annotations

DOT = "__dot__"


def encode_config(node):
    """JSON dict -> AiiDA-safe dict (dots in *any* key replaced by ``__dot__``)."""
    if isinstance(node, dict):
        return {(k.replace(".", DOT) if isinstance(k, str) else k): encode_config(v)
                for k, v in node.items()}
    if isinstance(node, list):
        return [encode_config(v) for v in node]
    return node


def decode_config(node):
    """Inverse of :func:`encode_config`."""
    if isinstance(node, dict):
        return {(k.replace(DOT, ".") if isinstance(k, str) else k): decode_config(v)
                for k, v in node.items()}
    if isinstance(node, list):
        return [decode_config(v) for v in node]
    return node
