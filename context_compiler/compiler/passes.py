"""
Redirects to context_compiler.query.passes.

.. deprecated::
    The ``compiler`` package has been renamed to ``context_compiler.query``.
    Use :mod:`context_compiler.query.passes` instead.
"""

from context_compiler.query.passes import (
    PassContext,
    QueryPass,
    DeterministicPass,
    ResolutionPass,
)

CompilerPass = QueryPass

__all__ = [
    "PassContext",
    "QueryPass",
    "CompilerPass",
    "DeterministicPass",
    "ResolutionPass",
]
