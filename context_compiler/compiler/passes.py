"""
Redirects to context_compiler.query.passes.

The compiler package has been renamed to ``context_compiler.query``.
Use ``context_compiler.query.passes`` instead.
"""

from context_compiler.query.passes import (
    PassContext,
    QueryPass,
    DeterministicPass,
    ResolutionPass,
)

# Alias for any code that still references CompilerPass.
CompilerPass = QueryPass

__all__ = [
    "PassContext",
    "QueryPass",
    "CompilerPass",
    "DeterministicPass",
    "ResolutionPass",
]
