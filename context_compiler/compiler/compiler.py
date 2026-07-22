"""
Redirects to context_compiler.query.resolver.

.. deprecated::
    The ``compiler`` package has been renamed to ``context_compiler.query``.
    Use :class:`context_compiler.query.resolver.Resolver` and
    :meth:`~context_compiler.query.resolver.Resolver.resolve_node` instead.
"""

from context_compiler.query.resolver import (
    Resolver,
    ResolutionError,
    _resolve_path,
    _extract_scalar,
    _decode_response,
)

Compiler = Resolver
CompilationError = ResolutionError

__all__ = [
    "Resolver",
    "Compiler",
    "ResolutionError",
    "CompilationError",
    "_resolve_path",
    "_extract_scalar",
    "_decode_response",
]
