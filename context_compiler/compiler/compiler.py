"""
Redirects to context_compiler.query.resolver.

The compiler package has been renamed to ``context_compiler.query``.
Use ``context_compiler.query.resolver.Resolver`` and ``resolve_node`` instead.
"""

from context_compiler.query.resolver import (
    Resolver,
    ResolutionError,
    _resolve_path,
    _extract_scalar,
    _decode_response,
)

# Alias for any code that still references Compiler or CompilationError.
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
