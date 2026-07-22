"""
Redirects to context_compiler.query.dependency_graph.

.. deprecated::
    The ``compiler`` package has been renamed to ``context_compiler.query``.
    Use :mod:`context_compiler.query.dependency_graph` instead.
"""

from context_compiler.query.dependency_graph import DependencyGraph, CycleError

__all__ = ["DependencyGraph", "CycleError"]
