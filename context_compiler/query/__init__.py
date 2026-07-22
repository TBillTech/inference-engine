"""Query sub-package: dependency graph, pass registry, and the main Resolver."""

from context_compiler.query.dependency_graph import DependencyGraph, CycleError
from context_compiler.query.passes import (
    QueryPass,
    DeterministicPass,
    ResolutionPass,
)
from context_compiler.query.resolver import Resolver

__all__ = [
    "DependencyGraph",
    "CycleError",
    "QueryPass",
    "DeterministicPass",
    "ResolutionPass",
    "Resolver",
]
