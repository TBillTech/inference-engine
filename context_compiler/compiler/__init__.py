"""Compiler sub-package: dependency graph, pass registry, and the main Compiler."""

from context_compiler.compiler.dependency_graph import DependencyGraph, CycleError
from context_compiler.compiler.passes import (
    CompilerPass,
    DeterministicPass,
    ResolutionPass,
    # Backward-compatible alias
    InferencePass,
)
from context_compiler.compiler.compiler import Compiler

__all__ = [
    "DependencyGraph",
    "CycleError",
    "CompilerPass",
    "DeterministicPass",
    "ResolutionPass",
    # Backward-compatible alias
    "InferencePass",
    "Compiler",
]
