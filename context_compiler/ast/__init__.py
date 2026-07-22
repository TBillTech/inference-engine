"""AST sub-package: typed node tree, paths, and schema definitions."""

from context_compiler.ast.nodes import Node, NodeState, ScalarNode, MappingNode, SequenceNode
from context_compiler.ast.prompt_node import (
    ResolvableNode,
    ResolvableNodeState,
    # Backward-compatible aliases
    PromptNode,
    PromptNodeState,
)
from context_compiler.ast.paths import Path, PathSegment
from context_compiler.ast.schema import Schema, FieldSpec

__all__ = [
    "Node",
    "NodeState",
    "ScalarNode",
    "MappingNode",
    "SequenceNode",
    "ResolvableNode",
    "ResolvableNodeState",
    # Backward-compatible aliases
    "PromptNode",
    "PromptNodeState",
    "Path",
    "PathSegment",
    "Schema",
    "FieldSpec",
]
