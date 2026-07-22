"""
context_resolver – A general-purpose demand-driven semantic query engine over a
typed semantic tree.

LLMs are one possible resolution engine; the resolver itself is agnostic to
the resolution strategy used.
"""

from context_resolver.ast.nodes import Node, NodeState
from context_resolver.ast.paths import Path
from context_resolver.context.context import Context

__all__ = ["Node", "NodeState", "Path", "Context"]
