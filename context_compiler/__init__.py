"""
context_compiler – A general-purpose demand-driven semantic compiler over a
typed semantic tree.

LLMs are one possible resolution engine; the compiler itself is agnostic to
the resolution strategy used.
"""

from context_compiler.ast.nodes import Node, NodeState
from context_compiler.ast.paths import Path
from context_compiler.context.context import Context

__all__ = ["Node", "NodeState", "Path", "Context"]
