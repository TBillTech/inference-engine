"""
Redirects to context_compiler.ast.resolvable_node.

The ``prompt_node`` module has been renamed to ``resolvable_node``.
``PromptNode`` and ``PromptNodeState`` aliases have been removed.
Use :class:`~context_compiler.ast.resolvable_node.ResolvableNode` and
:class:`~context_compiler.ast.resolvable_node.ResolvableNodeState` instead.
"""

from context_compiler.ast.resolvable_node import ResolvableNode, ResolvableNodeState

__all__ = ["ResolvableNode", "ResolvableNodeState"]
