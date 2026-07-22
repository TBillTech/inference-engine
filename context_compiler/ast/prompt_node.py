"""
Redirects to context_compiler.ast.resolvable_node.

.. deprecated::
    The ``prompt_node`` module has been renamed to ``resolvable_node``.
    ``PromptNode`` and ``PromptNodeState`` aliases have been removed.
    Use :mod:`context_compiler.ast.resolvable_node` directly.
"""

from context_compiler.ast.resolvable_node import ResolvableNode, ResolvableNodeState

__all__ = ["ResolvableNode", "ResolvableNodeState"]
