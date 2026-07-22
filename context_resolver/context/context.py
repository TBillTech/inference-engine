"""
Context – the top-level semantic tree with lazy, demand-driven resolution.

The :class:`Context` is the primary interface for application code.  It holds:

* the root :class:`~context_resolver.ast.nodes.Node` of the semantic tree
* a :class:`~context_resolver.query.resolver.Resolver` for resolving
  underspecified nodes
* a result cache keyed by :class:`~context_resolver.ast.paths.Path`

Query semantics
---------------
When :meth:`Context.query` is called:

1. The requested path is located in the tree.
2. If the node is ``FULLY_SPECIFIED`` *and* cached, return immediately.
3. Otherwise, hand the node to the resolver and store the result.
4. Return the (now-resolved) node.

The resolver is never invoked for nodes that are already fully specified,
ensuring the system is lazy and incremental.
"""

from __future__ import annotations

import logging
from typing import Any

from context_resolver.ast.nodes import (
    Node,
    NodeState,
    MappingNode,
    SequenceNode,
    ScalarNode,
)
from context_resolver.ast.paths import Path
from context_resolver.ast.resolvable_node import ResolvableNode
from context_resolver.query.resolver import Resolver, _resolve_path
from context_resolver.query.dependency_graph import DependencyGraph

logger = logging.getLogger(__name__)


class NodeNotFoundError(KeyError):
    """Raised when a queried path does not exist in the Context tree."""


class Context:
    """
    The demand-driven semantic context.

    Parameters
    ----------
    root:
        The root node of the semantic tree.  Typically a
        :class:`~context_resolver.ast.nodes.MappingNode`.
    resolver:
        The :class:`~context_resolver.query.resolver.Resolver` to use for
        resolving underspecified nodes.

    Examples
    --------
    >>> from context_resolver.ast.nodes import MappingNode, ScalarNode
    >>> root = MappingNode({"greeting": ScalarNode("Hello!")})
    >>> ctx = Context(root)
    >>> node = ctx.query(Path("greeting"))
    >>> node.value
    'Hello!'
    """

    def __init__(
        self,
        root: Node | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        self._root: Node = root or MappingNode()
        self._resolver: Resolver = resolver or Resolver()
        self._cache: dict[Path, Node] = {}

    # ------------------------------------------------------------------
    # Tree mutation
    # ------------------------------------------------------------------

    def set(self, path: Path, node: Node) -> None:
        """
        Assign *node* at *path* in the Context tree.

        If any ResolvableNodes transitively depend on *path*, their cached
        results are invalidated (marked stale).

        Parameters
        ----------
        path:
            The path at which to store the node.
        node:
            The node to store.
        """
        self._set_in_tree(self._root, path, node)
        # Invalidate the cache entry for this path.
        self._cache.pop(path, None)
        # Mark all transitive dependents as stale.
        stale_paths = self._resolver.dependency_graph.transitive_dependents_of(path)
        for stale_path in stale_paths:
            self._cache.pop(stale_path, None)
            stale_node = _resolve_path(self._root, stale_path)
            if isinstance(stale_node, ResolvableNode):
                stale_node.mark_stale()
        logger.debug("Set %s; invalidated %d dependent(s)", path, len(stale_paths))

    def _set_in_tree(self, root: Node, path: Path, value: Node) -> None:
        """Walk the tree and set *value* at the node addressed by *path*."""
        if path.is_empty:
            raise ValueError("Cannot set the root node via set(); create a new Context instead")

        parent_path = path.parent
        leaf = path.leaf
        parent = _resolve_path(root, parent_path) if not parent_path.is_empty else root

        if parent is None:
            raise NodeNotFoundError(
                f"Cannot set {path}: parent path {parent_path} does not exist"
            )

        if isinstance(leaf, str) and isinstance(parent, MappingNode):
            parent.set(leaf, value)
        elif isinstance(leaf, int) and isinstance(parent, SequenceNode):
            # Replace existing or append if index == len.
            while len(parent) < leaf:
                parent.append(ScalarNode(None))
            if leaf == len(parent):
                parent.append(value)
            else:
                parent._items[leaf] = value
        else:
            raise TypeError(
                f"Cannot set segment {leaf!r} on node of type "
                f"{type(parent).__name__}"
            )

    # ------------------------------------------------------------------
    # Query (lazy resolution entry point)
    # ------------------------------------------------------------------

    def query(self, path: Path) -> Node:
        """
        Return the node at *path*, resolving it on demand if necessary.

        This is the primary interface for demand-driven resolution.

        Parameters
        ----------
        path:
            The path to query.

        Returns
        -------
        Node
            The fully specified (or best-available) node at *path*.

        Raises
        ------
        NodeNotFoundError
            If *path* does not exist in the tree.
        ResolutionError
            If resolution fails for any reason.
        """
        # Fast path: check cache first.
        if path in self._cache:
            cached = self._cache[path]
            if cached.is_fully_specified:
                logger.debug("Cache hit for %s", path)
                return cached

        # Walk the tree.
        node = _resolve_path(self._root, path)
        if node is None:
            raise NodeNotFoundError(f"Path {path} not found in Context")

        # Already fully specified – cache and return.
        if node.is_fully_specified:
            self._cache[path] = node
            return node

        # Resolve the node.
        logger.debug("Resolving node at %s", path)
        resolved = self._resolver.resolve_node(node, path, self._root)

        # If resolution produced a fully-specified node, cache it.
        if resolved.is_fully_specified:
            self._cache[path] = resolved

        return resolved

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_value(self, path: Path) -> Any:
        """
        Query *path* and return the underlying Python value.

        For :class:`~context_resolver.ast.nodes.ScalarNode` this is the scalar
        value.  For resolved
        :class:`~context_resolver.ast.resolvable_node.ResolvableNode` instances
        this returns the result node.  For compound nodes it returns the node
        itself.
        """
        from context_resolver.query.resolver import _extract_scalar

        node = self.query(path)
        return _extract_scalar(node)

    @property
    def root(self) -> Node:
        """The root node of the Context tree."""
        return self._root

    @property
    def resolver(self) -> Resolver:
        """The resolver attached to this Context."""
        return self._resolver

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire Context tree to a plain dictionary."""
        return {"root": self._root.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any], resolver: Resolver | None = None) -> "Context":
        """Deserialize a :class:`Context` from a plain dictionary."""
        from context_resolver.ast.nodes import _node_from_dict

        root = _node_from_dict(data["root"])
        return cls(root=root, resolver=resolver)

    def __repr__(self) -> str:
        return f"Context(root={self._root!r})"
