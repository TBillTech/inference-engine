"""
Core node types for the Context Compiler semantic tree.

Every piece of semantic information inside the Context is represented as a
typed :class:`Node`.  Nodes are never raw dictionaries or plain strings – all
data must pass through a :class:`~context_compiler.ast.schema.Schema` decoder
before entering the tree.

Node states
-----------
* **FULLY_SPECIFIED** – the value is complete and can be used immediately.
* **PARTIALLY_SPECIFIED** – some fields are known; others still need resolution.
* **UNDERSPECIFIED** – the node contains no usable value yet.
"""

from __future__ import annotations

import json
from enum import Enum, auto
from typing import Any, Iterator, Sequence, Union


class NodeState(Enum):
    """Represents the completeness of a :class:`Node`'s value."""

    UNDERSPECIFIED = auto()
    """No value has been assigned to this node yet."""

    PARTIALLY_SPECIFIED = auto()
    """Some, but not all, fields of this node have been resolved."""

    FULLY_SPECIFIED = auto()
    """This node's value is complete and ready to use."""


class Node:
    """
    Abstract base class for all nodes in the semantic tree.

    Subclasses must implement :meth:`state`, :meth:`to_dict`, and
    :meth:`from_dict`.

    Parameters
    ----------
    metadata:
        Arbitrary key/value pairs used for debugging (e.g. provenance,
        timestamps, compiler pass annotations).
    """

    def __init__(self, *, metadata: dict[str, Any] | None = None) -> None:
        self._metadata: dict[str, Any] = metadata or {}

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> NodeState:
        """Return the current :class:`NodeState` of this node."""
        raise NotImplementedError

    @property
    def is_fully_specified(self) -> bool:
        """``True`` if this node requires no further compilation."""
        return self.state is NodeState.FULLY_SPECIFIED

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        """Read-only view of this node's metadata dict."""
        return dict(self._metadata)

    def with_metadata(self, **kwargs: Any) -> "Node":
        """Return a copy of this node with additional metadata entries merged in."""
        new_meta = {**self._metadata, **kwargs}
        return self._copy_with_metadata(new_meta)

    def _copy_with_metadata(self, metadata: dict[str, Any]) -> "Node":
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize this node to a plain Python dictionary."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Node":
        """Deserialize a node from a plain Python dictionary."""
        raise NotImplementedError

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize this node to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(state={self.state.name})"


# ---------------------------------------------------------------------------
# Concrete node types
# ---------------------------------------------------------------------------


class ScalarNode(Node):
    """
    A leaf node that holds a single scalar value (str, int, float, bool, None).

    Parameters
    ----------
    value:
        The scalar value.  Pass ``None`` to create an underspecified node.
    metadata:
        Optional debugging metadata.
    """

    #: Accepted Python types for scalar values.
    SCALAR_TYPES = (str, int, float, bool, type(None))

    def __init__(
        self,
        value: str | int | float | bool | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(metadata=metadata)
        if value is not None and not isinstance(value, self.SCALAR_TYPES):
            raise TypeError(
                f"ScalarNode value must be one of {self.SCALAR_TYPES}, "
                f"got {type(value).__name__}"
            )
        self._value: str | int | float | bool | None = value

    @property
    def value(self) -> str | int | float | bool | None:
        """The stored scalar value."""
        return self._value

    @property
    def state(self) -> NodeState:
        if self._value is None:
            return NodeState.UNDERSPECIFIED
        return NodeState.FULLY_SPECIFIED

    def _copy_with_metadata(self, metadata: dict[str, Any]) -> "ScalarNode":
        return ScalarNode(self._value, metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ScalarNode",
            "value": self._value,
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScalarNode":
        return cls(data["value"], metadata=data.get("metadata"))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScalarNode):
            return NotImplemented
        return self._value == other._value

    def __repr__(self) -> str:
        return f"ScalarNode(value={self._value!r}, state={self.state.name})"


class MappingNode(Node):
    """
    A node whose value is a mapping from string keys to child :class:`Node`\\ s.

    Parameters
    ----------
    fields:
        Initial mapping of field names to child nodes.
    metadata:
        Optional debugging metadata.
    """

    def __init__(
        self,
        fields: dict[str, Node] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(metadata=metadata)
        self._fields: dict[str, Node] = dict(fields) if fields else {}

    # ------------------------------------------------------------------
    # Field access
    # ------------------------------------------------------------------

    def get(self, key: str) -> Node | None:
        """Return the child node for *key*, or ``None`` if absent."""
        return self._fields.get(key)

    def set(self, key: str, node: Node) -> None:
        """Assign *node* to *key*, mutating this mapping in place."""
        self._fields[key] = node

    def keys(self) -> Sequence[str]:
        """Return the field names of this mapping."""
        return list(self._fields.keys())

    def items(self) -> Iterator[tuple[str, Node]]:
        """Iterate over ``(key, node)`` pairs."""
        return iter(self._fields.items())

    def __contains__(self, key: str) -> bool:
        return key in self._fields

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> NodeState:
        if not self._fields:
            return NodeState.UNDERSPECIFIED
        states = {child.state for child in self._fields.values()}
        if all(s is NodeState.FULLY_SPECIFIED for s in states):
            return NodeState.FULLY_SPECIFIED
        if all(s is NodeState.UNDERSPECIFIED for s in states):
            return NodeState.UNDERSPECIFIED
        return NodeState.PARTIALLY_SPECIFIED

    def _copy_with_metadata(self, metadata: dict[str, Any]) -> "MappingNode":
        return MappingNode(self._fields, metadata=metadata)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "MappingNode",
            "fields": {k: v.to_dict() for k, v in self._fields.items()},
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MappingNode":
        fields = {k: _node_from_dict(v) for k, v in data.get("fields", {}).items()}
        return cls(fields, metadata=data.get("metadata"))

    def __repr__(self) -> str:
        return (
            f"MappingNode(keys={list(self._fields.keys())!r}, "
            f"state={self.state.name})"
        )


class SequenceNode(Node):
    """
    A node whose value is an ordered sequence of child :class:`Node`\\ s.

    Parameters
    ----------
    items:
        Initial sequence of child nodes.
    metadata:
        Optional debugging metadata.
    """

    def __init__(
        self,
        items: list[Node] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(metadata=metadata)
        self._items: list[Node] = list(items) if items else []

    # ------------------------------------------------------------------
    # Item access
    # ------------------------------------------------------------------

    def get(self, index: int) -> Node | None:
        """Return the child node at *index*, or ``None`` if out of range."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def append(self, node: Node) -> None:
        """Append *node* to the end of this sequence."""
        self._items.append(node)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._items)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> NodeState:
        if not self._items:
            return NodeState.UNDERSPECIFIED
        states = {child.state for child in self._items}
        if all(s is NodeState.FULLY_SPECIFIED for s in states):
            return NodeState.FULLY_SPECIFIED
        if all(s is NodeState.UNDERSPECIFIED for s in states):
            return NodeState.UNDERSPECIFIED
        return NodeState.PARTIALLY_SPECIFIED

    def _copy_with_metadata(self, metadata: dict[str, Any]) -> "SequenceNode":
        return SequenceNode(self._items, metadata=metadata)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "SequenceNode",
            "items": [item.to_dict() for item in self._items],
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SequenceNode":
        items = [_node_from_dict(item) for item in data.get("items", [])]
        return cls(items, metadata=data.get("metadata"))

    def __repr__(self) -> str:
        return (
            f"SequenceNode(len={len(self._items)}, state={self.state.name})"
        )


# ---------------------------------------------------------------------------
# Polymorphic deserialization helper
# ---------------------------------------------------------------------------

# Registry maps the "type" field in serialized dicts to the corresponding class.
_NODE_REGISTRY: dict[str, type[Node]] = {}


def register_node_type(node_cls: type[Node]) -> type[Node]:
    """Register *node_cls* in the polymorphic deserialization registry."""
    _NODE_REGISTRY[node_cls.__name__] = node_cls
    return node_cls


def register_node_type_alias(alias: str, node_cls: type[Node]) -> None:
    """
    Register *alias* as an additional deserialization key for *node_cls*.

    This is used to maintain backward compatibility when a node class is
    renamed: old serialised data using the previous type name can still be
    deserialised by the new class.

    Parameters
    ----------
    alias:
        The legacy ``"type"`` field value to map to *node_cls*.
    node_cls:
        The class that should handle deserialization for *alias*.
    """
    _NODE_REGISTRY[alias] = node_cls


def _node_from_dict(data: dict[str, Any]) -> Node:
    """Deserialize a node from a dictionary, dispatching on the ``type`` field."""
    type_name: str = data.get("type", "")
    node_cls = _NODE_REGISTRY.get(type_name)
    if node_cls is None:
        raise ValueError(
            f"Unknown node type {type_name!r}. "
            f"Available types: {list(_NODE_REGISTRY.keys())}"
        )
    return node_cls.from_dict(data)


# Register the built-in node types.
for _cls in (ScalarNode, MappingNode, SequenceNode):
    register_node_type(_cls)
