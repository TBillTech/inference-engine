"""
ResolvableNode – a deferred semantic resolution node.

A :class:`ResolvableNode` represents any semantic value that is currently
underspecified and can be resolved by one or more
:class:`~context_compiler.inference.provider.ResolutionProvider` implementations.
It is **not** specifically tied to prompts or LLMs; any provider capable of
resolving the node may be used.

The node transitions through the following states:

* **PENDING** – declared but not yet resolved.
* **STALE** – a dependency has changed; the cached result is no longer valid.
* **RESOLVED** – resolution succeeded; the result is cached.
* **ERROR** – resolution failed; the error is stored for inspection.

A :class:`ResolvableNode` is **never** replaced by its result.  It retains its
provenance and cached output so the query engine can audit, replay, or diff
resolutions.
"""

from __future__ import annotations

import datetime
from enum import Enum, auto
from typing import Any

from context_compiler.ast.nodes import Node, NodeState
from context_compiler.ast.paths import Path
from context_compiler.ast.schema import Schema


class ResolvableNodeState(Enum):
    """Lifecycle state of a :class:`ResolvableNode`."""

    PENDING = auto()
    """The node has been declared but not yet resolved."""

    STALE = auto()
    """A dependency changed; the cached result is invalid."""

    RESOLVED = auto()
    """Resolution succeeded; the result is cached."""

    ERROR = auto()
    """Resolution failed; ``error`` carries the exception."""


class ResolvableNode(Node):
    """
    A node that defers semantic resolution to a resolution provider.

    A :class:`ResolvableNode` represents any underspecified semantic value
    that can be resolved by one or more
    :class:`~context_compiler.inference.provider.ResolutionProvider`
    implementations.  It is not tied to any specific resolution strategy
    (LLM prompting, database lookup, constraint solving, etc.).

    Parameters
    ----------
    template_ref:
        A string identifier for the :class:`~context_compiler.templates.template.Template`
        that drives this node's resolution step.
    input_bindings:
        A mapping from template variable names to :class:`~context_compiler.ast.paths.Path`
        objects that supply the variable values from the Context tree.
    output_schema:
        The expected output structure returned by the resolution provider.
    dependencies:
        Explicit list of :class:`Path` objects that, when changed, invalidate
        this node's cached result.  (The query engine may also derive implicit
        dependencies from *input_bindings*.)
    metadata:
        Optional debugging metadata (provenance, annotations, etc.).
    """

    def __init__(
        self,
        template_ref: str,
        input_bindings: dict[str, Path] | None = None,
        output_schema: Schema | None = None,
        dependencies: list[Path] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(metadata=metadata)
        self.template_ref: str = template_ref
        self.input_bindings: dict[str, Path] = dict(input_bindings or {})
        self.output_schema: Schema | None = output_schema
        self.dependencies: list[Path] = list(dependencies or [])

        # Runtime state
        self._resolution_state: ResolvableNodeState = ResolvableNodeState.PENDING
        self._result: Node | None = None
        self._error: Exception | None = None
        self._resolved_at: datetime.datetime | None = None
        self._provider: str | None = None
        self._model: str | None = None

    # ------------------------------------------------------------------
    # ResolvableNode state
    # ------------------------------------------------------------------

    @property
    def resolution_state(self) -> ResolvableNodeState:
        """Current lifecycle state of this ResolvableNode."""
        return self._resolution_state

    @property
    def result(self) -> Node | None:
        """The cached resolution result, or ``None`` if not yet resolved."""
        return self._result

    @property
    def error(self) -> Exception | None:
        """The resolution error, or ``None`` if not in ERROR state."""
        return self._error

    @property
    def resolved_at(self) -> datetime.datetime | None:
        """Timestamp of the last successful resolution."""
        return self._resolved_at

    @property
    def provider(self) -> str | None:
        """Name of the resolution provider that produced the cached result."""
        return self._provider

    @property
    def model(self) -> str | None:
        """Model identifier used during the last resolution call (LLM providers)."""
        return self._model

    # ------------------------------------------------------------------
    # Node.state (delegates to ResolvableNodeState)
    # ------------------------------------------------------------------

    @property
    def state(self) -> NodeState:
        """
        Map :class:`ResolvableNodeState` to :class:`~context_compiler.ast.nodes.NodeState`.

        * RESOLVED  → FULLY_SPECIFIED
        * PENDING / STALE → UNDERSPECIFIED
        * ERROR → UNDERSPECIFIED  (caller should inspect :attr:`error`)
        """
        if self._resolution_state is ResolvableNodeState.RESOLVED:
            return NodeState.FULLY_SPECIFIED
        return NodeState.UNDERSPECIFIED

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def mark_resolved(
        self,
        result: Node,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """
        Transition this node to RESOLVED state.

        The node is **not** discarded after resolution.  Its provenance and
        cached result are preserved so the query engine can audit or replay.

        Parameters
        ----------
        result:
            The decoded resolution result (a typed :class:`Node`).
        provider:
            Human-readable name of the provider (e.g. ``"openai"``, ``"mock"``).
        model:
            Model identifier (e.g. ``"gpt-4o"``).  May be ``None`` for
            non-LLM providers.
        """
        self._result = result
        self._resolution_state = ResolvableNodeState.RESOLVED
        self._error = None
        self._resolved_at = datetime.datetime.now(datetime.timezone.utc)
        self._provider = provider
        self._model = model

    def mark_stale(self) -> None:
        """Transition this node to STALE state, invalidating the cached result."""
        self._resolution_state = ResolvableNodeState.STALE
        self._result = None

    def mark_error(self, error: Exception) -> None:
        """Transition this node to ERROR state with the given exception."""
        self._resolution_state = ResolvableNodeState.ERROR
        self._error = error

    # ------------------------------------------------------------------
    # Effective dependency paths
    # ------------------------------------------------------------------

    def effective_dependencies(self) -> list[Path]:
        """
        Return the union of explicit *dependencies* and paths derived from
        *input_bindings*.
        """
        seen: set[Path] = set(self.dependencies)
        result: list[Path] = list(self.dependencies)
        for path in self.input_bindings.values():
            if path not in seen:
                seen.add(path)
                result.append(path)
        return result

    # ------------------------------------------------------------------
    # Node protocol
    # ------------------------------------------------------------------

    def _copy_with_metadata(self, metadata: dict[str, Any]) -> "ResolvableNode":
        node = ResolvableNode(
            self.template_ref,
            input_bindings=self.input_bindings,
            output_schema=self.output_schema,
            dependencies=self.dependencies,
            metadata=metadata,
        )
        node._resolution_state = self._resolution_state
        node._result = self._result
        node._error = self._error
        node._resolved_at = self._resolved_at
        node._provider = self._provider
        node._model = self._model
        return node

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ResolvableNode",
            "template_ref": self.template_ref,
            "input_bindings": {
                k: list(v.segments) for k, v in self.input_bindings.items()
            },
            "output_schema": (
                self.output_schema.to_dict() if self.output_schema else None
            ),
            "dependencies": [list(p.segments) for p in self.dependencies],
            "resolution_state": self._resolution_state.name,
            "result": self._result.to_dict() if self._result is not None else None,
            "error": str(self._error) if self._error else None,
            "resolved_at": (
                self._resolved_at.isoformat() if self._resolved_at else None
            ),
            "provider": self._provider,
            "model": self._model,
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResolvableNode":
        from context_compiler.ast.nodes import _node_from_dict

        input_bindings = {
            k: Path(*v) for k, v in data.get("input_bindings", {}).items()
        }
        dependencies = [Path(*p) for p in data.get("dependencies", [])]
        output_schema = (
            Schema.from_dict(data["output_schema"])
            if data.get("output_schema")
            else None
        )
        node = cls(
            template_ref=data["template_ref"],
            input_bindings=input_bindings,
            output_schema=output_schema,
            dependencies=dependencies,
            metadata=data.get("metadata"),
        )
        state_key = data.get("resolution_state", "PENDING")
        node._resolution_state = ResolvableNodeState[state_key]
        if data.get("result") is not None:
            node._result = _node_from_dict(data["result"])
        if data.get("error"):
            node._error = RuntimeError(data["error"])
        if data.get("resolved_at"):
            node._resolved_at = datetime.datetime.fromisoformat(data["resolved_at"])
        node._provider = data.get("provider")
        node._model = data.get("model")
        return node

    def __repr__(self) -> str:
        return (
            f"ResolvableNode(template_ref={self.template_ref!r}, "
            f"resolution_state={self._resolution_state.name})"
        )


# Register ResolvableNode in the polymorphic node registry.
from context_compiler.ast.nodes import register_node_type

register_node_type(ResolvableNode)
