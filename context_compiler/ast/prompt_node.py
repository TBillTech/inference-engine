"""
PromptNode – a deferred semantic inference node.

A :class:`PromptNode` represents work that has been declared but not yet
executed.  It transitions through three states:

* **PENDING** – declared but never compiled.
* **STALE** – a dependency has changed; the cached result is no longer valid.
* **RESOLVED** – inference succeeded; the result is cached.
* **ERROR** – compilation failed; the error is stored for inspection.

A PromptNode is **never** replaced by its result.  It retains its provenance
and cached output so the compiler can audit, replay, or diff compilations.
"""

from __future__ import annotations

import datetime
from enum import Enum, auto
from typing import Any

from context_compiler.ast.nodes import Node, NodeState
from context_compiler.ast.paths import Path
from context_compiler.ast.schema import Schema


class PromptNodeState(Enum):
    """Lifecycle state of a :class:`PromptNode`."""

    PENDING = auto()
    """The node has been declared but not yet compiled."""

    STALE = auto()
    """A dependency changed; the cached result is invalid."""

    RESOLVED = auto()
    """Inference succeeded; the result is cached."""

    ERROR = auto()
    """Compilation failed; ``error`` carries the exception."""


class PromptNode(Node):
    """
    A node that defers semantic resolution to an inference provider.

    Parameters
    ----------
    template_ref:
        A string identifier for the :class:`~context_compiler.templates.template.Template`
        that drives this node's inference step.
    input_bindings:
        A mapping from template variable names to :class:`~context_compiler.ast.paths.Path`
        objects that supply the variable values from the Context tree.
    output_schema:
        The expected output structure returned by the inference provider.
    dependencies:
        Explicit list of :class:`Path` objects that, when changed, invalidate
        this node's cached result.  (The compiler may also derive implicit
        dependencies from *input_bindings*.)
    metadata:
        Optional debugging metadata.
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
        self._prompt_state: PromptNodeState = PromptNodeState.PENDING
        self._result: Node | None = None
        self._error: Exception | None = None
        self._resolved_at: datetime.datetime | None = None
        self._provider: str | None = None
        self._model: str | None = None

    # ------------------------------------------------------------------
    # PromptNode state
    # ------------------------------------------------------------------

    @property
    def prompt_state(self) -> PromptNodeState:
        """Current lifecycle state of this PromptNode."""
        return self._prompt_state

    @property
    def result(self) -> Node | None:
        """The cached inference result, or ``None`` if not yet resolved."""
        return self._result

    @property
    def error(self) -> Exception | None:
        """The compilation error, or ``None`` if not in ERROR state."""
        return self._error

    @property
    def resolved_at(self) -> datetime.datetime | None:
        """Timestamp of the last successful resolution."""
        return self._resolved_at

    @property
    def provider(self) -> str | None:
        """Name of the inference provider that produced the cached result."""
        return self._provider

    @property
    def model(self) -> str | None:
        """Model identifier used during the last inference call."""
        return self._model

    # ------------------------------------------------------------------
    # Node.state (delegates to PromptNodeState)
    # ------------------------------------------------------------------

    @property
    def state(self) -> NodeState:
        """
        Map :class:`PromptNodeState` to :class:`~context_compiler.ast.nodes.NodeState`.

        * RESOLVED  → FULLY_SPECIFIED
        * PENDING / STALE → UNDERSPECIFIED
        * ERROR → UNDERSPECIFIED  (caller should inspect :attr:`error`)
        """
        if self._prompt_state is PromptNodeState.RESOLVED:
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

        Parameters
        ----------
        result:
            The decoded inference result (a typed :class:`Node`).
        provider:
            Human-readable name of the provider (e.g. ``"openai"``).
        model:
            Model identifier (e.g. ``"gpt-4o"``).
        """
        self._result = result
        self._prompt_state = PromptNodeState.RESOLVED
        self._error = None
        self._resolved_at = datetime.datetime.now(datetime.timezone.utc)
        self._provider = provider
        self._model = model

    def mark_stale(self) -> None:
        """Transition this node to STALE state, invalidating the cached result."""
        self._prompt_state = PromptNodeState.STALE
        self._result = None

    def mark_error(self, error: Exception) -> None:
        """Transition this node to ERROR state with the given exception."""
        self._prompt_state = PromptNodeState.ERROR
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

    def _copy_with_metadata(self, metadata: dict[str, Any]) -> "PromptNode":
        node = PromptNode(
            self.template_ref,
            input_bindings=self.input_bindings,
            output_schema=self.output_schema,
            dependencies=self.dependencies,
            metadata=metadata,
        )
        node._prompt_state = self._prompt_state
        node._result = self._result
        node._error = self._error
        node._resolved_at = self._resolved_at
        node._provider = self._provider
        node._model = self._model
        return node

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "PromptNode",
            "template_ref": self.template_ref,
            "input_bindings": {
                k: list(v.segments) for k, v in self.input_bindings.items()
            },
            "output_schema": (
                self.output_schema.to_dict() if self.output_schema else None
            ),
            "dependencies": [list(p.segments) for p in self.dependencies],
            "prompt_state": self._prompt_state.name,
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
    def from_dict(cls, data: dict[str, Any]) -> "PromptNode":
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
        node._prompt_state = PromptNodeState[data.get("prompt_state", "PENDING")]
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
            f"PromptNode(template_ref={self.template_ref!r}, "
            f"prompt_state={self._prompt_state.name})"
        )


# Register PromptNode in the polymorphic node registry.
from context_compiler.ast.nodes import register_node_type

register_node_type(PromptNode)
