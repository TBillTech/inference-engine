"""
Compiler pass infrastructure.

Compiler passes transform the Context tree in a composable, independently
testable way.  Each pass should:

* be deterministic (same input → same output) unless it wraps resolution
* have no hidden side effects beyond modifying the nodes it is given
* be independently testable without spinning up a full compiler

Pass types
----------
* :class:`DeterministicPass` – rule-based normalization / constraint passes.
* :class:`ResolutionPass` – drives provider resolution for unresolved
  :class:`~context_compiler.ast.prompt_node.ResolvableNode` instances.

Custom passes should subclass one of these and implement :meth:`run`.

Backward Compatibility
----------------------
``InferencePass`` is kept as an alias for :class:`ResolutionPass`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context_compiler.ast.nodes import Node
    from context_compiler.ast.paths import Path
    from context_compiler.ast.prompt_node import ResolvableNode
    from context_compiler.inference.provider import ResolutionProvider


class PassContext:
    """
    Lightweight context object passed to every compiler pass.

    Passes should read from and write to the Context *only* through this
    object so that the compiler can track modifications.

    Attributes
    ----------
    root:
        The root node of the current Context tree.
    changed_paths:
        Paths that were modified during this pass (populated by the pass).
    """

    def __init__(self, root: "Node") -> None:
        self.root: "Node" = root
        self.changed_paths: list["Path"] = []

    def record_change(self, path: "Path") -> None:
        """Mark *path* as having been modified by this pass."""
        self.changed_paths.append(path)


class CompilerPass:
    """
    Abstract base class for all compiler passes.

    Attributes
    ----------
    name:
        Human-readable name for logging and diagnostics.
    """

    name: str = "abstract-pass"

    def run(self, ctx: PassContext) -> None:
        """
        Execute this pass against the given :class:`PassContext`.

        Subclasses must implement this method.  After the method returns, any
        paths that were modified should be recorded via
        :meth:`PassContext.record_change`.

        Raises
        ------
        NotImplementedError
            If the subclass has not implemented this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement CompilerPass.run()"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class DeterministicPass(CompilerPass):
    """
    A compiler pass that applies deterministic, rule-based transformations.

    Sub-class and implement :meth:`run` to add normalization, validation, or
    other purely deterministic transformations that do not require resolution.

    Examples of deterministic passes:

    * Normalize string casing.
    * Fill in computed defaults (e.g. ``full_name = first + " " + last``).
    * Assert that required fields are present.
    """

    name: str = "deterministic-pass"


class ResolutionPass(CompilerPass):
    """
    A compiler pass that resolves unresolved
    :class:`~context_compiler.ast.prompt_node.ResolvableNode` instances using a
    :class:`~context_compiler.inference.provider.ResolutionProvider`.

    The compiler is provider-agnostic: any
    :class:`~context_compiler.inference.provider.ResolutionProvider`
    implementation can be used (LLM, database, constraint solver, etc.).

    Parameters
    ----------
    provider:
        The resolution provider to use.
    """

    name: str = "resolution-pass"

    def __init__(self, provider: "ResolutionProvider") -> None:
        self._provider = provider

    @property
    def provider(self) -> "ResolutionProvider":
        """The configured resolution provider."""
        return self._provider


#: Backward-compatible alias for :class:`ResolutionPass`.
InferencePass = ResolutionPass
