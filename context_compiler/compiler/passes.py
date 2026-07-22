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
* :class:`ResolutionPass` – drives strategy-based resolution for unresolved
  :class:`~context_compiler.ast.prompt_node.ResolvableNode` instances.

Custom passes should subclass one of these and implement :meth:`run`.

Backward Compatibility
----------------------
``InferencePass`` is kept as an alias for :class:`ResolutionPass`.

When a bare :class:`~context_compiler.inference.provider.ResolutionProvider`
is passed to :class:`ResolutionPass`, it is automatically wrapped in a
:class:`~context_compiler.inference.strategy.PromptStrategy` so that existing
call sites continue to work without modification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context_compiler.ast.nodes import Node
    from context_compiler.ast.paths import Path
    from context_compiler.ast.prompt_node import ResolvableNode
    from context_compiler.inference.provider import ResolutionProvider
    from context_compiler.inference.strategy import ResolutionStrategy


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
    :class:`~context_compiler.inference.strategy.ResolutionStrategy`.

    The compiler is strategy-agnostic: any
    :class:`~context_compiler.inference.strategy.ResolutionStrategy`
    implementation can be used (LLM prompt, Prolog query, database lookup, etc.).

    Parameters
    ----------
    strategy_or_provider:
        Either a :class:`~context_compiler.inference.strategy.ResolutionStrategy`
        or a :class:`~context_compiler.inference.provider.ResolutionProvider`.
        When a bare provider is given it is automatically wrapped in a
        :class:`~context_compiler.inference.strategy.PromptStrategy` for
        backward compatibility.
    """

    name: str = "resolution-pass"

    def __init__(
        self,
        strategy_or_provider: "ResolutionStrategy | ResolutionProvider",
    ) -> None:
        from context_compiler.inference.strategy import ResolutionStrategy, PromptStrategy
        from context_compiler.inference.provider import ResolutionProvider

        if isinstance(strategy_or_provider, ResolutionStrategy):
            self._strategy: "ResolutionStrategy" = strategy_or_provider
        elif isinstance(strategy_or_provider, ResolutionProvider):
            # Backward compat: wrap a bare provider in the default PromptStrategy.
            self._strategy = PromptStrategy(strategy_or_provider)
        else:
            raise TypeError(
                "ResolutionPass expects a ResolutionStrategy or ResolutionProvider; "
                f"got {type(strategy_or_provider).__name__}"
            )

    @property
    def strategy(self) -> "ResolutionStrategy":
        """The configured resolution strategy."""
        return self._strategy

    @property
    def provider(self) -> "ResolutionProvider":
        """
        The resolution provider exposed by the configured strategy.

        This is a backward-compatible accessor.  It returns
        ``self.strategy.provider`` and raises :exc:`AttributeError` if the
        strategy does not expose a single provider.
        """
        p = self._strategy.provider
        if p is None:
            raise AttributeError(
                f"{type(self._strategy).__name__} does not expose a single provider; "
                "use ResolutionPass.strategy instead."
            )
        return p


#: Backward-compatible alias for :class:`ResolutionPass`.
InferencePass = ResolutionPass
