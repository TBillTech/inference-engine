"""
Query pass infrastructure.

Query passes transform the Context tree in a composable, independently
testable way.  Each pass should:

* be deterministic (same input → same output) unless it wraps resolution
* have no hidden side effects beyond modifying the nodes it is given
* be independently testable without spinning up a full resolver

Pass types
----------
* :class:`DeterministicPass` – rule-based normalization / constraint passes.
* :class:`ResolutionPass` – drives strategy-based resolution for unresolved
    :class:`~context_resolver.ast.resolvable_node.ResolvableNode` instances.

Custom passes should subclass one of these and implement :meth:`run`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context_resolver.ast.nodes import Node
    from context_resolver.ast.paths import Path
    from context_resolver.ast.resolvable_node import ResolvableNode
    from context_resolver.inference.provider import ResolutionProvider
    from context_resolver.inference.strategy import ResolutionStrategy


class PassContext:
    """
    Lightweight context object passed to every query pass.

    Passes should read from and write to the Context *only* through this
    object so that the resolver can track modifications.

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


class QueryPass:
    """
    Abstract base class for all query passes.

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
            f"{type(self).__name__} must implement QueryPass.run()"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class DeterministicPass(QueryPass):
    """
    A query pass that applies deterministic, rule-based transformations.

    Sub-class and implement :meth:`run` to add normalization, validation, or
    other purely deterministic transformations that do not require resolution.

    Examples of deterministic passes:

    * Normalize string casing.
    * Fill in computed defaults (e.g. ``full_name = first + " " + last``).
    * Assert that required fields are present.
    """

    name: str = "deterministic-pass"


class ResolutionPass(QueryPass):
    """
    A query pass that resolves unresolved
    :class:`~context_resolver.ast.resolvable_node.ResolvableNode` instances using a
    :class:`~context_resolver.inference.strategy.ResolutionStrategy`.

    The resolver is strategy-agnostic: any
    :class:`~context_resolver.inference.strategy.ResolutionStrategy`
    implementation can be used (LLM prompt, Prolog query, database lookup, etc.).

    Parameters
    ----------
    strategy_or_provider:
        Either a :class:`~context_resolver.inference.strategy.ResolutionStrategy`
        or a :class:`~context_resolver.inference.provider.ResolutionProvider`.
        When a bare provider is given it is automatically wrapped in a
        :class:`~context_resolver.inference.strategy.PromptStrategy`.
    """

    name: str = "resolution-pass"

    def __init__(
        self,
        strategy_or_provider: "ResolutionStrategy | ResolutionProvider",
    ) -> None:
        from context_resolver.inference.strategy import ResolutionStrategy, PromptStrategy
        from context_resolver.inference.provider import ResolutionProvider

        if isinstance(strategy_or_provider, ResolutionStrategy):
            self._strategy: "ResolutionStrategy" = strategy_or_provider
        elif isinstance(strategy_or_provider, ResolutionProvider):
            # Wrap a bare provider in the default PromptStrategy.
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

        Returns ``self.strategy.provider`` and raises :exc:`AttributeError` if
        the strategy does not expose a single provider.
        """
        p = self._strategy.provider
        if p is None:
            raise AttributeError(
                f"{type(self._strategy).__name__} does not expose a single provider; "
                "use ResolutionPass.strategy instead."
            )
        return p
