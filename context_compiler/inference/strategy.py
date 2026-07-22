"""
Resolution strategy interface.

A :class:`ResolutionStrategy` describes *how* a
:class:`~context_compiler.ast.resolvable_node.ResolvableNode` should be resolved –
the semantic procedure.  It is separate from the
:class:`~context_compiler.inference.provider.ResolutionProvider`, which
describes *which engine* executes that procedure.

This decoupling means the same strategy can be executed by different providers:

    PromptStrategy
        ├── OpenAIProvider
        ├── AnthropicProvider
        └── LocalLLMProvider

and different strategies can target different providers:

    PrologQueryStrategy  →  SWIPrologProvider
    SQLLookupStrategy    →  SQLiteProvider

Concrete strategies must subclass :class:`ResolutionStrategy` and implement
:meth:`ResolutionStrategy.resolve`.

Built-in strategies
-------------------
* :class:`PromptStrategy` – the default strategy; renders a prompt and
  delegates to a single :class:`~context_compiler.inference.provider.ResolutionProvider`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from context_compiler.inference.provider import ResolutionProvider, ResolutionRequest, ResolutionResult


class ResolutionStrategy:
    """
    Abstract base class for all resolution strategies.

    A :class:`ResolutionStrategy` describes the *procedure* used to resolve a
    :class:`~context_compiler.ast.resolvable_node.ResolvableNode` – it is concerned
    with *what* to do, not *which engine* does the work.

    Possible implementations include:

    * :class:`PromptStrategy` – renders a prompt template and calls an LLM provider.
    * ``PrologQueryStrategy`` – formats a Prolog query and calls a Prolog engine.
    * ``DatabaseLookupStrategy`` – builds a SQL query and calls a database engine.
    * ``ConstraintStrategy`` – applies constraint solving via a constraint engine.
    * ``FallbackStrategy`` – tries a sequence of strategies until one succeeds.
    * ``CompositeStrategy`` – combines multiple strategies.

    Concrete subclasses must implement :meth:`resolve`.

    Provider Access
    ---------------
    The :attr:`provider` property is an optional hook for strategies that wrap
    a single :class:`~context_compiler.inference.provider.ResolutionProvider`.
    Strategies with multiple or no fixed providers should leave it as ``None``.
    """

    #: Human-readable strategy name; override in subclasses.
    name: str = "abstract"

    def resolve(self, request: "ResolutionRequest") -> "ResolutionResult":
        """
        Execute this strategy for the given resolution request.

        Parameters
        ----------
        request:
            The fully-described resolution request built by the compiler.

        Returns
        -------
        ResolutionResult
            A typed result.

        Raises
        ------
        NotImplementedError
            If the subclass has not implemented this method.
        RuntimeError
            If resolution fails for any strategy-specific reason.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement ResolutionStrategy.resolve()"
        )

    def can_apply(self, request: "ResolutionRequest") -> bool:
        """
        Return ``True`` if this strategy is applicable to *request*.

        The default implementation always returns ``True``.  Override this
        method to add strategy-specific capability declarations.
        """
        return True

    @property
    def provider(self) -> "ResolutionProvider | None":
        """
        The primary :class:`~context_compiler.inference.provider.ResolutionProvider`
        used by this strategy, or ``None`` if the strategy does not expose one.

        This is an optional hook for strategies that wrap a single provider.
        Strategies with multiple or no fixed providers should return ``None``.
        """
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class PromptStrategy(ResolutionStrategy):
    """
    The default resolution strategy: renders a prompt and delegates to a
    :class:`~context_compiler.inference.provider.ResolutionProvider`.

    This is the strategy that corresponds to the existing behaviour – it passes
    the :class:`~context_compiler.inference.provider.ResolutionRequest` directly
    to the wrapped provider's :meth:`~context_compiler.inference.provider.ResolutionProvider.resolve`
    method without any additional transformation.

    A single :class:`PromptStrategy` can be used with any LLM-compatible
    provider (OpenAI, Anthropic, local models, mock providers, etc.) by simply
    swapping the *provider* argument.

    Parameters
    ----------
    provider:
        The :class:`~context_compiler.inference.provider.ResolutionProvider`
        that will execute the prompt-based resolution.
    """

    name: str = "prompt"

    def __init__(self, provider: "ResolutionProvider") -> None:
        self._provider = provider

    @property
    def provider(self) -> "ResolutionProvider":
        """The provider that executes this strategy."""
        return self._provider

    def resolve(self, request: "ResolutionRequest") -> "ResolutionResult":
        """Delegate resolution to the wrapped provider."""
        return self._provider.resolve(request)

    def __repr__(self) -> str:
        return f"PromptStrategy(provider={self._provider!r})"
