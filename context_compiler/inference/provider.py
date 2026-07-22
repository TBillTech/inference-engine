"""
Resolution provider interface.

The query engine depends **only** on this interface; it never imports concrete
provider implementations directly.  This decoupling means you can swap
providers (OpenAI → Anthropic → Prolog → Database) without touching the
query engine.

All concrete providers must subclass :class:`ResolutionProvider` and implement
:meth:`ResolutionProvider.resolve`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context_compiler.ast.nodes import Node
    from context_compiler.ast.paths import Path


@dataclass
class ResolutionRequest:
    """
    All information a resolution provider needs to resolve an underspecified
    semantic node.

    This object is provider-agnostic.  LLM-based providers use :attr:`prompt`
    and :attr:`model`; other providers (database, constraint solver, etc.) use
    :attr:`output_schema`, :attr:`query_path`, :attr:`dependencies`, and
    :attr:`metadata` instead.

    Attributes
    ----------
    prompt:
        The rendered prompt string (or list of chat messages).  Used by LLM
        providers; may be ``None`` for non-LLM providers.
    output_schema:
        A JSON-Schema-compatible dict describing the expected response
        structure.  May be ``None`` if the provider is invoked without
        structured-output constraints.
    query_path:
        The :class:`~context_compiler.ast.paths.Path` of the node being
        resolved within the Context tree.
    dependencies:
        Paths of nodes that the target node depends on, included for providers
        that perform graph-aware reasoning.
    metadata:
        Optional provider-specific metadata (e.g. hints, tags, annotations).
    model:
        Optional model override.  If ``None``, the provider uses its default.
    temperature:
        Sampling temperature hint (LLM providers).
    extra:
        Provider-specific additional parameters.
    """

    prompt: str | list[dict[str, str]] | None = None
    output_schema: dict[str, Any] | None = None
    query_path: Path | None = None
    dependencies: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolutionResult:
    """
    The typed result returned by a resolution provider.

    Attributes
    ----------
    data:
        A plain Python dict representing the decoded structured output.
        The compiler will validate this against the expected schema before
        building typed nodes.  For non-LLM providers, this may be empty if
        :attr:`resolved_node` is provided directly.
    success:
        ``True`` if resolution succeeded, ``False`` otherwise.
    model:
        The model that produced this response (as reported by the provider).
        May be ``"unknown"`` for non-LLM providers.
    provider:
        The provider name (e.g. ``"openai"``, ``"mock"``).
    diagnostics:
        Optional list of human-readable diagnostic messages produced during
        resolution (warnings, partial failures, etc.).
    provenance:
        Arbitrary key/value pairs recording where and how the result was
        produced (e.g. model version, prompt hash, timestamp).
    confidence:
        Optional confidence or certainty score in ``[0, 1]`` reported by the
        provider.
    raw:
        The raw API response object, retained for debugging.
    """

    data: dict[str, Any]
    success: bool = True
    model: str = "unknown"
    provider: str = "unknown"
    diagnostics: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    raw: Any = None


class ResolutionProvider:
    """
    Abstract base class for all resolution providers.

    A :class:`ResolutionProvider` is any engine capable of resolving an
    underspecified :class:`~context_compiler.ast.resolvable_node.ResolvableNode`.

    Possible implementations include:

    * ``OpenAIProvider`` – resolves via the OpenAI Chat Completions API.
    * ``AnthropicProvider`` – resolves via the Anthropic Messages API.
    * ``MockProvider`` – returns pre-configured responses for testing.
    * ``SWIPrologProvider`` – resolves via a Prolog inference engine.
    * ``DatabaseProvider`` – resolves by querying a database.
    * ``ConstraintSolverProvider`` – resolves using a constraint solver.

    Concrete subclasses must implement :meth:`resolve`.  The compiler depends
    **only** on this interface and never imports concrete provider classes.

    Provider Selection
    ------------------
    The :meth:`can_resolve` method is provided as a hook for future provider
    selection logic.  Override it to declare which requests a provider is
    capable of handling.
    """

    #: Human-readable provider name; override in subclasses.
    name: str = "abstract"

    def resolve(self, request: ResolutionRequest) -> ResolutionResult:
        """
        Resolve the underspecified node described by *request*.

        Parameters
        ----------
        request:
            The fully-described resolution request built by the compiler.

        Returns
        -------
        ResolutionResult
            A typed result.  The ``data`` field must be a plain dict that will
            be decoded by the resolver, unless a ``resolved_node`` is provided
            directly.

        Raises
        ------
        NotImplementedError
            If the subclass has not implemented this method.
        RuntimeError
            If resolution fails for any provider-specific reason.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement ResolutionProvider.resolve()"
        )

    def can_resolve(self, request: ResolutionRequest) -> bool:
        """
        Return ``True`` if this provider is capable of resolving *request*.

        The default implementation always returns ``True``.  Override this
        method to add provider-specific capability declarations, enabling the
        resolver to select the most appropriate provider for a given request.

        This is a hook for future provider selection logic.
        """
        return True

    def supports_structured_output(self) -> bool:
        """
        Return ``True`` if this provider natively supports JSON-mode or
        function-calling structured output.

        The resolver uses this hint to decide whether to include schema
        constraints in the request.
        """
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
