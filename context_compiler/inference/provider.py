"""
Inference provider interface.

The compiler depends **only** on this interface; it never imports concrete
provider implementations directly.  This decoupling means you can swap
providers (OpenAI → Anthropic → local) without touching the compiler.

All concrete providers must subclass :class:`InferenceProvider` and implement
:meth:`InferenceProvider.infer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceRequest:
    """
    All information the compiler passes to an inference provider.

    Attributes
    ----------
    prompt:
        The rendered prompt string (or list of chat messages).
    output_schema:
        A JSON-Schema-compatible dict describing the expected response
        structure.  May be ``None`` if the provider is invoked without
        structured-output constraints.
    model:
        Optional model override.  If ``None``, the provider uses its default.
    temperature:
        Sampling temperature hint.
    extra:
        Provider-specific additional parameters.
    """

    prompt: str | list[dict[str, str]]
    output_schema: dict[str, Any] | None = None
    model: str | None = None
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    """
    The structured response returned by an inference provider.

    Attributes
    ----------
    data:
        A plain Python dict representing the decoded structured output.
        The compiler will validate this against the expected schema before
        building typed nodes.
    model:
        The model that produced this response (as reported by the provider).
    provider:
        The provider name (e.g. ``"openai"``, ``"mock"``).
    raw:
        The raw API response object, retained for debugging.
    """

    data: dict[str, Any]
    model: str = "unknown"
    provider: str = "unknown"
    raw: Any = None


class InferenceProvider:
    """
    Abstract base class for all inference providers.

    Concrete subclasses implement :meth:`infer` and optionally
    :meth:`supports_structured_output`.

    The compiler calls :meth:`infer` exactly once per unresolved
    :class:`~context_compiler.ast.prompt_node.PromptNode` during a compilation
    pass, then decodes the returned :class:`InferenceResponse` into typed nodes.
    """

    #: Human-readable provider name; override in subclasses.
    name: str = "abstract"

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        Perform inference and return a structured response.

        Parameters
        ----------
        request:
            The fully-rendered request built by the compiler.

        Returns
        -------
        InferenceResponse
            A validated, structured response.  The ``data`` field must be a
            plain dict that will be decoded by the compiler's decoder pass.

        Raises
        ------
        NotImplementedError
            If the subclass has not implemented this method.
        RuntimeError
            If the inference call fails for any provider-specific reason.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement InferenceProvider.infer()"
        )

    def supports_structured_output(self) -> bool:
        """
        Return ``True`` if this provider natively supports JSON-mode or
        function-calling structured output.

        The compiler uses this hint to decide whether to include schema
        constraints in the request.
        """
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
