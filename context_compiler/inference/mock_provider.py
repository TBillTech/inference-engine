"""
Mock resolution provider for testing and local development.

:class:`MockProvider` returns pre-configured responses without any network
calls.  You can also configure it to raise exceptions to test error paths.

Examples
--------
>>> provider = MockProvider(responses={"greet": {"greeting": "Hello, world!"}})
>>> resp = provider.resolve(ResolutionRequest(prompt="greet"))
>>> resp.data
{'greeting': 'Hello, world!'}
"""

from __future__ import annotations

from typing import Any, Callable

from context_compiler.inference.provider import (
    ResolutionProvider,
    ResolutionRequest,
    ResolutionResult,
)


class MockProvider(ResolutionProvider):
    """
    A deterministic, in-memory resolution provider for tests.

    Parameters
    ----------
    responses:
        A mapping from prompt strings (or prefixes) to response dicts.
        If the prompt matches a key exactly, that dict is returned.
    default_response:
        Fallback response dict when no key matches.
    raise_on_call:
        If set, :meth:`resolve` raises this exception instead of returning a
        response.  Useful for testing error-handling paths.
    model:
        Model name reported in responses.
    """

    name: str = "mock"

    def __init__(
        self,
        responses: dict[str, dict[str, Any]] | None = None,
        *,
        default_response: dict[str, Any] | None = None,
        raise_on_call: Exception | None = None,
        model: str = "mock-model",
    ) -> None:
        self._responses: dict[str, dict[str, Any]] = responses or {}
        self._default_response: dict[str, Any] = default_response or {}
        self._raise_on_call: Exception | None = raise_on_call
        self._model: str = model
        self.call_count: int = 0
        self.last_request: ResolutionRequest | None = None

    def resolve(self, request: ResolutionRequest) -> ResolutionResult:
        """
        Return a pre-configured response for *request*.

        The prompt is matched against :attr:`_responses` keys by exact string
        comparison.  If no match is found, :attr:`_default_response` is used.
        """
        self.call_count += 1
        self.last_request = request

        if self._raise_on_call is not None:
            raise self._raise_on_call

        prompt_key = (
            request.prompt
            if isinstance(request.prompt, str)
            else str(request.prompt)
        )
        data = self._responses.get(prompt_key, self._default_response)
        return ResolutionResult(
            data=dict(data),
            model=self._model,
            provider=self.name,
            raw=None,
        )

    def supports_structured_output(self) -> bool:
        return True

    def reset(self) -> None:
        """Reset call counters and last-request tracking."""
        self.call_count = 0
        self.last_request = None
