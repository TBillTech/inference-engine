"""
OpenAI resolution provider stub.

This module provides an :class:`OpenAIProvider` that wraps the OpenAI Python
SDK.  It is intentionally left as a stub so that the ``openai`` package is not
a required dependency; install it separately if you need live resolution.

To use this provider::

    from context_compiler.inference.openai_provider import OpenAIProvider
    provider = OpenAIProvider(api_key="sk-...", model="gpt-4o")
"""

from __future__ import annotations

from typing import Any

from context_compiler.inference.provider import (
    ResolutionProvider,
    ResolutionRequest,
    ResolutionResult,
)


class OpenAIProvider(ResolutionProvider):
    """
    Resolution provider backed by the OpenAI Chat Completions API.

    Parameters
    ----------
    api_key:
        Your OpenAI API key.  Defaults to ``None``, in which case the
        ``OPENAI_API_KEY`` environment variable is used.
    model:
        Default model to use (e.g. ``"gpt-4o"``).
    organization:
        Optional OpenAI organization ID.
    """

    name: str = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        organization: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._organization = organization
        self._client: Any = None  # Lazy-initialized on first resolve() call

    def _get_client(self) -> Any:
        """Lazily initialise and return the OpenAI client."""
        if self._client is None:
            try:
                import openai  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "The 'openai' package is required to use OpenAIProvider. "
                    "Install it with: pip install openai"
                ) from exc
            self._client = openai.OpenAI(
                api_key=self._api_key,
                organization=self._organization,
            )
        return self._client

    def resolve(self, request: ResolutionRequest) -> ResolutionResult:
        """
        Call the OpenAI Chat Completions API and return a structured result.

        When *request.output_schema* is provided and the chosen model supports
        JSON mode, the schema is passed as a response format constraint.

        Raises
        ------
        ValueError
            If *request.prompt* is ``None`` (OpenAI requires a prompt).
        """
        if request.prompt is None:
            raise ValueError(
                "OpenAIProvider requires a non-None prompt in the ResolutionRequest."
            )
        client = self._get_client()
        model = request.model or self._model

        # Build messages list
        if isinstance(request.prompt, list):
            messages = request.prompt
        else:
            messages = [{"role": "user", "content": request.prompt}]

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
        }

        if request.output_schema is not None and self.supports_structured_output():
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        content: str = response.choices[0].message.content or "{}"

        import json

        try:
            data: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"OpenAI response was not valid JSON: {content!r}"
            ) from exc

        return ResolutionResult(
            data=data,
            model=response.model,
            provider=self.name,
            raw=response,
        )

    def supports_structured_output(self) -> bool:
        return True
