"""
Local llama.cpp resolution provider.

:class:`LocalLlamaCppProvider` calls a locally-running llama.cpp server that
exposes an OpenAI-compatible HTTP API (the ``--server`` / ``-sp`` mode).  The
default endpoint is ``http://127.0.0.1:8080``, which matches the llama.cpp
server defaults.

Startup assumptions
-------------------
Start the llama.cpp server in OpenAI-compatible mode before running any code
that instantiates this provider::

    ./llama-server --model /path/to/model.gguf --host 127.0.0.1 --port 8080

The ``/v1/chat/completions`` endpoint must be reachable at the configured
``base_url``.  No authentication is required by default (the server runs
locally), but you can pass a custom ``api_key`` if your server is configured
to require one.

Environment variables
---------------------
``LLAMA_CPP_BASE_URL``
    Override the default ``http://127.0.0.1:8080`` base URL.
``LLAMA_CPP_API_KEY``
    Optional API key for the server (rarely needed for local use).
``LLAMA_CPP_MODEL``
    Default model name to report in requests (cosmetic; the server
    ignores it unless multiple models are loaded).

Example usage::

    from context_resolver.inference.llama_cpp_provider import LocalLlamaCppProvider
    from context_resolver.inference.strategy import PromptStrategy

    provider = LocalLlamaCppProvider(base_url="http://127.0.0.1:8080", model="local")
    strategy = PromptStrategy(provider)
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from context_resolver.inference.provider import (
    ResolutionProvider,
    ResolutionRequest,
    ResolutionResult,
)

_DEFAULT_BASE_URL = "http://127.0.0.1:8080"
_DEFAULT_MODEL = "local"


class LocalLlamaCppProvider(ResolutionProvider):
    """
    Resolution provider backed by a local llama.cpp OpenAI-compatible server.

    The provider targets the ``/v1/chat/completions`` endpoint, which is
    available when the llama.cpp server is started in server mode
    (``llama-server`` or ``./server``).

    Parameters
    ----------
    base_url:
        Base URL of the llama.cpp server.  Defaults to the
        ``LLAMA_CPP_BASE_URL`` environment variable, falling back to
        ``http://127.0.0.1:8080``.
    model:
        Model name to include in API requests.  Defaults to the
        ``LLAMA_CPP_MODEL`` environment variable, falling back to
        ``"local"``.  The llama.cpp server typically ignores this field
        when only one model is loaded, but it is forwarded for
        completeness.
    api_key:
        Optional API key sent as a ``Bearer`` token in the ``Authorization``
        header.  Defaults to the ``LLAMA_CPP_API_KEY`` environment variable
        (or ``None``).
    timeout:
        HTTP request timeout in seconds.  Defaults to ``120``.
    """

    name: str = "llama_cpp"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = (
            base_url
            or os.environ.get("LLAMA_CPP_BASE_URL", _DEFAULT_BASE_URL)
        ).rstrip("/")
        self._model = (
            model
            or os.environ.get("LLAMA_CPP_MODEL", _DEFAULT_MODEL)
        )
        self._api_key = api_key or os.environ.get("LLAMA_CPP_API_KEY")
        self._timeout = timeout

    def resolve(self, request: ResolutionRequest) -> ResolutionResult:
        """
        Send *request* to the local llama.cpp server and return a structured
        :class:`~context_resolver.inference.provider.ResolutionResult`.

        Raises
        ------
        ValueError
            If *request.prompt* is ``None``.
        RuntimeError
            If the server returns a non-200 status code or non-JSON body.
        ConnectionError
            If the server is not reachable.
        """
        if request.prompt is None:
            raise ValueError(
                "LocalLlamaCppProvider requires a non-None prompt in the "
                "ResolutionRequest."
            )

        model = request.model or self._model

        # Build the chat-completions messages list.
        if isinstance(request.prompt, list):
            messages: list[dict[str, str]] = request.prompt
        else:
            messages = [{"role": "user", "content": request.prompt}]

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
        }

        # Request JSON output when a schema is provided.
        if request.output_schema is not None and self.supports_structured_output():
            payload["response_format"] = {"type": "json_object"}

        url = f"{self._base_url}/v1/chat/completions"
        raw_response = self._post(url, payload)

        choices = raw_response.get("choices") or []
        if not choices:
            raise RuntimeError(
                "LocalLlamaCppProvider: server returned no choices in response."
            )

        content: str = (choices[0].get("message") or {}).get("content") or "{}"
        reported_model: str = raw_response.get("model") or model

        try:
            data: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"LocalLlamaCppProvider: response was not valid JSON: {content!r}"
            ) from exc

        return ResolutionResult(
            data=data,
            model=reported_model,
            provider=self.name,
            raw=raw_response,
        )

    def supports_structured_output(self) -> bool:
        """
        Return ``True`` – the llama.cpp server supports JSON-mode output
        via ``response_format: {type: "json_object"}``.
        """
        return True

    def __repr__(self) -> str:
        return (
            f"LocalLlamaCppProvider("
            f"base_url={self._base_url!r}, model={self._model!r})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Perform an HTTP POST to *url* with a JSON *payload* and return the
        decoded JSON response.

        Uses only the standard-library :mod:`urllib` so that no extra
        dependencies are required.
        """
        body = json.dumps(payload).encode("utf-8")
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key

        req = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                raw_bytes: bytes = resp.read()
        except URLError as exc:
            raise ConnectionError(
                f"LocalLlamaCppProvider: could not reach server at {url!r}. "
                "Make sure the llama.cpp server is running. "
                f"Original error: {exc}"
            ) from exc

        try:
            return json.loads(raw_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"LocalLlamaCppProvider: server returned non-JSON body: "
                f"{raw_bytes[:200]!r}"
            ) from exc
