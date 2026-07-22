"""Tests for context_resolver.inference.llama_cpp_provider."""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from context_resolver.inference.llama_cpp_provider import LocalLlamaCppProvider
from context_resolver.inference.provider import ResolutionRequest, ResolutionResult
from context_resolver.inference.strategy import PromptStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(data: dict, model: str = "llama-local") -> bytes:
    """Encode a minimal chat-completions response payload."""
    payload = {
        "model": model,
        "choices": [
            {"message": {"role": "assistant", "content": json.dumps(data)}}
        ],
    }
    return json.dumps(payload).encode("utf-8")


def _urlopen_stub(response_bytes: bytes):
    """Return a context-manager mock that yields a readable HTTP response."""
    resp = MagicMock()
    resp.read.return_value = response_bytes
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# Unit tests – LocalLlamaCppProvider
# ---------------------------------------------------------------------------

class TestLocalLlamaCppProvider:

    def test_default_name(self):
        provider = LocalLlamaCppProvider()
        assert provider.name == "llama_cpp"

    def test_supports_structured_output(self):
        provider = LocalLlamaCppProvider()
        assert provider.supports_structured_output() is True

    def test_repr(self):
        provider = LocalLlamaCppProvider(
            base_url="http://127.0.0.1:8080", model="test-model"
        )
        assert "LocalLlamaCppProvider" in repr(provider)
        assert "127.0.0.1:8080" in repr(provider)
        assert "test-model" in repr(provider)

    def test_resolve_returns_result(self):
        provider = LocalLlamaCppProvider(base_url="http://127.0.0.1:8080")
        response_data = {"greeting": "Hello!", "opening": "A brave world."}
        stub = _urlopen_stub(_make_response(response_data, model="llama-3"))

        with patch(
            "context_resolver.inference.llama_cpp_provider.urlopen",
            return_value=stub,
        ):
            result = provider.resolve(ResolutionRequest(prompt="Greet the hero."))

        assert isinstance(result, ResolutionResult)
        assert result.data == response_data
        assert result.provider == "llama_cpp"
        assert result.model == "llama-3"
        assert result.success is True

    def test_resolve_passes_temperature(self):
        provider = LocalLlamaCppProvider(base_url="http://127.0.0.1:8080")
        captured: list[dict] = []

        def fake_urlopen(req, timeout):
            body = json.loads(req.data.decode("utf-8"))
            captured.append(body)
            stub = _urlopen_stub(_make_response({"key": "value"}))
            return stub

        with patch(
            "context_resolver.inference.llama_cpp_provider.urlopen",
            side_effect=fake_urlopen,
        ):
            provider.resolve(
                ResolutionRequest(prompt="test", temperature=0.7)
            )

        assert captured[0]["temperature"] == pytest.approx(0.7)

    def test_resolve_passes_model_override(self):
        provider = LocalLlamaCppProvider(
            base_url="http://127.0.0.1:8080", model="default-model"
        )
        captured: list[dict] = []

        def fake_urlopen(req, timeout):
            body = json.loads(req.data.decode("utf-8"))
            captured.append(body)
            return _urlopen_stub(_make_response({}))

        with patch(
            "context_resolver.inference.llama_cpp_provider.urlopen",
            side_effect=fake_urlopen,
        ):
            provider.resolve(
                ResolutionRequest(prompt="test", model="override-model")
            )

        assert captured[0]["model"] == "override-model"

    def test_resolve_raises_value_error_on_none_prompt(self):
        provider = LocalLlamaCppProvider()
        with pytest.raises(ValueError, match="non-None prompt"):
            provider.resolve(ResolutionRequest(prompt=None))

    def test_resolve_raises_connection_error_on_url_error(self):
        from urllib.error import URLError

        provider = LocalLlamaCppProvider(base_url="http://127.0.0.1:8080")
        with patch(
            "context_resolver.inference.llama_cpp_provider.urlopen",
            side_effect=URLError("connection refused"),
        ):
            with pytest.raises(ConnectionError, match="could not reach server"):
                provider.resolve(ResolutionRequest(prompt="hello"))

    def test_resolve_raises_runtime_error_on_invalid_json_response(self):
        provider = LocalLlamaCppProvider(base_url="http://127.0.0.1:8080")
        # Server returns a valid HTTP response but the 'content' field is not JSON.
        bad_payload = {
            "model": "llama",
            "choices": [
                {"message": {"role": "assistant", "content": "not-json!!!"}}
            ],
        }
        stub = _urlopen_stub(json.dumps(bad_payload).encode())

        with patch(
            "context_resolver.inference.llama_cpp_provider.urlopen",
            return_value=stub,
        ):
            with pytest.raises(RuntimeError, match="not valid JSON"):
                provider.resolve(ResolutionRequest(prompt="hello"))

    def test_resolve_raises_runtime_error_on_empty_choices(self):
        provider = LocalLlamaCppProvider(base_url="http://127.0.0.1:8080")
        empty_choices_payload = {"model": "llama", "choices": []}
        stub = _urlopen_stub(json.dumps(empty_choices_payload).encode())

        with patch(
            "context_resolver.inference.llama_cpp_provider.urlopen",
            return_value=stub,
        ):
            with pytest.raises(RuntimeError, match="no choices"):
                provider.resolve(ResolutionRequest(prompt="hello"))

    def test_json_mode_added_when_schema_present(self):
        provider = LocalLlamaCppProvider(base_url="http://127.0.0.1:8080")
        captured: list[dict] = []

        def fake_urlopen(req, timeout):
            body = json.loads(req.data.decode("utf-8"))
            captured.append(body)
            return _urlopen_stub(_make_response({"key": "val"}))

        with patch(
            "context_resolver.inference.llama_cpp_provider.urlopen",
            side_effect=fake_urlopen,
        ):
            provider.resolve(
                ResolutionRequest(
                    prompt="test",
                    output_schema={"type": "object"},
                )
            )

        assert captured[0].get("response_format") == {"type": "json_object"}

    def test_json_mode_absent_when_no_schema(self):
        provider = LocalLlamaCppProvider(base_url="http://127.0.0.1:8080")
        captured: list[dict] = []

        def fake_urlopen(req, timeout):
            body = json.loads(req.data.decode("utf-8"))
            captured.append(body)
            return _urlopen_stub(_make_response({}))

        with patch(
            "context_resolver.inference.llama_cpp_provider.urlopen",
            side_effect=fake_urlopen,
        ):
            provider.resolve(ResolutionRequest(prompt="test"))

        assert "response_format" not in captured[0]

    def test_list_prompt_forwarded_as_messages(self):
        provider = LocalLlamaCppProvider(base_url="http://127.0.0.1:8080")
        captured: list[dict] = []
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        def fake_urlopen(req, timeout):
            body = json.loads(req.data.decode("utf-8"))
            captured.append(body)
            return _urlopen_stub(_make_response({}))

        with patch(
            "context_resolver.inference.llama_cpp_provider.urlopen",
            side_effect=fake_urlopen,
        ):
            provider.resolve(ResolutionRequest(prompt=messages))

        assert captured[0]["messages"] == messages

    def test_env_var_base_url(self, monkeypatch):
        monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://192.168.1.10:9090")
        provider = LocalLlamaCppProvider()
        assert "192.168.1.10:9090" in provider._base_url

    def test_env_var_model(self, monkeypatch):
        monkeypatch.setenv("LLAMA_CPP_MODEL", "my-custom-model")
        provider = LocalLlamaCppProvider()
        assert provider._model == "my-custom-model"

    def test_env_var_api_key(self, monkeypatch):
        monkeypatch.setenv("LLAMA_CPP_API_KEY", "test-key-123")
        provider = LocalLlamaCppProvider()
        assert provider._api_key == "test-key-123"

    def test_api_key_sent_in_header(self, monkeypatch):
        monkeypatch.setenv("LLAMA_CPP_API_KEY", "secret-key")
        provider = LocalLlamaCppProvider(base_url="http://127.0.0.1:8080")
        captured_headers: list[dict] = []

        def fake_urlopen(req, timeout):
            captured_headers.append(dict(req.headers))
            return _urlopen_stub(_make_response({}))

        with patch(
            "context_resolver.inference.llama_cpp_provider.urlopen",
            side_effect=fake_urlopen,
        ):
            provider.resolve(ResolutionRequest(prompt="test"))

        auth = captured_headers[0].get("Authorization") or captured_headers[0].get("authorization")
        assert auth is not None
        assert "secret-key" in auth


# ---------------------------------------------------------------------------
# Wiring tests – PromptStrategy integration
# ---------------------------------------------------------------------------

class TestPromptStrategyWiring:

    def test_prompt_strategy_wraps_provider(self):
        provider = LocalLlamaCppProvider()
        strategy = PromptStrategy(provider)
        assert strategy.provider is provider

    def test_prompt_strategy_delegates_resolve(self):
        provider = LocalLlamaCppProvider(base_url="http://127.0.0.1:8080")
        strategy = PromptStrategy(provider)

        response_data = {"answer": "42"}
        stub = _urlopen_stub(_make_response(response_data))

        with patch(
            "context_resolver.inference.llama_cpp_provider.urlopen",
            return_value=stub,
        ):
            result = strategy.resolve(ResolutionRequest(prompt="What is the answer?"))

        assert result.data == response_data
        assert result.provider == "llama_cpp"

    def test_resolution_pass_accepts_provider(self):
        """ResolutionPass should auto-wrap a bare LocalLlamaCppProvider."""
        from context_resolver.query.passes import ResolutionPass
        from context_resolver.inference.strategy import PromptStrategy

        provider = LocalLlamaCppProvider()
        rp = ResolutionPass(provider)
        assert isinstance(rp.strategy, PromptStrategy)
        assert rp.strategy.provider is provider

    def test_resolution_pass_accepts_strategy(self):
        """ResolutionPass should accept a pre-wrapped PromptStrategy."""
        from context_resolver.query.passes import ResolutionPass

        provider = LocalLlamaCppProvider()
        strategy = PromptStrategy(provider)
        rp = ResolutionPass(strategy)
        assert rp.strategy is strategy
