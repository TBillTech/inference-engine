"""Tests for context_compiler.inference.mock_provider."""

import pytest

from context_compiler.inference.mock_provider import MockProvider
from context_compiler.inference.provider import ResolutionRequest, ResolutionResult


@pytest.fixture
def provider():
    return MockProvider(
        responses={"hello": {"reply": "world"}},
        default_response={"reply": "default"},
    )


class TestMockProvider:
    def test_returns_matched_response(self, provider):
        req = ResolutionRequest(prompt="hello")
        resp = provider.resolve(req)
        assert resp.data == {"reply": "world"}
        assert resp.provider == "mock"

    def test_returns_default_for_unmatched_prompt(self, provider):
        req = ResolutionRequest(prompt="unknown")
        resp = provider.resolve(req)
        assert resp.data == {"reply": "default"}

    def test_call_count_increments(self, provider):
        provider.resolve(ResolutionRequest(prompt="hello"))
        provider.resolve(ResolutionRequest(prompt="hello"))
        assert provider.call_count == 2

    def test_last_request_tracked(self, provider):
        req = ResolutionRequest(prompt="hello")
        provider.resolve(req)
        assert provider.last_request is req

    def test_reset_clears_counters(self, provider):
        provider.resolve(ResolutionRequest(prompt="hello"))
        provider.reset()
        assert provider.call_count == 0
        assert provider.last_request is None

    def test_raise_on_call(self):
        exc = RuntimeError("provider error")
        p = MockProvider(raise_on_call=exc)
        with pytest.raises(RuntimeError, match="provider error"):
            p.resolve(ResolutionRequest(prompt="anything"))

    def test_supports_structured_output(self, provider):
        assert provider.supports_structured_output() is True

    def test_response_data_is_copy(self, provider):
        """Mutating the response dict should not affect the provider's registry."""
        req = ResolutionRequest(prompt="hello")
        resp = provider.resolve(req)
        resp.data["extra"] = "injected"
        resp2 = provider.resolve(req)
        assert "extra" not in resp2.data
