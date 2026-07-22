"""
Tests for the query-resolution architecture:

* ``ResolvableNode`` / ``ResolvableNodeState``
* ``ResolutionProvider`` / ``ResolutionRequest`` / ``ResolutionResult``
* ``ResolutionPass``
* ``ResolutionStrategy`` / ``PromptStrategy``
* ``Resolver`` (query engine)
"""

import pytest

from context_resolver.ast.nodes import ScalarNode, MappingNode, NodeState
from context_resolver.ast.paths import Path
from context_resolver.ast.resolvable_node import (
    ResolvableNode,
    ResolvableNodeState,
)
from context_resolver.ast.schema import Schema, FieldSpec
from context_resolver.inference.provider import (
    ResolutionProvider,
    ResolutionRequest,
    ResolutionResult,
)
from context_resolver.inference.strategy import ResolutionStrategy, PromptStrategy
from context_resolver.inference.mock_provider import MockProvider
from context_resolver.query.passes import ResolutionPass
from context_resolver.query.resolver import Resolver
from context_resolver.templates.template import Template, TemplateRegistry


# ---------------------------------------------------------------------------
# ResolvableNode basics
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_schema():
    return Schema(
        name="TestSchema",
        fields=[FieldSpec(name="value", type="str", required=True)],
    )


@pytest.fixture
def resolvable_node(simple_schema):
    return ResolvableNode(
        template_ref="test_template",
        input_bindings={"x": Path("data", "x")},
        output_schema=simple_schema,
        dependencies=[Path("data", "x")],
    )


class TestResolvableNodeInitialState:
    def test_initial_resolution_state_is_pending(self, resolvable_node):
        assert resolvable_node.resolution_state is ResolvableNodeState.PENDING

    def test_initial_node_state_is_underspecified(self, resolvable_node):
        assert resolvable_node.state is NodeState.UNDERSPECIFIED

    def test_result_is_none(self, resolvable_node):
        assert resolvable_node.result is None

    def test_error_is_none(self, resolvable_node):
        assert resolvable_node.error is None

    def test_repr_uses_class_name(self, resolvable_node):
        assert "ResolvableNode" in repr(resolvable_node)
        assert "resolution_state" in repr(resolvable_node)


class TestResolvableNodeTransitions:
    def test_mark_resolved(self, resolvable_node):
        result = ScalarNode("output")
        resolvable_node.mark_resolved(result, provider="mock", model="m1")
        assert resolvable_node.resolution_state is ResolvableNodeState.RESOLVED
        assert resolvable_node.state is NodeState.FULLY_SPECIFIED
        assert resolvable_node.result is result
        assert resolvable_node.provider == "mock"
        assert resolvable_node.model == "m1"
        assert resolvable_node.resolved_at is not None

    def test_mark_resolved_without_model(self, resolvable_node):
        """Non-LLM providers may not provide a model identifier."""
        resolvable_node.mark_resolved(ScalarNode("x"), provider="database")
        assert resolvable_node.provider == "database"
        assert resolvable_node.model is None

    def test_mark_stale(self, resolvable_node):
        resolvable_node.mark_resolved(ScalarNode("v"))
        resolvable_node.mark_stale()
        assert resolvable_node.resolution_state is ResolvableNodeState.STALE
        assert resolvable_node.result is None

    def test_mark_error(self, resolvable_node):
        exc = ValueError("boom")
        resolvable_node.mark_error(exc)
        assert resolvable_node.resolution_state is ResolvableNodeState.ERROR
        assert resolvable_node.error is exc
        assert resolvable_node.state is NodeState.UNDERSPECIFIED

    def test_node_not_discarded_after_resolution(self, resolvable_node):
        """The node must not be replaced – it must retain provenance."""
        original_id = id(resolvable_node)
        resolvable_node.mark_resolved(ScalarNode("v"))
        assert id(resolvable_node) == original_id
        assert resolvable_node.resolved_at is not None


class TestResolvableNodeSerialization:
    def test_to_dict_uses_resolvable_node_type(self, resolvable_node):
        d = resolvable_node.to_dict()
        assert d["type"] == "ResolvableNode"

    def test_to_dict_uses_resolution_state_key(self, resolvable_node):
        d = resolvable_node.to_dict()
        assert "resolution_state" in d
        assert d["resolution_state"] == "PENDING"

    def test_roundtrip_pending(self, resolvable_node):
        d = resolvable_node.to_dict()
        restored = ResolvableNode.from_dict(d)
        assert restored.template_ref == "test_template"
        assert restored.resolution_state is ResolvableNodeState.PENDING

    def test_roundtrip_resolved(self, resolvable_node):
        resolvable_node.mark_resolved(
            MappingNode({"value": ScalarNode("hi")}),
            provider="mock",
            model="m1",
        )
        d = resolvable_node.to_dict()
        restored = ResolvableNode.from_dict(d)
        assert restored.resolution_state is ResolvableNodeState.RESOLVED
        assert restored.provider == "mock"


# ---------------------------------------------------------------------------
# ResolutionRequest / ResolutionResult
# ---------------------------------------------------------------------------


class TestResolutionRequest:
    def test_basic_construction(self):
        req = ResolutionRequest(prompt="hello")
        assert req.prompt == "hello"
        assert req.output_schema is None
        assert req.query_path is None
        assert req.dependencies == []
        assert req.metadata == {}

    def test_with_generic_fields(self):
        path = Path("foo", "bar")
        req = ResolutionRequest(
            prompt=None,
            query_path=path,
            dependencies=[path],
            metadata={"hint": "database"},
        )
        assert req.query_path is path
        assert len(req.dependencies) == 1
        assert req.metadata["hint"] == "database"


class TestResolutionResult:
    def test_basic_construction(self):
        res = ResolutionResult(data={"key": "value"})
        assert res.data == {"key": "value"}
        assert res.success is True
        assert res.diagnostics == []
        assert res.provenance == {}
        assert res.confidence is None

    def test_failure_result(self):
        res = ResolutionResult(
            data={},
            success=False,
            diagnostics=["timeout"],
            provider="database",
        )
        assert res.success is False
        assert "timeout" in res.diagnostics

    def test_provenance_and_confidence(self):
        res = ResolutionResult(
            data={"x": 1},
            provenance={"model": "gpt-4o", "prompt_hash": "abc"},
            confidence=0.95,
        )
        assert res.provenance["model"] == "gpt-4o"
        assert res.confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# ResolutionProvider
# ---------------------------------------------------------------------------


class TestResolutionProvider:
    def test_resolve_raises_not_implemented(self):
        p = ResolutionProvider()
        with pytest.raises(NotImplementedError, match="resolve"):
            p.resolve(ResolutionRequest(prompt="x"))

    def test_can_resolve_defaults_to_true(self):
        p = ResolutionProvider()
        assert p.can_resolve(ResolutionRequest(prompt="anything")) is True

    def test_can_resolve_can_be_overridden(self):
        class SpecializedProvider(ResolutionProvider):
            name = "specialized"

            def resolve(self, request):
                return ResolutionResult(data={})

            def can_resolve(self, request):
                return request.metadata.get("domain") == "math"

        p = SpecializedProvider()
        assert p.can_resolve(ResolutionRequest(metadata={"domain": "math"})) is True
        assert p.can_resolve(ResolutionRequest(metadata={"domain": "text"})) is False


# ---------------------------------------------------------------------------
# ResolutionStrategy and PromptStrategy
# ---------------------------------------------------------------------------


class TestResolutionStrategy:
    def test_resolve_raises_not_implemented(self):
        s = ResolutionStrategy()
        with pytest.raises(NotImplementedError, match="resolve"):
            s.resolve(ResolutionRequest(prompt="x"))

    def test_can_apply_defaults_to_true(self):
        s = ResolutionStrategy()
        assert s.can_apply(ResolutionRequest(prompt="anything")) is True

    def test_can_apply_can_be_overridden(self):
        class MathStrategy(ResolutionStrategy):
            name = "math"

            def resolve(self, request):
                return ResolutionResult(data={})

            def can_apply(self, request):
                return request.metadata.get("domain") == "math"

        s = MathStrategy()
        assert s.can_apply(ResolutionRequest(metadata={"domain": "math"})) is True
        assert s.can_apply(ResolutionRequest(metadata={"domain": "text"})) is False

    def test_provider_defaults_to_none(self):
        s = ResolutionStrategy()
        assert s.provider is None

    def test_repr_includes_name(self):
        s = ResolutionStrategy()
        assert "abstract" in repr(s)


class TestPromptStrategy:
    def test_delegates_to_provider(self):
        provider = MockProvider(responses={"hello": {"result": "world"}})
        strategy = PromptStrategy(provider)
        request = ResolutionRequest(prompt="hello")
        result = strategy.resolve(request)
        assert result.data == {"result": "world"}
        assert provider.call_count == 1

    def test_provider_property_returns_wrapped_provider(self):
        provider = MockProvider()
        strategy = PromptStrategy(provider)
        assert strategy.provider is provider

    def test_can_apply_defaults_to_true(self):
        provider = MockProvider()
        strategy = PromptStrategy(provider)
        assert strategy.can_apply(ResolutionRequest(prompt="x")) is True

    def test_repr_includes_provider(self):
        provider = MockProvider()
        strategy = PromptStrategy(provider)
        assert "PromptStrategy" in repr(strategy)
        assert "MockProvider" in repr(strategy)

    def test_name_is_prompt(self):
        provider = MockProvider()
        strategy = PromptStrategy(provider)
        assert strategy.name == "prompt"

    def test_multiple_providers_same_strategy(self):
        """The same PromptStrategy interface works with different providers."""

        class RecordingProvider(ResolutionProvider):
            name = "recording"

            def __init__(self, tag: str) -> None:
                self.tag = tag
                self.called = False

            def resolve(self, request):
                self.called = True
                return ResolutionResult(data={"tag": self.tag}, provider=self.name)

        p1 = RecordingProvider("alpha")
        p2 = RecordingProvider("beta")

        s1 = PromptStrategy(p1)
        s2 = PromptStrategy(p2)

        r1 = s1.resolve(ResolutionRequest(prompt="x"))
        r2 = s2.resolve(ResolutionRequest(prompt="x"))

        assert p1.called and p2.called
        assert r1.data["tag"] == "alpha"
        assert r2.data["tag"] == "beta"


# ---------------------------------------------------------------------------
# ResolutionPass
# ---------------------------------------------------------------------------


class TestResolutionPass:
    def test_resolution_pass_stores_provider(self):
        provider = MockProvider()
        rp = ResolutionPass(provider)
        assert rp.provider is provider

    def test_resolution_pass_wraps_provider_in_prompt_strategy(self):
        """A bare provider is automatically wrapped in a PromptStrategy."""
        provider = MockProvider()
        rp = ResolutionPass(provider)
        assert isinstance(rp.strategy, PromptStrategy)
        assert rp.strategy.provider is provider

    def test_resolution_pass_accepts_strategy_directly(self):
        provider = MockProvider()
        strategy = PromptStrategy(provider)
        rp = ResolutionPass(strategy)
        assert rp.strategy is strategy

    def test_resolution_pass_strategy_property(self):
        provider = MockProvider()
        rp = ResolutionPass(provider)
        assert rp.strategy is not None

    def test_resolution_pass_provider_raises_for_no_provider_strategy(self):
        """Accessing .provider on a strategy without one raises AttributeError."""

        class NoProviderStrategy(ResolutionStrategy):
            name = "no-provider"

            def resolve(self, request):
                return ResolutionResult(data={})

        rp = ResolutionPass(NoProviderStrategy())
        with pytest.raises(AttributeError, match="provider"):
            _ = rp.provider

    def test_resolution_pass_rejects_invalid_type(self):
        with pytest.raises(TypeError):
            ResolutionPass("not-a-strategy-or-provider")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# End-to-end: resolve with new API names
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def _build_resolver(self):
        registry = TemplateRegistry()
        registry.register(Template(
            name="tmpl",
            template_str="Resolve: {x}",
        ))
        provider = MockProvider(
            responses={"Resolve: hello": {"result": "world"}}
        )
        return (
            Resolver(template_registry=registry, passes=[ResolutionPass(provider)]),
            provider,
        )

    def test_resolve_resolvable_node(self):
        resolver, provider = self._build_resolver()
        node = ResolvableNode(
            template_ref="tmpl",
            input_bindings={"x": Path("data", "x")},
        )
        root = MappingNode({
            "data": MappingNode({"x": ScalarNode("hello")}),
            "target": node,
        })
        resolved = resolver.resolve_node(node, Path("target"), root)
        assert isinstance(resolved, ResolvableNode)
        assert resolved.resolution_state is ResolvableNodeState.RESOLVED
        assert provider.call_count == 1

    def test_resolution_request_includes_query_path(self):
        resolver, provider = self._build_resolver()
        node = ResolvableNode(
            template_ref="tmpl",
            input_bindings={"x": Path("data", "x")},
        )
        root = MappingNode({
            "data": MappingNode({"x": ScalarNode("hello")}),
            "target": node,
        })
        resolver.resolve_node(node, Path("target"), root)
        assert provider.last_request is not None
        assert provider.last_request.query_path == Path("target")

    def test_resolve_with_explicit_prompt_strategy(self):
        """Passing a PromptStrategy directly to ResolutionPass works end-to-end."""
        registry = TemplateRegistry()
        registry.register(Template(name="tmpl", template_str="Resolve: {x}"))
        provider = MockProvider(responses={"Resolve: hello": {"result": "world"}})
        strategy = PromptStrategy(provider)
        resolver = Resolver(
            template_registry=registry,
            passes=[ResolutionPass(strategy)],
        )
        node = ResolvableNode(
            template_ref="tmpl",
            input_bindings={"x": Path("data", "x")},
        )
        root = MappingNode({
            "data": MappingNode({"x": ScalarNode("hello")}),
            "target": node,
        })
        resolved = resolver.resolve_node(node, Path("target"), root)
        assert resolved.resolution_state is ResolvableNodeState.RESOLVED
        assert provider.call_count == 1

    def test_custom_strategy_is_called_by_resolver(self):
        """A custom ResolutionStrategy is invoked by the resolver."""
        registry = TemplateRegistry()
        registry.register(Template(name="tmpl", template_str="Resolve: {x}"))

        class UpperCaseStrategy(ResolutionStrategy):
            """Transforms the prompt to upper-case and returns it as data."""

            name = "uppercase"

            def __init__(self) -> None:
                self.calls: list = []

            def resolve(self, request):
                self.calls.append(request)
                return ResolutionResult(
                    data={"result": (request.prompt or "").upper()},
                    provider="uppercase",
                )

        custom_strategy = UpperCaseStrategy()
        resolver = Resolver(
            template_registry=registry,
            passes=[ResolutionPass(custom_strategy)],
        )
        node = ResolvableNode(
            template_ref="tmpl",
            input_bindings={"x": Path("data", "x")},
        )
        root = MappingNode({
            "data": MappingNode({"x": ScalarNode("hello")}),
            "target": node,
        })
        resolved = resolver.resolve_node(node, Path("target"), root)
        assert resolved.resolution_state is ResolvableNodeState.RESOLVED
        assert len(custom_strategy.calls) == 1
        assert custom_strategy.calls[0].prompt == "Resolve: hello"
