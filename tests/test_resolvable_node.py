"""
Tests for the generalized architecture rename:

* ``ResolvableNode`` (was ``PromptNode``)
* ``ResolvableNodeState`` (was ``PromptNodeState``)
* ``ResolutionProvider`` / ``ResolutionRequest`` / ``ResolutionResult``
  (were ``InferenceProvider`` / ``InferenceRequest`` / ``InferenceResponse``)
* ``ResolutionPass`` (was ``InferencePass``)

Backward-compatibility aliases are also checked.
"""

import pytest

from context_compiler.ast.nodes import ScalarNode, MappingNode, NodeState
from context_compiler.ast.paths import Path
from context_compiler.ast.prompt_node import (
    ResolvableNode,
    ResolvableNodeState,
    PromptNode,
    PromptNodeState,
)
from context_compiler.ast.schema import Schema, FieldSpec
from context_compiler.inference.provider import (
    ResolutionProvider,
    ResolutionRequest,
    ResolutionResult,
    InferenceProvider,
    InferenceRequest,
    InferenceResponse,
)
from context_compiler.inference.mock_provider import MockProvider
from context_compiler.compiler.passes import ResolutionPass, InferencePass
from context_compiler.compiler.compiler import Compiler
from context_compiler.templates.template import Template, TemplateRegistry


# ---------------------------------------------------------------------------
# Aliases are identical objects
# ---------------------------------------------------------------------------


class TestBackwardCompatAliases:
    def test_prompt_node_is_resolvable_node(self):
        assert PromptNode is ResolvableNode

    def test_prompt_node_state_is_resolvable_node_state(self):
        assert PromptNodeState is ResolvableNodeState

    def test_inference_provider_is_resolution_provider(self):
        assert InferenceProvider is ResolutionProvider

    def test_inference_request_is_resolution_request(self):
        assert InferenceRequest is ResolutionRequest

    def test_inference_response_is_resolution_result(self):
        assert InferenceResponse is ResolutionResult

    def test_inference_pass_is_resolution_pass(self):
        assert InferencePass is ResolutionPass


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

    def test_prompt_state_alias_matches_resolution_state(self, resolvable_node):
        """prompt_state is a backward-compat alias for resolution_state."""
        assert resolvable_node.prompt_state is resolvable_node.resolution_state

    def test_initial_node_state_is_underspecified(self, resolvable_node):
        assert resolvable_node.state is NodeState.UNDERSPECIFIED

    def test_result_is_none(self, resolvable_node):
        assert resolvable_node.result is None

    def test_error_is_none(self, resolvable_node):
        assert resolvable_node.error is None

    def test_repr_uses_new_class_name(self, resolvable_node):
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

    def test_legacy_prompt_state_key_is_accepted(self, resolvable_node):
        """Old serialised data using 'prompt_state' must still deserialise."""
        d = resolvable_node.to_dict()
        # Simulate legacy format
        d["type"] = "PromptNode"
        d["prompt_state"] = d.pop("resolution_state")
        restored = ResolvableNode.from_dict(d)
        assert restored.resolution_state is ResolvableNodeState.PENDING

    def test_legacy_prompt_node_type_is_accepted(self, resolvable_node):
        """'PromptNode' type key must deserialise to ResolvableNode."""
        from context_compiler.ast.nodes import _node_from_dict

        d = resolvable_node.to_dict()
        d["type"] = "PromptNode"
        d["prompt_state"] = d.pop("resolution_state")
        restored = _node_from_dict(d)
        assert isinstance(restored, ResolvableNode)


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

    def test_inference_request_alias_works(self):
        """InferenceRequest is an alias; existing call sites continue to work."""
        req = InferenceRequest(prompt="legacy")
        assert req.prompt == "legacy"


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

    def test_inference_response_alias_works(self):
        """InferenceResponse is an alias; existing call sites continue to work."""
        res = InferenceResponse(data={"a": 1}, model="gpt-4", provider="openai")
        assert res.model == "gpt-4"


# ---------------------------------------------------------------------------
# ResolutionProvider
# ---------------------------------------------------------------------------


class TestResolutionProvider:
    def test_resolve_raises_not_implemented(self):
        p = ResolutionProvider()
        with pytest.raises(NotImplementedError, match="resolve"):
            p.resolve(ResolutionRequest(prompt="x"))

    def test_infer_alias_delegates_to_resolve(self):
        """infer() is a backward-compat shim that calls resolve()."""

        class MyProvider(ResolutionProvider):
            name = "my"

            def resolve(self, request):
                return ResolutionResult(data={"ok": True}, provider="my")

        p = MyProvider()
        result = p.infer(ResolutionRequest(prompt="hi"))
        assert result.data == {"ok": True}

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
# ResolutionPass / InferencePass alias
# ---------------------------------------------------------------------------


class TestResolutionPass:
    def test_resolution_pass_stores_provider(self):
        provider = MockProvider()
        rp = ResolutionPass(provider)
        assert rp.provider is provider

    def test_inference_pass_alias(self):
        provider = MockProvider()
        ip = InferencePass(provider)
        assert isinstance(ip, ResolutionPass)
        assert ip.provider is provider


# ---------------------------------------------------------------------------
# End-to-end: compile with new API names
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def _build_compiler(self):
        registry = TemplateRegistry()
        registry.register(Template(
            name="tmpl",
            template_str="Resolve: {x}",
        ))
        provider = MockProvider(
            responses={"Resolve: hello": {"result": "world"}}
        )
        return (
            Compiler(template_registry=registry, passes=[ResolutionPass(provider)]),
            provider,
        )

    def test_compile_resolvable_node(self):
        compiler, provider = self._build_compiler()
        node = ResolvableNode(
            template_ref="tmpl",
            input_bindings={"x": Path("data", "x")},
        )
        root = MappingNode({
            "data": MappingNode({"x": ScalarNode("hello")}),
            "target": node,
        })
        compiled = compiler.compile_node(node, Path("target"), root)
        assert isinstance(compiled, ResolvableNode)
        assert compiled.resolution_state is ResolvableNodeState.RESOLVED
        assert provider.call_count == 1

    def test_resolution_request_includes_query_path(self):
        compiler, provider = self._build_compiler()
        node = ResolvableNode(
            template_ref="tmpl",
            input_bindings={"x": Path("data", "x")},
        )
        root = MappingNode({
            "data": MappingNode({"x": ScalarNode("hello")}),
            "target": node,
        })
        compiler.compile_node(node, Path("target"), root)
        assert provider.last_request is not None
        assert provider.last_request.query_path == Path("target")
