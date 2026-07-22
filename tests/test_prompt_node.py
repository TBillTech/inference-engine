"""Tests for context_resolver.ast.resolvable_node."""

import pytest

from context_resolver.ast.nodes import ScalarNode, MappingNode, NodeState
from context_resolver.ast.paths import Path
from context_resolver.ast.resolvable_node import ResolvableNode, ResolvableNodeState
from context_resolver.ast.schema import Schema, FieldSpec


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
    def test_initial_state_is_pending(self, resolvable_node):
        assert resolvable_node.resolution_state is ResolvableNodeState.PENDING

    def test_initial_node_state_is_underspecified(self, resolvable_node):
        assert resolvable_node.state is NodeState.UNDERSPECIFIED
        assert not resolvable_node.is_fully_specified

    def test_result_is_none_initially(self, resolvable_node):
        assert resolvable_node.result is None

    def test_error_is_none_initially(self, resolvable_node):
        assert resolvable_node.error is None


class TestResolvableNodeTransitions:
    def test_mark_resolved(self, resolvable_node):
        result = ScalarNode("output")
        resolvable_node.mark_resolved(result, provider="mock", model="m1")
        assert resolvable_node.resolution_state is ResolvableNodeState.RESOLVED
        assert resolvable_node.state is NodeState.FULLY_SPECIFIED
        assert resolvable_node.is_fully_specified
        assert resolvable_node.result is result
        assert resolvable_node.provider == "mock"
        assert resolvable_node.model == "m1"
        assert resolvable_node.resolved_at is not None

    def test_mark_stale(self, resolvable_node):
        result = ScalarNode("output")
        resolvable_node.mark_resolved(result)
        resolvable_node.mark_stale()
        assert resolvable_node.resolution_state is ResolvableNodeState.STALE
        assert resolvable_node.result is None

    def test_mark_error(self, resolvable_node):
        exc = ValueError("boom")
        resolvable_node.mark_error(exc)
        assert resolvable_node.resolution_state is ResolvableNodeState.ERROR
        assert resolvable_node.error is exc

    def test_mark_error_state_is_underspecified(self, resolvable_node):
        resolvable_node.mark_error(ValueError("err"))
        assert resolvable_node.state is NodeState.UNDERSPECIFIED


class TestEffectiveDependencies:
    def test_includes_explicit_dependencies(self, resolvable_node):
        deps = resolvable_node.effective_dependencies()
        assert Path("data", "x") in deps

    def test_no_duplicates(self):
        node = ResolvableNode(
            template_ref="t",
            input_bindings={"a": Path("data", "a")},
            dependencies=[Path("data", "a")],  # same as binding
        )
        deps = node.effective_dependencies()
        assert deps.count(Path("data", "a")) == 1

    def test_union_of_both(self):
        node = ResolvableNode(
            template_ref="t",
            input_bindings={"a": Path("x")},
            dependencies=[Path("y")],
        )
        deps = node.effective_dependencies()
        paths = set(deps)
        assert Path("x") in paths
        assert Path("y") in paths


class TestResolvableNodeSerialization:
    def test_roundtrip_pending(self, resolvable_node):
        data = resolvable_node.to_dict()
        restored = ResolvableNode.from_dict(data)
        assert restored.template_ref == "test_template"
        assert restored.resolution_state is ResolvableNodeState.PENDING
        assert restored.result is None

    def test_roundtrip_resolved(self, resolvable_node):
        resolvable_node.mark_resolved(
            MappingNode({"value": ScalarNode("hello")}),
            provider="mock",
            model="m1",
        )
        data = resolvable_node.to_dict()
        restored = ResolvableNode.from_dict(data)
        assert restored.resolution_state is ResolvableNodeState.RESOLVED
        assert restored.provider == "mock"
        assert isinstance(restored.result, MappingNode)

    def test_roundtrip_error(self, resolvable_node):
        resolvable_node.mark_error(RuntimeError("oops"))
        data = resolvable_node.to_dict()
        restored = ResolvableNode.from_dict(data)
        assert restored.resolution_state is ResolvableNodeState.ERROR
        assert restored.error is not None
