"""Tests for context_compiler.ast.prompt_node."""

import pytest

from context_compiler.ast.nodes import ScalarNode, MappingNode, NodeState
from context_compiler.ast.paths import Path
from context_compiler.ast.prompt_node import PromptNode, PromptNodeState
from context_compiler.ast.schema import Schema, FieldSpec


@pytest.fixture
def simple_schema():
    return Schema(
        name="TestSchema",
        fields=[FieldSpec(name="value", type="str", required=True)],
    )


@pytest.fixture
def prompt_node(simple_schema):
    return PromptNode(
        template_ref="test_template",
        input_bindings={"x": Path("data", "x")},
        output_schema=simple_schema,
        dependencies=[Path("data", "x")],
    )


class TestPromptNodeInitialState:
    def test_initial_state_is_pending(self, prompt_node):
        assert prompt_node.prompt_state is PromptNodeState.PENDING

    def test_initial_node_state_is_underspecified(self, prompt_node):
        assert prompt_node.state is NodeState.UNDERSPECIFIED
        assert not prompt_node.is_fully_specified

    def test_result_is_none_initially(self, prompt_node):
        assert prompt_node.result is None

    def test_error_is_none_initially(self, prompt_node):
        assert prompt_node.error is None


class TestPromptNodeTransitions:
    def test_mark_resolved(self, prompt_node):
        result = ScalarNode("output")
        prompt_node.mark_resolved(result, provider="mock", model="m1")
        assert prompt_node.prompt_state is PromptNodeState.RESOLVED
        assert prompt_node.state is NodeState.FULLY_SPECIFIED
        assert prompt_node.is_fully_specified
        assert prompt_node.result is result
        assert prompt_node.provider == "mock"
        assert prompt_node.model == "m1"
        assert prompt_node.resolved_at is not None

    def test_mark_stale(self, prompt_node):
        result = ScalarNode("output")
        prompt_node.mark_resolved(result)
        prompt_node.mark_stale()
        assert prompt_node.prompt_state is PromptNodeState.STALE
        assert prompt_node.result is None

    def test_mark_error(self, prompt_node):
        exc = ValueError("boom")
        prompt_node.mark_error(exc)
        assert prompt_node.prompt_state is PromptNodeState.ERROR
        assert prompt_node.error is exc

    def test_mark_error_state_is_underspecified(self, prompt_node):
        prompt_node.mark_error(ValueError("err"))
        assert prompt_node.state is NodeState.UNDERSPECIFIED


class TestEffectiveDependencies:
    def test_includes_explicit_dependencies(self, prompt_node):
        deps = prompt_node.effective_dependencies()
        assert Path("data", "x") in deps

    def test_no_duplicates(self):
        node = PromptNode(
            template_ref="t",
            input_bindings={"a": Path("data", "a")},
            dependencies=[Path("data", "a")],  # same as binding
        )
        deps = node.effective_dependencies()
        assert deps.count(Path("data", "a")) == 1

    def test_union_of_both(self):
        node = PromptNode(
            template_ref="t",
            input_bindings={"a": Path("x")},
            dependencies=[Path("y")],
        )
        deps = node.effective_dependencies()
        paths = set(deps)
        assert Path("x") in paths
        assert Path("y") in paths


class TestPromptNodeSerialization:
    def test_roundtrip_pending(self, prompt_node):
        data = prompt_node.to_dict()
        restored = PromptNode.from_dict(data)
        assert restored.template_ref == "test_template"
        assert restored.prompt_state is PromptNodeState.PENDING
        assert restored.result is None

    def test_roundtrip_resolved(self, prompt_node):
        prompt_node.mark_resolved(
            MappingNode({"value": ScalarNode("hello")}),
            provider="mock",
            model="m1",
        )
        data = prompt_node.to_dict()
        restored = PromptNode.from_dict(data)
        assert restored.prompt_state is PromptNodeState.RESOLVED
        assert restored.provider == "mock"
        assert isinstance(restored.result, MappingNode)

    def test_roundtrip_error(self, prompt_node):
        prompt_node.mark_error(RuntimeError("oops"))
        data = prompt_node.to_dict()
        restored = PromptNode.from_dict(data)
        assert restored.prompt_state is PromptNodeState.ERROR
        assert restored.error is not None
