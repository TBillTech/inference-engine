"""Tests for context_compiler.context.context.Context."""

import pytest

from context_compiler.ast.nodes import MappingNode, ScalarNode, SequenceNode
from context_compiler.ast.paths import Path
from context_compiler.ast.prompt_node import PromptNode, PromptNodeState
from context_compiler.ast.schema import Schema, FieldSpec
from context_compiler.compiler.compiler import Compiler
from context_compiler.compiler.passes import InferencePass
from context_compiler.context.context import Context, NodeNotFoundError
from context_compiler.inference.mock_provider import MockProvider
from context_compiler.templates.template import Template, TemplateRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_context():
    """A context with a flat mapping of scalar nodes."""
    root = MappingNode({
        "name": ScalarNode("Alice"),
        "age": ScalarNode(30),
    })
    return Context(root)


@pytest.fixture
def nested_context():
    """A context with a nested mapping."""
    root = MappingNode({
        "player": MappingNode({
            "name": ScalarNode("Bob"),
            "score": ScalarNode(100),
        }),
        "items": SequenceNode([ScalarNode("sword"), ScalarNode("shield")]),
    })
    return Context(root)


@pytest.fixture
def prompt_context():
    """A context with a PromptNode wired up to a mock provider."""
    schema = Schema(
        name="GreetSchema",
        fields=[FieldSpec("greeting", type="str", required=True)],
    )
    prompt = PromptNode(
        template_ref="greet",
        input_bindings={"name": Path("name")},
        output_schema=schema,
    )
    root = MappingNode({
        "name": ScalarNode("Carol"),
        "greeting": prompt,
    })

    registry = TemplateRegistry()
    registry.register(Template("greet", "Say hello to {name}."))

    mock = MockProvider(
        responses={"Say hello to Carol.": {"greeting": "Hello, Carol!"}},
    )
    compiler = Compiler(template_registry=registry, passes=[InferencePass(mock)])
    return Context(root=root, compiler=compiler)


# ---------------------------------------------------------------------------
# Basic query tests
# ---------------------------------------------------------------------------


class TestContextQueryScalar:
    def test_query_existing_scalar(self, simple_context):
        node = simple_context.query(Path("name"))
        assert isinstance(node, ScalarNode)
        assert node.value == "Alice"

    def test_query_missing_path_raises(self, simple_context):
        with pytest.raises(NodeNotFoundError):
            simple_context.query(Path("nonexistent"))

    def test_query_nested_path(self, nested_context):
        node = nested_context.query(Path("player", "name"))
        assert node.value == "Bob"

    def test_query_sequence_index(self, nested_context):
        node = nested_context.query(Path("items", 0))
        assert node.value == "sword"

    def test_query_caches_result(self, simple_context):
        n1 = simple_context.query(Path("name"))
        n2 = simple_context.query(Path("name"))
        assert n1 is n2


# ---------------------------------------------------------------------------
# Mutation and cache invalidation
# ---------------------------------------------------------------------------


class TestContextSet:
    def test_set_updates_value(self, simple_context):
        simple_context.set(Path("name"), ScalarNode("Dave"))
        node = simple_context.query(Path("name"))
        assert node.value == "Dave"

    def test_set_invalidates_cache(self, simple_context):
        old = simple_context.query(Path("name"))
        simple_context.set(Path("name"), ScalarNode("Eve"))
        new = simple_context.query(Path("name"))
        assert new.value == "Eve"
        assert old is not new

    def test_set_missing_parent_raises(self, simple_context):
        with pytest.raises(NodeNotFoundError):
            simple_context.set(Path("ghost", "child"), ScalarNode("x"))


# ---------------------------------------------------------------------------
# PromptNode compilation
# ---------------------------------------------------------------------------


class TestContextPromptNode:
    def test_query_triggers_inference(self, prompt_context):
        node = prompt_context.query(Path("greeting"))
        assert isinstance(node, PromptNode)
        assert node.prompt_state is PromptNodeState.RESOLVED
        result = node.result
        assert result is not None

    def test_resolved_greeting_value(self, prompt_context):
        node = prompt_context.query(Path("greeting"))
        assert isinstance(node, PromptNode)
        greeting = node.result.get("greeting").value  # type: ignore[union-attr]
        assert greeting == "Hello, Carol!"

    def test_provider_called_once(self, prompt_context):
        prompt_context.query(Path("greeting"))
        # Get the mock provider via the compiler pass.
        inference_pass = prompt_context.compiler._passes[0]
        assert inference_pass.provider.call_count == 1

    def test_second_query_served_from_cache(self, prompt_context):
        n1 = prompt_context.query(Path("greeting"))
        n2 = prompt_context.query(Path("greeting"))
        assert n1 is n2
        # Provider should still have been called only once.
        inference_pass = prompt_context.compiler._passes[0]
        assert inference_pass.provider.call_count == 1

    def test_mutation_marks_prompt_stale(self, prompt_context):
        prompt_context.query(Path("greeting"))
        # Mutate the dependency.
        prompt_context.set(Path("name"), ScalarNode("Frank"))
        # Inspect the node in the tree directly without re-querying,
        # because querying would trigger re-compilation (which would
        # fail since the mock has no canned response for "Frank").
        from context_compiler.compiler.compiler import _resolve_path

        greeting_node = _resolve_path(prompt_context.root, Path("greeting"))
        assert isinstance(greeting_node, PromptNode)
        # Node should be stale: cache is invalidated and result is cleared.
        assert greeting_node.prompt_state is PromptNodeState.STALE


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestContextSerialization:
    def test_to_dict_and_from_dict(self, simple_context):
        data = simple_context.to_dict()
        restored = Context.from_dict(data)
        node = restored.query(Path("name"))
        assert node.value == "Alice"

    def test_get_value_helper(self, simple_context):
        value = simple_context.get_value(Path("age"))
        assert value == 30
