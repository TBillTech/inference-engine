"""Tests for context_resolver.context.context.Context."""

import pytest

from context_resolver.ast.nodes import MappingNode, ScalarNode, SequenceNode
from context_resolver.ast.paths import Path
from context_resolver.ast.resolvable_node import ResolvableNode, ResolvableNodeState
from context_resolver.ast.schema import Schema, FieldSpec
from context_resolver.query.resolver import Resolver
from context_resolver.query.passes import ResolutionPass
from context_resolver.context.context import Context, NodeNotFoundError
from context_resolver.inference.mock_provider import MockProvider
from context_resolver.templates.template import Template, TemplateRegistry


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
def resolvable_context():
    """A context with a ResolvableNode wired up to a mock provider."""
    schema = Schema(
        name="GreetSchema",
        fields=[FieldSpec("greeting", type="str", required=True)],
    )
    node = ResolvableNode(
        template_ref="greet",
        input_bindings={"name": Path("name")},
        output_schema=schema,
    )
    root = MappingNode({
        "name": ScalarNode("Carol"),
        "greeting": node,
    })

    registry = TemplateRegistry()
    registry.register(Template("greet", "Say hello to {name}."))

    mock = MockProvider(
        responses={"Say hello to Carol.": {"greeting": "Hello, Carol!"}},
    )
    resolver = Resolver(template_registry=registry, passes=[ResolutionPass(mock)])
    return Context(root=root, resolver=resolver)


@pytest.fixture
def auto_configured_context():
    """A context where ResolvableNode bindings are inferred at construction time."""
    schema = Schema(
        name="GreetSchema",
        fields=[FieldSpec("greeting", type="str", required=True)],
    )
    template = Template("greet", "Say hello to {name}.")
    node = ResolvableNode(template, schema)
    root = MappingNode({
        "name": ScalarNode("Carol"),
        "greeting": node,
    })

    registry = TemplateRegistry()
    registry.register(template)
    mock = MockProvider(
        responses={"Say hello to Carol.": {"greeting": "Hello, Carol!"}},
    )
    resolver = Resolver(template_registry=registry, passes=[ResolutionPass(mock)])
    return Context(root=root, resolver=resolver)


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

    def test_has_path_matches_unresolved_resolvable_schema_field(self):
        schema = Schema(
            name="Out",
            fields=[FieldSpec(name="vibe", type="str", required=True)],
        )
        unresolved = ResolvableNode(
            template="vibe",
            output_schema=schema,
            input_bindings={},
            dependencies=[],
        )
        ctx = Context(MappingNode({"atmosphere": unresolved}))

        assert ctx.has_path(Path("atmosphere", "vibe")) is True
        assert ctx.has_path(Path("atmosphere", "missing")) is False

    def test_has_path_matches_nested_unresolved_resolvable_schema_field(self):
        details_schema = Schema(
            name="Details",
            fields=[FieldSpec(name="smell", type="str", required=True)],
        )
        schema = Schema(
            name="Out",
            fields=[FieldSpec(name="details", type=details_schema, required=True)],
        )
        unresolved = ResolvableNode(
            template="scene",
            output_schema=schema,
            input_bindings={},
            dependencies=[],
        )
        ctx = Context(MappingNode({"scene": unresolved}))

        assert ctx.has_path(Path("scene", "details", "smell")) is True
        assert ctx.has_path(Path("scene", "details", "sound")) is False


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
# ResolvableNode resolution
# ---------------------------------------------------------------------------


class TestContextResolvableNode:
    def test_query_triggers_resolution(self, resolvable_context):
        node = resolvable_context.query(Path("greeting"))
        assert isinstance(node, ResolvableNode)
        assert node.resolution_state is ResolvableNodeState.RESOLVED
        result = node.result
        assert result is not None

    def test_resolved_greeting_value(self, resolvable_context):
        node = resolvable_context.query(Path("greeting"))
        assert isinstance(node, ResolvableNode)
        greeting = node.result.get("greeting").value  # type: ignore[union-attr]
        assert greeting == "Hello, Carol!"

    def test_provider_called_once(self, resolvable_context):
        resolvable_context.query(Path("greeting"))
        # Get the mock provider via the resolver pass.
        resolution_pass = resolvable_context.resolver._passes[0]
        assert resolution_pass.provider.call_count == 1

    def test_second_query_served_from_cache(self, resolvable_context):
        n1 = resolvable_context.query(Path("greeting"))
        n2 = resolvable_context.query(Path("greeting"))
        assert n1 is n2
        # Provider should still have been called only once.
        resolution_pass = resolvable_context.resolver._passes[0]
        assert resolution_pass.provider.call_count == 1

    def test_mutation_marks_node_stale(self, resolvable_context):
        resolvable_context.query(Path("greeting"))
        # Mutate the dependency.
        resolvable_context.set(Path("name"), ScalarNode("Frank"))
        # Inspect the node in the tree directly without re-querying,
        # because querying would trigger re-resolution (which would
        # fail since the mock has no canned response for "Frank").
        from context_resolver.query.resolver import _resolve_path

        greeting_node = _resolve_path(resolvable_context.root, Path("greeting"))
        assert isinstance(greeting_node, ResolvableNode)
        # Node should be stale: cache is invalidated and result is cleared.
        assert greeting_node.resolution_state is ResolvableNodeState.STALE

    def test_context_auto_configures_unconfigured_resolvable_node(self, auto_configured_context):
        greeting_node = auto_configured_context.root.get("greeting")
        assert isinstance(greeting_node, ResolvableNode)
        assert greeting_node.is_configured()
        assert greeting_node.input_bindings == {"name": Path("name")}
        assert greeting_node.dependencies == [Path("name")]


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
