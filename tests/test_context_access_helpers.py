from context_resolver.ast.nodes import MappingNode, ScalarNode
from context_resolver.ast.paths import Path
from context_resolver.ast.resolvable_node import ResolvableNode
from context_resolver.ast.schema import FieldSpec, Schema
from context_resolver.context.context import Context, NodeNotFoundError
from context_resolver.inference.mock_provider import MockProvider
from context_resolver.query.passes import ResolutionPass
from context_resolver.query.resolver import Resolver
from context_resolver.templates.template import Template, TemplateRegistry


def test_query_text_returns_scalar_as_string() -> None:
    ctx = Context(MappingNode({"age": ScalarNode(30)}))

    assert ctx.root.query_text(ctx, "age") == "30"
    assert ctx.query_text("age") == "30"


def test_query_text_walks_nested_mapping_segments() -> None:
    ctx = Context(
        MappingNode(
            {
                "town": MappingNode(
                    {
                        "name": ScalarNode("Arkham"),
                    }
                ),
            }
        )
    )

    assert ctx.root.query_text(ctx, "town", "name") == "Arkham"
    assert ctx.query_text("town", "name") == "Arkham"


def test_query_text_walks_into_resolved_node_result() -> None:
    schema = Schema(
        name="TownSchema",
        fields=[
            FieldSpec("name", type="str", required=True),
            FieldSpec("location", type="str", required=True),
        ],
    )
    town = ResolvableNode(
        template_ref="town",
        input_bindings={"seed": Path("seed")},
        output_schema=schema,
    )
    root = MappingNode(
        {
            "seed": ScalarNode("coastal"),
            "town": town,
        }
    )
    registry = TemplateRegistry()
    registry.register(Template("town", "Town seed: {seed}"))
    mock = MockProvider(
        responses={
            "Town seed: coastal": {"name": "Innsmouth", "location": "Massachusetts"},
        }
    )
    resolver = Resolver(template_registry=registry, passes=[ResolutionPass(mock)])
    ctx = Context(root=root, resolver=resolver)

    assert ctx.root.query_text(ctx, "town", "name") == "Innsmouth"
    assert ctx.query_text("town", "name") == "Innsmouth"
    assert ctx.query_text("town") == "Innsmouth"


def test_query_text_raises_for_missing_child_segment() -> None:
    ctx = Context(MappingNode({"town": MappingNode({})}))

    try:
        ctx.query_text("town", "name")
    except NodeNotFoundError:
        pass
    else:
        raise AssertionError("query_text should raise NodeNotFoundError for missing path")