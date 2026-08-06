from context_resolver.ast.nodes import MappingNode, ScalarNode
from context_resolver.ast.paths import Path
from context_resolver.ast.resolvable_node import ResolvableNode
from context_resolver.ast.schema import FieldSpec, Schema
from context_resolver.context.context import Context

from apps.little_house_in_the_eerie.window_print_functions import get_location_summary


def test_get_location_summary_uses_mappingnode_items_and_updates_location_data() -> None:
    ctx = Context(
        MappingNode(
            {
                "locations": MappingNode(
                    {
                        "Little House Bed and Breakfast": MappingNode(
                            {
                                "name": ScalarNode("Little House Bed and Breakfast"),
                                "description": ScalarNode("An old B&B by the bay."),
                            }
                        ),
                        "Town Square": MappingNode(
                            {
                                "name": ScalarNode("Town Square"),
                                "description": ScalarNode("A windy plaza."),
                            }
                        ),
                    }
                ),
                "print_summaries": MappingNode(
                    {
                        "location_data": ScalarNode(""),
                        "location_summary": ScalarNode("formatted location summary"),
                    }
                ),
            }
        )
    )

    summary = get_location_summary(ctx, "Little")

    assert summary == "formatted location summary"
    location_data = ctx.query(Path("print_summaries", "location_data"))
    assert isinstance(location_data, ScalarNode)
    assert "Little House Bed and Breakfast" in str(location_data.value)


def test_get_location_summary_uses_full_context_path_for_resolved_fields() -> None:
    description = ResolvableNode(
        template_ref="location_description",
        output_schema=Schema(
            name="LocationDescription",
            fields=[FieldSpec(name="value", type="str", required=True)],
        ),
        input_bindings={},
        dependencies=[],
    )
    description.mark_resolved(ScalarNode("An old B&B by the bay."))

    ctx = Context(
        MappingNode(
            {
                "locations": MappingNode(
                    {
                        "Little House Bed and Breakfast": MappingNode(
                            {
                                "name": ScalarNode("Little House Bed and Breakfast"),
                                "description": description,
                            }
                        ),
                    }
                ),
                "print_summaries": MappingNode(
                    {
                        "location_data": ScalarNode(""),
                        "location_summary": ScalarNode("formatted location summary"),
                    }
                ),
            }
        )
    )

    summary = get_location_summary(ctx, "Little")

    assert summary == "formatted location summary"
    location_data = ctx.query(Path("print_summaries", "location_data"))
    assert isinstance(location_data, ScalarNode)
    assert str(location_data.value) == (
        "name: Little House Bed and Breakfast\n"
        "description: An old B&B by the bay.\n"
    )
