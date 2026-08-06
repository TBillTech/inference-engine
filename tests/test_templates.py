"""Tests for context_resolver.templates.template."""

import pytest

from context_resolver.templates.template import (
    Template,
    TemplateRegistry,
    JSONOutputTemplate,
    JSONOutputFunction,
)
from context_resolver.ast.schema import Schema, FieldSpec
from context_resolver.ast.nodes import MappingNode, ScalarNode
from context_resolver.ast.paths import Path
from context_resolver.ast.resolvable_node import ResolvableNode
from context_resolver.context.context import Context


class TestTemplate:
    def test_render_with_bindings(self):
        t = Template("greet", "Hello, {name}!")
        assert t.render({"name": "World"}) == "Hello, World!"

    def test_render_missing_variable_raises(self):
        t = Template("greet", "Hello, {name}!")
        with pytest.raises(KeyError, match="name"):
            t.render({})

    def test_render_extra_bindings_ignored(self):
        t = Template("greet", "Hello, {name}!")
        # Extra keys don't cause errors with format_map.
        assert t.render({"name": "Alice", "extra": "ignored"}) == "Hello, Alice!"

    def test_serialization_roundtrip(self):
        t = Template("greet", "Hello, {name}!", description="A greeting")
        restored = Template.from_dict(t.to_dict())
        assert restored.name == "greet"
        assert restored.template_str == "Hello, {name}!"
        assert restored.description == "A greeting"

    def test_repr(self):
        t = Template("greet", "Hello!")
        assert "greet" in repr(t)

    def test_infer_input_bindings_from_existing_paths(self):
        t = Template("case", "Town: {town.name}; Weather: {atmosphere.weather}")
        ctx = Context(
            MappingNode(
                {
                    "town": MappingNode({"name": ScalarNode("Eerie")}),
                    "atmosphere": MappingNode({"weather": ScalarNode("Fog")}),
                }
            )
        )

        bindings = t.infer_input_bindings(ctx)
        assert bindings == {
            "town.name": Path("town", "name"),
            "atmosphere.weather": Path("atmosphere", "weather"),
        }

    def test_infer_input_bindings_ignores_missing_paths(self):
        t = Template("case", "Town: {town.name}; Missing: {town.backstory}")
        ctx = Context(MappingNode({"town": MappingNode({"name": ScalarNode("Eerie")})}))

        bindings = t.infer_input_bindings(ctx)
        assert bindings == {"town.name": Path("town", "name")}

    def test_infer_input_bindings_matches_unresolved_resolvable_schema_field(self):
        out_schema = Schema(
            name="SceneOut",
            fields=[FieldSpec(name="description", type="str", required=True)],
        )
        scene_node = ResolvableNode(
            template="scene",
            output_schema=out_schema,
            input_bindings={},
            dependencies=[],
        )
        ctx = Context(MappingNode({"scene": scene_node}))
        t = Template("narrate", "Describe: {scene.description}")

        bindings = t.infer_input_bindings(ctx)
        assert bindings == {"scene.description": Path("scene", "description")}

    def test_infer_input_bindings_matches_nested_unresolved_schema_path(self):
        details_schema = Schema(
            name="SceneDetails",
            fields=[FieldSpec(name="smell", type="str", required=True)],
        )
        out_schema = Schema(
            name="SceneOut",
            fields=[FieldSpec(name="details", type=details_schema, required=True)],
        )
        scene_node = ResolvableNode(
            template="scene",
            output_schema=out_schema,
            input_bindings={},
            dependencies=[],
        )
        ctx = Context(MappingNode({"scene": scene_node}))
        t = Template("narrate", "Smell: {scene.details.smell}")

        bindings = t.infer_input_bindings(ctx)
        assert bindings == {"scene.details.smell": Path("scene", "details", "smell")}


class TestTemplateRegistry:
    def test_register_and_get(self):
        reg = TemplateRegistry()
        t = Template("t1", "prompt")
        reg.register(t)
        assert reg.get("t1") is t

    def test_get_missing_returns_none(self):
        reg = TemplateRegistry()
        assert reg.get("missing") is None

    def test_contains(self):
        reg = TemplateRegistry()
        reg.register(Template("t1", "prompt"))
        assert "t1" in reg
        assert "t2" not in reg

    def test_len(self):
        reg = TemplateRegistry()
        assert len(reg) == 0
        reg.register(Template("t1", "prompt"))
        assert len(reg) == 1

    def test_repr(self):
        reg = TemplateRegistry()
        reg.register(Template("t1", "prompt"))
        assert "t1" in repr(reg)


# ---------------------------------------------------------------------------
# JSONOutputTemplate
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_schema():
    return Schema(
        name="IntroOutput",
        fields=[
            FieldSpec(name="greeting", type="str", required=True,
                      description="A greeting message"),
            FieldSpec(name="opening", type="str", required=False,
                      description="An opening line"),
        ],
    )


class TestJSONOutputTemplate:

    def test_is_subclass_of_template(self, simple_schema):
        t = JSONOutputTemplate("t", "Base {name}.", schema=simple_schema)
        assert isinstance(t, Template)

    def test_render_contains_base_prompt(self, simple_schema):
        t = JSONOutputTemplate("t", "Greet {name}.", schema=simple_schema)
        rendered = t.render({"name": "Alice"})
        assert rendered.startswith("Greet Alice.")

    def test_render_appends_json_block(self, simple_schema):
        t = JSONOutputTemplate("t", "Task.", schema=simple_schema)
        rendered = t.render({})
        assert "Respond with a valid JSON object" in rendered
        assert '"greeting"' in rendered
        assert '"opening"' in rendered
        assert "Output only the JSON object" in rendered

    def test_render_includes_field_types(self, simple_schema):
        t = JSONOutputTemplate("t", "Task.", schema=simple_schema)
        rendered = t.render({})
        assert "(str, required)" in rendered
        assert "(str, optional)" in rendered

    def test_render_includes_field_descriptions(self, simple_schema):
        t = JSONOutputTemplate("t", "Task.", schema=simple_schema)
        rendered = t.render({})
        assert "A greeting message" in rendered
        assert "An opening line" in rendered

    def test_render_omits_description_when_empty(self):
        schema = Schema("S", fields=[FieldSpec("val", "int", required=True, description="")])
        t = JSONOutputTemplate("t", "Task.", schema=schema)
        rendered = t.render({})
        # No description suffix should appear after the field spec
        assert '"val" (int, required)' in rendered
        assert '"val" (int, required):' not in rendered

    def test_render_base_and_json_separated_by_blank_line(self, simple_schema):
        t = JSONOutputTemplate("t", "Base.", schema=simple_schema)
        rendered = t.render({})
        assert "\n\n" in rendered

    def test_render_raises_on_missing_binding(self, simple_schema):
        t = JSONOutputTemplate("t", "Hello, {name}.", schema=simple_schema)
        with pytest.raises(KeyError, match="name"):
            t.render({})

    def test_schema_property(self, simple_schema):
        t = JSONOutputTemplate("t", "Task.", schema=simple_schema)
        assert t.schema is simple_schema

    def test_repr(self, simple_schema):
        t = JSONOutputTemplate("t", "Task.", schema=simple_schema)
        r = repr(t)
        assert "JSONOutputTemplate" in r
        assert "IntroOutput" in r

    def test_registered_in_registry(self, simple_schema):
        reg = TemplateRegistry()
        t = JSONOutputTemplate("greet", "Task.", schema=simple_schema)
        reg.register(t)
        assert reg.get("greet") is t

    def test_no_schema_fields_renders_empty_list(self):
        schema = Schema("Empty", fields=[])
        t = JSONOutputTemplate("t", "Task.", schema=schema)
        rendered = t.render({})
        assert "Respond with a valid JSON object" in rendered
        assert "Output only the JSON object" in rendered


class TestJSONOutputFunction:
    def test_infer_input_bindings_from_arglist(self):
        schema = Schema(
            name="SunDial",
            fields=[FieldSpec(name="sun_dial", type="str", required=True)],
        )
        fn = JSONOutputFunction(
            name="sun",
            arglist=["atmosphere.hour", "atmosphere.pm"],
            python_fn=lambda hour, pm: "Morning" if not pm else "Evening",
            schema=schema,
        )
        ctx = Context(
            MappingNode(
                {
                    "atmosphere": MappingNode(
                        {
                            "hour": ScalarNode(8),
                            "pm": ScalarNode(False),
                        }
                    )
                }
            )
        )

        bindings = fn.infer_input_bindings(ctx)
        assert bindings == {
            "atmosphere.hour": Path("atmosphere", "hour"),
            "atmosphere.pm": Path("atmosphere", "pm"),
        }

    def test_evaluate_wraps_single_field_result(self):
        schema = Schema(
            name="SunDial",
            fields=[FieldSpec(name="sun_dial", type="str", required=True)],
        )
        fn = JSONOutputFunction(
            name="sun",
            arglist=["h"],
            python_fn=lambda hour: "Morning" if hour < 12 else "Afternoon",
            schema=schema,
        )

        out = fn.evaluate({"h": 8})
        assert out == {"sun_dial": "Morning"}

    def test_evaluate_uses_dict_result_directly(self):
        schema = Schema(
            name="Pair",
            fields=[
                FieldSpec(name="a", type="int", required=True),
                FieldSpec(name="b", type="int", required=True),
            ],
        )
        fn = JSONOutputFunction(
            name="pair",
            arglist=["x", "y"],
            python_fn=lambda x, y: {"a": x, "b": y},
            schema=schema,
        )

        out = fn.evaluate({"x": 1, "y": 2})
        assert out == {"a": 1, "b": 2}
