"""Tests for context_compiler.templates.template."""

import pytest

from context_compiler.templates.template import Template, TemplateRegistry


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
