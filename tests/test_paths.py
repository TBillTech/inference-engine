"""Tests for context_resolver.ast.paths."""

import pytest

from context_resolver.ast.paths import Path


class TestPath:
    def test_basic_construction(self):
        p = Path("a", "b", "c")
        assert p.segments == ("a", "b", "c")

    def test_integer_segments(self):
        p = Path("items", 0, "name")
        assert p.segments == ("items", 0, "name")

    def test_invalid_segment_type(self):
        with pytest.raises(TypeError):
            Path("a", 1.5)  # type: ignore[arg-type]

    def test_empty_path(self):
        p = Path()
        assert p.is_empty
        assert str(p) == "<root>"

    def test_head(self):
        p = Path("x", "y", "z")
        assert p.head == "x"

    def test_head_empty_raises(self):
        with pytest.raises(IndexError):
            Path().head

    def test_tail(self):
        p = Path("x", "y", "z")
        assert p.tail == Path("y", "z")

    def test_tail_of_single(self):
        p = Path("x")
        assert p.tail == Path()

    def test_parent(self):
        p = Path("a", "b", "c")
        assert p.parent == Path("a", "b")

    def test_parent_empty_raises(self):
        with pytest.raises(IndexError):
            Path().parent

    def test_leaf(self):
        p = Path("a", "b", "c")
        assert p.leaf == "c"

    def test_leaf_integer(self):
        p = Path("items", 3)
        assert p.leaf == 3

    def test_division_operator_with_string(self):
        p = Path("a", "b") / "c"
        assert p.segments == ("a", "b", "c")

    def test_division_operator_with_int(self):
        p = Path("items") / 0
        assert p.segments == ("items", 0)

    def test_division_operator_with_path(self):
        p = Path("a") / Path("b", "c")
        assert p.segments == ("a", "b", "c")

    def test_str_representation_strings_only(self):
        assert str(Path("a", "b", "c")) == "a.b.c"

    def test_str_representation_with_int(self):
        assert str(Path("items", 2, "name")) == "items[2].name"

    def test_str_representation_int_at_start(self):
        assert str(Path(0, "name")) == "[0].name"

    def test_repr(self):
        p = Path("x", 1)
        assert repr(p) == "Path('x', 1)"

    def test_equality(self):
        assert Path("a", "b") == Path("a", "b")
        assert Path("a", "b") != Path("a", "c")

    def test_hashable(self):
        d = {Path("a", "b"): "value"}
        assert d[Path("a", "b")] == "value"

    def test_len(self):
        assert len(Path("a", "b", "c")) == 3
        assert len(Path()) == 0

    def test_startswith(self):
        p = Path("player", "inventory", 0)
        assert p.startswith(Path("player"))
        assert p.startswith(Path("player", "inventory"))
        assert not p.startswith(Path("world"))
