"""Tests for the serialization sub-package."""

import json
import pytest

from context_compiler.ast.nodes import MappingNode, ScalarNode, SequenceNode
from context_compiler.ast.paths import Path
from context_compiler.serialization.serializer import Serializer
from context_compiler.serialization.diff import diff_contexts, ContextDiff


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------


class TestSerializer:
    def test_roundtrip_scalar(self):
        s = Serializer()
        node = ScalarNode("hello")
        data = s.to_dict(node)
        restored = s.from_dict(data)
        assert isinstance(restored, ScalarNode)
        assert restored.value == "hello"

    def test_roundtrip_mapping(self):
        s = Serializer()
        node = MappingNode({"a": ScalarNode(1), "b": ScalarNode(2)})
        data = s.to_dict(node)
        restored = s.from_dict(data)
        assert isinstance(restored, MappingNode)
        assert restored.get("a").value == 1

    def test_json_roundtrip(self):
        s = Serializer()
        node = SequenceNode([ScalarNode("x"), ScalarNode("y")])
        json_str = s.to_json(node)
        restored = s.from_json(json_str)
        assert isinstance(restored, SequenceNode)
        assert restored.get(0).value == "x"

    def test_json_is_deterministic(self):
        s = Serializer()
        node = MappingNode({"b": ScalarNode(2), "a": ScalarNode(1)})
        assert s.to_json(node) == s.to_json(node)

    def test_version_mismatch_raises(self):
        s = Serializer(version="2")
        node = ScalarNode("hi")
        data = s.to_dict(node)
        s2 = Serializer(version="1")
        with pytest.raises(ValueError, match="version mismatch"):
            s2.from_dict(data)

    def test_version_in_envelope(self):
        s = Serializer()
        node = ScalarNode("x")
        data = s.to_dict(node)
        assert data["version"] == Serializer.CURRENT_VERSION


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


class TestDiffContexts:
    def test_identical_trees_produce_no_diffs(self):
        a = MappingNode({"x": ScalarNode(1)})
        b = MappingNode({"x": ScalarNode(1)})
        assert diff_contexts(a, b) == []

    def test_changed_scalar(self):
        a = MappingNode({"x": ScalarNode(1)})
        b = MappingNode({"x": ScalarNode(2)})
        diffs = diff_contexts(a, b)
        assert len(diffs) == 1
        assert diffs[0].kind == "changed"
        assert diffs[0].path == Path("x")
        assert diffs[0].old_value == 1
        assert diffs[0].new_value == 2

    def test_added_key(self):
        a = MappingNode({"x": ScalarNode(1)})
        b = MappingNode({"x": ScalarNode(1), "y": ScalarNode(2)})
        diffs = diff_contexts(a, b)
        kinds = {d.kind for d in diffs}
        assert "added" in kinds

    def test_removed_key(self):
        a = MappingNode({"x": ScalarNode(1), "y": ScalarNode(2)})
        b = MappingNode({"x": ScalarNode(1)})
        diffs = diff_contexts(a, b)
        kinds = {d.kind for d in diffs}
        assert "removed" in kinds

    def test_nested_change(self):
        a = MappingNode({"inner": MappingNode({"v": ScalarNode("old")})})
        b = MappingNode({"inner": MappingNode({"v": ScalarNode("new")})})
        diffs = diff_contexts(a, b)
        assert len(diffs) == 1
        assert diffs[0].path == Path("inner", "v")

    def test_sequence_change(self):
        a = SequenceNode([ScalarNode(1), ScalarNode(2)])
        b = SequenceNode([ScalarNode(1), ScalarNode(99)])
        diffs = diff_contexts(a, b)
        assert len(diffs) == 1
        assert diffs[0].kind == "changed"

    def test_diff_str_representation(self):
        diff = ContextDiff(Path("x"), "changed", 1, 2)
        s = str(diff)
        assert "~" in s
        assert "x" in s
