"""Tests for context_compiler.ast.nodes."""

import json
import pytest

from context_compiler.ast.nodes import (
    NodeState,
    ScalarNode,
    MappingNode,
    SequenceNode,
    _node_from_dict,
)


# ---------------------------------------------------------------------------
# ScalarNode
# ---------------------------------------------------------------------------


class TestScalarNode:
    def test_underspecified_when_value_is_none(self):
        node = ScalarNode()
        assert node.state is NodeState.UNDERSPECIFIED
        assert not node.is_fully_specified

    def test_fully_specified_with_value(self):
        node = ScalarNode("hello")
        assert node.state is NodeState.FULLY_SPECIFIED
        assert node.is_fully_specified

    def test_fully_specified_with_zero(self):
        """Falsy but non-None values should still be fully specified."""
        node = ScalarNode(0)
        assert node.state is NodeState.FULLY_SPECIFIED

    def test_fully_specified_with_false(self):
        node = ScalarNode(False)
        assert node.state is NodeState.FULLY_SPECIFIED

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            ScalarNode([1, 2, 3])  # type: ignore[arg-type]

    def test_equality(self):
        assert ScalarNode("x") == ScalarNode("x")
        assert ScalarNode("x") != ScalarNode("y")

    def test_serialization_roundtrip(self):
        node = ScalarNode(42, metadata={"source": "test"})
        data = node.to_dict()
        restored = ScalarNode.from_dict(data)
        assert restored.value == 42

    def test_json_roundtrip(self):
        node = ScalarNode("world")
        json_str = node.to_json()
        data = json.loads(json_str)
        assert data["value"] == "world"

    def test_with_metadata(self):
        node = ScalarNode("hello")
        node2 = node.with_metadata(key="value")
        assert node2.metadata == {"key": "value"}
        assert node.metadata == {}


# ---------------------------------------------------------------------------
# MappingNode
# ---------------------------------------------------------------------------


class TestMappingNode:
    def test_empty_is_underspecified(self):
        node = MappingNode()
        assert node.state is NodeState.UNDERSPECIFIED

    def test_all_specified_is_fully_specified(self):
        node = MappingNode({"a": ScalarNode(1), "b": ScalarNode(2)})
        assert node.state is NodeState.FULLY_SPECIFIED

    def test_mixed_is_partially_specified(self):
        node = MappingNode({"a": ScalarNode(1), "b": ScalarNode(None)})
        assert node.state is NodeState.PARTIALLY_SPECIFIED

    def test_get_returns_child(self):
        child = ScalarNode("hi")
        node = MappingNode({"key": child})
        assert node.get("key") is child

    def test_get_missing_returns_none(self):
        node = MappingNode()
        assert node.get("missing") is None

    def test_set_adds_key(self):
        node = MappingNode()
        node.set("x", ScalarNode(99))
        assert node.get("x").value == 99

    def test_contains(self):
        node = MappingNode({"a": ScalarNode(1)})
        assert "a" in node
        assert "b" not in node

    def test_serialization_roundtrip(self):
        node = MappingNode({"name": ScalarNode("Alice"), "age": ScalarNode(30)})
        data = node.to_dict()
        restored = MappingNode.from_dict(data)
        assert restored.get("name").value == "Alice"
        assert restored.get("age").value == 30


# ---------------------------------------------------------------------------
# SequenceNode
# ---------------------------------------------------------------------------


class TestSequenceNode:
    def test_empty_is_underspecified(self):
        node = SequenceNode()
        assert node.state is NodeState.UNDERSPECIFIED

    def test_all_specified_is_fully_specified(self):
        node = SequenceNode([ScalarNode(1), ScalarNode(2)])
        assert node.state is NodeState.FULLY_SPECIFIED

    def test_mixed_is_partially_specified(self):
        node = SequenceNode([ScalarNode(1), ScalarNode(None)])
        assert node.state is NodeState.PARTIALLY_SPECIFIED

    def test_get_by_index(self):
        node = SequenceNode([ScalarNode("a"), ScalarNode("b")])
        assert node.get(0).value == "a"
        assert node.get(1).value == "b"

    def test_get_out_of_range_returns_none(self):
        node = SequenceNode()
        assert node.get(5) is None

    def test_append(self):
        node = SequenceNode()
        node.append(ScalarNode("x"))
        assert len(node) == 1
        assert node.get(0).value == "x"

    def test_iteration(self):
        items = [ScalarNode(i) for i in range(3)]
        node = SequenceNode(items)
        values = [n.value for n in node]
        assert values == [0, 1, 2]

    def test_serialization_roundtrip(self):
        node = SequenceNode([ScalarNode(10), ScalarNode(20)])
        data = node.to_dict()
        restored = SequenceNode.from_dict(data)
        assert restored.get(0).value == 10
        assert restored.get(1).value == 20


# ---------------------------------------------------------------------------
# Polymorphic deserialization
# ---------------------------------------------------------------------------


class TestNodeFromDict:
    def test_scalar(self):
        data = {"type": "ScalarNode", "value": "hello", "metadata": {}}
        node = _node_from_dict(data)
        assert isinstance(node, ScalarNode)
        assert node.value == "hello"

    def test_mapping(self):
        data = {
            "type": "MappingNode",
            "fields": {"x": {"type": "ScalarNode", "value": 1, "metadata": {}}},
            "metadata": {},
        }
        node = _node_from_dict(data)
        assert isinstance(node, MappingNode)
        assert node.get("x").value == 1

    def test_sequence(self):
        data = {
            "type": "SequenceNode",
            "items": [{"type": "ScalarNode", "value": True, "metadata": {}}],
            "metadata": {},
        }
        node = _node_from_dict(data)
        assert isinstance(node, SequenceNode)
        assert node.get(0).value is True

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown node type"):
            _node_from_dict({"type": "GhostNode", "metadata": {}})
