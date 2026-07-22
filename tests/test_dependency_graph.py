"""Tests for context_resolver.query.dependency_graph."""

import pytest

from context_resolver.ast.paths import Path
from context_resolver.query.dependency_graph import DependencyGraph, CycleError


@pytest.fixture
def graph():
    return DependencyGraph()


class TestDependencyGraphBasics:
    def test_empty_graph(self, graph):
        assert graph.dependencies_of(Path("x")) == frozenset()
        assert graph.dependents_of(Path("x")) == frozenset()

    def test_add_single_dependency(self, graph):
        graph.add_dependency(Path("a"), Path("b"))
        assert Path("b") in graph.dependencies_of(Path("a"))
        assert Path("a") in graph.dependents_of(Path("b"))

    def test_add_multiple_dependencies(self, graph):
        graph.add_dependency(Path("a"), Path("b"))
        graph.add_dependency(Path("a"), Path("c"))
        deps = graph.dependencies_of(Path("a"))
        assert Path("b") in deps
        assert Path("c") in deps

    def test_remove_dependency(self, graph):
        graph.add_dependency(Path("a"), Path("b"))
        graph.remove_dependency(Path("a"), Path("b"))
        assert graph.dependencies_of(Path("a")) == frozenset()

    def test_transitive_dependents(self, graph):
        # a -> b -> c  (c depends on b, b depends on a)
        graph.add_dependency(Path("b"), Path("a"))
        graph.add_dependency(Path("c"), Path("b"))
        transitive = graph.transitive_dependents_of(Path("a"))
        assert Path("b") in transitive
        assert Path("c") in transitive

    def test_register_node(self, graph):
        graph.register_node(Path("x"))
        assert Path("x") in graph.all_nodes()


class TestCycleDetection:
    def test_direct_cycle_raises(self, graph):
        graph.add_dependency(Path("a"), Path("b"))
        with pytest.raises(CycleError):
            graph.add_dependency(Path("b"), Path("a"))

    def test_indirect_cycle_raises(self, graph):
        graph.add_dependency(Path("a"), Path("b"))
        graph.add_dependency(Path("b"), Path("c"))
        with pytest.raises(CycleError):
            graph.add_dependency(Path("c"), Path("a"))

    def test_no_cycle_for_dag(self, graph):
        # a -> b, a -> c, b -> d, c -> d  (diamond, no cycle)
        graph.add_dependency(Path("a"), Path("b"))
        graph.add_dependency(Path("a"), Path("c"))
        graph.add_dependency(Path("b"), Path("d"))
        graph.add_dependency(Path("c"), Path("d"))
        # No exception raised

    def test_cycle_error_contains_cycle(self, graph):
        graph.add_dependency(Path("a"), Path("b"))
        with pytest.raises(CycleError) as exc_info:
            graph.add_dependency(Path("b"), Path("a"))
        assert exc_info.value.cycle is not None
        assert len(exc_info.value.cycle) >= 2

    def test_failed_add_rolls_back(self, graph):
        graph.add_dependency(Path("a"), Path("b"))
        try:
            graph.add_dependency(Path("b"), Path("a"))
        except CycleError:
            pass
        # The rolled-back edge should not exist.
        assert Path("a") not in graph.dependencies_of(Path("b"))


class TestSerialization:
    def test_roundtrip(self, graph):
        graph.add_dependency(Path("x"), Path("y"))
        data = graph.to_dict()
        restored = DependencyGraph.from_dict(data)
        assert Path("y") in restored.dependencies_of(Path("x"))
