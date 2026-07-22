"""
Dependency graph for the Context Compiler.

The :class:`DependencyGraph` tracks which :class:`~context_compiler.ast.prompt_node.PromptNode`
paths depend on which other paths so that:

* stale detection is automatic when a dependency changes
* cycle detection prevents infinite compilation loops

Design
------
The graph is a directed graph where an edge ``A → B`` means "node at path A
depends on the value at path B".  If we detect a cycle (A → B → … → A) we
raise :class:`CycleError` rather than recursing indefinitely.
"""

from __future__ import annotations

from typing import Iterator

from context_compiler.ast.paths import Path


class CycleError(Exception):
    """Raised when a cyclic dependency is detected in the dependency graph."""

    def __init__(self, cycle: list[Path]) -> None:
        path_str = " → ".join(str(p) for p in cycle)
        super().__init__(f"Cyclic dependency detected: {path_str}")
        self.cycle: list[Path] = cycle


class DependencyGraph:
    """
    Directed dependency graph over :class:`~context_compiler.ast.paths.Path` nodes.

    Each entry records: *dependent* depends on *dependency*.

    The graph can be queried to find:
    * what a given path depends on (its *dependencies*)
    * what depends on a given path (its *dependents*)

    Cycle detection is performed eagerly when :meth:`add_dependency` is called.
    """

    def __init__(self) -> None:
        # edges[dependent] = set of paths that `dependent` depends on
        self._edges: dict[Path, set[Path]] = {}
        # reverse_edges[dependency] = set of paths that depend on `dependency`
        self._reverse_edges: dict[Path, set[Path]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_dependency(self, dependent: Path, dependency: Path) -> None:
        """
        Record that *dependent* requires the value at *dependency*.

        Parameters
        ----------
        dependent:
            The path of the node that needs to be compiled.
        dependency:
            The path of the node whose value is consumed during compilation
            of *dependent*.

        Raises
        ------
        CycleError
            If adding this edge would create a cycle.
        """
        if dependent not in self._edges:
            self._edges[dependent] = set()
        if dependency not in self._reverse_edges:
            self._reverse_edges[dependency] = set()

        self._edges[dependent].add(dependency)
        self._reverse_edges[dependency].add(dependent)

        # Check for cycles starting from the newly added edge.
        cycle = self._find_cycle(dependent)
        if cycle:
            # Roll back the edge before raising.
            self._edges[dependent].discard(dependency)
            self._reverse_edges[dependency].discard(dependent)
            raise CycleError(cycle)

    def remove_dependency(self, dependent: Path, dependency: Path) -> None:
        """Remove the dependency edge *dependent* → *dependency* if it exists."""
        self._edges.get(dependent, set()).discard(dependency)
        self._reverse_edges.get(dependency, set()).discard(dependent)

    def register_node(self, path: Path) -> None:
        """Ensure *path* appears in the graph even if it has no edges yet."""
        self._edges.setdefault(path, set())

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def dependencies_of(self, path: Path) -> frozenset[Path]:
        """Return the set of paths that *path* directly depends on."""
        return frozenset(self._edges.get(path, set()))

    def dependents_of(self, path: Path) -> frozenset[Path]:
        """Return the set of paths that directly depend on *path*."""
        return frozenset(self._reverse_edges.get(path, set()))

    def transitive_dependents_of(self, path: Path) -> frozenset[Path]:
        """
        Return all paths that transitively depend on *path*.

        This is the set of nodes that must be marked stale when *path* changes.
        """
        visited: set[Path] = set()
        stack: list[Path] = list(self._reverse_edges.get(path, set()))
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self._reverse_edges.get(current, set()))
        return frozenset(visited)

    def all_nodes(self) -> frozenset[Path]:
        """Return all known paths in the graph."""
        return frozenset(self._edges.keys()) | frozenset(self._reverse_edges.keys())

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def _find_cycle(self, start: Path) -> list[Path] | None:
        """
        DFS-based cycle detection starting from *start*.

        Returns the cycle as an ordered list of paths if one is found, or
        ``None`` otherwise.
        """
        visited: set[Path] = set()
        path: list[Path] = []

        def dfs(node: Path) -> bool:
            if node in visited:
                return False
            if node in path:
                # Found cycle – extract it.
                cycle_start = path.index(node)
                return True  # signal: check `path`

            path.append(node)
            for dep in self._edges.get(node, set()):
                if dep in path:
                    path.append(dep)
                    return True
                if dfs(dep):
                    return True
            path.pop()
            visited.add(node)
            return False

        if dfs(start):
            # Find the start of the cycle in `path`.
            last = path[-1]
            cycle_start_idx = path.index(last)
            return path[cycle_start_idx:]
        return None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, list[list[str | int]]]:
        """Serialize the graph to a plain dictionary."""
        return {
            "edges": [
                [list(dep.segments), list(dependency.segments)]
                for dep, deps in self._edges.items()
                for dependency in deps
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DependencyGraph":
        """Deserialize a :class:`DependencyGraph` from a plain dictionary."""
        graph = cls()
        for dep_segs, dependency_segs in data.get("edges", []):
            graph.add_dependency(Path(*dep_segs), Path(*dependency_segs))
        return graph

    def __repr__(self) -> str:
        edge_count = sum(len(v) for v in self._edges.values())
        return f"DependencyGraph(nodes={len(self._edges)}, edges={edge_count})"
