"""
Context diff utility.

:func:`diff_contexts` compares two serialized Context trees and returns a
human-readable list of differences.  This is primarily useful for debugging
and auditing incremental compilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from context_compiler.ast.nodes import Node, MappingNode, SequenceNode, ScalarNode
from context_compiler.ast.paths import Path


@dataclass
class ContextDiff:
    """
    Represents a single difference between two Context trees.

    Attributes
    ----------
    path:
        The path at which the difference was found.
    kind:
        One of ``"added"``, ``"removed"``, or ``"changed"``.
    old_value:
        The value in the *old* tree (``None`` for additions).
    new_value:
        The value in the *new* tree (``None`` for removals).
    """

    path: Path
    kind: str
    old_value: Any
    new_value: Any

    def __str__(self) -> str:
        if self.kind == "added":
            return f"+ {self.path}: {self.new_value!r}"
        if self.kind == "removed":
            return f"- {self.path}: {self.old_value!r}"
        return f"~ {self.path}: {self.old_value!r} → {self.new_value!r}"


def diff_contexts(old: Node, new: Node) -> list[ContextDiff]:
    """
    Compare two node trees and return a flat list of :class:`ContextDiff` entries.

    The comparison is recursive and path-aware.  Only leaf-level
    :class:`~context_compiler.ast.nodes.ScalarNode` values are compared for
    changes; structural differences (added/removed keys or indices) are also
    reported.

    Parameters
    ----------
    old:
        The baseline node tree.
    new:
        The updated node tree.

    Returns
    -------
    list[ContextDiff]
        Ordered list of differences found.  Empty if the trees are identical.
    """
    diffs: list[ContextDiff] = []
    _diff_nodes(old, new, Path(), diffs)
    return diffs


def _diff_nodes(
    old: Node | None,
    new: Node | None,
    path: Path,
    diffs: list[ContextDiff],
) -> None:
    if old is None and new is None:
        return

    if old is None:
        diffs.append(ContextDiff(path, "added", None, _summarise(new)))
        return

    if new is None:
        diffs.append(ContextDiff(path, "removed", _summarise(old), None))
        return

    # Both exist – compare by type.
    if type(old) is not type(new):
        diffs.append(
            ContextDiff(path, "changed", _summarise(old), _summarise(new))
        )
        return

    if isinstance(old, ScalarNode) and isinstance(new, ScalarNode):
        if old.value != new.value:
            diffs.append(ContextDiff(path, "changed", old.value, new.value))
        return

    if isinstance(old, MappingNode) and isinstance(new, MappingNode):
        old_keys = set(old.keys())
        new_keys = set(new.keys())
        for key in sorted(old_keys - new_keys):
            diffs.append(
                ContextDiff(path / key, "removed", _summarise(old.get(key)), None)
            )
        for key in sorted(new_keys - old_keys):
            diffs.append(
                ContextDiff(path / key, "added", None, _summarise(new.get(key)))
            )
        for key in sorted(old_keys & new_keys):
            _diff_nodes(old.get(key), new.get(key), path / key, diffs)
        return

    if isinstance(old, SequenceNode) and isinstance(new, SequenceNode):
        max_len = max(len(old), len(new))
        for i in range(max_len):
            _diff_nodes(old.get(i), new.get(i), path / i, diffs)
        return

    # Fallback for other node types (e.g. ResolvableNode).
    old_dict = old.to_dict()
    new_dict = new.to_dict()
    if old_dict != new_dict:
        diffs.append(ContextDiff(path, "changed", _summarise(old), _summarise(new)))


def _summarise(node: Node | None) -> Any:
    """Return a compact representation of *node* for diff output."""
    if node is None:
        return None
    from context_compiler.ast.nodes import ScalarNode

    if isinstance(node, ScalarNode):
        return node.value
    return f"<{type(node).__name__}>"
