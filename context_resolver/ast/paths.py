"""
Typed path abstraction for addressing nodes inside the Context tree.

Use :class:`Path` instead of raw strings so that path manipulation is
type-safe, composable, and easily testable.

Examples
--------
>>> p = Path("player", "inventory", 2)
>>> p.segments
('player', 'inventory', 2)
>>> child = p / "sword"
>>> child.segments
('player', 'inventory', 2, 'sword')
"""

from __future__ import annotations

from typing import Union

# A path segment is either a string (mapping key) or an int (sequence index).
PathSegment = Union[str, int]


class Path:
    """
    An immutable, typed sequence of path segments.

    Parameters
    ----------
    *segments:
        One or more path segments (strings or integers).

    Examples
    --------
    >>> p = Path("world", "npcs", 0, "name")
    >>> str(p)
    'world.npcs[0].name'
    """

    __slots__ = ("_segments",)

    def __init__(self, *segments: PathSegment) -> None:
        for i, seg in enumerate(segments):
            if not isinstance(seg, (str, int)):
                raise TypeError(
                    f"Path segment at index {i} must be str or int, "
                    f"got {type(seg).__name__!r}"
                )
        self._segments: tuple[PathSegment, ...] = segments

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def segments(self) -> tuple[PathSegment, ...]:
        """Immutable tuple of this path's segments."""
        return self._segments

    @property
    def is_empty(self) -> bool:
        """``True`` if this path has no segments."""
        return len(self._segments) == 0

    @property
    def head(self) -> PathSegment:
        """The first segment of this path."""
        if not self._segments:
            raise IndexError("Cannot take head of an empty Path")
        return self._segments[0]

    @property
    def tail(self) -> "Path":
        """A new :class:`Path` containing all segments after the first."""
        return Path(*self._segments[1:])

    @property
    def parent(self) -> "Path":
        """A new :class:`Path` containing all segments except the last."""
        if not self._segments:
            raise IndexError("Cannot take parent of an empty Path")
        return Path(*self._segments[:-1])

    @property
    def leaf(self) -> PathSegment:
        """The last segment of this path."""
        if not self._segments:
            raise IndexError("Cannot take leaf of an empty Path")
        return self._segments[-1]

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def __truediv__(self, other: PathSegment | "Path") -> "Path":
        """
        Extend this path using the ``/`` operator.

        >>> Path("a", "b") / "c"
        Path('a', 'b', 'c')
        >>> Path("a") / Path("b", "c")
        Path('a', 'b', 'c')
        """
        if isinstance(other, Path):
            return Path(*self._segments, *other._segments)
        return Path(*self._segments, other)

    def startswith(self, prefix: "Path") -> bool:
        """Return ``True`` if this path starts with *prefix*."""
        return self._segments[: len(prefix)] == prefix._segments

    def __len__(self) -> int:
        return len(self._segments)

    # ------------------------------------------------------------------
    # Equality & hashing (needed for use as dict keys / set members)
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Path):
            return self._segments == other._segments
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._segments)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """
        Return a human-readable dotted path string.

        Integer segments are rendered as ``[n]`` subscripts.

        >>> str(Path("a", "b", 2, "c"))
        'a.b[2].c'
        """
        if not self._segments:
            return "<root>"
        parts: list[str] = []
        for i, seg in enumerate(self._segments):
            if isinstance(seg, int):
                if parts:
                    # Append as subscript to the previous token.
                    parts[-1] = f"{parts[-1]}[{seg}]"
                else:
                    parts.append(f"[{seg}]")
            else:
                parts.append(seg)
        return ".".join(parts)

    def __repr__(self) -> str:
        segs = ", ".join(repr(s) for s in self._segments)
        return f"Path({segs})"
