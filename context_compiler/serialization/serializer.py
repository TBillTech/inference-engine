"""
Context serializer.

:class:`Serializer` provides deterministic serialization (JSON and dict) for
the Context tree.  Deterministic means that identical trees always produce
byte-for-byte identical output, which is necessary for diffing and caching.
"""

from __future__ import annotations

import json
from typing import Any

from context_compiler.ast.nodes import _node_from_dict, Node


class Serializer:
    """
    Serializes and deserializes the Context tree.

    Attributes
    ----------
    version:
        Format version written into every serialized envelope.  Consumers
        should reject envelopes with unknown versions.
    """

    CURRENT_VERSION: str = "1"

    def __init__(self, version: str = CURRENT_VERSION) -> None:
        self.version: str = version

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self, node: Node) -> dict[str, Any]:
        """
        Serialize *node* to a versioned plain-dict envelope.

        The envelope schema is::

            {
                "version": "<str>",
                "node": { <node dict> }
            }
        """
        return {
            "version": self.version,
            "node": node.to_dict(),
        }

    def to_json(self, node: Node, *, indent: int | None = 2) -> str:
        """
        Serialize *node* to a deterministic JSON string.

        Keys are sorted at every level so that identical trees always produce
        identical byte sequences.
        """
        return json.dumps(self.to_dict(node), indent=indent, sort_keys=True)

    # ------------------------------------------------------------------
    # Deserialization
    # ------------------------------------------------------------------

    def from_dict(self, data: dict[str, Any]) -> Node:
        """
        Deserialize a node from a versioned plain-dict envelope.

        Raises
        ------
        ValueError
            If the envelope version does not match :attr:`version`.
        """
        envelope_version = data.get("version", "")
        if envelope_version != self.version:
            raise ValueError(
                f"Serializer version mismatch: expected {self.version!r}, "
                f"got {envelope_version!r}"
            )
        return _node_from_dict(data["node"])

    def from_json(self, json_str: str) -> Node:
        """Deserialize a node from a JSON string produced by :meth:`to_json`."""
        return self.from_dict(json.loads(json_str))
