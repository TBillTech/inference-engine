"""
Schema definitions for validating and decoding structured data returned by
inference providers.

A :class:`Schema` describes the expected shape of a piece of semantic data.
The compiler uses schemas to:

* validate inference output before it enters the Context tree
* decode raw dictionaries into typed :class:`~context_resolver.ast.nodes.Node`
  instances
* generate output-format hints for inference providers (e.g. JSON Schema)

Design notes
------------
Schemas are intentionally kept simple at this stage.  Future passes can enrich
them with coercions, cross-field constraints, or references to other schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Field specification
# ---------------------------------------------------------------------------


@dataclass
class FieldSpec:
    """
    Describes a single field in a :class:`Schema`.

    Attributes
    ----------
    name:
        The field name as it appears in the decoded mapping.
    type:
        A string tag describing the expected Python type
        (``"str"``, ``"int"``, ``"float"``, ``"bool"``, ``"list"``,
        ``"mapping"``, or the name of a nested schema).
    required:
        If ``True``, a :class:`SchemaValidationError` is raised when the field
        is absent from the raw data.
    description:
        Human-readable description for documentation and prompt generation.
    """

    name: str
    type: str = "str"
    required: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize this field spec to a plain dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldSpec":
        """Deserialize a :class:`FieldSpec` from a plain dictionary."""
        return cls(
            name=data["name"],
            type=data.get("type", "str"),
            required=data.get("required", True),
            description=data.get("description", ""),
        )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class SchemaValidationError(Exception):
    """Raised when raw data does not conform to a :class:`Schema`."""


@dataclass
class Schema:
    """
    Describes the expected structure of inference output.

    Attributes
    ----------
    name:
        A human-readable name for this schema (used in error messages and
        documentation).
    fields:
        Ordered list of :class:`FieldSpec` entries that define the expected
        mapping keys.
    description:
        Optional human-readable description of what this schema represents.
    version:
        Schema version string for compatibility tracking.
    """

    name: str
    fields: list[FieldSpec] = field(default_factory=list)
    description: str = ""
    version: str = "1.0"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, data: dict[str, Any]) -> None:
        """
        Validate *data* against this schema.

        Raises
        ------
        SchemaValidationError
            If a required field is missing or a field value has the wrong type.
        """
        if not isinstance(data, dict):
            raise SchemaValidationError(
                f"Schema '{self.name}' expected a dict, got {type(data).__name__}"
            )
        for spec in self.fields:
            if spec.name not in data:
                if spec.required:
                    raise SchemaValidationError(
                        f"Schema '{self.name}': required field '{spec.name}' is missing"
                    )
                continue
            value = data[spec.name]
            self._check_type(spec, value)

    def _check_type(self, spec: FieldSpec, value: Any) -> None:
        """Raise :class:`SchemaValidationError` if *value* has wrong type."""
        expected = spec.type
        type_map: dict[str, type | tuple[type, ...]] = {
            "str": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "mapping": dict,
        }
        if expected in type_map:
            allowed = type_map[expected]
            if not isinstance(value, allowed):
                raise SchemaValidationError(
                    f"Schema '{self.name}': field '{spec.name}' expected type "
                    f"'{expected}', got {type(value).__name__!r}"
                )

    # ------------------------------------------------------------------
    # JSON Schema generation (minimal stub)
    # ------------------------------------------------------------------

    def to_json_schema(self) -> dict[str, Any]:
        """
        Return a minimal JSON Schema dict describing this schema.

        This stub can be extended to generate full JSON Schema objects for
        passing to inference providers that support structured output.
        """
        properties: dict[str, Any] = {}
        required_fields: list[str] = []

        type_map = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "mapping": "object",
        }

        for spec in self.fields:
            json_type = type_map.get(spec.type, "string")
            prop: dict[str, Any] = {"type": json_type}
            if spec.description:
                prop["description"] = spec.description
            properties[spec.name] = prop
            if spec.required:
                required_fields.append(spec.name)

        schema: dict[str, Any] = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": self.name,
            "type": "object",
            "properties": properties,
        }
        if required_fields:
            schema["required"] = required_fields
        if self.description:
            schema["description"] = self.description
        return schema

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize this schema to a plain dictionary."""
        return {
            "name": self.name,
            "fields": [f.to_dict() for f in self.fields],
            "description": self.description,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Schema":
        """Deserialize a :class:`Schema` from a plain dictionary."""
        return cls(
            name=data["name"],
            fields=[FieldSpec.from_dict(f) for f in data.get("fields", [])],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
        )
