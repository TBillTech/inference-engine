"""Tests for context_resolver.ast.schema."""

import pytest

from context_resolver.ast.schema import Schema, FieldSpec, SchemaValidationError


@pytest.fixture
def person_schema():
    return Schema(
        name="Person",
        fields=[
            FieldSpec(name="name", type="str", required=True),
            FieldSpec(name="age", type="int", required=True),
            FieldSpec(name="nickname", type="str", required=False),
        ],
    )


class TestFieldSpec:
    def test_defaults(self):
        f = FieldSpec(name="x")
        assert f.type == "str"
        assert f.required is True
        assert f.description == ""

    def test_serialization_roundtrip(self):
        f = FieldSpec(name="score", type="float", required=False, description="A score")
        restored = FieldSpec.from_dict(f.to_dict())
        assert restored.name == "score"
        assert restored.type == "float"
        assert restored.required is False
        assert restored.description == "A score"

    def test_nested_schema_type_roundtrip(self):
        address = Schema(
            name="Address",
            fields=[FieldSpec(name="city", type="str", required=True)],
        )
        f = FieldSpec(name="address", type=address, required=True)

        restored = FieldSpec.from_dict(f.to_dict())
        assert isinstance(restored.type, Schema)
        assert restored.type.name == "Address"
        assert restored.type.fields[0].name == "city"


class TestSchemaValidation:
    def test_valid_data_passes(self, person_schema):
        person_schema.validate({"name": "Alice", "age": 30})

    def test_missing_required_field_raises(self, person_schema):
        with pytest.raises(SchemaValidationError, match="required field 'name'"):
            person_schema.validate({"age": 30})

    def test_optional_field_absence_is_ok(self, person_schema):
        person_schema.validate({"name": "Bob", "age": 25})

    def test_wrong_type_raises(self, person_schema):
        with pytest.raises(SchemaValidationError, match="expected type 'int'"):
            person_schema.validate({"name": "Alice", "age": "thirty"})

    def test_non_dict_raises(self, person_schema):
        with pytest.raises(SchemaValidationError, match="expected a dict"):
            person_schema.validate("not a dict")  # type: ignore[arg-type]

    def test_float_accepts_int(self):
        schema = Schema("S", fields=[FieldSpec("score", type="float")])
        schema.validate({"score": 42})  # int is acceptable for float

    def test_bool_type_check(self):
        schema = Schema("S", fields=[FieldSpec("flag", type="bool")])
        schema.validate({"flag": True})
        with pytest.raises(SchemaValidationError):
            schema.validate({"flag": "yes"})

    def test_nested_schema_validation(self):
        address_schema = Schema(
            "Address",
            fields=[FieldSpec("city", type="str", required=True)],
        )
        person_schema = Schema(
            "Person",
            fields=[FieldSpec("address", type=address_schema, required=True)],
        )

        person_schema.validate({"address": {"city": "Salem"}})
        with pytest.raises(SchemaValidationError):
            person_schema.validate({"address": "not-a-dict"})


class TestJsonSchemaGeneration:
    def test_generates_valid_structure(self, person_schema):
        js = person_schema.to_json_schema()
        assert js["type"] == "object"
        assert "name" in js["properties"]
        assert js["properties"]["name"]["type"] == "string"
        assert "name" in js["required"]
        assert "nickname" not in js["required"]

    def test_description_included(self):
        schema = Schema(
            "S",
            fields=[FieldSpec("x", type="str", description="An x value")],
            description="Top-level description",
        )
        js = schema.to_json_schema()
        assert js["description"] == "Top-level description"
        assert js["properties"]["x"]["description"] == "An x value"

    def test_nested_schema_generates_nested_object_properties(self):
        profile = Schema(
            "Profile",
            fields=[FieldSpec("bio", type="str", required=True)],
            description="Profile data",
        )
        schema = Schema(
            "User",
            fields=[FieldSpec("profile", type=profile, required=True)],
        )

        js = schema.to_json_schema()
        assert js["properties"]["profile"]["type"] == "object"
        assert "bio" in js["properties"]["profile"]["properties"]
        assert "bio" in js["properties"]["profile"]["required"]


class TestSchemaSerialization:
    def test_roundtrip(self, person_schema):
        data = person_schema.to_dict()
        restored = Schema.from_dict(data)
        assert restored.name == "Person"
        assert len(restored.fields) == 3
        assert restored.fields[0].name == "name"
