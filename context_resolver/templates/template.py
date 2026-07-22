"""
Prompt template definitions.

A :class:`Template` maps variable bindings to a rendered prompt string.
Templates are intentionally kept separate from the compiler so that prompt
engineering decisions are not coupled to compilation logic.

The default implementation uses Python's built-in :meth:`str.format_map` for
substitution, which is simple and deterministic.  Future implementations might
support Jinja2, Mustache, or structured chat-message templates.

Built-in template types
-----------------------
* :class:`Template` – the base class; renders a format string.
* :class:`JSONOutputTemplate` – extends :class:`Template` by appending a
  structured JSON output instruction block derived from a
  :class:`~context_resolver.ast.schema.Schema`.  Use this with LLM providers
  so the model always knows exactly which JSON fields to return.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context_resolver.ast.schema import Schema


class Template:
    """
    A named, parameterized prompt template.

    Parameters
    ----------
    name:
        A unique identifier for this template (must match the
        :attr:`~context_resolver.ast.resolvable_node.ResolvableNode.template_ref`).
    template_str:
        A Python format string.  Variables are referenced as ``{variable_name}``.
    description:
        Human-readable description of what this template does.

    Examples
    --------
    >>> t = Template("greet", "Say hello to {name} in {language}.")
    >>> t.render({"name": "Alice", "language": "French"})
    'Say hello to Alice in French.'
    """

    def __init__(
        self, name: str, template_str: str, description: str = ""
    ) -> None:
        self.name: str = name
        self.template_str: str = template_str
        self.description: str = description

    def render(self, bindings: dict[str, Any]) -> str:
        """
        Render this template with *bindings* and return the prompt string.

        Parameters
        ----------
        bindings:
            A mapping from variable names to their resolved values.

        Returns
        -------
        str
            The rendered prompt.

        Raises
        ------
        KeyError
            If a required variable is missing from *bindings*.
        """
        try:
            return self.template_str.format_map(bindings)
        except KeyError as exc:
            raise KeyError(
                f"Template '{self.name}' requires variable {exc} "
                f"which is not present in bindings: {list(bindings.keys())}"
            ) from exc

    def to_dict(self) -> dict[str, str]:
        """Serialize this template to a plain dictionary."""
        return {
            "name": self.name,
            "template_str": self.template_str,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "Template":
        """Deserialize a :class:`Template` from a plain dictionary."""
        return cls(
            name=data["name"],
            template_str=data["template_str"],
            description=data.get("description", ""),
        )

    def __repr__(self) -> str:
        return f"Template(name={self.name!r})"


class TemplateRegistry:
    """
    A registry of named :class:`Template` objects.

    The compiler looks up templates by name via :meth:`get`.  Templates can be
    registered at startup or loaded from a file.
    """

    def __init__(self) -> None:
        self._templates: dict[str, Template] = {}

    def register(self, template: Template) -> None:
        """Add *template* to the registry."""
        self._templates[template.name] = template

    def get(self, name: str) -> Template | None:
        """Return the template with the given *name*, or ``None``."""
        return self._templates.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._templates

    def __len__(self) -> int:
        return len(self._templates)

    def __repr__(self) -> str:
        return f"TemplateRegistry(templates={list(self._templates.keys())})"


class JSONOutputTemplate(Template):
    """
    A :class:`Template` that automatically appends a JSON output instruction
    block to the rendered prompt.

    Many LLMs need an explicit instruction describing which JSON keys to emit
    and their expected types.  :class:`JSONOutputTemplate` generates that
    instruction automatically from a :class:`~context_resolver.ast.schema.Schema`
    so you don't have to repeat field names and types inside the template string
    itself.

    The appended block looks like::

        Respond with a valid JSON object containing the following fields:
          - "greeting" (str, required): A greeting message for the player
          - "opening" (str, optional): An opening line that sets the scene
        Output only the JSON object with no additional text or markdown.

    Parameters
    ----------
    name:
        Unique template identifier (must match the node's ``template_ref``).
    template_str:
        A Python format string for the main part of the prompt.  Write the
        task-specific instructions here; the JSON output block is added
        automatically.
    schema:
        The :class:`~context_resolver.ast.schema.Schema` that describes the
        expected response.  Each field in the schema becomes a bullet in the
        output instruction.
    description:
        Human-readable description of what this template does.

    Examples
    --------
    >>> from context_resolver.ast.schema import Schema, FieldSpec
    >>> schema = Schema("Out", fields=[FieldSpec("greeting", "str", description="A hello")])
    >>> t = JSONOutputTemplate("greet", "Say hello to {name}.", schema=schema)
    >>> print(t.render({"name": "Alice"}))
    Say hello to Alice.
    <BLANKLINE>
    Respond with a valid JSON object containing the following fields:
      - "greeting" (str, required): A hello
    Output only the JSON object with no additional text or markdown.
    """

    def __init__(
        self,
        name: str,
        template_str: str,
        schema: "Schema",
        description: str = "",
    ) -> None:
        super().__init__(name, template_str, description)
        self._schema = schema

    @property
    def schema(self) -> "Schema":
        """The output schema used to generate the JSON instruction block."""
        return self._schema

    def render(self, bindings: dict[str, Any]) -> str:
        """
        Render the base template and append JSON output instructions.

        Parameters
        ----------
        bindings:
            A mapping from variable names to their resolved values.

        Returns
        -------
        str
            The rendered prompt with the JSON output instruction appended.
        """
        base = super().render(bindings)
        instructions = self._build_json_instructions()
        return f"{base}\n\n{instructions}"

    def _build_json_instructions(self) -> str:
        """Build the JSON output instruction block from the schema fields."""
        lines = [
            "Respond with a valid JSON object containing the following fields:"
        ]
        for field_spec in self._schema.fields:
            req_label = "required" if field_spec.required else "optional"
            desc_suffix = f": {field_spec.description}" if field_spec.description else ""
            lines.append(
                f'  - "{field_spec.name}" ({field_spec.type}, {req_label}){desc_suffix}'
            )
        lines.append(
            "Output only the JSON object with no additional text or markdown."
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"JSONOutputTemplate(name={self.name!r}, schema={self._schema.name!r})"
