"""
Prompt template definitions.

A :class:`Template` maps variable bindings to a rendered prompt string.
Templates are intentionally kept separate from the compiler so that prompt
engineering decisions are not coupled to compilation logic.

The default implementation uses Python's built-in :meth:`str.format_map` for
substitution, which is simple and deterministic.  Future implementations might
support Jinja2, Mustache, or structured chat-message templates.
"""

from __future__ import annotations

from typing import Any


class Template:
    """
    A named, parameterized prompt template.

    Parameters
    ----------
    name:
        A unique identifier for this template (must match the
        :attr:`~context_compiler.ast.prompt_node.PromptNode.template_ref`).
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
