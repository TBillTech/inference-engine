"""Templates sub-package: prompt template definitions and registry."""

from context_resolver.templates.template import (
    Template,
    TemplateRegistry,
    JSONOutputTemplate,
    JSONOutputFunction,
)

__all__ = ["Template", "TemplateRegistry", "JSONOutputTemplate", "JSONOutputFunction"]
