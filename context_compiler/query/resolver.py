"""
The Context Resolver.

:class:`Resolver` is the central engine that orchestrates query passes
and drives resolution for unresolved
:class:`~context_compiler.ast.resolvable_node.ResolvableNode` instances.

The resolver is intentionally **lazy**: it only resolves nodes that are
demanded by a :meth:`~context_compiler.context.context.Context.query` call.
It never eagerly traverses the entire tree.

The resolver depends **only** on the
:class:`~context_compiler.inference.strategy.ResolutionStrategy` interface
(via :class:`~context_compiler.query.passes.ResolutionPass`).
It has no knowledge of LLMs, prompts, or any specific resolution provider.

Resolution algorithm (per query)
---------------------------------
1. Locate the node at the requested path in the Context tree.
2. If the node is already ``FULLY_SPECIFIED``, return it immediately.
3. Otherwise:

   a. Run all registered :class:`~context_compiler.query.passes.DeterministicPass` objects.
   b. If the node is now ``FULLY_SPECIFIED``, cache and return it.
   c. If the node is a :class:`~context_compiler.ast.resolvable_node.ResolvableNode`:

      * Resolve all dependency paths (recursively, via query).
      * Build the resolution request from the template and bindings.
      * Call the :class:`~context_compiler.inference.strategy.ResolutionStrategy`.
      * Decode the response into typed nodes.
      * Validate the decoded nodes against the output schema.
      * Mark the ResolvableNode as ``RESOLVED`` and cache its result.
   d. Run constraint validation passes.
4. Return the requested node.
"""

from __future__ import annotations

import logging
from typing import Any

from context_compiler.ast.nodes import Node, NodeState, MappingNode, SequenceNode, _node_from_dict
from context_compiler.ast.paths import Path
from context_compiler.ast.resolvable_node import ResolvableNode, ResolvableNodeState
from context_compiler.ast.schema import SchemaValidationError
from context_compiler.query.dependency_graph import DependencyGraph, CycleError
from context_compiler.query.passes import QueryPass, DeterministicPass, ResolutionPass, PassContext
from context_compiler.inference.provider import ResolutionRequest, ResolutionResult
from context_compiler.templates.template import Template, TemplateRegistry

logger = logging.getLogger(__name__)


class ResolutionError(Exception):
    """Raised when query-based resolution fails for any reason."""


class Resolver:
    """
    The demand-driven, incremental context resolver.

    The resolver is strategy-agnostic: it depends only on the
    :class:`~context_compiler.inference.strategy.ResolutionStrategy` interface
    (via :class:`~context_compiler.query.passes.ResolutionPass`) and never
    imports any concrete provider or strategy implementation.

    Parameters
    ----------
    template_registry:
        Registry of named :class:`~context_compiler.templates.template.Template` objects.
    passes:
        Ordered list of :class:`~context_compiler.query.passes.QueryPass` objects
        to run during each resolution cycle.
    """

    def __init__(
        self,
        template_registry: TemplateRegistry | None = None,
        passes: list[QueryPass] | None = None,
    ) -> None:
        self._template_registry: TemplateRegistry = (
            template_registry or TemplateRegistry()
        )
        self._passes: list[QueryPass] = list(passes or [])
        self._dependency_graph: DependencyGraph = DependencyGraph()
        # Track which paths are currently being resolved to detect cycles.
        self._in_progress: set[Path] = set()

    # ------------------------------------------------------------------
    # Pass management
    # ------------------------------------------------------------------

    def add_pass(self, query_pass: QueryPass) -> None:
        """Append *query_pass* to the end of the pass pipeline."""
        self._passes.append(query_pass)

    @property
    def dependency_graph(self) -> DependencyGraph:
        """The live dependency graph maintained by this resolver."""
        return self._dependency_graph

    # ------------------------------------------------------------------
    # Core resolution
    # ------------------------------------------------------------------

    def resolve_node(self, node: Node, path: Path, root: Node) -> Node:
        """
        Resolve *node* at *path* against the Context *root*.

        Returns the node after all applicable passes have been applied.
        The caller is responsible for storing the result back into the tree.

        Parameters
        ----------
        node:
            The node to resolve.
        path:
            The path of *node* within the Context tree.
        root:
            The root node of the Context tree (used for resolving bindings).

        Raises
        ------
        ResolutionError
            If resolution fails due to a missing template, provider error,
            schema validation failure, or cycle.
        CycleError
            If a cyclic ResolvableNode dependency is detected.
        """
        if node.is_fully_specified:
            return node

        # Cycle guard
        if path in self._in_progress:
            cycle = list(self._in_progress) + [path]
            raise CycleError(cycle)
        self._in_progress.add(path)

        try:
            return self._resolve_node_inner(node, path, root)
        finally:
            self._in_progress.discard(path)

    def _resolve_node_inner(self, node: Node, path: Path, root: Node) -> Node:
        # 1. Run deterministic passes.
        pass_ctx = PassContext(root)
        for p in self._passes:
            if isinstance(p, DeterministicPass):
                p.run(pass_ctx)

        if node.is_fully_specified:
            return node

        # 2. Handle ResolvableNodes specially.
        if isinstance(node, ResolvableNode):
            return self._resolve_resolvable_node(node, path, root)

        return node

    def _resolve_resolvable_node(
        self, node: ResolvableNode, path: Path, root: Node
    ) -> ResolvableNode:
        """Drive resolution for a single ResolvableNode."""
        # Find the resolution pass (first one wins).
        resolution_pass: ResolutionPass | None = None
        for p in self._passes:
            if isinstance(p, ResolutionPass):
                resolution_pass = p
                break

        if resolution_pass is None:
            raise ResolutionError(
                f"No ResolutionPass configured; cannot resolve ResolvableNode at {path}"
            )

        strategy = resolution_pass.strategy

        # Resolve dependencies.
        bound_values: dict[str, Any] = {}
        for var_name, dep_path in node.input_bindings.items():
            self._dependency_graph.add_dependency(path, dep_path)
            dep_node = _resolve_path(root, dep_path)
            if dep_node is None:
                raise ResolutionError(
                    f"ResolvableNode at {path}: binding '{var_name}' references "
                    f"unknown path {dep_path}"
                )
            # Recursively resolve if needed.
            if not dep_node.is_fully_specified:
                dep_node = self.resolve_node(dep_node, dep_path, root)
            bound_values[var_name] = _extract_scalar(dep_node)

        # Look up and render the template.
        template = self._template_registry.get(node.template_ref)
        if template is None:
            raise ResolutionError(
                f"Template {node.template_ref!r} not found in registry"
            )

        rendered_prompt = template.render(bound_values)

        # Build schema hint.
        output_schema_dict = (
            node.output_schema.to_json_schema()
            if node.output_schema is not None
            else None
        )

        request = ResolutionRequest(
            prompt=rendered_prompt,
            output_schema=output_schema_dict,
            query_path=path,
            dependencies=node.effective_dependencies(),
        )

        # Call the strategy (which delegates to a provider).
        try:
            result: ResolutionResult = strategy.resolve(request)
        except Exception as exc:
            node.mark_error(exc)
            raise ResolutionError(
                f"Resolution failed for ResolvableNode at {path}: {exc}"
            ) from exc

        # Validate the response against the schema.
        if node.output_schema is not None:
            try:
                node.output_schema.validate(result.data)
            except SchemaValidationError as exc:
                node.mark_error(exc)
                raise ResolutionError(
                    f"Schema validation failed for ResolvableNode at {path}: {exc}"
                ) from exc

        # Decode response data into a typed node.
        result_node = _decode_response(result.data)

        node.mark_resolved(
            result_node,
            provider=result.provider,
            model=result.model,
        )
        logger.debug(
            "Resolved ResolvableNode at %s via provider=%s model=%s",
            path,
            result.provider,
            result.model,
        )
        return node


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_path(root: Node, path: Path) -> Node | None:
    """Walk *root* following *path* and return the node found, or ``None``."""
    current: Node = root
    for segment in path.segments:
        if isinstance(segment, str) and isinstance(current, MappingNode):
            child = current.get(segment)
            if child is None:
                return None
            current = child
        elif isinstance(segment, int) and isinstance(current, SequenceNode):
            child = current.get(segment)
            if child is None:
                return None
            current = child
        else:
            return None
    return current


def _extract_scalar(node: Node) -> Any:
    """
    Extract a plain Python value from a fully-specified node.

    For a :class:`~context_compiler.ast.nodes.ScalarNode` this is just
    :attr:`~context_compiler.ast.nodes.ScalarNode.value`.
    For a resolved :class:`~context_compiler.ast.resolvable_node.ResolvableNode`
    this recursively extracts the result.
    For compound nodes a dict/list is returned.
    """
    from context_compiler.ast.nodes import ScalarNode, MappingNode, SequenceNode

    if isinstance(node, ResolvableNode) and node.result is not None:
        return _extract_scalar(node.result)
    if isinstance(node, ScalarNode):
        return node.value
    if isinstance(node, MappingNode):
        return {k: _extract_scalar(v) for k, v in node.items()}
    if isinstance(node, SequenceNode):
        return [_extract_scalar(item) for item in node]
    return None


def _decode_response(data: dict[str, Any]) -> Node:
    """
    Decode a plain dict returned by a resolution provider into typed nodes.

    Scalar values become :class:`~context_compiler.ast.nodes.ScalarNode` instances.
    Nested dicts become :class:`~context_compiler.ast.nodes.MappingNode` instances.
    Lists become :class:`~context_compiler.ast.nodes.SequenceNode` instances.
    """
    from context_compiler.ast.nodes import ScalarNode, MappingNode, SequenceNode

    def _decode_value(value: Any) -> Node:
        if isinstance(value, dict):
            fields = {k: _decode_value(v) for k, v in value.items()}
            return MappingNode(fields)
        if isinstance(value, list):
            items = [_decode_value(item) for item in value]
            return SequenceNode(items)
        return ScalarNode(value)

    return _decode_value(data)
