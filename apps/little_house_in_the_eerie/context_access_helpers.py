from __future__ import annotations

from enum import Enum, auto
from typing import Any

from context_resolver.ast.nodes import MappingNode, Node, ScalarNode, SequenceNode, _node_from_dict
from context_resolver.ast.paths import Path
from context_resolver.ast.resolvable_node import ResolvableNode, ResolvableNodeState
from context_resolver.context.context import Context
from context_resolver.templates.template import JSONOutputTemplate


def fmt(node: Any) -> str:
    """Return a display string for *node* without triggering resolution."""
    if node is None:
        return "<not found>"
    if isinstance(node, ResolvableNode):
        if node.resolution_state is ResolvableNodeState.RESOLVED:
            return fmt(node.result)
        return f"PromptNode({node.resolution_state.name})"
    if isinstance(node, ScalarNode):
        return str(node.value) if node.value is not None else "<unspecified>"
    if isinstance(node, SequenceNode):
        items = [fmt(item) for item in node]
        return "[" + ", ".join(items) + "]" if items else "[]"
    if isinstance(node, MappingNode):
        parts = ", ".join(f"{k}: {fmt(v)}" for k, v in node.items())
        return "{" + parts + "}"
    return repr(node)


def scalar(ctx: Context, *segments: str) -> Any:
    """Query a ScalarNode and return its raw Python value."""
    node = ctx.query(Path(*segments))
    return node.value if isinstance(node, ScalarNode) else None


def set_keyed_node(ctx: Context, node: Node, *segments: str) -> None:
    """Set a node using Context.set so caches are invalidated."""
    if not segments:
        raise ValueError("set_keyed_node requires at least one path segment")
    ctx.set(Path(*segments), node)


def materialize_node(node: Node) -> Node:
    """Return a detached snapshot of a node's resolved value when available."""
    source: Node
    if isinstance(node, ResolvableNode) and node.result is not None:
        source = node.result
    else:
        source = node
    return _node_from_dict(source.to_dict())


def invalidate(ctx: Context, *segments: str) -> None:
    """Force a ResolvableNode to be regenerated on next query."""
    node = ctx.query(Path(*segments))
    if isinstance(node, ResolvableNode):
        node.mark_stale()


def set_not_hint(ctx: Context, not_hint: str) -> None:
    set_keyed_node(ctx, ScalarNode(not_hint), "interview", "but_not_hint")


def get_not_hint(ctx: Context) -> str:
    return ctx.query_text("interview", "but_not_hint")


def append_not_hint(ctx: Context, not_hint: str) -> None:
    old_hint = get_not_hint(ctx)
    set_not_hint(ctx, old_hint + "\n" + not_hint)


def confirm_and_freeze(
    ctx: Context,
    source_segments: tuple[str, ...],
    target_segments: tuple[str, ...],
    *,
    clear_not_hint: bool = True,
) -> Node:
    """Resolve source node, store a detached snapshot at target, and return it."""
    node = ctx.query(Path(*source_segments))
    frozen = materialize_node(node)
    set_keyed_node(ctx, frozen, *target_segments)
    if clear_not_hint:
        set_not_hint(ctx, "")
    return frozen


def append_sequence_node(ctx: Context, segments: tuple[str, ...], node: Node) -> int:
    """Append a detached node to a sequence path, creating the sequence if needed."""
    current = ctx.query(Path(*segments))
    items: list[Node] = []
    if isinstance(current, SequenceNode):
        items = [materialize_node(item) for item in current]
    elif isinstance(current, ScalarNode) and current.value is None:
        items = []
    else:
        items = [materialize_node(current)]
    items.append(materialize_node(node))
    set_keyed_node(ctx, SequenceNode(items), *segments)
    return len(items)


class NameMode(Enum):
    FIRST = auto()
    FORMAL = auto()
    LAST = auto()
    FULL = auto()


def query_investigator_name(ctx: Context, mode: NameMode, *segments: str) -> str:
    if len(segments) == 0:
        segments = ("investigator", "name")
    if mode == NameMode.FIRST:
        return ctx.query_text(*segments, "first")
    if mode == NameMode.FORMAL:
        honorific = ctx.query_text(*segments, "honorific").strip()
        if not honorific:
            first = ctx.query_text(*segments, "first").strip()
            honorific = first[0] if first else ""
        return (honorific + ctx.query_text(*segments, "last")).strip()
    if mode == NameMode.LAST:
        return ctx.query_text(*segments, "last")
    full_name = (
        f"{ctx.query_text(*segments, 'honorific')} "
        f"{ctx.query_text(*segments, 'first')} "
        f"{ctx.query_text(*segments, 'middle')} "
        f"{ctx.query_text(*segments, 'last')} "
        f"{ctx.query_text(*segments, 'suffix')}"
    ).strip()
    return full_name


def query_town(ctx: Context, *segments: str) -> str:
    if len(segments) == 0:
        segments = ("town",)
    return (
        f"Name: {ctx.query_text(*segments, 'name')}\n"
        f"Location: {ctx.query_text(*segments, 'location')}\n"
        f"Economics: {ctx.query_text(*segments, 'economic')}\n"
        f"Backstory: {ctx.query_text(*segments, 'backstory')}\n"
    )


def query_town_short(ctx: Context, *segments: str) -> str:
    if len(segments) == 0:
        segments = ("town",)
    return (
        f"{ctx.query_text(*segments, 'name')}, "
        f"{ctx.query_text(*segments, 'location')}"
    )


def query_newspaper(ctx: Context) -> str:
    segments = ("interview", "newspaper")
    return (
        f"{ctx.query_text(*segments, 'title')}\n"
        f"{ctx.query_text(*segments, 'publisher')} "
        f"{ctx.query_text(*segments, 'circulation')} "
        f"{ctx.query_text(*segments, 'date')} "
    )


def query_case(ctx: Context, *segments: str) -> str:
    if len(segments) == 0:
        segments = ("interview", "case1")
    return (
        f"Case: {ctx.query_text(*segments, 'name')}\n"
        f"Summary: {ctx.query_text(*segments, 'description')}"
    )


def query_advantage(ctx: Context, *segments: str) -> str:
    return (
        f"Item: {ctx.query_text(*segments, 'book_or_equipment')}\n"
        f"Skill: {ctx.query_text(*segments, 'implied_skill')}\n"
        f"Advantage: {ctx.query_text(*segments, 'advantage')}\n"
    )


def simple_schema(name: str, field_name: str, template_str: str) -> tuple[JSONOutputTemplate, Any]:
    from context_resolver.ast.schema import FieldSpec, Schema

    key = name.replace(" ", "")
    schema = Schema(
        name=key,
        fields=[
            FieldSpec(
                name=field_name,
                type="str",
                required=True,
                description=f"A {name}.",
            )
        ],
    )
    template = JSONOutputTemplate(
        name=key,
        template_str=template_str,
        schema=schema,
        description=f"A generated {name}.",
    )
    return (template, schema)

def function_schema(name: str, field_name: str, function: str, arglist: [str]) -> tuple[JSONOutputFunction, Any]:
    from context_resolver.ast.schema import FieldSpec, Schema

    key = name.replace(" ", "")
    schema = Schema(
        name=key,
        fields=[
            FieldSpec(
                name=field_name,
                type="str",
                required=True,
                description=f"A {name}.",
            )
        ],
    )
    fn = JSONOutputFunction(
        name=key,
        arglist=arglist,
        python_fn=function,
        schema=schema,
        description=f"A computation of {name}.",
    )
    return (fn, schema)


def template_schema_tuple(template_str: str, schema: Any, description: str) -> tuple[JSONOutputTemplate, Any]:
    template = JSONOutputTemplate(
        name=schema.name,
        template_str=template_str,
        schema=schema,
        description=description,
    )
    return (template, schema)


def meta_data(temp: float, name: str) -> dict[str, Any]:
    # Provider-level defaults still apply when max_tokens is omitted, so set
    # per-template budgets explicitly where longer structured outputs are expected.
    max_tokens_by_name = {
        "investigator_arch": 24,
        "investigator_name": 80,
        "brochure": 512,
        "interest": 256,
    }

    extra: dict[str, Any] = {
        "seed": -1,
        "top_p": 0.98,
        "top_k": 80,
        "min_p": 0.05,
        "repeat_penalty": 1.12,
    }
    if name in max_tokens_by_name:
        extra["max_tokens"] = max_tokens_by_name[name]

    return {
        "temperature": temp,
        "extra": extra,
    }
