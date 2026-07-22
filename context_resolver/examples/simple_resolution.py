"""
End-to-end example: lazy resolution with a mock resolution provider.

This script demonstrates:

* Building a Context tree with an unresolved ResolvableNode.
* Registering a Template.
* Registering a MockProvider.
* Querying the Context – which triggers lazy resolution.
* Inspecting the resolved ResolvableNode.
* Mutating a dependency and observing that the cache is invalidated.

Run with::

    python -m context_resolver.examples.simple_resolution
"""

from __future__ import annotations

from context_resolver.ast.nodes import MappingNode, ScalarNode
from context_resolver.ast.paths import Path
from context_resolver.ast.resolvable_node import ResolvableNode
from context_resolver.ast.schema import Schema, FieldSpec
from context_resolver.query.resolver import Resolver
from context_resolver.query.passes import ResolutionPass
from context_resolver.context.context import Context
from context_resolver.inference.mock_provider import MockProvider
from context_resolver.templates.template import Template, TemplateRegistry


def build_example_context() -> Context:
    """
    Construct a Context tree with the following structure::

        root
        ├── player
        │   ├── name   (ScalarNode: "Alice")
        │   └── intro  (ResolvableNode: template="greet", input_bindings={"name": Path("player","name")})
        └── world
            └── setting  (ScalarNode: "a fantasy kingdom")

    The ``intro`` node is initially underspecified.  Querying it will
    trigger the mock resolution provider to produce a greeting.
    """
    # --- Schema for the intro output ---
    intro_schema = Schema(
        name="IntroOutput",
        fields=[
            FieldSpec(name="greeting", type="str", required=True,
                      description="A greeting message for the player"),
            FieldSpec(
                name="opening",
                type="str",
                required=True,
                description="An opening line that sets the scene",
            ),
        ],
        description="Expected output from the intro template",
    )

    # --- ResolvableNode that needs to be resolved ---
    intro_node = ResolvableNode(
        template_ref="greet",
        input_bindings={
            "name": Path("player", "name"),
            "setting": Path("world", "setting"),
        },
        output_schema=intro_schema,
        dependencies=[Path("player", "name"), Path("world", "setting")],
    )

    # --- Build the tree ---
    root = MappingNode({
        "player": MappingNode({
            "name": ScalarNode("Alice"),
            "intro": intro_node,
        }),
        "world": MappingNode({
            "setting": ScalarNode("a fantasy kingdom"),
        }),
    })

    # --- Template registry ---
    registry = TemplateRegistry()
    registry.register(Template(
        name="greet",
        template_str=(
            "You are a hero in the great Kingdom of {setting}. "
            "Greet the hero named {name}."
        ),
        description="Generates an intro for the player",
    ))

    # --- Mock provider returns a canned response ---
    rendered_prompt = (
        "You are a hero in the great Kingdom of a fantasy kingdom. "
        "Greet the hero named Alice."
    )
    mock_provider = MockProvider(
        responses={
            rendered_prompt: {
                "greeting": "Hail, brave Alice! Welcome to the kingdom!",
                "opening": "You are a hero in the great Kingdom of a fantasy kingdom.",
            }
        }
    )

    # --- Resolver pipeline ---
    resolver = Resolver(
        template_registry=registry,
        passes=[ResolutionPass(mock_provider)],
    )

    return Context(root=root, resolver=resolver)


def main() -> None:
    """Run the end-to-end demonstration."""
    print("=" * 60)
    print("Context Compiler – Simple Query Resolution Example")
    print("=" * 60)

    ctx = build_example_context()

    # 1. Query a fully-specified scalar – no resolution needed.
    print("\n[1] Querying player.name (already fully specified)...")
    name_node = ctx.query(Path("player", "name"))
    print(f"    → {name_node!r}")

    # 2. Query the ResolvableNode – triggers lazy resolution.
    print("\n[2] Querying player.intro (triggers resolution)...")
    intro_node = ctx.query(Path("player", "intro"))
    print(f"    → {intro_node!r}")

    assert hasattr(intro_node, "result"), "Expected a ResolvableNode"
    result = intro_node.result  # type: ignore[union-attr]
    greeting_text = result.get("greeting").value  # type: ignore[union-attr]
    opening_text = result.get("opening").value  # type: ignore[union-attr]
    print(f"    Resolved greeting: {greeting_text!r}")
    print(f"    Resolved opening: {opening_text!r}")
    print(f"    Provider: {intro_node.provider!r}")  # type: ignore[union-attr]
    print(f"    Resolved at: {intro_node.resolved_at}")  # type: ignore[union-attr]

    # 3. Second query – should be served from cache.
    print("\n[3] Querying player.intro again (should be cached)...")
    cached = ctx.query(Path("player", "intro"))
    print(f"    → {cached!r} (same object: {cached is intro_node})")

    # 4. Mutate a dependency – cache should be invalidated.
    print("\n[4] Changing player.name to 'Bob' (invalidates intro cache)...")
    ctx.set(Path("player", "name"), ScalarNode("Bob"))
    # Inspect the node directly to observe the stale state without triggering
    # re-resolution (the mock only has a canned response for "Alice").
    from context_resolver.query.resolver import _resolve_path
    stale_node = _resolve_path(ctx.root, Path("player", "intro"))
    print(f"    intro resolution_state after mutation: {stale_node.resolution_state.name}")  # type: ignore[union-attr]
    print("    (Re-querying would trigger re-resolution with the new name.)")

    print("\n[5] Context tree (serialized):")
    import json
    print(json.dumps(ctx.to_dict(), indent=2, sort_keys=True)[:600] + "...\n")

    print("Done.")


if __name__ == "__main__":
    main()
