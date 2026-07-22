"""
End-to-end example: lazy compilation with a mock inference provider.

This script demonstrates:

* Building a Context tree with an unresolved PromptNode.
* Registering a Template.
* Registering a MockProvider.
* Querying the Context – which triggers lazy compilation.
* Inspecting the resolved PromptNode.
* Mutating a dependency and observing that the cache is invalidated.

Run with::

    python -m context_compiler.examples.simple_prompt
"""

from __future__ import annotations

from context_compiler.ast.nodes import MappingNode, ScalarNode
from context_compiler.ast.paths import Path
from context_compiler.ast.prompt_node import PromptNode
from context_compiler.ast.schema import Schema, FieldSpec
from context_compiler.compiler.compiler import Compiler
from context_compiler.compiler.passes import InferencePass
from context_compiler.context.context import Context
from context_compiler.inference.mock_provider import MockProvider
from context_compiler.templates.template import Template, TemplateRegistry


def build_example_context() -> Context:
    """
    Construct a Context tree with the following structure::

        root
        ├── player
        │   ├── name   (ScalarNode: "Alice")
        │   └── greeting  (PromptNode: template="greet", input_bindings={"name": Path("player","name")})
        └── world
            └── setting  (ScalarNode: "a fantasy kingdom")

    The ``greeting`` node is initially underspecified.  Querying it will
    trigger the mock inference provider to produce a greeting.
    """
    # --- Schema for the greeting output ---
    greeting_schema = Schema(
        name="GreetingOutput",
        fields=[
            FieldSpec(name="greeting", type="str", required=True,
                      description="A greeting message for the player"),
        ],
        description="Expected output from the greeting template",
    )

    # --- PromptNode that needs to be resolved ---
    greeting_prompt = PromptNode(
        template_ref="greet",
        input_bindings={
            "name": Path("player", "name"),
            "setting": Path("world", "setting"),
        },
        output_schema=greeting_schema,
        dependencies=[Path("player", "name"), Path("world", "setting")],
    )

    # --- Build the tree ---
    root = MappingNode({
        "player": MappingNode({
            "name": ScalarNode("Alice"),
            "greeting": greeting_prompt,
        }),
        "world": MappingNode({
            "setting": ScalarNode("a fantasy kingdom"),
        }),
    })

    # --- Template registry ---
    registry = TemplateRegistry()
    registry.register(Template(
        name="greet",
        template_str="You are a narrator in {setting}. Greet the hero named {name}.",
        description="Generates a greeting for the player",
    ))

    # --- Mock provider returns a canned response ---
    rendered_prompt = (
        "You are a narrator in a fantasy kingdom. Greet the hero named Alice."
    )
    mock_provider = MockProvider(
        responses={
            rendered_prompt: {"greeting": "Hail, brave Alice! Welcome to the kingdom!"}
        }
    )

    # --- Compiler pipeline ---
    compiler = Compiler(
        template_registry=registry,
        passes=[InferencePass(mock_provider)],
    )

    return Context(root=root, compiler=compiler)


def main() -> None:
    """Run the end-to-end demonstration."""
    print("=" * 60)
    print("Context Compiler – Simple Prompt Example")
    print("=" * 60)

    ctx = build_example_context()

    # 1. Query a fully-specified scalar – no compilation needed.
    print("\n[1] Querying player.name (already fully specified)...")
    name_node = ctx.query(Path("player", "name"))
    print(f"    → {name_node!r}")

    # 2. Query the PromptNode – triggers lazy compilation.
    print("\n[2] Querying player.greeting (triggers inference)...")
    greeting_node = ctx.query(Path("player", "greeting"))
    print(f"    → {greeting_node!r}")

    assert hasattr(greeting_node, "result"), "Expected a PromptNode"
    result = greeting_node.result  # type: ignore[union-attr]
    greeting_text = result.get("greeting").value  # type: ignore[union-attr]
    print(f"    Resolved greeting: {greeting_text!r}")
    print(f"    Provider: {greeting_node.provider!r}")  # type: ignore[union-attr]
    print(f"    Resolved at: {greeting_node.resolved_at}")  # type: ignore[union-attr]

    # 3. Second query – should be served from cache.
    print("\n[3] Querying player.greeting again (should be cached)...")
    cached = ctx.query(Path("player", "greeting"))
    print(f"    → {cached!r} (same object: {cached is greeting_node})")

    # 4. Mutate a dependency – cache should be invalidated.
    print("\n[4] Changing player.name to 'Bob' (invalidates greeting cache)...")
    ctx.set(Path("player", "name"), ScalarNode("Bob"))
    # Inspect the node directly to observe the stale state without triggering
    # re-compilation (the mock only has a canned response for "Alice").
    from context_compiler.compiler.compiler import _resolve_path
    stale_node = _resolve_path(ctx.root, Path("player", "greeting"))
    print(f"    greeting prompt_state after mutation: {stale_node.prompt_state.name}")  # type: ignore[union-attr]
    print("    (Re-querying would trigger re-compilation with the new name.)")

    print("\n[5] Context tree (serialized):")
    import json
    print(json.dumps(ctx.to_dict(), indent=2, sort_keys=True)[:600] + "...\n")

    print("Done.")


if __name__ == "__main__":
    main()
