"""
End-to-end example: lazy resolution with a local llama.cpp provider.

This script demonstrates how to swap the mock provider used in
``simple_resolution.py`` for a :class:`LocalLlamaCppProvider` that calls a
locally-running llama.cpp server.  Everything at the strategy layer
(:class:`PromptStrategy`, :class:`ResolutionPass`) stays identical – only the
provider changes.

Startup assumptions
-------------------
Before running this script, start the llama.cpp server in OpenAI-compatible
mode::

    # Download a GGUF model (e.g. from https://huggingface.co/TheBloke)
    ./llama-server --model /path/to/model.gguf --host 127.0.0.1 --port 8080

The server must be reachable at ``http://127.0.0.1:8080`` (the default) or at
the URL set by the ``LLAMA_CPP_BASE_URL`` environment variable.

Configuration
-------------
``LLAMA_CPP_BASE_URL``
    Override the server URL (default: ``http://127.0.0.1:8080``).
``LLAMA_CPP_MODEL``
    Model name forwarded in API requests (default: ``"local"``).
``LLAMA_CPP_API_KEY``
    Optional bearer token for servers that require authentication.

Run with::

    python -m context_resolver.examples.local_llama_resolution
"""

from __future__ import annotations

import json

from context_resolver.ast.nodes import MappingNode, ScalarNode
from context_resolver.ast.paths import Path
from context_resolver.ast.resolvable_node import ResolvableNode
from context_resolver.ast.schema import Schema, FieldSpec
from context_resolver.query.resolver import Resolver
from context_resolver.query.passes import ResolutionPass
from context_resolver.context.context import Context
from context_resolver.inference.llama_cpp_provider import LocalLlamaCppProvider
from context_resolver.inference.strategy import PromptStrategy
from context_resolver.templates.template import TemplateRegistry, JSONOutputTemplate


def build_example_context(
    base_url: str | None = None,
    model: str | None = None,
) -> Context:
    """
    Construct a Context tree and wire it to a local llama.cpp provider.

    The tree structure is identical to ``simple_resolution.py``::

        root
        ├── player
        │   ├── name   (ScalarNode: "Alice")
        │   └── intro  (ResolvableNode: template="greet")
        └── world
            └── setting  (ScalarNode: "a fantasy kingdom")

    Parameters
    ----------
    base_url:
        llama.cpp server base URL.  When ``None`` the provider reads
        ``LLAMA_CPP_BASE_URL`` from the environment, defaulting to
        ``http://127.0.0.1:8080``.
    model:
        Model name to send with each request.  When ``None`` the provider
        reads ``LLAMA_CPP_MODEL`` from the environment, defaulting to
        ``"local"``.
    """
    # --- Schema for the intro output ---
    intro_schema = Schema(
        name="IntroOutput",
        fields=[
            FieldSpec(
                name="greeting",
                type="str",
                required=True,
                description="A greeting message for the player",
            ),
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
    # JSONOutputTemplate appends a JSON output instruction block automatically
    # based on the schema fields, so the template_str only needs to describe
    # the task itself.
    registry = TemplateRegistry()
    registry.register(JSONOutputTemplate(
        name="greet",
        template_str=(
            "You are a narrator for a story in the Kingdom of {setting}. "
            "Greet the hero named {name}."
        ),
        schema=intro_schema,
        description="Generates an intro for the player with structured JSON output",
    ))

    # --- Local llama.cpp provider wired through PromptStrategy ---
    # Swapping to a different provider (e.g. OpenAIProvider) requires only
    # changing the constructor call below; the strategy and resolver layers
    # remain unchanged.
    llama_provider = LocalLlamaCppProvider(
        base_url=base_url,
        model=model,
    )
    strategy = PromptStrategy(llama_provider)

    # --- Resolver pipeline ---
    resolver = Resolver(
        template_registry=registry,
        passes=[ResolutionPass(strategy)],
    )

    return Context(root=root, resolver=resolver)


def main() -> None:
    """Run the end-to-end demonstration against a local llama.cpp server."""
    print("=" * 60)
    print("Context Resolver – Local llama.cpp Resolution Example")
    print("=" * 60)
    print()
    print("This example requires a locally-running llama.cpp server.")
    print("Start it with:")
    print("    ./llama-server --model /path/to/model.gguf --host 127.0.0.1 --port 8080")
    print()

    ctx = build_example_context()

    # 1. Query a fully-specified scalar – no resolution needed.
    print("[1] Querying player.name (already fully specified)...")
    name_node = ctx.query(Path("player", "name"))
    print(f"    → {name_node!r}")

    # 2. Query the ResolvableNode – triggers lazy resolution via llama.cpp.
    print("\n[2] Querying player.intro (triggers resolution via llama.cpp)...")
    print("    (This will call http://127.0.0.1:8080/v1/chat/completions)")
    try:
        intro_node = ctx.query(Path("player", "intro"))
    except ConnectionError as exc:
        print(f"\n    [ERROR] Could not reach the llama.cpp server: {exc}")
        print("    Make sure the server is running and try again.")
        return

    print(f"    → {intro_node!r}")

    assert hasattr(intro_node, "result"), "Expected a ResolvableNode"
    result = intro_node.result  # type: ignore[union-attr]
    greeting_text = result.get("greeting")
    opening_text = result.get("opening")
    print(f"    Resolved greeting: {greeting_text!r}")
    print(f"    Resolved opening:  {opening_text!r}")
    print(f"    Provider:          {intro_node.provider!r}")  # type: ignore[union-attr]
    print(f"    Resolved at:       {intro_node.resolved_at}")  # type: ignore[union-attr]

    # 3. Second query – should be served from cache.
    print("\n[3] Querying player.intro again (should be cached)...")
    cached = ctx.query(Path("player", "intro"))
    print(f"    → same object: {cached is intro_node}")

    # 4. Serialise the context tree.
    print("\n[4] Context tree (serialized, first 600 chars):")
    print(json.dumps(ctx.to_dict(), indent=2, sort_keys=True)[:600] + "...\n")

    print("Done.")


if __name__ == "__main__":
    main()
