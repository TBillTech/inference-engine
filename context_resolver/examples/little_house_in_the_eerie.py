"""
Little House in the Eerie – multi-query REPL demonstrating lazy AST inference.

This example shows how the inference engine resolves a structured Context lazily,
using multiple queries that correspond to AST subtrees.  Each ``look`` command
triggers :meth:`~context_resolver.context.context.Context.query` for the
relevant subtree; ResolvableNodes (PromptNodes) resolve only the first time they
are queried and cache their results in the AST for subsequent calls.

Run with::

    python -m context_resolver.examples.little_house_in_the_eerie

Or pipe a script through stdin for non-interactive use::

    echo -e "look scene\\nlook event\\nchoose action 2\\nquit" | \\
        python -m context_resolver.examples.little_house_in_the_eerie

Commands
--------
help                  Show available commands.
look notebook         Investigator's narrative notebook (case, motivation, clues).
look status           Investigator's attributes and secrets (lazy inference).
look atmosphere       Atmosphere description (lazy inference).
look scene            Scene description – sensory detail is a PromptNode.
look event            Current scene event and investigation progress.
look action           Available actions (also printed automatically).
choose action <n>     Mark action *n* as the chosen action.
question <text>       Ask the Co-GM a yes/no oracle question.
quit / exit           End the session.

"look" is a synonym for "query".
"""

from __future__ import annotations

import random
import sys
from typing import Any

from context_resolver.ast.nodes import MappingNode, ScalarNode, SequenceNode
from context_resolver.ast.paths import Path
from context_resolver.ast.resolvable_node import ResolvableNode, ResolvableNodeState
from context_resolver.ast.schema import FieldSpec, Schema
from context_resolver.context.context import Context
from context_resolver.inference.mock_provider import MockProvider
from context_resolver.query.passes import ResolutionPass
from context_resolver.query.resolver import Resolver, _resolve_path
from context_resolver.templates.template import Template, TemplateRegistry

_SEP = "-" * 60
_ORACLE_ANSWERS = [
    "Yes, and … things are even better than expected.",
    "Yes, but … there is a complication.",
    "No, but … you gain some unexpected advantage.",
    "No, and … the situation worsens.",
]


# ---------------------------------------------------------------------------
# Node display helpers (no inference triggered)
# ---------------------------------------------------------------------------


def _fmt(node: Any) -> str:
    """Return a display string for *node* **without** triggering resolution."""
    if node is None:
        return "<not found>"
    if isinstance(node, ResolvableNode):
        if node.resolution_state is ResolvableNodeState.RESOLVED:
            return _fmt(node.result)
        return f"PromptNode({node.resolution_state.name})"
    if isinstance(node, ScalarNode):
        return str(node.value) if node.value is not None else "<unspecified>"
    if isinstance(node, SequenceNode):
        items = [_fmt(item) for item in node]
        return "[" + ", ".join(items) + "]" if items else "[]"
    if isinstance(node, MappingNode):
        parts = ", ".join(f"{k}: {_fmt(v)}" for k, v in node.items())
        return "{" + parts + "}"
    return repr(node)


def _query_text(ctx: Context, *segments: str) -> str:
    """
    Query *path* (triggering lazy inference if the node is a PromptNode) and
    return a human-readable string.

    For a resolved :class:`ResolvableNode` the first scalar field of the result
    MappingNode is returned.  For a :class:`ScalarNode` the value is returned
    directly.
    """
    path = Path(*segments)
    node = ctx.query(path)
    if isinstance(node, ResolvableNode) and node.result is not None:
        result = node.result
        if isinstance(result, MappingNode):
            for _, child in result.items():
                if isinstance(child, ScalarNode) and child.value is not None:
                    return str(child.value)
        return _fmt(result)
    if isinstance(node, ScalarNode):
        return str(node.value) if node.value is not None else "<unspecified>"
    return _fmt(node)


def _scalar(ctx: Context, *segments: str) -> Any:
    """Query a ScalarNode and return its raw Python value."""
    node = ctx.query(Path(*segments))
    return node.value if isinstance(node, ScalarNode) else None  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------


def build_initial_context() -> Context:
    """
    Construct the opening game state as a Context AST.

    The tree mixes three kinds of nodes:

    * **Fully-specified ScalarNodes** – deterministic game state (name, stats, …).
    * **SequenceNodes** – ordered collections (clues, NPCs, available actions).
    * **ResolvableNodes** (PromptNodes) – lazily resolved via the MockProvider:

      - ``investigator.secrets``  – depends on investigator name + case title.
      - ``atmosphere.vibe``       – depends on weather + time of day.
      - ``scene.sensory``         – depends on location + time of day.

    Tree layout::

        root
        ├── investigator
        │   ├── name          ScalarNode("Doris Waverly")
        │   ├── occupation    ScalarNode("Former Detective")
        │   ├── sanity        ScalarNode(85)
        │   ├── health        ScalarNode(90)
        │   ├── instability   ScalarNode(2)
        │   ├── secrets       ResolvableNode("investigator_secrets")  ← PromptNode
        │   └── clues         SequenceNode([…])
        ├── case
        │   ├── title         ScalarNode("The Ashen Lane Affair")
        │   └── motivation    ScalarNode("Your sister vanished …")
        ├── atmosphere
        │   ├── time_of_day   ScalarNode("Dusk")
        │   ├── weather       ScalarNode("Overcast, with a cold drizzle")
        │   └── vibe          ResolvableNode("atmosphere_vibe")       ← PromptNode
        ├── scene
        │   ├── location      ScalarNode("The abandoned house on Ashen Lane")
        │   ├── episode       ScalarNode(1)
        │   ├── sensory       ResolvableNode("scene_sensory")         ← PromptNode
        │   └── npcs          SequenceNode([MappingNode(…)])
        ├── event
        │   ├── current       ScalarNode("Investigation Progress")
        │   ├── progress      ScalarNode("You see a suspect at the Location")
        │   ├── consequences  ScalarNode(None)  ← underspecified
        │   └── escalation    ScalarNode(1)
        └── action
            ├── available     SequenceNode([four action strings])
            └── chosen        ScalarNode(None)  ← underspecified
    """
    # ------------------------------------------------------------------
    # Output schemas for the three ResolvableNodes
    # ------------------------------------------------------------------
    secrets_schema = Schema(
        name="InvestigatorSecrets",
        fields=[
            FieldSpec(
                name="secret",
                type="str",
                required=True,
                description="A dark personal secret the investigator carries",
            ),
        ],
        description="Output schema for investigator_secrets template",
    )
    vibe_schema = Schema(
        name="AtmosphereVibe",
        fields=[
            FieldSpec(
                name="vibe",
                type="str",
                required=True,
                description="One evocative sentence capturing the atmosphere",
            ),
        ],
        description="Output schema for atmosphere_vibe template",
    )
    sensory_schema = Schema(
        name="SceneSensory",
        fields=[
            FieldSpec(
                name="description",
                type="str",
                required=True,
                description="Vivid sensory details of the scene",
            ),
        ],
        description="Output schema for scene_sensory template",
    )

    # ------------------------------------------------------------------
    # PromptNode 1 – investigator's dark secret
    # ------------------------------------------------------------------
    secrets_node = ResolvableNode(
        template_ref="investigator_secrets",
        input_bindings={
            "investigator_name": Path("investigator", "name"),
            "case_title": Path("case", "title"),
        },
        output_schema=secrets_schema,
        dependencies=[
            Path("investigator", "name"),
            Path("case", "title"),
        ],
    )

    # ------------------------------------------------------------------
    # PromptNode 2 – atmospheric vibe
    # ------------------------------------------------------------------
    vibe_node = ResolvableNode(
        template_ref="atmosphere_vibe",
        input_bindings={
            "weather": Path("atmosphere", "weather"),
            "time_of_day": Path("atmosphere", "time_of_day"),
        },
        output_schema=vibe_schema,
        dependencies=[
            Path("atmosphere", "weather"),
            Path("atmosphere", "time_of_day"),
        ],
    )

    # ------------------------------------------------------------------
    # PromptNode 3 – scene sensory details
    # ------------------------------------------------------------------
    sensory_node = ResolvableNode(
        template_ref="scene_sensory",
        input_bindings={
            "location": Path("scene", "location"),
            "time_of_day": Path("atmosphere", "time_of_day"),
        },
        output_schema=sensory_schema,
        dependencies=[
            Path("scene", "location"),
            Path("atmosphere", "time_of_day"),
        ],
    )

    # ------------------------------------------------------------------
    # Full AST
    # ------------------------------------------------------------------
    root = MappingNode({
        "investigator": MappingNode({
            "name": ScalarNode("Doris Waverly"),
            "occupation": ScalarNode("Former Detective"),
            "sanity": ScalarNode(85),
            "health": ScalarNode(90),
            "instability": ScalarNode(2),
            "secrets": secrets_node,               # PromptNode
            "clues": SequenceNode([
                ScalarNode("A torn photograph found near the front gate"),
            ]),
        }),
        "case": MappingNode({
            "title": ScalarNode("The Ashen Lane Affair"),
            "motivation": ScalarNode(
                "Your sister vanished near this house three months ago."
            ),
        }),
        "atmosphere": MappingNode({
            "time_of_day": ScalarNode("Dusk"),
            "weather": ScalarNode("Overcast, with a cold drizzle"),
            "vibe": vibe_node,                     # PromptNode
        }),
        "scene": MappingNode({
            "location": ScalarNode("The abandoned house on Ashen Lane"),
            "episode": ScalarNode(1),
            "sensory": sensory_node,               # PromptNode
            "npcs": SequenceNode([
                MappingNode({
                    "name": ScalarNode("Shadowy Figure"),
                    "doing": ScalarNode("watching from an upstairs window"),
                }),
            ]),
        }),
        "event": MappingNode({
            "current": ScalarNode("Investigation Progress"),
            "progress": ScalarNode("You see a suspect at the Location"),
            "consequences": ScalarNode(None),      # underspecified
            "escalation": ScalarNode(1),
        }),
        "action": MappingNode({
            "available": SequenceNode([
                ScalarNode("Approach the front door cautiously"),
                ScalarNode("Observe the shadowy figure from a distance"),
                ScalarNode("Check the perimeter of the house"),
                ScalarNode("Call out to the figure in the window"),
            ]),
            "chosen": ScalarNode(None),            # underspecified until player acts
        }),
    })

    # ------------------------------------------------------------------
    # Template registry
    # The template strings use Python str.format_map substitution.
    # The rendered prompts below are what the MockProvider matches against.
    # ------------------------------------------------------------------
    registry = TemplateRegistry()
    registry.register(Template(
        name="investigator_secrets",
        template_str=(
            "You are the Co-GM for a horror investigation game. "
            "Generate a dark personal secret for investigator {investigator_name}, "
            "who is investigating the case titled '{case_title}'."
        ),
        description="Generates a secret backstory element for the investigator",
    ))
    registry.register(Template(
        name="atmosphere_vibe",
        template_str=(
            "Describe the atmosphere of a horror investigation scene in one evocative "
            "sentence. The time is {time_of_day}. The weather is: {weather}."
        ),
        description="Generates a one-line atmospheric description",
    ))
    registry.register(Template(
        name="scene_sensory",
        template_str=(
            "Describe the sensory details of the following location in one vivid "
            "sentence. Location: {location}. Time of day: {time_of_day}."
        ),
        description="Generates sensory details for the current scene location",
    ))

    # ------------------------------------------------------------------
    # MockProvider – pre-configured responses for every PromptNode.
    #
    # Keys are the exact rendered prompts produced by Template.render().
    # The default_response acts as a safety net for any unmatched prompt.
    # ------------------------------------------------------------------
    mock_provider = MockProvider(
        responses={
            # investigator_secrets rendered with Doris Waverly / The Ashen Lane Affair
            (
                "You are the Co-GM for a horror investigation game. "
                "Generate a dark personal secret for investigator Doris Waverly, "
                "who is investigating the case titled 'The Ashen Lane Affair'."
            ): {
                "secret": (
                    "Doris knows her sister was not taken by chance – "
                    "she unknowingly led the cult to the house herself."
                ),
            },
            # atmosphere_vibe rendered with Dusk / Overcast, with a cold drizzle
            (
                "Describe the atmosphere of a horror investigation scene in one evocative "
                "sentence. The time is Dusk. The weather is: Overcast, with a cold drizzle."
            ): {
                "vibe": (
                    "The dying light bleeds through grey clouds, turning the drizzle "
                    "to silver needles that prick the skin and whisper of things "
                    "better left unseen."
                ),
            },
            # scene_sensory rendered with The abandoned house on Ashen Lane / Dusk
            (
                "Describe the sensory details of the following location in one vivid "
                "sentence. Location: The abandoned house on Ashen Lane. Time of day: Dusk."
            ): {
                "description": (
                    "A faint smell of mildew and cold ash seeps from the cracked doorway, "
                    "while the wind makes the broken shutters clatter like dry bones."
                ),
            },
        },
        default_response={
            # Fallback values satisfy each schema's required field in case the
            # rendered prompt does not match a key (e.g. after game-state changes).
            "secret": "(no response – prompt key not matched)",
            "vibe": "(no response – prompt key not matched)",
            "description": "(no response – prompt key not matched)",
        },
    )

    # ------------------------------------------------------------------
    # Resolver
    # ------------------------------------------------------------------
    resolver = Resolver(
        template_registry=registry,
        passes=[ResolutionPass(mock_provider)],
    )

    return Context(root=root, resolver=resolver)


# ---------------------------------------------------------------------------
# Window print functions  (one per query subtree)
# ---------------------------------------------------------------------------


def print_notebook(ctx: Context) -> None:
    """
    Window 1 – Investigator's Notebook.

    Shows the case title, investigator motivation, and collected clues.
    All nodes in this subtree are fully-specified ScalarNodes – no inference
    is triggered.
    """
    print(_SEP)
    print("INVESTIGATOR'S NOTEBOOK")
    print(_SEP)
    print(f"Case:       {_scalar(ctx, 'case', 'title')}")
    print(f"Motivation: {_scalar(ctx, 'case', 'motivation')}")
    clues_node = ctx.query(Path("investigator", "clues"))
    print("Clues:")
    if isinstance(clues_node, SequenceNode) and len(clues_node) > 0:
        for i, clue in enumerate(clues_node, 1):
            print(f"  {i}. {_fmt(clue)}")
    else:
        print("  (none yet)")
    print()


def print_status(ctx: Context) -> None:
    """
    Window 2 – Investigator's Status.

    Displays attributes (health, sanity, instability) and resolves the
    ``investigator.secrets`` PromptNode lazily on the first call.
    """
    print(_SEP)
    print("INVESTIGATOR STATUS")
    print(_SEP)
    print(f"Name:        {_scalar(ctx, 'investigator', 'name')}")
    print(f"Occupation:  {_scalar(ctx, 'investigator', 'occupation')}")
    print(f"Health:      {_scalar(ctx, 'investigator', 'health')}")
    print(f"Sanity:      {_scalar(ctx, 'investigator', 'sanity')}")
    print(f"Instability: {_scalar(ctx, 'investigator', 'instability')} / 5")

    # Demonstrate lazy inference: show the node state before querying, then resolve.
    secrets_path = Path("investigator", "secrets")
    pre_node = _resolve_path(ctx.root, secrets_path)
    print(f"\nSecrets [before query]: {_fmt(pre_node)}")
    print(f"Secrets [after  query]: {_query_text(ctx, 'investigator', 'secrets')}")
    print()


def print_atmosphere(ctx: Context) -> None:
    """
    Window 3 – Atmosphere Description.

    Shows time of day and weather (fully specified), then resolves the
    ``atmosphere.vibe`` PromptNode lazily.
    """
    print(_SEP)
    print("ATMOSPHERE")
    print(_SEP)
    print(f"Time of day: {_scalar(ctx, 'atmosphere', 'time_of_day')}")
    print(f"Weather:     {_scalar(ctx, 'atmosphere', 'weather')}")

    vibe_path = Path("atmosphere", "vibe")
    pre_node = _resolve_path(ctx.root, vibe_path)
    print(f"Vibe [before query]: {_fmt(pre_node)}")
    print(f"Vibe [after  query]: {_query_text(ctx, 'atmosphere', 'vibe')}")
    print()


def print_scene(ctx: Context) -> None:
    """
    Window 4 – Scene Description.

    Shows location (fully specified), resolves the ``scene.sensory`` PromptNode
    lazily (demonstrating before/after inference), then lists the NPCs present.
    """
    print(_SEP)
    print("SCENE DESCRIPTION")
    print(_SEP)
    print(f"Location: {_scalar(ctx, 'scene', 'location')}")
    print(f"Episode:  {_scalar(ctx, 'scene', 'episode')}")

    # Sensory is a PromptNode – illustrate the lazy resolution flow.
    sensory_path = Path("scene", "sensory")
    pre_node = _resolve_path(ctx.root, sensory_path)
    print(f"Sensory [before query]: {_fmt(pre_node)}")
    print(f"Sensory [after  query]: {_query_text(ctx, 'scene', 'sensory')}")

    # NPCs – fully specified SequenceNode, no inference.
    npcs_node = ctx.query(Path("scene", "npcs"))
    print("NPCs present:")
    if isinstance(npcs_node, SequenceNode) and len(npcs_node) > 0:
        for npc in npcs_node:
            if isinstance(npc, MappingNode):
                name = _fmt(npc.get("name"))
                doing = _fmt(npc.get("doing"))
                print(f"  * {name} – {doing}")
    else:
        print("  (none)")
    print()


def print_event(ctx: Context) -> None:
    """
    Window 5 – Event Description.

    Shows the current scene event and investigation progress roll.
    The ``consequences`` field is underspecified (ScalarNode(None)) until
    the player advances the scene.
    """
    print(_SEP)
    print("SCENE EVENT")
    print(_SEP)
    print(f"Current event:  {_scalar(ctx, 'event', 'current')}")
    print(f"Progress roll:  {_scalar(ctx, 'event', 'progress')}")
    consequences = _scalar(ctx, "event", "consequences")
    print(f"Consequences:   {consequences if consequences is not None else '<pending>'}")
    print(f"Escalation:     {_scalar(ctx, 'event', 'escalation')} / 5")
    print()


def print_action(ctx: Context) -> None:
    """
    Window 6 – Action Description (always printed).

    Lists the available actions and marks the chosen one if set.
    This window is automatically displayed after every REPL command.
    """
    print(_SEP)
    print("AVAILABLE ACTIONS")
    print(_SEP)
    available = ctx.query(Path("action", "available"))
    chosen_n = _scalar(ctx, "action", "chosen")
    if isinstance(available, SequenceNode):
        for i, action in enumerate(available, 1):
            marker = "->" if i == chosen_n else "  "
            print(f" {marker} {i}. {_fmt(action)}")
    if chosen_n is not None:
        print(f"\nChosen: action {chosen_n}")
    print()


def print_gm(ctx: Context, last_message: str | None = None) -> None:
    """
    Window 7 – Co-GM / Command Window (always printed).

    Displays the last oracle response or meta-command result.
    In the REPL this window appears immediately after the action window
    and serves as the visual prompt separator.
    """
    print(_SEP)
    print("CO-GM")
    print(_SEP)
    if last_message:
        print(f"  {last_message}")
    else:
        print("  (awaiting input – type 'help' for commands)")
    print()


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

_HELP_TEXT = """
Commands:
  help                  Show this help.
  look notebook         Investigator's narrative notebook (case, clues).
  look status           Investigator's attributes and secrets (lazy inference).
  look atmosphere       Atmosphere and vibe (lazy inference).
  look scene            Scene description – sensory detail is a PromptNode.
  look event            Current scene event and progress.
  look action           Available actions.
  choose action <n>     Mark action n as the chosen action.
  question <text>       Ask the Co-GM a yes/no oracle question.
  quit / exit           End the session.

'look' and 'query' are synonyms.
PromptNodes (marked as 'PromptNode(PENDING)') resolve the first time
they are queried and cache their result for subsequent queries.
"""


def run_repl(ctx: Context) -> None:
    """Start the interactive REPL loop."""
    print("=" * 60)
    print("  LITTLE HOUSE IN THE EERIE")
    print("  A Solo Horror Investigation")
    print("=" * 60)
    print()
    print("The abandoned house on Ashen Lane looms before you.")
    print("Type 'help' for commands.\n")

    gm_message: str | None = None
    print_action(ctx)
    print_gm(ctx, gm_message)

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFarewell, investigator.")
            break

        if not raw:
            continue

        tokens = raw.lower().split(maxsplit=1)
        cmd = tokens[0]
        rest = tokens[1].strip() if len(tokens) > 1 else ""

        if cmd in ("quit", "exit"):
            print("Farewell, investigator.")
            break

        elif cmd == "help":
            print(_HELP_TEXT)
            gm_message = None

        elif cmd in ("look", "query"):
            target = rest
            if target == "notebook":
                print_notebook(ctx)
            elif target == "status":
                print_status(ctx)
            elif target == "atmosphere":
                print_atmosphere(ctx)
            elif target == "scene":
                print_scene(ctx)
            elif target == "event":
                print_event(ctx)
            elif target == "action":
                # action window prints below; skip the duplicate here
                pass
            else:
                print(
                    f"Unknown target '{rest}'. "
                    "Try: notebook, status, atmosphere, scene, event, action\n"
                )

        elif cmd == "choose" and rest.startswith("action "):
            num_str = rest[len("action "):].strip()
            try:
                n = int(num_str)
                available = ctx.query(Path("action", "available"))
                if isinstance(available, SequenceNode) and 1 <= n <= len(available):
                    ctx.set(Path("action", "chosen"), ScalarNode(n))
                    chosen_text = _fmt(available.get(n - 1))
                    gm_message = f"Action {n} chosen: {chosen_text}"
                else:
                    max_n = len(available) if isinstance(available, SequenceNode) else "?"
                    print(f"Invalid action number '{n}'. Choose between 1 and {max_n}.\n")
                    continue
            except ValueError:
                print(f"'{num_str}' is not a valid number.\n")
                continue

        elif cmd == "question":
            if not rest:
                print("Usage: question <your yes/no question>\n")
                continue
            answer = random.choice(_ORACLE_ANSWERS)
            gm_message = f"Q: {rest}\n  A: {answer}"

        else:
            print(f"Unknown command '{raw}'. Type 'help' for commands.\n")
            continue

        # Action and Co-GM windows always print last after every recognised command.
        print_action(ctx)
        print_gm(ctx, gm_message)


def main() -> None:
    """Entry point: build the game context and start the REPL."""
    ctx = build_initial_context()
    run_repl(ctx)


if __name__ == "__main__":
    main()
