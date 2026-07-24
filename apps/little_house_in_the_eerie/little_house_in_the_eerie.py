"""
Little House in the Eerie – A solor Horror Investigation.

This program implements a solo RPG where the player investigates the paranormal
in a small town somewhere -- eerie.

Run with::

    python -m apps.little_house_in_the_eerie.little_house_in_the_eerie

Or pipe a script through stdin for non-interactive use::

    echo -e "look scene\\nlook event\\nchoose action 2\\nquit" | \\
        python -m apps.little_house_in_the_eerie.little_house_in_the_eerie

Introduction Commands
--------
help
look wakeup_scene
look atmosphere
look status
question <text>
quit / exit

Story Mode Commands
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
from context_resolver.query.passes import ResolutionPass
from context_resolver.query.resolver import Resolver, _resolve_path
from context_resolver.inference.llama_cpp_provider import LocalLlamaCppProvider
from context_resolver.inference.strategy import PromptStrategy
from context_resolver.templates.template import TemplateRegistry, JSONOutputTemplate

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

place_holder = ScalarNode(None)

def simple_schema(name, field_name, template_str):
    key = name.replace(' ', '')
    t = JSONOutputTemplate(
        name=key,
        template_str = template_str,
        description=f"A generated {name}."
    ),
    s = Schema(
        name=key,
        fields=[
            FieldSpec(
                name=field_name,
                type="str",
                required=True,
                decription=f"A {name}."
            )
        ]
    )
    return (t, s)


def build_initial_context() -> Context:
    """
    Construct the opening game state as a Context AST.
    """
    # ------------------------------------------------------------------
    # Template registry
    # The template strings use Python str.format_map substitution.
    # ------------------------------------------------------------------
    registry = TemplateRegistry()
    brochure = (JSONOutputTemplate(
        name="TownBrochure",
        template_str=(
            "Your are the Co-GM for a paranormal investigation game centered on a little town. "
            "Choose a location {interview.destination_hint}, and remember to keep it isolated and preferably hidden "
            "from the rest of the world; that’s how its inhabitants like to live. For example, "
            "the location could be vast desert near 7 shooter peak, OR concealed in a dense forest "
            "near terror falls, OR in a canyon near the cliffs of twilight, OR on a remote island etc...  "
            "Choose a name for it, like Sapphire Bluffs or Little Hill Marsh. "
            "Invent the Economic basis for the town in just a few words, for example Diary Farming OR Mining OR ... "
            "The Town has a dark backstory that the townsfolk seem to have forgotten or keep hiding. "
            "For example, it was once a hub for smuggling operations, OR maybe they had witch trials, OR "
            "organized crime controls local busineses, OR something else equally scandalous. "
        ),
        description="Generates a brochure for one of the towns the player can investigate."
    ),
    Schema(
        name="TownBrochure",
        fields=[
            FieldSpec(
                name="name",
                type="str",
                required=True,
                decription="The Town name is composed of at least two parts, with an optional third part",
            ),
            FieldSpec(
                name="location",
                type="str",
                required=True,
                description="Where is the town located atmospherically (exact geography is left vague)"
            ),
            FieldSpec(
                name="economic",
                type="str",
                require=True,
                description="The Economic foundation of the town"
            ),
            FieldSpec(
                name="backstory",
                type="str",
                require=True,
                description="The dark backstory of the town (possibly paranormal)"
            ),
        ],
        description="Output schema for a town in the brochure (possible nearby location for the house)"
    ))
    registry.register(brochure[0])

    case_headline = (JSONOutputTemplate(
        name="CaseHeadline",
        template_str=(
            "You are a Co-GM for a paranormal investigation game centered on a little town. "
            "The town is named {town.name}, situated in: "
            "{town.location} "
            "For their livelihood, the denizens rely on: "
            "{town.economy} "
            "Those who know about the town are aware that: "
            "{town.backstory} "
            "Invent a Case to be investigated by a paranormal investigator, (surface level, not the root). "
            "Life in the Town is calm and slow, like trees swaying in the wind. "
            "However, the peaceful routine of its inhabitants was recently disturbed by an extraordinary "
            "event, rooted in a case for the investigator to delve into. For example, maybe a person is missing, "
            "OR someone is kidnapped, OR mysterious power outages, etc... What is the headline that "
            "drew the investigator here? What is the sensational story in the newspaper?"
        ),
        description="Generate a case headline for the investigator."
    ),
    Schema(
        name="CaseHeadline",
        fields=[
            FieldSpec(
                name="name",
                type="str",
                require=True,
                desription="The headline for the case under investigation relating to this town"
            ),
            FieldSpec(
                name="description",
                type="str",
                require=True,
                description="The sensational (and superficial) description of the case."
            )
        ],
    ))
    registry.register(case_headline[0])

    personal_secret = simple_schema("Investigator Secret", "secret", (
            "You are the Co-GM for a paranormal investigation game. "
            "Generate a dark personal secret for investigator {investigator.first_name} {investigator.last_name}, "
            "who is investigating the case titled '{case.name}'."
        ))
    registry.register(personal_secret[0])

    atmosphere_vibe = simple_schema("Atmosphere Vibe", "vibe", (
            "Describe the atmosphere of a paranormal investigation scene in one evocative sentence. "
            "The time is {atmosphere.hour_of_day}:{atmosphere.minute_of_day} PM is {atmosphere.post_meridian}. "
            "The weather is: {weather}."
        ))
    registry.register(atmosphere_vibe[0])

    investigator_archetype = simple_schema("Investigator Archetype", "archetype", (
            "In just a word or two, suggest an archetype for the {interview.archetype_hint} investigator. "
            "Archetypes are things like: Federal Agent, Doctor, College Student, Amateur Sleuth, Disgraced Policeman. "            
        ))
    registry.register(investigator_archetype[0])

    attributes_modifiers = (JSONOutputTemplate(
        name="AttributesModifiers",
        template_str = (
            "The investigator (a {investigator.archetype}) has Agility, Mind, Strength and Prescence attributes."
            "The investigator also has some secrets: {investigator.secrets}."
            "Choose plus modifiers for these attributes between 0 and 3. Use numbers that seem to make sense when describing this person's advantages."
        ),
        description="Investigator attribute modifiers"
    ),
    Schema(
        name="AttributeModifiers",
        fields=[
            FieldSpec(
                name="agility",
                type="int",
                required=True,
                description="Dexterity and Athletics"
            ),
            FieldSpec(
                name="mind",
                type="int",
                required=True,
                description="Critical thinking and wisdom"
            ),
            FieldSpec(
                name="strength",
                type="int",
                required=True,
                description="Health and Strength"
            ),
            FieldSpec(
                name="prescence",
                type="int",
                required=True,
                description="Charisma and Persuasion"
            )
        ]
    ))

    interest = simple_schema("Interest", "interest", (
            "The investigator (a {investigator.archetype}) has an {interview.interest_hint} interest in this case. "
            "The investigator also has some secrets: {investigator.secrets}. "
            "The current case is {case.name}. {case.description} "
            "Invent a reason why the investigator is so invested in this case."
        ))
    registry.register(interest[0])

    vibe = (JSONOutputTemplate(
        name="Vibe", 
        template_str = (
            "Describe the weather, vibe, and urgency each in one evocative sentence. "
            "Don't just repeat the same sentence. "
            "The current scene location is: {scene.location} . the current scene specifics are {scene.specfics} . "
            "The time is {atmosphere.hour_of_day}:{atmosphere.minute_of_day} PM is {atmosphere.post_meridian}. "
            "The previous weather is: {atmosphere.prior_weather}. "
            "The weather is {atmosphere.weather_changing} from what it was."
            "If the PM is past 6 PM, or past 6 AM, then make sure to correct the weather if necessary.  "
            "For example, it should be something like 'twinkling stars' past 6 PM because it is night, and could no longer be sunny. "
            "But weather could still be not changing and still clear."
            "If the scene is indoors, be careful not to describe raindrops hitting things, for example. "
            "The vibe should be the vague atmosphere of tension or peace or levity, and not repeat any current scene location or specifics. "
            "The urgency should be the gut viscreal feeling of the investigator versus how soon the time will run out.  A short sentence."
            "The current timestep is {atmosphere.time_limit} out of 12, when 'it is game over!' "
        )),
    Schema(
        name="Vibe",
        fields=[
            FieldSpec(
                name="weather",
                type="str",
                required=True,
                description="A simple description of just the weather conditions."
            ),
            FieldSpec(
                name="vibe",
                type="str",
                required=True,
                description="The feeling in the atmosphere like tense, peaseful, or joyous."
            ),
            FieldSpec(
                name="urgency",
                type="str",
                required=True,
                description="The viscreal feeling of how close to 'game over!'"
            )
        ]
        ))
    registry.register(vibe[0]s)

    # ------------------------------------------------------------------
    # Full AST
    # ------------------------------------------------------------------
    root = MappingNode({
        "interview": MappingNode({
            "destination_hint": ScalarNode(" somewhere remote "),
            "brochure1": ResolutionNode(*brochure),
            "brochure2": ResolutionNode(*brochure),
            "brochure3": ResolutionNode(*brochure),
            "brochure4": ResolutionNode(*brochure),
            "case1": ResolutionNode(*case_headline),
            "case2": ResolutionNode(*case_headline),
            "case3": ResolutionNode(*case_headline),
            "case4": ResolutionNode(*case_headline),
            "archetype_hint": ScalarNode(" curious "),
            "interest_hint": ScalarNode(" keen ")
        }),
        # Town is filled in via player choosing a brochure output
        "town": MappingNode({
            "name": place_holder,
            "location": place_holder,
            "economic": place_holder,
            "backstory": place_holder,
        }),
        # case is filled in via player choosing a case output
        "case": MappingNode({
            "name": place_holder,
            "description": place_holder
        })
        # investigator is filled in during initial interview
        "investigator": MappingNode({
            "first_name": place_holder,
            "last_name": place_holder,
            "archetype": ResolutionNode(*investigator_archetype),
            "attributes": ResolutionNode(*attributes_modifiers),
            "wounds": ScalarNode(0),
            "instability": ScalarNode(0),
            "luck": ScalarNode(9),
            "secrets": place_holder,
            "interest": ResolutionNode(*interest),
            "advantages": SequenceNode([]),
            "disadvantages": SequenceNode([])
        }),
        "notebook": MappingNode({
            "logbook": SequenceNode([]),
            "daybook": SequenceNode([]),
            "diary": SequenceNode([]),
            "clues": SequenceNode([])
        }),
        "public": MappingNode({
            "rumors": SequenceNode([]),
            "conspiracies": SequenceNode([]),
            "player": SequenceNode([])
        })
        "locations": MappingNode({
            "Little House": MappingNode({
                "name": ScalarNode("Little House"),
                "description": little_house_description_node
            })
        }),
        "npcs": MappingNode({
            "Mable Jenner": MappingNode({
                "first_name": ScalarNode("Mable"),
                "last_name": ScalarNode("Jenner"),
                "archetype": ScalarNode("Elderly Citizen"),
                "attitude": ScalarNode(5),
                "personality": ScalarNode("Clumsy"),
                "motivation": ScalarNode("Duty"),
                "location": ScalarNode("Little House")
            }),
            "Daryl Jenner": MappingNode({
                "first_name": ScalarNode("Daryl"),
                "last_name": ScalarNode("Jenner"),
                "archetype": ScalarNode("Elderly Citizen"),
                "attitude": ScalarNode(4),
                "personality": ScalarNode("Introverted"),
                "motivation": ScalarNode("Survival"),
                "location": ScalarNode("Little House")
            })
        }),
        "atmosphere": MappingNode({
            "hour_of_day": ScalarNode(8),
            "minute_of_day": ScalarNode(20),
            "post_meridian": ScalarNode(False),
            "weather_changing": ScalarNode("Not changing")
            "prior_weather": ScalarNode("Sunny")
            "vibe": ResolutionNode(*vibe),
            "time_limit": ScalarNode(6),
        }),
        "scene": MappingNode({
            "location": ScalarNode("Little House"),
            "specifics": ScalarNode("You are sitting in a soft bed in a bedroom in the Little House,"
                " chatting with Mable Jenners.")
            "elements": SequenceNode(["Mable on rocking chair", "newspaper within reach", 
                "comfortable bed", "my suitcase", "eclectic electric lamp", 
                "alarm clock reading 8:20 A.M.", "Mable holding a bible with a bookmark",
                "something about the light and the view of outside from the window feels eerie"]),
            "npcs": SequenceNode(["Mable Jenner"]),
            "description": scene_node
        }),
        "event": MappingNode({
            "description": ScalarNode("Investigation Progress"),
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
    # LLMResolver
    # ------------------------------------------------------------------
    # --- Local llama.cpp provider wired through PromptStrategy ---
    # Swapping to a different provider (e.g. OpenAIProvider) requires only
    # changing the constructor call below; the strategy and resolver layers
    # remain unchanged.
    llama_provider = LocalLlamaCppProvider(
        base_url=None,
        model=None,
    )
    strategy = PromptStrategy(llama_provider)

    resolver = Resolver(
        template_registry=registry,
        passes=[ResolutionPass(strategy)],
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
    print("You wake up with a dry mouth, and sticky eyes.  Gradually, the room comes into focus.")
    print("You sit up in a soft, clean bed, in a small room, with rustic furnishings.")
    print("An old woman in a flower dress is rocking in a chair, watching you as you awake.")
    print("She seems relived so see you stirring.")
    print("You try to remember how you got here; it is gradually coming back to you.")
    print('The woman says, "Hello, my name is Mable Jenners. I\'m sorry we had to meet under such ... frightening circumstances."')
    print('She continues, "How are you feeling?"')
    print('You reply, "Ugh. Minor pains all over. I feel like I was in a car wreck!"')
    print('Mable responds, "Oh, dear! Well, that is because _you were_ in a car wreck! I must say!"')
    print('You reply, "Oh, wow! Uh ... where am I?"')
    print('Mable replies, "Well, what do you remember?"')
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
