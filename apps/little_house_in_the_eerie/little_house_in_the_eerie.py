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
from typing import Any

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable
import shutil
import textwrap
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

def _set_keyed_node(ctx: Context, node: Node, *segments: str) -> None:
    """Set a named (or indexed) Node in the Context"""
    parent_segments = segments[:-1]
    parent_node = ctx.query(Path(*parent_segments))
    if isinstance(parent_node, MappingNode):
        parent_node.set(segments[-1], node)
    if isinstance(parent_node, SequenceNode):
        parent_node.set(int(segments[-1]), node)

def _invalidate(ctx: Context, *segments: str) -> None:
    """Force a ResolveableNode to be regenerated."""
    node = ctx.query(Path(*segments))
    if isinstance(node, ResolvableNode):
        node.mark_stale()

# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------

place_holder = ScalarNode(None)

def simple_schema(name, field_name, template_str):
    key = name.replace(' ', '')
    s = Schema(
        name=key,
        fields=[
            FieldSpec(
                name=field_name,
                type="str",
                required=True,
                description=f"A {name}."
            )
        ]
    )
    t = JSONOutputTemplate(
        name=key,
        template_str=template_str,
        schema=s,
        description=f"A generated {name}."
    )
    return (t, s)

def template_schema_tuple(template_str, schema, description):
    t = JSONOutputTemplate(
        name=schema.name,
        template_str=template_str,
        schema=schema,
        description=description
    )
    return (t, schema)

class NameMode(Enum):
    FIRST = auto()
    FORMAL = auto()
    LAST = auto()
    FULL = auto()

def query_investigator_name(ctx: Context, mode: NameMode, *segments: str):
    if len(segments) == 0:
        segments = ['investigator', 'name']
    if mode == NameMode.FIRST:
        return _scalar(ctx, *segments, 'first')
    if mode == NameMode.FORMAL:
        return f"{_scalar(ctx, *segments, 'honorific')} {_scalar(ctx, *segments, 'last')}"
    if mode == NameMode.LAST:
        return _scalar(ctx, *segments, 'last')
    full_name = (
        f"{_scalar(ctx, *segments, 'honorific')} {_scalar(ctx, *segments, 'first')} "
        f"{_scalar(ctx, *segments, 'middle')} {_scalar(ctx, *segments, 'last')} "
        f"{_scalar(ctx, *segments, 'suffix')}" )
    return full_name


def _query_resolved_field_text(ctx: Context, *segments: str, field_name: str) -> str:
    """Query a resolved mapping node and return one named scalar field as text."""
    node = ctx.query(Path(*segments))
    if isinstance(node, ResolvableNode) and node.result is not None:
        result = node.result
        if isinstance(result, MappingNode):
            child = result.get(field_name)
            if isinstance(child, ScalarNode) and child.value is not None:
                return str(child.value)
        return _fmt(result)
    return _fmt(node)

def build_initial_context() -> Context:
    """Construct the opening game state as a Context AST."""
    registry = TemplateRegistry()

    investigator_name = template_schema_tuple(
        template_str=(
            "You are the Co-GM for a paranormal investigation game centered on a little town. "
            "The investigator is a {investigator.archetype}. "
            "The first name {interview.first_name_hint}. "
            "The rest of the name {interview.rest_name_hint}. "
            "What is the investigator's name? A prefix and/or a suffix could be provided. "
            "A middle name may also be provided. "
        ),
        schema=Schema(
            name="InvestigatorName",
            fields=[
                FieldSpec(name="honorific", type="str", required=False, description="Honorific"),
                FieldSpec(name="first", type="str", required=True, description="First name"),
                FieldSpec(name="middle", type="str", required=False, description="Middle name"),
                FieldSpec(name="last", type="str", required=True, description="Last name"),
                FieldSpec(name="suffix", type="str", required=False, description="Suffix")
            ]
        ),
        description="Generates an investigator name."
    )

    brochure = template_schema_tuple(
            template_str=(
                "You are the Co-GM for a paranormal investigation game centered on a little town. "
                "Choose a location {interview.destination_hint}, and keep it isolated and hidden. "
                "Invent a town name, an economic basis, and a dark backstory."
            ),
            schema=Schema(
                name="TownBrochure",
                fields=[
                    FieldSpec(name="name", type="str", required=True, description="Town name."),
                    FieldSpec(name="location", type="str", required=True, description="Town location."),
                    FieldSpec(name="economic", type="str", required=True, description="Economic basis."),
                    FieldSpec(name="backstory", type="str", required=True, description="Dark backstory."),
                ],
                description="Output schema for a nearby town.",
            ),
            description="Generates one town brochure option.",
        )
    registry.register(brochure[0])

    newspaper_title = template_schema_tuple(
        template_str=(
            "You are a Co-GM for a paranormal investigation game. You need to come up with an in-fiction newspaper details. "
            "The newspaper needs a Brand Name/Title, but not _too_ respectable.  It should lean speculative and too credulous. "
            "Also, invent the publisher, circulation and date. The date should be {investigation.date_hint}. "
        ),
        schema=Schema(
            name="Newspaper_Title",
            fields=[
                FieldSpec(name="title", type="str", required=True, description="Newspaper Title"),
                FieldSpec(name="publisher", type="str", required=True, description="Publisher"),
                FieldSpec(name="circulation", type="str", required=True, description="The city or means of circulation"),
                FieldSpec(name="date", type="The date of this issue")
            ],
            description="Details for the investigators favored newspaper of record."
        ),
        description="Generates a newspaper fiction."
    )
    registry.register(newspaper_title[0])

    case_headline = template_schema_tuple(
            template_str=(
                "You are a Co-GM for a paranormal investigation game. "
                "The investigator archetype is {investigator.archetype}. "
                "Town: {town.name}, location: {town.location}, economy: {town.economic}, backstory: {town.backstory}. "
                "Invent a current case headline and a short sensational description for {interview.case_hint}."
            ),
            schema=Schema(
                name="CaseHeadline",
                fields=[
                    FieldSpec(name="name", type="str", required=True, description="Case headline."),
                    FieldSpec(name="description", type="str", required=True, description="Case summary."),
                ],
            ),
            description="Generate a case headline for the investigator.",
        )
    registry.register(case_headline[0])

    personal_secret = simple_schema(
        "Investigator Secret",
        "secret",
        (
            "Generate a dark personal secret for investigator "
            "{investigator.first_name} {investigator.last_name}, "
            "who is investigating '{case.name}'."
        ),
    )
    registry.register(personal_secret[0])

    investigator_archetype = simple_schema(
        "Investigator Archetype",
        "archetype",
        (
            "In one to three words, suggest a specific investigator archetype for this character. "
            "Think of things like Federal Agent, Computer Hacker, Amateur Sleuth, Student, Police Officer."
            "Primary vibe: {interview.archetype_hint}. "
        ),
    )
    registry.register(investigator_archetype[0])

    attributes_modifiers = template_schema_tuple(
            template_str=(
                "The investigator (a {investigator.archetype}) has Agility, Mind, Strength and Prescence attributes. "
                "The investigator also has some secrets: {investigator.secrets}. "
                "Choose modifiers between 0 and 3 for each attribute."
            ),
            schema=Schema(
                name="AttributeModifiers",
                fields=[
                    FieldSpec(name="agility", type="int", required=True, description="Dexterity and athletics."),
                    FieldSpec(name="mind", type="int", required=True, description="Critical thinking and wisdom."),
                    FieldSpec(name="strength", type="int", required=True, description="Health and strength."),
                    FieldSpec(name="prescence", type="int", required=True, description="Charisma and persuasion."),
                ],
            ),
            description="Investigator attribute modifiers",
        )
    registry.register(attributes_modifiers[0])

    interest = simple_schema(
        "Interest",
        "interest",
        (
            "The investigator (a {investigator.archetype}) has an {interview.interest_hint} interest in this case. "
            "The current case is {case.name}. {case.description}. "
            "Invent a reason they are personally invested."
        ),
    )
    registry.register(interest[0])

    advantage = template_schema_tuple(
            template_str=(
                "Describe an advantage that the investigator has, usually based on equipment they packed with them "
                "when they left home to began this investigation. " 
                "If the investigator has a skill advantage, symbolize this by suggesting a reference textbook."
                "But don't ignore the possibility of weapons or crime scene investigation tools either."
                "The investigator is a {investigator.archetype} on the case {case.name}. {case.description}. "
                "The investigator remembers they possess {interview.advantage_hint}. "
            ),
            schema=Schema(
                name="Advantage",
                fields=[
                    FieldSpec(name="advantage", type="str", required=True, description="A very brief description of the advantage."),
                    FieldSpec(name="description", type="str", required=True, description="A description of the advantage."),
                    FieldSpec(name="item", type="str", required=True, description="An objectified literal or symbolic representation of the advantage.")
                ],
            ),
            description="One of the investigator's advantages, possibly literal or psychological."
        )
    registry.register(advantage[0])

    vibe = template_schema_tuple(
            template_str=(
                "Describe the weather, vibe, and urgency each in one sentence. "
                "Scene location: {scene.location}. Scene specifics: {scene.specifics}. "
                "Time: {atmosphere.hour_of_day}:{atmosphere.minute_of_day}, PM flag: {atmosphere.post_meridian}. "
                "Prior weather: {atmosphere.prior_weather}. Weather trend: {atmosphere.weather_changing}. "
                "Current timestep: {atmosphere.time_limit} out of 12."
            ),
            schema=Schema(
                name="Vibe",
                fields=[
                    FieldSpec(name="weather", type="str", required=True, description="Weather conditions."),
                    FieldSpec(name="vibe", type="str", required=True, description="Atmosphere feeling."),
                    FieldSpec(name="urgency", type="str", required=True, description="How urgent the moment feels."),
                ],
            ),
            description="Weather, mood, and urgency for the current scene.",
        )
    registry.register(vibe[0])

    little_house_description = simple_schema(
        "Little House Description",
        "description",
        "Provide one atmospheric sentence describing the Little House in {town.name}.",
    )
    registry.register(little_house_description[0])

    scene_description = simple_schema(
        "Scene Description",
        "description",
        "Write one vivid sentence for the current scene at {scene.location}: {scene.specifics}.",
    )
    registry.register(scene_description[0])

    invent_date = simple_schema(
        "Start Date",
        "date",
        "Invent an exact date just a couple days after {interview.newspaper.date}."
    )
    registry.register(invent_date[0])

    date_generator = simple_schema(
        "Current Date",
        "date",
        "What is the exact next day after {atmosphere.prior_date.date}. Be sure to increment the month and decrement to 1 of the month if necessary."
    )

    root = MappingNode(
        {
            "interview": MappingNode(
                {
                    "first_name_hint": ScalarNode(" starts with the letter T "),
                    "last_name_hint": ScalarNode(" ends with the letter R "),
                    "name_guess1": ResolvableNode(*investigator_name),
                    "name_guess2": ResolvableNode(*investigator_name),
                    "name_guess3": ResolvableNode(*investigator_name),
                    "name_guess4": ResolvableNode(*investigator_name),
                    "destination_hint": ScalarNode(" somewhere remote "),
                    "brochure1": ResolvableNode(*brochure),
                    "brochure2": ResolvableNode(*brochure),
                    "brochure3": ResolvableNode(*brochure),
                    "brochure4": ResolvableNode(*brochure),
                    "date_hint": ScalarNode(" in the late 20th century "),
                    "newspaper": ResolvableNode(*newspaper_title),
                    "case_hint": ScalarNode (" a strange case "),
                    "case1": ResolvableNode(*case_headline),
                    "case2": ResolvableNode(*case_headline),
                    "case3": ResolvableNode(*case_headline),
                    "case4": ResolvableNode(*case_headline),
                    "archetype_hint": ScalarNode(" investigator of the unknown and unknowable, but not an Occultist "),
                    "archetype1": ResolvableNode(
                        *investigator_archetype,
                        metadata={
                            "temperature": 1.0,
                            "extra": {
                                "seed": -1,
                                "max_tokens": 24,
                                "top_p": 0.98,
                                "top_k": 80,
                                "min_p": 0.05,
                                "repeat_penalty": 1.12,
                            },
                        },
                    ),
                    "archetype2": ResolvableNode(
                        *investigator_archetype,
                        metadata={
                            "temperature": 3.0,
                            "extra": {
                                "seed": -1,
                                "max_tokens": 24,
                                "top_p": 0.98,
                                "top_k": 80,
                                "min_p": 0.05,
                                "repeat_penalty": 1.12,
                            },
                        },
                    ),
                    "archetype3": ResolvableNode(
                        *investigator_archetype,
                        metadata={
                            "temperature": 5.0,
                            "extra": {
                                "seed": -1,
                                "max_tokens": 24,
                                "top_p": 0.98,
                                "top_k": 80,
                                "min_p": 0.05,
                                "repeat_penalty": 1.12,
                            },
                        },
                    ),
                    "archetype4": ResolvableNode(
                        *investigator_archetype,
                        metadata={
                            "temperature": 7.0,
                            "extra": {
                                "seed": -1,
                                "max_tokens": 24,
                                "top_p": 0.98,
                                "top_k": 80,
                                "min_p": 0.05,
                                "repeat_penalty": 1.12,
                            },
                        },
                    ),
                    "interest_hint": ScalarNode(" keen "),
                    "interest1": ResolvableNode(*interest),
                    "interest2": ResolvableNode(*interest),
                    "interest3": ResolvableNode(*interest),
                    "interest4": ResolvableNode(*interest),
                    "secrets_hint": ScalarNode(" dark "),
                    "secret1": ResolvableNode(*personal_secret),
                    "secret2": ResolvableNode(*personal_secret),
                    "secret3": ResolvableNode(*personal_secret),
                    "secret4": ResolvableNode(*personal_secret),
                    "advantage_hint": ScalarNode(" a useful item "),
                    "advantage1": ResolvableNode(*advantage),
                    "advantage2": ResolvableNode(*advantage),
                    "advantage3": ResolvableNode(*advantage),
                    "advantage4": ResolvableNode(*advantage),
                }
            ),
            "town": MappingNode(
                {
                    "name": place_holder,
                    "location": place_holder,
                    "economic": place_holder,
                    "backstory": place_holder,
                }
            ),
            "case": MappingNode(
                {
                    "name": place_holder,
                    "case_date": place_holder,
                    "description": place_holder,
                }
            ),
            "investigator": MappingNode(
                {
                    "name": place_holder,
                    "archetype": place_holder,
                    "attributes": ResolvableNode(*attributes_modifiers),
                    "wounds": ScalarNode(0),
                    "instability": ScalarNode(0),
                    "luck": ScalarNode(9),
                    "secrets": place_holder,
                    "interest": place_holder,
                    "advantages": SequenceNode([]),
                    "disadvantages": SequenceNode([]),
                }
            ),
            "notebook": MappingNode(
                {
                    "logbook": SequenceNode([]), # A list of daybooks organized by date.
                    "daybook": SequenceNode([]), # Every choice, clue, confirmed elements, and summarized descriptions of things
                    "diary": SequenceNode([]),
                    "clues": SequenceNode([]),
                }
            ),
            "public": MappingNode(
                {
                    "rumors": SequenceNode([]),
                    "conspiracies": SequenceNode([]),
                    "player": SequenceNode([]),
                }
            ),
            "locations": MappingNode(
                {
                    "Little House": MappingNode(
                        {
                            "name": ScalarNode("Little House"),
                            "description": ResolvableNode(*little_house_description),
                        }
                    )
                }
            ),
            "npcs": MappingNode(
                {
                    "Mable Jenner": MappingNode(
                        {
                            "first_name": ScalarNode("Mable"),
                            "last_name": ScalarNode("Jenner"),
                            "archetype": ScalarNode("Elderly Citizen"),
                            "attitude": ScalarNode(5),
                            "personality": ScalarNode("Clumsy"),
                            "motivation": ScalarNode("Duty"),
                            "location": ScalarNode("Little House"),
                        }
                    ),
                    "Daryl Jenner": MappingNode(
                        {
                            "first_name": ScalarNode("Daryl"),
                            "last_name": ScalarNode("Jenner"),
                            "archetype": ScalarNode("Elderly Citizen"),
                            "attitude": ScalarNode(4),
                            "personality": ScalarNode("Introverted"),
                            "motivation": ScalarNode("Survival"),
                            "location": ScalarNode("Little House"),
                        }
                    ),
                }
            ),
            "atmosphere": MappingNode(
                {
                    "hour_of_day": ScalarNode(8),
                    "minute_of_day": ScalarNode(20),
                    "post_meridian": ScalarNode(False),
                    "prior_date": ResolvableNode(*invent_date),
                    "date": ResolvableNode(*date_generator),
                    "weather_changing": ScalarNode("Not changing"),
                    "prior_weather": ScalarNode("Sunny"),
                    "vibe": ResolvableNode(*vibe),
                    "time_limit": ScalarNode(6),
                }
            ),
            "scene": MappingNode(
                {
                    "location": ScalarNode("Little House"),
                    "specifics": ScalarNode(
                        "You are sitting in a soft bed in a bedroom in the Little House, "
                        "chatting with Mable Jenners."
                    ),
                    "elements": SequenceNode(
                        [
                            "Mable on rocking chair",
                            "newspaper within reach",
                            "comfortable bed",
                            "my suitcase",
                            "eclectic electric lamp",
                            "alarm clock reading 8:20 A.M.",
                            "Mable holding a bible with a bookmark",
                            "something about the light and the view of outside from the window feels eerie",
                        ]
                    ),
                    "npcs": SequenceNode(["Mable Jenner"]),
                    "description": ResolvableNode(*scene_description),
                }
            ),
            "event": MappingNode(
                {
                    "description": ScalarNode("Investigation Progress"),
                    "progress": ScalarNode("You see a suspect at the Location"),
                    "consequences": ScalarNode(None),
                    "escalation": ScalarNode(1),
                }
            ),
            "action": MappingNode(
                {
                    "available": SequenceNode(
                        [
                            "Approach the front door cautiously",
                            "Observe the shadowy figure from a distance",
                            "Check the perimeter of the house",
                            "Call out to the figure in the window",
                        ]
                    ),
                    "chosen": ScalarNode(None),
                }
            ),
        }
    )


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

    Shows case details and investigator notes.
    """
    print(_SEP)
    print("INVESTIGATOR'S NOTEBOOK")
    print(_SEP)
    print(f"Case:        {_scalar(ctx, 'case', 'name')}")
    print(f"Summary:     {_scalar(ctx, 'case', 'description')}")
    print(f"Interest:    {_query_text(ctx, 'investigator', 'interest')}")

    clues_node = ctx.query(Path("notebook", "clues"))
    print("\nClues:")
    if isinstance(clues_node, SequenceNode) and len(clues_node) > 0:
        for i, clue in enumerate(clues_node, 1):
            print(f"  {i}. {_fmt(clue)}")
    else:
        print("  (none yet)")

    for key in ("logbook", "daybook", "diary"):
        entries = ctx.query(Path("notebook", key))
        print(f"\n{key.title()}:")
        if isinstance(entries, SequenceNode) and len(entries) > 0:
            for i, entry in enumerate(entries, 1):
                print(f"  {i}. {_fmt(entry)}")
        else:
            print("  (empty)")
    print()


def print_status(ctx: Context) -> None:
    """
    Window 2 – Investigator's Status.

    Displays the investigator identity, traits, and condition track.
    """
    print(_SEP)
    print("INVESTIGATOR STATUS")
    print(_SEP)
    first = _scalar(ctx, "investigator", "first_name")
    last = _scalar(ctx, "investigator", "last_name")
    print(f"Name:         {first} {last}")
    print(f"Archetype:    {_query_text(ctx, 'investigator', 'archetype')}")
    print(f"Interest:     {_query_text(ctx, 'investigator', 'interest')}")
    print(f"Wounds:       {_scalar(ctx, 'investigator', 'wounds')}")
    print(f"Instability:  {_scalar(ctx, 'investigator', 'instability')}")
    print(f"Luck:         {_scalar(ctx, 'investigator', 'luck')}")

    attrs = ctx.query(Path("investigator", "attributes"))
    print("\nAttributes:")
    if isinstance(attrs, ResolvableNode):
        if attrs.result is None:
            print("  (pending resolution)")
        else:
            print(f"  {_fmt(attrs.result)}")
    else:
        print(f"  {_fmt(attrs)}")

    print("\nAdvantages:")
    adv = ctx.query(Path("investigator", "advantages"))
    if isinstance(adv, SequenceNode) and len(adv) > 0:
        for i, item in enumerate(adv, 1):
            print(f"  {i}. {_fmt(item)}")
    else:
        print("  (none)")

    print("\nDisadvantages:")
    dis = ctx.query(Path("investigator", "disadvantages"))
    if isinstance(dis, SequenceNode) and len(dis) > 0:
        for i, item in enumerate(dis, 1):
            print(f"  {i}. {_fmt(item)}")
    else:
        print("  (none)")

    # Demonstrate lazy inference: show the node state before querying, then resolve.
    secrets_path = Path("investigator", "secrets")
    pre_node = _resolve_path(ctx.root, secrets_path)
    print(f"\nSecrets [before query]: {_fmt(pre_node)}")
    print(f"Secrets [after  query]: {_query_text(ctx, 'investigator', 'secrets')}")
    print()


def print_atmosphere(ctx: Context) -> None:
    """
    Window 3 – Atmosphere Description.

    Shows time and weather context, then resolves atmosphere.vibe lazily.
    """
    print(_SEP)
    print("ATMOSPHERE")
    print(_SEP)
    hour = _scalar(ctx, "atmosphere", "hour_of_day")
    minute = _scalar(ctx, "atmosphere", "minute_of_day")
    is_pm = _scalar(ctx, "atmosphere", "post_meridian")
    am_pm = "PM" if is_pm else "AM"
    print(f"Time:         {hour}:{int(minute):02d} {am_pm}" if isinstance(minute, int) else f"Time:         {hour}:{minute} {am_pm}")
    print(f"Prior weather: {_scalar(ctx, 'atmosphere', 'prior_weather')}")
    print(f"Trend:         {_scalar(ctx, 'atmosphere', 'weather_changing')}")
    print(f"Time limit:    {_scalar(ctx, 'atmosphere', 'time_limit')} / 12")

    vibe_path = Path("atmosphere", "vibe")
    pre_node = _resolve_path(ctx.root, vibe_path)
    print(f"Vibe [before query]: {_fmt(pre_node)}")
    print(f"Vibe [after  query]: {_query_text(ctx, 'atmosphere', 'vibe')}")
    print()


def print_scene(ctx: Context) -> None:
    """
    Window 4 – Scene Description.

    Shows location, scene details, and currently present NPCs.
    """
    print(_SEP)
    print("SCENE DESCRIPTION")
    print(_SEP)
    print(f"Location: {_scalar(ctx, 'scene', 'location')}")
    print(f"Details:  {_scalar(ctx, 'scene', 'specifics')}")

    # Description is a PromptNode in the new scene model.
    description_path = Path("scene", "description")
    pre_node = _resolve_path(ctx.root, description_path)
    print(f"Description [before query]: {_fmt(pre_node)}")
    print(f"Description [after  query]: {_query_text(ctx, 'scene', 'description')}")

    elements = ctx.query(Path("scene", "elements"))
    print("\nElements:")
    if isinstance(elements, SequenceNode) and len(elements) > 0:
        for i, item in enumerate(elements, 1):
            print(f"  {i}. {_fmt(item)}")
    else:
        print("  (none)")

    npcs_node = ctx.query(Path("scene", "npcs"))
    print("\nNPCs present:")
    if isinstance(npcs_node, SequenceNode) and len(npcs_node) > 0:
        for i, npc in enumerate(npcs_node, 1):
            print(f"  {i}. {_fmt(npc)}")
    else:
        print("  (none)")
    print()


def print_event(ctx: Context) -> None:
    """
    Window 5 – Event Description.

    Shows the event narrative and current escalation state.
    """
    print(_SEP)
    print("SCENE EVENT")
    print(_SEP)
    print(f"Description:    {_scalar(ctx, 'event', 'description')}")
    print(f"Progress:       {_scalar(ctx, 'event', 'progress')}")
    consequences = _scalar(ctx, "event", "consequences")
    print(f"Consequences:   {consequences if consequences is not None else '<pending>'}")
    print(f"Escalation:     {_scalar(ctx, 'event', 'escalation')} / 12")
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


# ---------------------------------------------------------------------------
# Game State Machine
# ---------------------------------------------------------------------------

# "The interview is the part of the game where the program 'extracts' the details of the situation from "
# "the player, basically a kind of interview based character creation process."
# "It follows these steps:"
# "1) The Player is welcomed to the game, and the Player 'wakes up' and is asked 'do you remember your name?'"
# "2) Archetype: The player is expected to then choose from one of 8 archetypes 'Oh yeah, I am a __, I remember that much!'"
# "2.5) OR the player could pick the option: 'No those aren't right ... <provide archetype_hint> and try to remember the truth!'"
# "3) Player Names: The Player is shown a list of names 1 to 4 and selects 'Oh, I think my name is __ __' "
# "3.5) OR the player could pick the option: 'No those aren't right ... friends I remember maybe? <provide name hints> and concentrate!'"
# "4) The player is welcomed by name, and is asked 'what do you remember?' and shown a brochure with 4 choices."
# "5) Town: The player is expected to answer with number 1 to 4 'Oh yeah, __ was where I was headed, I remember now!'"
# "5.5) OR the player could pick the option: 'None of those sound right ... <provide destination hint> and turn the page.' and go back to step 2)"
# "6) Newspaper Title: The player is expected to confirm the newspaper title."
# "6.5) OR the player says 'No, that's can't be.  My eyes are still fuzzy though, maybe I read it wrong. The day should be <provide date_hint>'"
# "7) Case: The player is expected to answer with number 1 to 4 'I\'m investigating the case of _'"
# "7.5) OR the player could pick the option: 'None of these ring a bell ... <provide case_hint> and turn to the next newspaper page.'"
# "8) Interest: The player is expected to answer with number 1 to 4 'I\'m not ready to tell Mable this, but my real reason for being here is __'"
# "8.5) OR the player could pick the option: 'Maybe somebody would expect something like that but they don't know my interest is <provide interest_hint>'"
# "9) Secrets: The player is expected to answer with number 1 to 4 'I\'ll never willingly tell a soul, but _.'"
# "9.5) OR the player could pick the option: 'Ha! My real secret is more <provide secret_hint>.'"
# "10) Then, until enough secrets are rolled, repeat step 9."
# "11) Advantages: The player is expected to answer with number 1 to 4 'I packed the __ because __.'"
# "11.5) OR the player could pick the option: 'No I felt I needed something for <suitcase_hint>.' and regenerate."
# "11.6) OR the player could pick the option: 'I didn't bring anything else with me.'"
# "12) Then, until up to 3 advantages have been rolled, repeat step 11."

class Phase(Enum):
    GAME_BEGIN = auto()
    INTERVIEW_CHOOSE_ARCHETYPE = auto()
    INTERVIEW_CHOOSE_NAME = auto()
    INTERVIEW_CHOOSE_TOWN = auto()
    INTERVIEW_CHOOSE_NEWSPAPER = auto()
    INTERVIEW_CHOOSE_CASE = auto()
    INTERVIEW_CHOOSE_INTEREST = auto()
    INTERVIEW_CHOOSE_SECRETS = auto()
    INTERVIEW_CHOOSE_ADVANTAGES = auto()
    GAME_OVER = auto()

class Verb(Enum):
    CMD_NOOP = auto()
    CMD_QUIT = auto()
    CMD_CHOICE = auto()
    CMD_NOPE = auto()
    CMD_CONFIRM = auto()
    CMD_HELP = auto()
    CMD_LOOK = auto()
    CMD_DO_CHOICE = auto()
    CMD_DO_GENERIC = auto()
    QUESTION_IS = auto()
    QUESTION = auto()

def to_verb(tokens: [str]) -> Verb:
    if tokens[0].upper() == 'QUIT' or tokens[0].upper() == 'EXIT':
        return Verb.CMD_QUIT
    if tokens[0].isnumeric():
        return Verb.CMD_CHOICE
    if tokens[0].upper() == 'N' or tokens[0].upper() == 'NO' or tokens[0].upper() == "NOPE":
        return Verb.CMD_NOPE
    if tokens[0].upper() == 'YES' or tokens[0].upper() == "Y":
        return Verb.CMD_CONFIRM
    if tokens[0].upper() == 'H' or tokens[0].upper() == "HELP":
        return Verb.CMD_HELP
    if tokens[0].upper() == 'L' or tokens[0].upper() == "LOOK" or tokens[0].upper() == 'Q' or tokens[0].upper() == 'QUERY':
        return Verb.CMD_LOOK
    if tokens[0].upper() == 'DO' and (len(tokens) <= 1):
        return Verb.CMD_DO_CHOICE
    if tokens[0].upper() == 'DO':
        return Verb.CMD_DO_GENERIC
    if tokens[0].upper() == 'IS' and tokens[-1][-1] == '?':
        return Verb.QUESTION_IS
    if tokens[-1][-1] == '?':
        return Verb.QUESTION
    return Verb.CMD_NOOP
    

@dataclass
class MachineState:
    phase: Phase = Phase.GAME_BEGIN

@dataclass
class Transition:
    next_phase: Phase
    prompt: str
    options: [Command]

@dataclass
class Command:
    verb: Verb = Verb.CMD_NOOP
    ordinal: int = 0
    arg: str = ""

cmd_noop = Command()

def to_command(tokens: [str]) -> Command:
    verb = to_verb(tokens)
    if (verb == Verb.CMD_CHOICE):
        ordinal = int(tokens[0])
    else:
        ordinal = 0
    if len(tokens) > 1:
        arg = tokens[1]
    else:
        arg = ""
    return Command(verb, ordinal, arg)

def is_compatible_command(prototype: Command, cmd: Command) -> bool:
    if prototype.verb == cmd.verb:
        if prototype.verb == Verb.CMD_CHOICE:
            return prototype.ordinal == cmd.ordinal
        return True
    return False

def is_valid_option(valid_options: [Command], cmd: Command) -> bool:
    if not valid_options and cmd.verb == Verb.CMD_NOOP:
        return True
    for valid in valid_options:
        if is_compatible_command(valid, cmd):
            return True
    return False

def option_to_prompt(option: Command) -> str:
    if option.verb == Verb.CMD_CHOICE:
        return f"{option.ordinal}: {option.arg}"
    if option.verb == Verb.CMD_NOPE:
        return f"No: {option.arg}"
    if option.verb == Verb.CMD_CONFIRM:
        return f"Yes: {option.arg}"
    return "" 

def to_prompt(options: [Command]) -> str:
    result = ""
    for option in options:
        result += option_to_prompt(option) + "\n"
    return result

Handler = Callable[[Context, MachineState, Command], Transition]


def handle_game_begin(ctx: Context, ms: MachineState, cmd: Command) -> Transition:
    if cmd.verb == Verb.CMD_CONFIRM:
        return handle_interview_choose_archetype(ctx, ms=MachineState(Phase.INTERVIEW_CHOOSE_ARCHETYPE), cmd=cmd_noop)
    prompt = (
        "You wake up with a dry mouth, and sticky eyes. Gradually, the room comes into focus. "
        "You sit up in a soft, clean bed, in a small room, with rustic furnishings. "
        "An old woman in a flower dress is rocking in a chair, watching you as you awake. "
        "She seems relived to see you stirring. "
        "You try to remember how you got here; it is gradually coming back to you. "
        'The woman says, "Hello, my name is Mable Jenners. I\'m sorry we had to meet under such ... frightening circumstances." '
        'Mable continues, "You took a nasty bump to the head. Do you remember why you are here?')
    options = [Command(verb=Verb.CMD_CONFIRM, arg="I'm here to start an investigation!")]
    return Transition(next_phase=ms.phase, prompt=prompt, options=options)

def handle_interview_choose_name(ctx: Context, ms: MachineState, cmd: Command) -> Transition:
    prompt = (
        "You're name? What was it exactly? You are pretty sure that: "
    )
    if cmd.verb == Verb.CMD_CHOICE:
        if cmd.ordinal == 1 and cmd.arg.strip() != "":
            _set_keyed_node(ctx, ScalarNode(cmd.arg), 'interview', 'first_name_hint')
        if cmd.ordinal == 2 and cmd.arg.strip() != "":
            _set_keyed_node(ctx, ScalarNode(cmd.arg), 'interview', 'last_name_hint')
        if cmd.ordinal == 3:
            node = ctx.query(Path('interview', 'name_guess1'))
        if cmd.ordinal == 4:
            node = ctx.query(Path('interview', 'name_guess2'))
        if cmd.ordinal == 5:
            node = ctx.query(Path('interview', 'name_guess3'))
        if cmd.ordinal == 6:
            node = ctx.query(Path('interview', 'name_guess4'))
        if cmd.ordinal >= 3 or cmd.ordinal <= 6:
            _set_keyed_node(ctx, node, 'investigator', 'name')
            return handle_interview_choose_town(ctx, MachineState(Phase.INTERVIEW_CHOOSE_ARCHETYPE), cmd_noop)
        options = [Command(verb=Verb.CMD_CHOICE, ordinal=1, arg=f"My first name: {_scalar(ctx, 'interview', 'first_name_hint')}")
                ,Command(verb=Verb.CMD_CHOICE, ordinal=2, arg=f"The rest of my name: {_scalar(ctx), 'interview', 'last_name_hint'}")
                ,Command(verb=Verb.CMD_NOPE, arg="At least you know that much; try thinking again ...")]
        return Transition(next_phase=ms.phase, prompt=prompt, options=options)
    if cmd.verb == Verb.CMD_NOPE or cmd.verb == Verb.CMD_CHOICE:
        _invalidate(ctx, 'interview', 'name_guess1')
        _invalidate(ctx, 'interview', 'name_guess2')
        _invalidate(ctx, 'interview', 'name_guess3')
        _invalidate(ctx, 'interview', 'name_guess4')
    options = [Command(verb=Verb.CMD_CHOICE, ordinal=1, arg=f"My first name: {_scalar(ctx, 'interview', 'first_name_hint')}")
              ,Command(verb=Verb.CMD_CHOICE, ordinal=2, arg=f"The rest of my name: {_scalar(ctx), 'interview', 'last_name_hint'}")
              ,Command(verb=Verb.CMD_CHOICE, ordinal=3, arg=query_investigator_name(ctx, NameMode.FULL, 'interview', 'name_guess1'))
              ,Command(verb=Verb.CMD_CHOICE, ordinal=4, arg=query_investigator_name(ctx, NameMode.FULL, 'interview', 'name_guess2'))
              ,Command(verb=Verb.CMD_CHOICE, ordinal=5, arg=query_investigator_name(ctx, NameMode.FULL, 'interview', 'name_guess3'))
              ,Command(verb=Verb.CMD_CHOICE, ordinal=6, arg=query_investigator_name(ctx, NameMode.FULL, 'interview', 'name_guess4'))
              ,Command(verb=Verb.CMD_NOPE, arg="Those sound like people with similiar interests; try thinking again ...")]
    return Transition(next_phase=ms.phase, prompt=prompt, options=options)

def handle_interview_choose_archetype(ctx: Context, ms: MachineState, cmd: Command) -> Transition:
    prompt = (
        "You know you are an investigator of some kind, but what kind exactly? Was it: "
    )
    if cmd.verb == Verb.CMD_CHOICE:
        if cmd.ordinal == 1 and cmd.arg.strip() != "":
            _set_keyed_node(ctx, ScalarNode(cmd.arg), 'interview', 'archetype_hint')
        if cmd.ordinal == 2:
            archetype = _query_text(ctx, 'interview', 'archetype1')
        if cmd.ordinal == 3:
            archetype = _query_text(ctx, 'interview', 'archetype2')
        if cmd.ordinal == 4:
            archetype = _query_text(ctx, 'interview', 'archetype3')
        if cmd.ordinal == 5:
            archetype = _query_text(ctx, 'interview', 'archetype4')
        if 2 <= cmd.ordinal <= 5:
            _set_keyed_node(ctx, ScalarNode(archetype), 'investigator', 'archetype')
            return handle_interview_choose_name(ctx, MachineState(Phase.INTERVIEW_CHOOSE_NAME), cmd_noop)
        options = [Command(verb=Verb.CMD_CHOICE, ordinal=1, arg=f"I describe myself as: {_scalar(ctx, 'interview', 'archetype_hint')}")
                ,Command(verb=Verb.CMD_NOPE, arg="You recall your methods and approach; try thinking again ...")]
        return Transition(next_phase=ms.phase, prompt=prompt, options=options)

    if cmd.verb == Verb.CMD_NOPE:
        _invalidate(ctx, 'interview', 'archetype1')
        _invalidate(ctx, 'interview', 'archetype2')
        _invalidate(ctx, 'interview', 'archetype3')
        _invalidate(ctx, 'interview', 'archetype4')
    options = [Command(verb=Verb.CMD_CHOICE, ordinal=1, arg=f"I describe myself as: {_scalar(ctx, 'interview', 'archetype_hint')}")
              ,Command(verb=Verb.CMD_CHOICE, ordinal=2, arg=f"{_query_text(ctx, 'interview', 'archetype1')}?")
              ,Command(verb=Verb.CMD_CHOICE, ordinal=3, arg=f"{_query_text(ctx, 'interview', 'archetype2')}?")
              ,Command(verb=Verb.CMD_CHOICE, ordinal=4, arg=f"{_query_text(ctx, 'interview', 'archetype3')}?")
              ,Command(verb=Verb.CMD_CHOICE, ordinal=5, arg=f"{_query_text(ctx, 'interview', 'archetype4')}?")
              ,Command(verb=Verb.CMD_NOPE, arg="None of those are exactly right; try thinking again ...")]
    return Transition(next_phase=ms.phase, prompt=prompt, options=options)


    # print('She continues, "How are you feeling?"')
    # print('You reply, "Ugh. Minor pains all over. I feel like I was in a car wreck!"')
    # print('Mable responds, "Oh, dear! Well, that is because _you were_ in a car wreck! I must say!"')
    # print('You reply, "Oh, wow! Uh ... where am I?"')
    # print('Mable replies, "Well, what do you remember?"')
    # print('I was travelling to a little town somewhere?  To investigate a crime?  Or something mysterious?')
    # print('Mable hands you a travel brochure and says: "We found this in your car. Does it help jog your memory?"')
    # print('You being reading the travel brochure. The Title is "Little Known Towns in Eerie Places!"')
    # print('Then it says, "Do you have investigative proclivities?  A need to _see_ beyond the veil?  Well, these are the places you should seriously consider visiting!"')
    # print("Type 'help' for commands and options.\n")

def handle_game_over(ctx: Context, ms: MachineState, cmd: Command) -> Transition:
    return Transition(Phase.GAME_OVER, should_quit=True)

HANDLERS: dict[Phase, Handler] = {
    Phase.GAME_BEGIN: handle_game_begin,
    Phase.INTERVIEW_CHOOSE_ARCHETYPE: handle_interview_choose_archetype,
    Phase.INTERVIEW_CHOOSE_NAME: handle_interview_choose_name,
    Phase.GAME_OVER: handle_game_over,
}

# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def format_prompt(prompt: str, usable_width: int) -> str:
    lines = []
    for raw_line in prompt.splitlines():
        if not raw_line.strip():
            lines.append("")
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        wrapped = textwrap.fill(
            raw_line.strip(),
            width=max(20, usable_width - indent),
            initial_indent=" " * indent,
            subsequent_indent=" " * indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.append(wrapped)

    return "\n".join(lines)


_HELP_TEXT = """
Commands:
  help                  Show this help.
    look notebook         Case, motivation, clues, and notes.
    look status           Investigator profile, stats, traits, and secrets.
    look atmosphere       Time, weather trend, and vibe (lazy inference).
    look scene            Scene details and description (lazy inference).
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

    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    usable_width = max(20, width - 2)

    print("=" * usable_width)
    print("  LITTLE HOUSE IN THE EERIE")
    print("  A Paranormal Investigation")
    print("=" * usable_width)
    print()

    ms = MachineState(phase=Phase.GAME_BEGIN)
    handler = HANDLERS[ms.phase]
    tr = handler(ctx, ms, cmd_noop)
    ms.phase = tr.next_phase
    prompt = tr.prompt + "\n"
    prompt += to_prompt(tr.options)
    print(format_prompt(prompt, usable_width))
    valid_options = tr.options

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFarewell, investigator.")
            break

        if not raw:
            continue

        tokens = raw.strip().split(maxsplit=1)
        cmd = to_command(tokens)

        if cmd.verb == Verb.CMD_QUIT:
            print("\nFarewell, investigator.")
            break

        elif cmd.verb == Verb.CMD_HELP:
            print(format_prompt(_HELP_TEXT, usable_width))
            
        elif cmd.verb == Verb.CMD_LOOK:
            target = cmd.arg
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
        
        elif not is_valid_option(valid_options, cmd):
            print(format_prompt(f"Command is not valid at this time: {to_str(cmd)}.", usable_width))
            continue

        handler = HANDLERS[ms.phase]
        tr = handler(ctx, ms, cmd)
        ms.phase = tr.next_phase
        prompt = tr.prompt + "\n"
        prompt += to_prompt(tr.options)
        print(format_prompt(prompt, usable_width))
        valid_options = tr.options


def main() -> None:
    """Entry point: build the game context and start the REPL."""
    ctx = build_initial_context()
    run_repl(ctx)


if __name__ == "__main__":
    main()
