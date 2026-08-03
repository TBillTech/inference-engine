from __future__ import annotations

from context_resolver.ast.nodes import SequenceNode
from context_resolver.ast.paths import Path
from context_resolver.ast.resolvable_node import ResolvableNode
from context_resolver.context.context import Context
from context_resolver.query.resolver import _resolve_path

from apps.little_house_in_the_eerie.context_access_helpers import fmt, query_text, scalar


SEP = "-" * 60


def print_notebook(ctx: Context) -> None:
    """Window 1: investigator notebook."""
    print(SEP)
    print("INVESTIGATOR'S NOTEBOOK")
    print(SEP)
    print(f"Case:        {scalar(ctx, 'case', 'name')}")
    print(f"Summary:     {scalar(ctx, 'case', 'description')}")
    print(f"Interest:    {query_text(ctx, 'investigator', 'interest')}")

    clues_node = ctx.query(Path("notebook", "clues"))
    print("\nClues:")
    if isinstance(clues_node, SequenceNode) and len(clues_node) > 0:
        for i, clue in enumerate(clues_node, 1):
            print(f"  {i}. {fmt(clue)}")
    else:
        print("  (none yet)")

    for key in ("logbook", "daybook", "diary"):
        entries = ctx.query(Path("notebook", key))
        print(f"\n{key.title()}:")
        if isinstance(entries, SequenceNode) and len(entries) > 0:
            for i, entry in enumerate(entries, 1):
                print(f"  {i}. {fmt(entry)}")
        else:
            print("  (empty)")
    print()


def print_status(ctx: Context) -> None:
    """Window 2: investigator status."""
    print(SEP)
    print("INVESTIGATOR STATUS")
    print(SEP)
    first = scalar(ctx, "investigator", "first_name")
    last = scalar(ctx, "investigator", "last_name")
    print(f"Name:         {first} {last}")
    print(f"Archetype:    {query_text(ctx, 'investigator', 'archetype')}")
    print(f"Interest:     {query_text(ctx, 'investigator', 'interest')}")
    print(f"Wounds:       {scalar(ctx, 'investigator', 'wounds')}")
    print(f"Instability:  {scalar(ctx, 'investigator', 'instability')}")
    print(f"Luck:         {scalar(ctx, 'investigator', 'luck')}")

    attrs = ctx.query(Path("investigator", "attributes"))
    print("\nAttributes:")
    if isinstance(attrs, ResolvableNode):
        if attrs.result is None:
            print("  (pending resolution)")
        else:
            print(f"  {fmt(attrs.result)}")
    else:
        print(f"  {fmt(attrs)}")

    print("\nAdvantages:")
    adv = ctx.query(Path("investigator", "advantages"))
    if isinstance(adv, SequenceNode) and len(adv) > 0:
        for i, item in enumerate(adv, 1):
            print(f"  {i}. {fmt(item)}")
    else:
        print("  (none)")

    print("\nDisadvantages:")
    dis = ctx.query(Path("investigator", "disadvantages"))
    if isinstance(dis, SequenceNode) and len(dis) > 0:
        for i, item in enumerate(dis, 1):
            print(f"  {i}. {fmt(item)}")
    else:
        print("  (none)")

    secrets_path = Path("investigator", "secrets")
    pre_node = _resolve_path(ctx.root, secrets_path)
    print(f"\nSecrets [before query]: {fmt(pre_node)}")
    print(f"Secrets [after  query]: {query_text(ctx, 'investigator', 'secrets')}")
    print()


def print_atmosphere(ctx: Context) -> None:
    """Window 3: atmosphere."""
    print(SEP)
    print("ATMOSPHERE")
    print(SEP)
    hour = scalar(ctx, "atmosphere", "hour_of_day")
    minute = scalar(ctx, "atmosphere", "minute_of_day")
    is_pm = scalar(ctx, "atmosphere", "post_meridian")
    am_pm = "PM" if is_pm else "AM"
    if isinstance(minute, int):
        print(f"Time:         {hour}:{minute:02d} {am_pm}")
    else:
        print(f"Time:         {hour}:{minute} {am_pm}")
    print(f"Prior weather: {scalar(ctx, 'atmosphere', 'prior_weather')}")
    print(f"Trend:         {scalar(ctx, 'atmosphere', 'weather_changing')}")
    print(f"Time limit:    {scalar(ctx, 'atmosphere', 'time_limit')} / 12")

    vibe_path = Path("atmosphere", "vibe")
    pre_node = _resolve_path(ctx.root, vibe_path)
    print(f"Vibe [before query]: {fmt(pre_node)}")
    print(f"Vibe [after  query]: {query_text(ctx, 'atmosphere', 'vibe')}")
    print()


def print_scene(ctx: Context) -> None:
    """Window 4: scene."""
    print(SEP)
    print("SCENE DESCRIPTION")
    print(SEP)
    print(f"Location: {scalar(ctx, 'scene', 'location')}")
    print(f"Details:  {scalar(ctx, 'scene', 'specifics')}")

    description_path = Path("scene", "description")
    pre_node = _resolve_path(ctx.root, description_path)
    print(f"Description [before query]: {fmt(pre_node)}")
    print(f"Description [after  query]: {query_text(ctx, 'scene', 'description')}")

    elements = ctx.query(Path("scene", "elements"))
    print("\nElements:")
    if isinstance(elements, SequenceNode) and len(elements) > 0:
        for i, item in enumerate(elements, 1):
            print(f"  {i}. {fmt(item)}")
    else:
        print("  (none)")

    npcs_node = ctx.query(Path("scene", "npcs"))
    print("\nNPCs present:")
    if isinstance(npcs_node, SequenceNode) and len(npcs_node) > 0:
        for i, npc in enumerate(npcs_node, 1):
            print(f"  {i}. {fmt(npc)}")
    else:
        print("  (none)")
    print()


def print_event(ctx: Context) -> None:
    """Window 5: event."""
    print(SEP)
    print("SCENE EVENT")
    print(SEP)
    print(f"Description:    {scalar(ctx, 'event', 'description')}")
    print(f"Progress:       {scalar(ctx, 'event', 'progress')}")
    consequences = scalar(ctx, "event", "consequences")
    print(f"Consequences:   {consequences if consequences is not None else '<pending>'}")
    print(f"Escalation:     {scalar(ctx, 'event', 'escalation')} / 12")
    print()


def print_action(ctx: Context) -> None:
    """Window 6: available actions."""
    print(SEP)
    print("AVAILABLE ACTIONS")
    print(SEP)
    available = ctx.query(Path("action", "available"))
    chosen_n = scalar(ctx, "action", "chosen")
    if isinstance(available, SequenceNode):
        for i, action in enumerate(available, 1):
            marker = "->" if i == chosen_n else "  "
            print(f" {marker} {i}. {fmt(action)}")
    if chosen_n is not None:
        print(f"\nChosen: action {chosen_n}")
    print()
