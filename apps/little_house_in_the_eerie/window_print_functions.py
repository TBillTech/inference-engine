from __future__ import annotations

from context_resolver.ast.nodes import SequenceNode, ScalarNode
from context_resolver.ast.paths import Path
from context_resolver.ast.resolvable_node import ResolvableNode
from context_resolver.context.context import Context
from context_resolver.query.resolver import _resolve_path

from apps.little_house_in_the_eerie.context_access_helpers import fmt, set_keyed_node


def _unique_preserve_order(values):
    """Return unique values in first-seen order without requiring hashability."""
    unique_values = []
    for value in values:
        if not any(existing == value for existing in unique_values):
            unique_values.append(value)
    return unique_values


def get_notebook_diary(ctx: Context, idx: int = -1) -> str:
    """Investigator diary page."""
    date_list = ctx.query(Path("notebook", "diary_dates"))
    dates = _unique_preserve_order(date_list)
    if len(dates) <= 0:
        return "Diary empty"
    idx = min(idx, len(dates)-1)
    idx = max(idx, -len(dates))
    element = dates[idx]
    entries = ctx.query(Path("notebook", "diary"))
    dated_entries = zip(date_list, entries)
    notebook_diary = ""
    for (date, entry) in dated_entries:
        if date == element:
            notebook_diary += fmt(entry)
            notebook_diary += "\n"
    return notebook_diary

def get_notebook_daybook(ctx: Context) -> str:
    """Investigator day book."""
    day_book = ""
    daybook_node = ctx.query(Path("notebook", "daybook"))
    for i, blurb in enumerate(daybook_node, 1):
        day_book += f"{fmt(blurb)}\n"
    return day_book

def get_notebook_log(ctx: Context, idx: int = -1) -> str:
    """Investigator log page."""
    date_list = ctx.query(Path("notebook", "logbook_dates"))
    dates = _unique_preserve_order(date_list)
    idx = min(idx, len(dates)-1)
    idx = max(idx, -len(dates))
    element = dates[idx]
    entries = ctx.query(Path("notebook", "logbook"))
    dated_entries = zip(date_list, entries)
    notebook_log = ""
    for (date, entry) in dated_entries:
        if date == element:
            notebook_log += entry
            notebook_log += "\n"
    return notebook_log

def get_clue_summary(ctx: Context) -> str:
    clue_summary = ""
    clues_node = ctx.query(Path("notebook", "clues"))
    clue_summary += "Clues:\n"
    if isinstance(clues_node, SequenceNode) and len(clues_node) > 0:
        for i, clue in enumerate(clues_node, 1):
            clue_summary += f"  {i}. {fmt(clue)}\n"
    else:
        clue_summary += "  (none yet)\n"
    return clue_summary

def get_case_summary(ctx: Context) -> str:
    return ctx.query_text('print_summaries', 'case_summary')

def get_investigator_summary(ctx: Context) -> str:
    return ctx.query_text('print_summaries', 'investigator_summary')

def get_secrets_summary(ctx: Context) -> str:
    return ctx.query_text('print_summaries', 'secrets_summary')
    
def get_town_summary(ctx: Context) -> str:
    town_data = ""
    town_data += f"Name: {ctx.query_text('town', 'name')}\n"
    town_data += f"Location: {ctx.query_text('town', 'location')}\n"
    town_data += f"Economics: {ctx.query_text('town', 'economic')}\n"
    node = ctx.query(Path('locations'))
    location_keys = list(node.keys())
    town_data += f"Places: {location_keys}\n"
    old_town_data = ctx.query_text('print_summaries', 'town_data')
    if town_data != old_town_data:
        set_keyed_node(ctx, ScalarNode(town_data), 'print_summaries', 'town_data')
    return ctx.query_text('print_summaries', 'town_summary')

def get_location_summary(ctx: Context, search: str) -> str:
    node = ctx.query(Path('locations'))
    best_location = ("", None)
    best_match = 0
    for (location_key, location_value) in node.items():
        match = 0
        for i in range(len(search)):
            if i >= len(location_key):
                break
            if location_key[i:i+1].lower() == search[i:i+1].lower():
                match += 1
        if match > best_match:
            best_location = (location_key, location_value)
            best_match = match
    if best_location[1] == None:
        return ""
    location_node = ctx.query(Path('locations', best_location[0]))
    location_text = ""
    for (key, value) in location_node.items():
        location_text += f"{key}: {ctx.query_text('locations', best_location[0], key)}\n"
    set_keyed_node(ctx, ScalarNode(location_text), 'print_summaries', 'location_data')
    return ctx.query_text('print_summaries', 'location_summary')

def get_scene_summary(ctx: Context) -> str:
    return ctx.query_text("print_summaries", "scene_summary")

def get_vibe_summary(ctx: Context) -> str:
    return ctx.query_text("print_summaries", "vibe_summary")