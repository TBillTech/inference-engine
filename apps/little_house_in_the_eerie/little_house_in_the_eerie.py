"""
Little House in the Eerie - A solo Horror Investigation.

This module keeps the command REPL and delegates:
- context access/query helpers
- context construction
- window print functions
- game state machine

Run with::

    python -m apps.little_house_in_the_eerie.little_house_in_the_eerie
"""

from __future__ import annotations

import pickle
import random
import shutil
import textwrap
from pathlib import Path as FilePath

from context_resolver.ast.nodes import MappingNode, SequenceNode
from context_resolver.ast.resolvable_node import ResolvableNode
from context_resolver.context.context import Context

from apps.little_house_in_the_eerie.context_construction import build_initial_context
from apps.little_house_in_the_eerie.game_state_machine import (
    HANDLERS,
    Phase,
    Verb,
    apply_phase_changes,
    is_valid_option,
    to_command,
    to_str,
    to_prompt,
    cmd_noop,
    MachineState,
    Command,
    PromptOptions,
)
from apps.little_house_in_the_eerie.window_print_functions import (
    get_notebook_diary,
    get_notebook_daybook,
    get_notebook_log,
    get_clue_summary,
    get_case_summary,
    get_investigator_summary,
    get_secrets_summary,
    get_town_summary,
    get_location_summary,
    get_scene_summary,
    get_vibe_summary,
)


_ORACLE_ANSWERS = [
    "Yes, and ... things are even better than expected.",
    "Yes, but ... there is a complication.",
    "No, but ... you gain some unexpected advantage.",
    "No, and ... the situation worsens.",
]

_SAVE_FILE_NAME = "little_house_in_the_eerie.pkl"


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


def render_prompt_options(po: PromptOptions, usable_width: int) -> list[Command]:
    prompt = po.prompt + "\n" + to_prompt(po.options)
    print(format_prompt(prompt, usable_width))
    return po.options


def _save_path(slot: str) -> FilePath:
    slot = slot.strip()
    if not slot:
        return FilePath.cwd() / _SAVE_FILE_NAME
    return FilePath.cwd() / f"save_{slot}.pkl"


def _save_game(ctx: Context, phase: Phase, slot: str) -> FilePath:
    save_path = _save_path(slot)
    payload = {
        "phase": phase.name,
        "context": ctx.to_dict(),
    }
    with save_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return save_path


def _refresh_resolvable_bindings(ctx: Context) -> None:
    """Re-infer bindings/dependencies from current templates after load."""
    stack = [ctx.root]
    while stack:
        node = stack.pop()
        if isinstance(node, MappingNode):
            stack.extend(child for _, child in node.items())
            continue
        if isinstance(node, SequenceNode):
            stack.extend(node)
            continue
        if isinstance(node, ResolvableNode):
            old_bindings = node.input_bindings
            old_dependencies = node.dependencies
            old_configured = node._configured
            node.input_bindings = {}
            node.dependencies = []
            node._configured = False
            try:
                node.configure(ctx)
            except ValueError:
                # If a template is not in the current registry, keep loaded bindings.
                node.input_bindings = old_bindings
                node.dependencies = old_dependencies
                node._configured = old_configured
            if node.result is not None:
                stack.append(node.result)


def _load_game(slot: str) -> tuple[Context, Phase]:
    load_path = _save_path(slot)
    with load_path.open("rb") as handle:
        payload = pickle.load(handle)

    fresh_context = build_initial_context()
    loaded_context = Context.from_dict(payload["context"], resolver=fresh_context.resolver)
    _refresh_resolvable_bindings(loaded_context)
    phase_name = payload.get("phase", Phase.GAME_BEGIN.name)
    phase = Phase[phase_name]
    return loaded_context, phase


def _prompt_from_current_state(ctx: Context, ms: MachineState, usable_width: int) -> list[Command]:
    handler = HANDLERS[ms.phase]
    tr = apply_phase_changes(ctx, ms, handler(ctx, ms, cmd_noop))
    return render_prompt_options(tr, usable_width)


_HELP_TEXT = """
Commands are generally case insensitive. Some commands have arguments.
Usually you should first type a command, followed by a space and then the argument afterward (if the command takes it).
Commands:
  help                  Show this help.
  quit / exit           End the session.
  save <n>              Save the game to save_<n>.pkl in the current directory.
  load <n>              Load the game from save_<n>.pkl in the current directory.
  look diary <n>        Look at the diary page n counting up from the beginning, or if negative, backward from the end.
  look daybook          Look at the full current day log
  look daybook <n>      Look at the full day log for day n counting up from the beginning, or if negative, backward from the end.
  look clues            Look at a summary of the current clues you have unconvered.
  look case             Look at the summary of the case so far.
  look investigator     Look at the summary of the investigator as a character page.
  look secrets          Examine the secrets of the investigator.
  look town             Look at the summary of the town and the places you know about.
  look location <place> Look at the summary of a place you know about in the town.
  look scene            Look at the current scene summary. 
  y(es)                 Confirm the Yes entry or action.
  n(o(pe))              Choose the Nope entry or action. Often used to regenerate options.
  <n> argument          Replace entry n, or do action n, using the supplied typed argument.
  question <text>       Ask the Co-GM a yes/no oracle question.
  do argument           Inside the fiction, initiate the task described in argument.          

'look' and 'query' are synonyms.
"""

_BREIF_HELP = """
This game runs using a command line interface, where you type a command and then press enter.
Here are the most commonly used commands:
  help or h - will print out the full help instructions.
  quit or exit - will exit the game
  <n> argument - Will replace entry n, or do action n, using the supplied typed argument.
  y - will confirm the Yes entry or option
  n - will choose the Nope entry or action, which commonly regenerates options.
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
    print(format_prompt(_BREIF_HELP, usable_width))
    print()

    ms = MachineState(phase=Phase.GAME_BEGIN)
    valid_options = _prompt_from_current_state(ctx, ms, usable_width)

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

        elif cmd.verb == Verb.CMD_SAVE:
            try:
                save_path = _save_game(ctx, ms.phase, cmd.arg)
            except OSError as exc:
                print(format_prompt(f"Could not save game: {exc}", usable_width))
                print()
                continue
            print(format_prompt(f"Saved game to {save_path}", usable_width))
            print()
            continue

        elif cmd.verb == Verb.CMD_LOAD:
            try:
                ctx, ms.phase = _load_game(cmd.arg)
            except FileNotFoundError:
                print(format_prompt(f"Could not load game: {_save_path(cmd.arg)} does not exist.", usable_width))
                print()
                continue
            except (OSError, KeyError, ValueError, pickle.PickleError) as exc:
                print(format_prompt(f"Could not load game: {exc}", usable_width))
                print()
                continue

            valid_options = _prompt_from_current_state(ctx, ms, usable_width)
            continue

        elif cmd.verb == Verb.CMD_HELP:
            print(format_prompt(_HELP_TEXT, usable_width))
            print()

        elif cmd.verb == Verb.CMD_LOOK:
            target = cmd.arg.split()
            print_str = None
            if target[0] == "diary":
                print_str = get_notebook_diary(ctx, int(target[1]) if len(target) > 1 else -1)
            elif target[0] == "daybook":
                print_str = get_notebook_daybook(ctx)
            elif target[0] == "daybook":
                print_str = get_notebook_log(ctx, int(target[1]) if len(target) > 1 else -1)
            elif target[0] == "clues":
                print_str = get_clue_summary(ctx)
            elif target[0] == "case":
                print_str = get_case_summary(ctx)
            elif target[0] == "investigator":
                print_str = get_investigator_summary(ctx)
            elif target[0] == "secrets":
                print_str = get_secrets_summary(ctx)
            elif target[0] == "town":
                print_str = get_town_summary(ctx)
            elif target[0] == "location":
                print_str = get_location_summary(ctx, target[1])
            elif target[0] == "scene":
                print_str = get_scene_summary(ctx)
            elif target[0] == "vibe":
                print_str = get_vibe_summary(ctx)
            if print_str == None:
                print(
                    format_prompt(
                        f"Unknown target '{target}'. Try: notebook, status, atmosphere, scene, event, action\n",
                        usable_width,
                    )
                )
            else:
                print(format_prompt(print_str+"\n", usable_width))

        elif cmd.verb in (Verb.QUESTION, Verb.QUESTION_IS):
            answer = random.choice(_ORACLE_ANSWERS)
            print(format_prompt(f"Oracle: {answer}", usable_width))
            print()

        elif not is_valid_option(valid_options, cmd):
            if cmd.verb == Verb.CMD_NOOP:
                print(format_prompt("I don't know how to do that command at this time.", usable_width))
            else:
                print(format_prompt(f"At this time, I do not know how to do this: {to_str(cmd)}.", usable_width))
            print(format_prompt(_BREIF_HELP, usable_width))
            print()
            continue

        handler = HANDLERS[ms.phase]
        tr = apply_phase_changes(ctx, ms, handler(ctx, ms, cmd))
        valid_options = render_prompt_options(tr, usable_width)


def main() -> None:
    """Entry point: build the game context and start the REPL."""
    ctx = build_initial_context()
    run_repl(ctx)


if __name__ == "__main__":
    main()
