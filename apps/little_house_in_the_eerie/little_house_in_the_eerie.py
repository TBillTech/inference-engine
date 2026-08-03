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

import random
import shutil
import textwrap

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
    print_action,
    print_atmosphere,
    print_event,
    print_notebook,
    print_scene,
    print_status,
)


_ORACLE_ANSWERS = [
    "Yes, and ... things are even better than expected.",
    "Yes, but ... there is a complication.",
    "No, but ... you gain some unexpected advantage.",
    "No, and ... the situation worsens.",
]


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


_HELP_TEXT = """
Commands:
  help                  Show this help.
    look notebook         Case, motivation, clues, and notes.
    look status           Investigator profile, stats, traits, and secrets.
    look atmosphere       Time, weather trend, and vibe (lazy inference).
    look scene            Scene details and description (lazy inference).
  look event            Current scene event and progress.
  look action           Available actions.
  y(es)                 Confirm the Yes entry or action.
  n(o(pe))              Choose the Nope entry or action. Often used to regenerate options.
  <n> argument          Replace entry n, or do action n, using the supplied typed argument.
  question <text>       Ask the Co-GM a yes/no oracle question.
  quit / exit           End the session.

'look' and 'query' are synonyms.
PromptNodes (marked as 'PromptNode(PENDING)') resolve the first time
they are queried and cache their result for subsequent queries.
"""

_BREIF_HELP = """
This game runs using a command line interface, where you type a command and then press enter.
help or h - will print out the full help instructions.
quit or exit - will exit the game
<n> argument - Will replace entry n, or do action n, using the supplied typed argument.
y - will confirm the Yes entry or option
n - will choose the nope entry or action, which commonly regenerates options.
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
    handler = HANDLERS[ms.phase]
    tr = apply_phase_changes(ctx, ms, handler(ctx, ms, cmd_noop))
    valid_options = render_prompt_options(tr, usable_width)

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
            print()

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
                pass
            else:
                print(
                    format_prompt(
                        f"Unknown target '{target}'. Try: notebook, status, atmosphere, scene, event, action\n",
                        usable_width,
                    )
                )

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
