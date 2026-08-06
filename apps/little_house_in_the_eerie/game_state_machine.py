from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

from context_resolver.ast.nodes import ScalarNode, SequenceNode
from context_resolver.ast.paths import Path
from context_resolver.context.context import Context

from apps.little_house_in_the_eerie.context_access_helpers import (
    NameMode,
    append_not_hint,
    append_sequence_node,
    confirm_and_freeze,
    get_not_hint,
    invalidate,
    query_advantage,
    query_case,
    query_investigator_name,
    query_newspaper,
    query_town,
    query_town_short,
    set_keyed_node,
    set_not_hint,
)


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
    CMD_SAVE = auto()
    CMD_LOAD = auto()
    CMD_CHOICE = auto()
    CMD_NOPE = auto()
    CMD_CONFIRM = auto()
    CMD_HELP = auto()
    CMD_LOOK = auto()
    CMD_DO_CHOICE = auto()
    CMD_DO_GENERIC = auto()
    QUESTION_IS = auto()
    QUESTION = auto()


@dataclass
class MachineState:
    phase: Phase = Phase.GAME_BEGIN


@dataclass
class Transition:
    next_phase: Phase

    @classmethod
    def phase_change(cls, next_phase: Phase) -> "Transition":
        return cls(next_phase=next_phase)

    @classmethod
    def prompt_options(cls, prompt: str, options: list["Command"]) -> "PromptOptions":
        return PromptOptions(prompt=prompt, options=options)


@dataclass
class Command:
    verb: Verb = Verb.CMD_NOOP
    ordinal: int = 0
    arg: str = ""


@dataclass
class PromptOptions:
    prompt: str
    options: list[Command] = field(default_factory=list)


cmd_noop = Command()


Handler = Callable[[Context, MachineState, Command], Transition | PromptOptions]
PhaseTransitionHandler = Callable[[Context, Phase], None]


def to_verb(tokens: list[str]) -> Verb:
    if tokens[0].upper() in ("QUIT", "EXIT"):
        return Verb.CMD_QUIT
    if tokens[0].upper() == "SAVE":
        return Verb.CMD_SAVE
    if tokens[0].upper() == "LOAD":
        return Verb.CMD_LOAD
    if tokens[0].isnumeric():
        return Verb.CMD_CHOICE
    if tokens[0].upper() in ("N", "NO", "NOPE"):
        return Verb.CMD_NOPE
    if tokens[0].upper() in ("YES", "Y"):
        return Verb.CMD_CONFIRM
    if tokens[0].upper() in ("H", "HELP"):
        return Verb.CMD_HELP
    if tokens[0].upper() in ("L", "LOOK", "Q", "QUERY"):
        return Verb.CMD_LOOK
    if tokens[0].upper() == "DO" and len(tokens) <= 1:
        return Verb.CMD_DO_CHOICE
    if tokens[0].upper() == "DO":
        return Verb.CMD_DO_GENERIC
    if tokens[0].upper() == "IS" and tokens[-1][-1] == "?":
        return Verb.QUESTION_IS
    if tokens[-1][-1] == "?":
        return Verb.QUESTION
    return Verb.CMD_NOOP


def to_command(tokens: list[str]) -> Command:
    verb = to_verb(tokens)
    ordinal = int(tokens[0]) if verb == Verb.CMD_CHOICE else 0
    arg = tokens[1] if len(tokens) > 1 else ""
    return Command(verb, ordinal, arg)


def to_str(cmd: Command) -> str:
    if cmd.verb == Verb.CMD_CHOICE:
        return f"Choice {cmd.ordinal}: {cmd.arg}"
    if not cmd.arg:
        return str(cmd.verb)
    return f"{str(cmd.verb)}: {cmd.arg}"


def is_compatible_command(prototype: Command, cmd: Command) -> bool:
    if prototype.verb == cmd.verb:
        if prototype.verb == Verb.CMD_CHOICE:
            return prototype.ordinal == cmd.ordinal
        return True
    return False


def is_valid_option(valid_options: list[Command], cmd: Command) -> bool:
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


def to_prompt(options: list[Command]) -> str:
    result = ""
    for option in options:
        result += option_to_prompt(option) + "\n"
    return result


def sequence_len(ctx: Context, *segments: str) -> int:
    node = ctx.query(Path(*segments))
    return len(node) if isinstance(node, SequenceNode) else 0


def post_interview_initialization(ctx: Context) -> None:
    """Run one-time setup when leaving interview flow."""
    secret_count = sequence_len(ctx, "investigator", "secrets")
    set_keyed_node(ctx, ScalarNode(secret_count), "investigator", "instability")

    ctx.query_text("investigator", "attributes")

    name = query_investigator_name(ctx, NameMode.FULL)
    archetype = ctx.query_text("investigator", "archetype")
    case_name = ctx.query_text("case", "name")
    case_summary = ctx.query_text("case", "description")
    interest = ctx.query_text("investigator", "interest", "interest")
    advantages = sequence_len(ctx, "investigator", "advantages")
    atmosphere_date = ctx.query_text("atmosphere", "date")
    daybook_info = (
        f"{name}'s {archetype} Investigative Journal\n"
        f"Day: {atmosphere_date}\n"
    )
    summary = (
        "I have recovered my memory. I was in an accident where the bridge to the bed and breakfast collapsed while I was trying to cross it. "
        "I feel fine, and though the rental car was totalled, it was also insured. This is not my focus, and I plan to let the insurance company deal with it. "
        "Mable and Earl Jenner, the prioprietors of the B&B, seem relieved I'm not planning to sue them separately. "
        "They have promised to let me stay for free for one month, and Earl Jenner has agreed to drive me around town. "
        "Mable has also helped me recover my memory by gently asking questions ... but there is still something suspicious about her attitude. "
        f"At any rate, I have arrived in {query_town_short(ctx)}\n"
        f"I am here to pursue my next case: {case_name}. {case_summary}.\n"
        f"I have a personal stake in this case: {interest}. "
    )
    set_keyed_node(ctx, ScalarNode(summary), "notebook", "diary_raw")
    diary_summary = ctx.query_text("notebook", "diary_entry")

    append_sequence_node(ctx, ("notebook", "daybook"), ScalarNode(daybook_info))
    append_sequence_node(ctx, ("notebook", "daybook"), ScalarNode(summary))
    append_sequence_node(ctx, ("notebook", "diary_dates"), ScalarNode(ctx.query_text("atmosphere", "date")))
    append_sequence_node(ctx, ("notebook", "diary"), ScalarNode(diary_summary))


def on_exit_interview_choose_advantages(ctx: Context, phase: Phase) -> None:
    """OUT hook for interview teardown and setup before gameplay starts."""
    post_interview_initialization(ctx)


OUT_TRANSITION_HANDLERS: dict[Phase, list[PhaseTransitionHandler]] = {
    Phase.INTERVIEW_CHOOSE_ADVANTAGES: [on_exit_interview_choose_advantages],
}

IN_TRANSITION_HANDLERS: dict[Phase, list[PhaseTransitionHandler]] = {}


def run_phase_transition_handlers(
    ctx: Context,
    phase: Phase,
    handlers: dict[Phase, list[PhaseTransitionHandler]],
) -> None:
    for handler in handlers.get(phase, []):
        handler(ctx, phase)


def apply_phase_changes(ctx: Context, ms: MachineState, tr: Transition | PromptOptions) -> PromptOptions:
    """Resolve phase changes until a prompt/options payload is produced."""
    current = tr
    while isinstance(current, Transition):
        from_phase = ms.phase
        run_phase_transition_handlers(ctx, from_phase, OUT_TRANSITION_HANDLERS)
        ms.phase = current.next_phase
        run_phase_transition_handlers(ctx, ms.phase, IN_TRANSITION_HANDLERS)
        current = HANDLERS[ms.phase](ctx, ms, cmd_noop)
    return current


def handle_game_begin(ctx: Context, ms: MachineState, cmd: Command) -> Transition | PromptOptions:
    if cmd.verb == Verb.CMD_CONFIRM:
        set_not_hint(ctx, "Occultist")
        return Transition(Phase.INTERVIEW_CHOOSE_ARCHETYPE)
    prompt = (
        "You wake up with a dry mouth, and sticky eyes. Gradually, the room comes into focus. "
        "You sit up in a soft, clean bed, in a small room, with rustic furnishings. "
        "An old woman in a flower dress is rocking in a chair, watching you as you awake. "
        "She seems relived to see you stirring. "
        "You try to remember how you got here; it is gradually coming back to you. "
        'The woman says, "Hello, my name is Mable Jenners. I\'m sorry we had to meet under such ... frightening circumstances." '
        'Mable continues, "You took a nasty bump to the head. Do you remember why you are here?'
    )
    options = [Command(verb=Verb.CMD_CONFIRM, arg="I think I'm here to start an investigation?")]
    return PromptOptions(prompt=prompt, options=options)


def handle_interview_choose_name(ctx: Context, ms: MachineState, cmd: Command) -> Transition | PromptOptions:
    if cmd.verb == Verb.CMD_CHOICE and 1 <= cmd.ordinal <= 5 and cmd.arg.strip() == "":
        return handle_interview_choose_name(ctx, ms, cmd_noop)

    prompt = "Your name? What was it exactly? You are pretty sure that: "
    if cmd.verb == Verb.CMD_CHOICE:
        if cmd.ordinal == 1:
            set_not_hint(ctx, cmd.arg)
        if cmd.ordinal == 2:
            set_keyed_node(ctx, ScalarNode(cmd.arg), "interview", "first_name_hint")
        if cmd.ordinal == 3:
            set_keyed_node(ctx, ScalarNode(cmd.arg), "interview", "last_name_hint")
        if cmd.ordinal == 4:
            set_keyed_node(ctx, ScalarNode(cmd.arg), "interview", "honorific_hint")
        if cmd.ordinal == 5:
            set_keyed_node(ctx, ScalarNode(cmd.arg), "interview", "suffix_hint")
        invalidate(ctx, "interview", "name_guess1")
    if cmd.verb == Verb.CMD_CONFIRM:
        confirm_and_freeze(ctx, ("interview", "name_guess1"), ("investigator", "name"), clear_not_hint=False)
        set_not_hint(ctx, "Oakhaven, the logging town")
        return Transition(Phase.INTERVIEW_CHOOSE_TOWN)
    if cmd.verb == Verb.CMD_NOPE:
        append_not_hint(ctx, query_investigator_name(ctx, NameMode.FULL, "interview", "name_guess1"))
        invalidate(ctx, "interview", "name_guess1")

    options = [
        Command(verb=Verb.CMD_CHOICE, ordinal=1, arg=f"My name is not: {get_not_hint(ctx)}"),
        Command(verb=Verb.CMD_CHOICE, ordinal=2, arg=f"My first name: {ctx.query_text('interview', 'first_name_hint')}"),
        Command(verb=Verb.CMD_CHOICE, ordinal=3, arg=f"The rest of my name: {ctx.query_text('interview', 'last_name_hint')}"),
        Command(verb=Verb.CMD_CHOICE, ordinal=4, arg=f"My honorific is: {ctx.query_text('interview', 'honorific_hint')}"),
        Command(verb=Verb.CMD_CHOICE, ordinal=5, arg=f"My suffix is: {ctx.query_text('interview', 'suffix_hint')}"),
        Command(verb=Verb.CMD_CONFIRM, arg=query_investigator_name(ctx, NameMode.FULL, "interview", "name_guess1")),
        Command(verb=Verb.CMD_NOPE, arg="That sounds like someone with a similiar interest but; try thinking again ..."),
    ]
    return PromptOptions(prompt=prompt, options=options)


def handle_interview_choose_archetype(ctx: Context, ms: MachineState, cmd: Command) -> Transition | PromptOptions:
    if cmd.verb == Verb.CMD_CHOICE and cmd.ordinal in (1, 2) and cmd.arg.strip() == "":
        return handle_interview_choose_archetype(ctx, ms, cmd_noop)

    prompt = "You know you are an investigator of some kind, but what kind exactly?"
    if cmd.verb == Verb.CMD_CHOICE:
        if cmd.ordinal == 1 and cmd.arg.strip() != "":
            set_keyed_node(ctx, ScalarNode(cmd.arg), "interview", "but_not_hint")
        if cmd.ordinal == 2 and cmd.arg.strip() != "":
            set_keyed_node(ctx, ScalarNode(cmd.arg), "interview", "archetype_hint")
        invalidate(ctx, "interview", "archetype1")
    if cmd.verb == Verb.CMD_CONFIRM:
        archetype = ctx.query_text("interview", "archetype1")
        set_keyed_node(ctx, ScalarNode(archetype), "investigator", "archetype")
        set_not_hint(ctx, "")
        return Transition(Phase.INTERVIEW_CHOOSE_NAME)
    if cmd.verb == Verb.CMD_NOPE:
        append_not_hint(ctx, ctx.query_text("interview", "archetype1"))
        invalidate(ctx, "interview", "archetype1")

    options = [
        Command(verb=Verb.CMD_CHOICE, ordinal=1, arg=f"You are _not_ a: {get_not_hint(ctx)}"),
        Command(verb=Verb.CMD_CHOICE, ordinal=2, arg=f"You describe yourself as: {ctx.query_text('interview', 'archetype_hint')}"),
        Command(verb=Verb.CMD_CONFIRM, arg=f"{ctx.query_text('interview', 'archetype1')}?"),
        Command(verb=Verb.CMD_NOPE, arg="You recall your methods and approach; try thinking again ..."),
    ]
    return PromptOptions(prompt=prompt, options=options)


def handle_interview_choose_town(ctx: Context, ms: MachineState, cmd: Command) -> Transition | PromptOptions:
    if cmd.verb == Verb.CMD_CHOICE and cmd.ordinal in (1, 2) and cmd.arg.strip() == "":
        return handle_interview_choose_town(ctx, ms, cmd_noop)

    prompt = (
        '"Nice to meet you?" you say with a querilous turn, '
        '"My name is '
        + query_investigator_name(ctx, NameMode.FULL)
    )
    prompt += f' and I am a {ctx.query_text("investigator", "archetype")}."\n'
    prompt += (
        '"Hmm, yes, I see", she continues, "How are you feeling?"\n'
        'You reply, "Ugh. Minor pains all over. I feel like I was in a car wreck!"\n'
        'Mable responds, "Oh, dear! Well, that is because _you were_ in a car wreck!"\n'
        '"I must say!" Mable looks out the window, as if uncomfortable or maybe nervous.\n'
        'You exclaim, "What?! Well ... where am I now?"\n'
        'Mable replies, "Hum. Hmm. Well, what do you remember?"\n'
        "Mable's responses seem a little ... off to you, but she asks a good question. "
        "You seem to recall you were travelling to a little town somewhere? "
        "To investigate a crime? Or something mysterious?\n"
        'Mable notices your confusion and hands you a travel brochure and says: '
        '"We found this in your car. Does it help jog your memory?"\n'
        'It is a travel brochure, and you begin to remember as you read it. The Title is "Little Known Towns in Eerie Places!"\n'
        'It says, "Do you have investigative proclivities?  A need to _see_ beyond the veil?  Well, these are the places no investigator should ignore!"'
    )

    if cmd.verb == Verb.CMD_CHOICE:
        if cmd.ordinal == 1:
            set_not_hint(ctx, cmd.arg)
        if cmd.ordinal == 2 and cmd.arg.strip() != "":
            set_keyed_node(ctx, ScalarNode(cmd.arg), "interview", "destination_hint")
        invalidate(ctx, "interview", "brochure1")
    if cmd.verb == Verb.CMD_CONFIRM:
        confirm_and_freeze(ctx, ("interview", "brochure1"), ("town",))
        set_not_hint(ctx, "Occult Gazette")
        return Transition(Phase.INTERVIEW_CHOOSE_NEWSPAPER)
    if cmd.verb == Verb.CMD_NOPE:
        append_not_hint(ctx, query_town_short(ctx, "interview", "brochure1"))
        invalidate(ctx, "interview", "brochure1")

    options = [
        Command(verb=Verb.CMD_CHOICE, ordinal=1, arg=f"The town was not: {get_not_hint(ctx)}"),
        Command(verb=Verb.CMD_CHOICE, ordinal=2, arg=f"You recall the town was: {ctx.query_text('interview', 'destination_hint')}"),
        Command(verb=Verb.CMD_CONFIRM, arg=f"{query_town(ctx, 'interview', 'brochure1')}"),
        Command(verb=Verb.CMD_NOPE, arg="That wasn't the place; turn to the next page ..."),
    ]
    return PromptOptions(prompt=prompt, options=options)


def handle_interview_choose_newspaper(ctx: Context, ms: MachineState, cmd: Command) -> Transition | PromptOptions:
    if cmd.verb == Verb.CMD_CHOICE and cmd.ordinal in (1, 2) and cmd.arg.strip() == "":
        return handle_interview_choose_newspaper(ctx, ms, cmd_noop)

    prompt = (
        f"Yes, you remember now! This is {ctx.query_text('town', 'name')}. "
        "The question remains: Why did you come here? Not just for the sight seeing! "
        "With a start (or is it a shudder) you notice a newspaper sitting on the nightstand. "
        "You can barely read the Newspaper Title from where you sit. "
    )

    if cmd.verb == Verb.CMD_CHOICE:
        if cmd.ordinal == 1:
            set_not_hint(ctx, cmd.arg)
        if cmd.ordinal == 2 and cmd.arg.strip() != "":
            set_keyed_node(ctx, ScalarNode(cmd.arg), "interview", "date_hint")
        invalidate(ctx, "interview", "newspaper")
    if cmd.verb == Verb.CMD_CONFIRM:
        confirm_and_freeze(ctx, ("interview", "newspaper"), ("newspaper",))
        set_not_hint(ctx, " petrification or humming or being hollowed out or vibration ")
        return Transition(Phase.INTERVIEW_CHOOSE_CASE)
    if cmd.verb == Verb.CMD_NOPE:
        append_not_hint(ctx, query_newspaper(ctx))
        invalidate(ctx, "interview", "newspaper")

    options = [
        Command(verb=Verb.CMD_CHOICE, ordinal=1, arg=f"The newspaper isn't: {get_not_hint(ctx)}"),
        Command(verb=Verb.CMD_CHOICE, ordinal=2, arg=f"The day must be: {ctx.query_text('interview', 'date_hint')}"),
        Command(verb=Verb.CMD_CONFIRM, arg=f"{query_newspaper(ctx)}"),
        Command(verb=Verb.CMD_NOPE, arg="You must be seeing things ..."),
    ]
    return PromptOptions(prompt=prompt, options=options)


def handle_interview_choose_case(ctx: Context, ms: MachineState, cmd: Command) -> Transition | PromptOptions:
    if cmd.verb == Verb.CMD_CHOICE and cmd.ordinal in (1, 2) and cmd.arg.strip() == "":
        return handle_interview_choose_case(ctx, ms, cmd_noop)

    prompt = "You pick up the newspaper, and turn to the first article. " if cmd.verb == Verb.CMD_NOOP else "You turn to the next article. "
    prompt += "Is this the case that convinced you to travel all the way out here to this eerie town? "

    if cmd.verb == Verb.CMD_CHOICE:
        if cmd.ordinal == 1:
            set_not_hint(ctx, cmd.arg)
        if cmd.ordinal == 2 and cmd.arg.strip() != "":
            set_keyed_node(ctx, ScalarNode(cmd.arg), "interview", "case_hint")
        invalidate(ctx, "interview", "case1")
    if cmd.verb == Verb.CMD_CONFIRM:
        confirm_and_freeze(ctx, ("interview", "case1"), ("case",))
        return Transition(Phase.INTERVIEW_CHOOSE_INTEREST)
    if cmd.verb == Verb.CMD_NOPE:
        append_not_hint(ctx, query_case(ctx, "interview", "case1"))
        invalidate(ctx, "interview", "case1")

    options = [
        Command(verb=Verb.CMD_CHOICE, ordinal=1, arg=f"The case is not: {get_not_hint(ctx)}"),
        Command(verb=Verb.CMD_CHOICE, ordinal=2, arg=f"The case should be: {ctx.query_text('interview', 'case_hint')}"),
        Command(verb=Verb.CMD_CONFIRM, arg=f"{query_case(ctx, 'interview', 'case1')}"),
        Command(verb=Verb.CMD_NOPE, arg="None of these ring a bell; turn to the next page ..."),
    ]
    return PromptOptions(prompt=prompt, options=options)


def handle_interview_choose_interest(ctx: Context, ms: MachineState, cmd: Command) -> Transition | PromptOptions:
    if cmd.verb == Verb.CMD_CHOICE and cmd.ordinal in (1, 2) and cmd.arg.strip() == "":
        return handle_interview_choose_interest(ctx, ms, cmd_noop)

    prompt = "That brings up another question: There was a deeper reason you are personally invested. What was it?"

    if cmd.verb == Verb.CMD_CHOICE:
        if cmd.ordinal == 1:
            set_not_hint(ctx, cmd.arg)
        if cmd.ordinal == 2 and cmd.arg.strip() != "":
            set_keyed_node(ctx, ScalarNode(cmd.arg), "interview", "interest_hint")
        invalidate(ctx, "interview", "interest1")
    if cmd.verb == Verb.CMD_CONFIRM:
        confirm_and_freeze(ctx, ("interview", "interest1"), ("investigator", "interest"))
        set_not_hint(ctx, " involving the syndicate or law enforcement ")
        return Transition.phase_change(Phase.INTERVIEW_CHOOSE_SECRETS)
    if cmd.verb == Verb.CMD_NOPE:
        append_not_hint(ctx, ctx.query_text("interview", "interest1", "interest"))
        invalidate(ctx, "interview", "interest1")

    options = [
        Command(verb=Verb.CMD_CHOICE, ordinal=1, arg=f"My reason is not: {get_not_hint(ctx)}"),
        Command(verb=Verb.CMD_CHOICE, ordinal=2, arg=f"My reason is more: {ctx.query_text('interview', 'interest_hint')}"),
        Command(verb=Verb.CMD_CONFIRM, arg=f"{ctx.query_text('interview', 'interest1', 'interest')}"),
        Command(verb=Verb.CMD_NOPE, arg="Maybe somebody would expect that; but that's not it ..."),
    ]
    return Transition.prompt_options(prompt, options)


def handle_interview_choose_secrets(ctx: Context, ms: MachineState, cmd: Command) -> Transition | PromptOptions:
    target_secret_count = 2
    current_secrets = ctx.query(Path("investigator", "secrets"))
    secret_count = len(current_secrets) if isinstance(current_secrets, SequenceNode) else 0

    if cmd.verb == Verb.CMD_CHOICE and cmd.ordinal in (1, 2) and cmd.arg.strip() == "":
        return handle_interview_choose_secrets(ctx, ms, cmd_noop)

    prompt = (
        "Oh. Right. And now that you recall more, you also remember some other things ... "
        "Maybe a case of amnesia wasn't the worst thing that can happen to a person ... "
        f"(You are choosing secrets. {secret_count}/{target_secret_count} choices confirmed)."
    )

    if cmd.verb == Verb.CMD_CHOICE:
        if cmd.ordinal == 1:
            set_not_hint(ctx, cmd.arg)
        if cmd.ordinal == 2 and cmd.arg.strip() != "":
            set_keyed_node(ctx, ScalarNode(cmd.arg), "interview", "secrets_hint")
        invalidate(ctx, "interview", "secret1")

    if cmd.verb == Verb.CMD_CONFIRM:
        secret_text = ctx.query_text("interview", "secret1", "secret")
        secret_count = append_sequence_node(ctx, ("investigator", "secrets"), ScalarNode(secret_text))
        append_not_hint(ctx, secret_text)
        invalidate(ctx, "interview", "secret1")
        if secret_count >= target_secret_count:
            set_not_hint(ctx, "")
            return Transition(Phase.INTERVIEW_CHOOSE_ADVANTAGES)
        return handle_interview_choose_secrets(ctx, ms, cmd_noop)

    if cmd.verb == Verb.CMD_NOPE:
        append_not_hint(ctx, ctx.query_text("interview", "secret1", "secret"))
        invalidate(ctx, "interview", "secret1")

    options = [
        Command(verb=Verb.CMD_CHOICE, ordinal=1, arg=f"My secret is not: {get_not_hint(ctx)}"),
        Command(verb=Verb.CMD_CHOICE, ordinal=2, arg=f"My secret should be more: {ctx.query_text('interview', 'secrets_hint')}"),
        Command(verb=Verb.CMD_CONFIRM, arg=f"{ctx.query_text('interview', 'secret1', 'secret')}"),
        Command(verb=Verb.CMD_NOPE, arg="Ha! My real secret is darker than that ..."),
    ]
    return PromptOptions(prompt=prompt, options=options)


def handle_interview_choose_advantages(ctx: Context, ms: MachineState, cmd: Command) -> Transition | PromptOptions:
    target_advantage_count = 3
    current_advantages = ctx.query(Path("investigator", "advantages"))
    advantage_count = len(current_advantages) if isinstance(current_advantages, SequenceNode) else 0

    if cmd.verb == Verb.CMD_CHOICE and cmd.ordinal in (1, 2) and cmd.arg.strip() == "":
        return handle_interview_choose_advantages(ctx, ms, cmd_noop)

    prompt = ""
    if cmd.verb == Verb.CMD_NOOP:
        prompt += "A fit of restlessness strikes you, and you nearly leap out of bed as you move to your suitcase and unpack a few things. "
    prompt += "You _were_ trying to plan ahead it seems. You can almost see an inventory of your instincts and skill in what you packed. "
    prompt += f"(You are choosing advantages. {advantage_count}/{target_advantage_count} choices confirmed). "

    if cmd.verb == Verb.CMD_CHOICE:
        if cmd.ordinal == 1:
            set_not_hint(ctx, cmd.arg)
        if cmd.ordinal == 2 and cmd.arg.strip() != "":
            set_keyed_node(ctx, ScalarNode(cmd.arg), "interview", "advantage_hint")
        if cmd.ordinal == 3:
            set_not_hint(ctx, "")
            return Transition(Phase.GAME_OVER)
        invalidate(ctx, "interview", "advantage1")

    if cmd.verb == Verb.CMD_CONFIRM:
        node = ctx.query(Path("interview", "advantage1"))
        advantage_count = append_sequence_node(ctx, ("investigator", "advantages"), node)
        append_not_hint(ctx, query_advantage(ctx, "interview", "advantage1"))
        invalidate(ctx, "interview", "advantage1")
        if advantage_count >= target_advantage_count:
            set_not_hint(ctx, "")
            return Transition(Phase.GAME_OVER)
        return handle_interview_choose_advantages(ctx, ms, cmd_noop)

    if cmd.verb == Verb.CMD_NOPE:
        append_not_hint(ctx, ctx.query_text("interview", "advantage1", "advantage"))
        invalidate(ctx, "interview", "advantage1")

    options = [
        Command(verb=Verb.CMD_CHOICE, ordinal=1, arg=f"I did not pack: {get_not_hint(ctx)}"),
        Command(verb=Verb.CMD_CHOICE, ordinal=2, arg=f"I needed something for: {ctx.query_text('interview', 'advantage_hint')}"),
        Command(verb=Verb.CMD_CHOICE, ordinal=3, arg="I didn't bring anything else with me."),
        Command(verb=Verb.CMD_CONFIRM, arg=f"{query_advantage(ctx, 'interview', 'advantage1')}"),
        Command(verb=Verb.CMD_NOPE, arg="No, I packed something else ..."),
    ]
    return PromptOptions(prompt=prompt, options=options)


def handle_game_over(ctx: Context, ms: MachineState, cmd: Command) -> Transition | PromptOptions:
    prompt = (
        "Your memory settles into place. You have your case, your reasons, your secrets, and your tools. "
        "The real investigation can begin now."
    )
    options = [Command(verb=Verb.CMD_QUIT, arg="Farewell, investigator.")]
    return PromptOptions(prompt=prompt, options=options)


HANDLERS: dict[Phase, Handler] = {
    Phase.GAME_BEGIN: handle_game_begin,
    Phase.INTERVIEW_CHOOSE_ARCHETYPE: handle_interview_choose_archetype,
    Phase.INTERVIEW_CHOOSE_NAME: handle_interview_choose_name,
    Phase.INTERVIEW_CHOOSE_TOWN: handle_interview_choose_town,
    Phase.INTERVIEW_CHOOSE_NEWSPAPER: handle_interview_choose_newspaper,
    Phase.INTERVIEW_CHOOSE_CASE: handle_interview_choose_case,
    Phase.INTERVIEW_CHOOSE_INTEREST: handle_interview_choose_interest,
    Phase.INTERVIEW_CHOOSE_SECRETS: handle_interview_choose_secrets,
    Phase.INTERVIEW_CHOOSE_ADVANTAGES: handle_interview_choose_advantages,
    Phase.GAME_OVER: handle_game_over,
}
