from __future__ import annotations

from context_resolver.ast.nodes import MappingNode, ScalarNode, SequenceNode
from context_resolver.ast.resolvable_node import ResolvableNode
from context_resolver.ast.schema import FieldSpec, Schema
from context_resolver.context.context import Context
from context_resolver.inference.llama_cpp_provider import LocalLlamaCppProvider
from context_resolver.inference.strategy import PromptStrategy
from context_resolver.query.passes import ResolutionPass
from context_resolver.query.resolver import Resolver
from context_resolver.templates.template import TemplateRegistry

from apps.little_house_in_the_eerie.context_access_helpers import meta_data, simple_schema, template_schema_tuple


PLACE_HOLDER = ScalarNode(None)


def build_initial_context() -> Context:
    """Construct the opening game state as a Context AST."""
    registry = TemplateRegistry()

    investigator_name = template_schema_tuple(
        template_str=(
            "You are the Co-GM for a paranormal investigation game centered on a little town. "
            "The following is the list of prior suggestions you should avoid:\n"
            "*** List of prior suggestions (if any):\n"
            "{interview.but_not_hint}"
            "*** End list of prior suggestions.\n"
            "The investigator is a {investigator.archetype}. "
            "Fill in this honorific: {interview.honorific_hint}. "
            "The first name {interview.first_name_hint}. "
            "The rest of the name {interview.last_name_hint}. "
            "Fill in this suffix: {interview.suffix_hint}. "
            "What is the investigator's name? "
            "If the optional honorific is None or Nothing, then leave it blank."
            "If the optional suffix is None or Nothing, then leave it blank. "
            "Either fill in the optional with someting like the above, or leave them blank. "
            "A middle name may also be provided. "
        ),
        schema=Schema(
            name="InvestigatorName",
            fields=[
                FieldSpec(name="honorific", type="str", required=False, description="Honorific"),
                FieldSpec(name="first", type="str", required=True, description="First name"),
                FieldSpec(name="middle", type="str", required=False, description="Middle name"),
                FieldSpec(name="last", type="str", required=True, description="Last name"),
                FieldSpec(name="suffix", type="str", required=False, description="Suffix"),
            ],
        ),
        description="Generates an investigator name.",
    )

    brochure = template_schema_tuple(
        template_str=(
            "You are the Co-GM for a paranormal investigation game centered on a little town. "
            "The following is the list of prior suggestions you should avoid:\n"
            "*** List of prior suggestions (if any):\n"
            "{interview.but_not_hint}"
            "*** End list of prior suggestions.\n"
            "Choose a location {interview.destination_hint}, and keep it isolated and hidden. "
            "Invent a town name, an economic basis, and a dark backstory."
            "However, the dark backstory should not be overtly supernatural, and should date back more than one generation. "
            "The backstory should be something scandalous, and possibly criminal, "
            "but it should be normal human corruption (which is bad and common enough). "
            "We will layer on the supernatural investigation _later_. "
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
            "The following is the list of prior suggestions you should avoid:\n"
            "*** List of prior suggestions (if any):\n"
            "{interview.but_not_hint}"
            "*** End list of prior suggestions.\n"
            "The newspaper needs a Brand Name/Title, but not _too_ respectable.  It should lean speculative and too credulous. "
            "Also, invent the publisher, circulation and date. The date should be {interview.date_hint}. "
        ),
        schema=Schema(
            name="Newspaper_Title",
            fields=[
                FieldSpec(name="title", type="str", required=True, description="Newspaper Title"),
                FieldSpec(name="publisher", type="str", required=True, description="Publisher"),
                FieldSpec(name="circulation", type="str", required=True, description="The city or means of circulation"),
                FieldSpec(name="date", type="The date of this issue"),
            ],
            description="Details for the investigators favored newspaper of record.",
        ),
        description="Generates a newspaper fiction.",
    )
    registry.register(newspaper_title[0])

    case_headline = template_schema_tuple(
        template_str=(
            "You are a Co-GM for a paranormal investigation game. "
            "The following is the list of prior suggestions you should avoid:\n"
            "*** List of prior suggestions (if any):\n"
            "{interview.but_not_hint}"
            "*** End list of prior suggestions.\n"
            "The investigator archetype is {investigator.archetype}. "
            "Town: {town.name}, location: {town.location}, economy: {town.economic}, secret backstory: {town.backstory}. "
            "Invent a current sketchy headline and a short sensational description for {interview.case_hint}. "
            "This case _should_ at least lean toward the supernatural, and may use the town backstory as a canvas and background."
            "Ideally the case exacerbates and somehow flows from the negative energy of the town as the seed where the eerie breaks into our world "
            "through the weakening and decay of human virtues. "
            "However, the secret backstory should not be explicitly referred to in this newspaper article. We want the investigator to have to dig. "
            "Do not to use the word Case in the name. Keep it a simple title without extra phrases and colons. "
            "To keep track of it, use the secret part to very briefly describe the secret component. "
            "Remember this is a newspaper article _before_ any investigations happened; but give enough detail to pull the investigator in. "
        ),
        schema=Schema(
            name="CaseHeadline",
            fields=[
                FieldSpec(name="name", type="str", required=True, description="Headline."),
                FieldSpec(name="secret_part", type="str", required=True, description="The Secret part of the case."),
                FieldSpec(name="description", type="str", required=True, description="Public knowledge summary."),
            ],
        ),
        description="Generate a short case article for the investigator.",
    )
    registry.register(case_headline[0])

    personal_secret = simple_schema(
        "Investigator Secret",
        "secret",
        (
            "The following is the list of prior suggestions you should avoid:\n"
            "*** List of prior suggestions (if any):\n"
            "{interview.but_not_hint}"
            "*** End list of prior suggestions.\n"
            "Generate a dark personal secret for investigator "
            "{investigator.name}, who is investigating '{case.name}'."
            "The secret should NOT be something directly related to the case. "
            "The secret should be something to do with the world outside and to do with being a {investigator.archetype}. "
            "The secret should be something personal, that, through no fault of my own, led to a misunderstanding or even a sad ending. "
            "The secret should be something that would set my career back, or the my closest friends would be scandalized to learn it. "
            "The secret should be something that I probably could not have avoided, but haunts me anyway. "
            "But the secret should be something that if kept secret will not cause any obvious problems. "
            "Also, keep the secret concise ... don't describe downstream consequences. The brute facts are enough to haunt me. "
            "Report this as if you are the one describing your own secret in first person. "
        ),
    )
    registry.register(personal_secret[0])

    investigator_archetype = simple_schema(
        "Investigator Archetype",
        "archetype",
        (
            "The following is the list of prior suggestions you should avoid:\n"
            "*** List of prior suggestions (if any):\n"
            "{interview.but_not_hint}"
            "*** End list of prior suggestions.\n"
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
            "The following is the list of prior suggestions you should avoid:\n"
            "*** List of prior suggestions (if any):\n"
            "{interview.but_not_hint}"
            "*** End list of prior suggestions.\n"
            "The investigator (a {investigator.archetype}) has an {interview.interest_hint} interest in this case. "
            "The current case is {case.name}. {case.description} "
            "Invent a reason the investigator is personally invested. It should concise and somewhat personal. "
            "Use First person, as if you are the one who cares about this case. "
        ),
    )
    registry.register(interest[0])

    advantage = template_schema_tuple(
        template_str=(
            "Describe either a book and a skill, or a piece of investigative equipment and a skill "
            "that the investigator has, something that could fit in a suitcase. "
            "The following is the list of prior suggestions you should avoid:\n"
            "*** List of prior suggestions (if any):\n"
            "{interview.but_not_hint}"
            "*** End list of prior suggestions.\n"
            "You are an investigator, a {investigator.archetype}, on the case {case.name}. {case.description}. "
            "You remember you packed something you needed for {interview.advantage_hint}. "
            "Use first person as if you are the one describing your own book or object. "
            "Describe the object, which is either a book or an object as an almost completely normal piece of equipment. "
            "Then describe the skill or ability that goes along with it, and investigative expertise relating to the object. "
            "Also describe what advantage this gives the you versus which kinds of activities with one simple example. "
        ),
        schema=Schema(
            name="Advantage",
            fields=[
                FieldSpec(name="book_or_equipment", type="str", required=True, description="Either the title of the book, or a ery brief description of the tool."),
                FieldSpec(name="implied_skill", type="str", required=True, description="The skill that is implied and utilizes the book or equipment."),
                FieldSpec(name="advantage", type="str", required=True, description="A description of the advantage this gives, and some simple examples."),
            ],
        ),
        description="One of the investigator's advantages, possibly literal or psychological.",
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
        "Invent an exact date just a couple days after {newspaper.date}.",
    )
    registry.register(invent_date[0])

    date_generator = simple_schema(
        "Current Date",
        "date",
        "What is the exact next day after {atmosphere.prior_date}. Be sure to increment the month and decrement to 1 of the month if necessary.",
    )
    registry.register(date_generator[0])

    diary_generator = simple_schema(
        "Diary Brief",
        "entry",
        (
            "Summarize the following personal log below in first person as a short diary entry. "
            "Try to be detail oriented, but not exhaustive, this should be a summary not a complete retelling. "
            "Feel free to add emotional color where natural. "
            "Preface the entry with the date: {atmosphere.prior_date}, and the time (hour: {atmosphere.hour_of_day}, minute: {atmosphere.minute_of_day}, is pm? {atmosphere.post_meridian}). "
            "Here is the log: \n{notebook.diary_raw}"
        ),
    )
    registry.register(diary_generator[0])

    case_summary = simple_schema(
        "Case Summary",
        "case_so_far",
        (
            "Summarize the following case details in a coherent first person description (like a report to a superior). "
            "Feel free to merge or harmonize clues that seem related. "
            "Try to be detail oriented, but not exhaustive, this should be a story not a set of bullet points. "
            "Feel free to add obvious interpretive connections. "
            "Preface the entry with the case name: {case.name}, on date {case.date}. "
            "The case description is: {case.description}. "
            "The clues related to the case are: {notebook.clues}. "
            "Describe the case like you are the one who found the clues. "
        )
    )
    registry.register(case_summary[0])

    investigator_summary = simple_schema(
        "Investigator Summary",
        "investigator_stats",
        (
            "Clean up and orient the following investigator details in a summary. "
            "If you see empty or schematic fields, represent it minimally (Like if disadvantages is an empty list). "
            "For wounds, 0 means healthy, 1 means bad wound, 2 means severe wound, 3 means dying. "
            "For instability, 0 means completely stable, 6 means game-over insane. "
            "For luck, 0 means no luck at all, 6 means normal luck. "
            "This is a kind of character sheet.\n"
            "Name: {investigator.name}\n"
            "Job: {investigator.archetype}\n"
            "Attributes: {investigator.attributes}\n"
            "Wounds: {investigator.wounds}\n"
            "Instability: {investigator.instability}\n"
            "Luck: {investigator.luck}\n"
            "Advantages: {investigator.advantages}\n"
            "Disadvantages: {investigator.disadvantages}\n"
            "Iterest: {investigator.interest}\n"
        )
    )
    registry.register(investigator_summary[0])

    secrets_summary = simple_schema(
        "Investigator Secrets",
        "investigator_secrets",
        (
            "Clean up and summaries the following investigator secrets. "
            "This is a kind of forbidden diary. Use first person like they are your secrets.\n"
            "Secrets: {investigator.secrets}\n"
        )
    )
    registry.register(secrets_summary[0])

    location_summary = simple_schema(
        "Location Summary",
        'location_summary',
        (
            "Clean up and prose format the following location name and description.\n"
            "{print_summaries.location_data}"
        )
    )
    registry.register(location_summary[0])

    town_summary = simple_schema(
        "Town Summary",
        "summary",
        (
            "Clean up and prose format the following town name and description.\n"
            "{print_summaries.town_data}"
        )
    )
    registry.register(town_summary[0])

    npc_list = simple_schema(
        "NPC List",
        "summary",
        (
            "List the names of the NPCs in the current game state.\n"
            "{npcs}"
        )
    )
    registry.register(npc_list[0])

    npc = simple_schema(
        "NPC Summary",
        "summary",
        (
            "Clean up and prose format the following NPC name and description.\n"
            "Name: {print_summaries.npc_name}\n"
            "Description: {print_summaries.npc_data}"
        )
    )
    registry.register(npc[0])

    scene_description = simple_schema(
        "Scene Description",
        'scene_description',
        (
            "Collect and describe the following scene, combining the details as logically as possible.\n"
            "Location: {scene.location}\n"
            "Overall description: {scene.specifics}\n"
            "Scene vibe: {print_summaries.vibe_summary}\n"
            "Scene details: {scene.elements}\n"
        )
    )
    registry.register(scene_description[0])

    root = MappingNode(
        {
            "interview": MappingNode(
                {
                    "but_not_hint": ScalarNode(""),
                    "honorific_hint": ScalarNode(" nothing "),
                    "first_name_hint": ScalarNode(" starts with the letter J, but not just the letter "),
                    "last_name_hint": ScalarNode(" a short middle name, last name ends with the letter r "),
                    "suffix_hint": ScalarNode(" nothing "),
                    "name_guess1": ResolvableNode(*investigator_name, metadata=meta_data(1.0, "investigator_name")),
                    "destination_hint": ScalarNode(" somewhere remote, town of 1000, most of the townspeople are completely healthy and normal, old money dates back a century "),
                    "brochure1": ResolvableNode(*brochure, metadata=meta_data(4.0, "brochure")),
                    "date_hint": ScalarNode(" in the late 20th century "),
                    "newspaper": ResolvableNode(*newspaper_title),
                    "case_hint": ScalarNode(" an inexplicable death "),
                    "case1": ResolvableNode(*case_headline),
                    "archetype_hint": ScalarNode(" federal investigator, SCP division "), # " investigator of the unknown and unknowable "
                    "archetype1": ResolvableNode(*investigator_archetype, metadata=meta_data(1.0, "investigator_arch")),
                    "interest_hint": ScalarNode(" it reminds me of something my colleague once reported ... "),
                    "interest1": ResolvableNode(*interest, metadata=meta_data(4.0, "interest")),
                    "secrets_hint": ScalarNode(" My fakeness could come out. I tried to be reliable but failed. "),  # " dark "
                    "secret1": ResolvableNode(*personal_secret),
                    "advantage_hint": ScalarNode(" a useful item "),
                    "advantage1": ResolvableNode(*advantage),
                }
            ),
            "town": MappingNode(
                {
                    "name": PLACE_HOLDER,
                    "location": PLACE_HOLDER,
                    "economic": PLACE_HOLDER,
                    "backstory": PLACE_HOLDER,
                }
            ),
            "case": MappingNode(
                {
                    "name": PLACE_HOLDER,
                    "case_date": PLACE_HOLDER,
                    "description": PLACE_HOLDER,
                }
            ),
            "newspaper": MappingNode(
                {
                    "title": PLACE_HOLDER,
                    "publisher": PLACE_HOLDER,
                    "circulation": PLACE_HOLDER,
                    "date": PLACE_HOLDER,
                }
            ),
            "investigator": MappingNode(
                {
                    "name": PLACE_HOLDER,
                    "archetype": PLACE_HOLDER,
                    "attributes": ResolvableNode(*attributes_modifiers),
                    "wounds": ScalarNode(0),
                    "instability": ScalarNode(0),
                    "luck": ScalarNode(9),
                    "secrets": PLACE_HOLDER,
                    "interest": PLACE_HOLDER,
                    "advantages": SequenceNode([]),
                    "disadvantages": SequenceNode([]),
                }
            ),
            "notebook": MappingNode(
                {
                    "logbook_dates": SequenceNode([]), # A matching list of the date for each logbook
                    "logbook": SequenceNode([]), # concatenation of the daybooks
                    "daybook": SequenceNode([]), # Summaries, clues, and questions
                    "diary_raw": ScalarNode(""), # Raw text for generating a diary entry
                    "diary_entry": ResolvableNode(*diary_generator),
                    "diary_dates": SequenceNode([]), # A matching list of the date for each summary
                    "diary": SequenceNode([]), # A list of summaries of the daybook
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
                    "npcs_data": SequenceNode("")
                }
            ),
            # This "event" node is not directly "lookable" by the player.  
            # It is used to track progress through the scene and phase details
            "event": MappingNode(
                {
                    "description": ScalarNode("Investigation Progress"),
                    "progress": ScalarNode("You see a suspect at the Location"),
                    "consequences": ScalarNode(None),
                    "escalation": ScalarNode(1),
                }
            ),
            "print_summaries": MappingNode(
                {
                    "case_summary": ResolvableNode(*case_summary),
                    "investigator_summary": ResolvableNode(*investigator_summary),
                    "secrets_summary": ResolvableNode(*secrets_summary),
                    "town_data": PLACE_HOLDER,
                    "town_summary": ResolvableNode(*town_summary),
                    "location_data": PLACE_HOLDER,
                    "location_summary": ResolvableNode(*location_summary),
                    "scene_summary": ResolvableNode(*scene_description),
                    "vibe_summary": ResolvableNode(*vibe),
                    "all_npcs": ResolvableNode(*npc_list),
                    "npc_name": PLACE_HOLDER,
                    "npc_data": PLACE_HOLDER,
                    "npc_summary": ResolvableNode(*npc),
                }
            ),
        }
    )

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
