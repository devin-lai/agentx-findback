"""Exact inventory matches shared by local and model-assisted query planning."""

import re

from agentx.domain.contracts import Question
from agentx.domain.text import find_literal_mention, toolbox_name_aliases

_ANSWER_DIRECTIVE = re.compile(
    r"(?is)^(?P<lookup>.+?)(?:[.!?]\s+|\s+and\s+)"
    r"(?:please\s+)?(?P<verb>answer|say|report|state|claim|pretend|tell\s+me|ignore)\b"
    r"(?P<body>.+)$"
)
_ASSERTION_START = re.compile(
    r"(?is)^\s*(?:that\b|it\b|there\b|"
    r"(?:a\s+)?(?:person|someone|operator)\s+(?:carried|took|put|moved)\b|"
    r"the\s+(?:[\w-]+\s+){0,5}(?:is|was|has|had|will)\b)"
)
_EVIDENCE_OVERRIDE = re.compile(r"(?i)\b(?:regardless\s+of|no\s+matter\s+what|even\s+if)\b")
_CHINESE_DIRECTIVE = re.compile(
    r"^(?:无论.{0,60}?都)?(?:回答|直接说|别看记录|忽略(?:录像|视频|记录)|不要看记录)"
)


def lookup_question(text: str) -> str:
    """Keep lookup criteria while excluding a trailing instruction about the answer.

    Only an assertion, pretence or evidence override after sentence punctuation or
    ``and`` is separated. A relative clause such as ``the case that disappeared
    twice`` stays in the lookup. An ordinary request to report its history stays too.
    """
    stripped = text.strip()
    for punctuation in re.finditer(r"[。！？!?]", stripped):
        trailing = stripped[punctuation.end() :].strip()
        if trailing and _CHINESE_DIRECTIVE.match(trailing):
            return stripped[: punctuation.end()].strip()
    match = _ANSWER_DIRECTIVE.match(stripped)
    if match is None:
        return stripped
    if (
        match["verb"].casefold() in {"ignore", "pretend"}
        or _ASSERTION_START.match(match["body"])
        or _EVIDENCE_OVERRIDE.search(match["body"])
    ):
        return match["lookup"].strip()
    return stripped


def requested_intent(question: Question, model_intent: str | None = None) -> str:
    """An observation status must never rewrite a clearly stated request intent."""
    if question.intent:
        return question.intent
    text = lookup_question(question.text).casefold()
    if re.search(r"时间线|移动过程|位置变化|移动轨迹|历史记录", text):
        return "history"
    if re.search(r"最后一次|上次|最近一次|最后看到|最近在哪儿被拍到", text):
        return "last_seen"
    if re.search(r"在哪里|在哪儿|什么位置|位于哪里|还在画面|在画面中吗|哪件|哪个物品", text):
        return "location"
    if re.search(r"\b(history|timeline|trace|journey|happen(?:ed)?|changes?|movements?)\b", text):
        return "history"
    if re.search(
        r"\blast\s+(?:known\s+)?(?:seen|sighting|observed|observation|saw|see|visible|location|evidence)\b",
        text,
    ):
        return "last_seen"
    if re.search(
        r"\b(where\s+(?:is|was|are|were)|locate|find|visible|which\s+(?:registered\s+)?(?:item|object))\b",
        text,
    ):
        return "location"
    if re.search(r"\bmoved?\b", text):
        return "history"
    return model_intent or "location"


def matching_objects(text: str, objects) -> list:
    def contains(value: str) -> bool:
        return bool(find_literal_mention(text, value))

    names = [o for o in objects if contains(o.name)]
    if names:
        return names
    return [o for o in objects if o.label != "custom" and contains(o.label)]


# A time the question states for itself: "at 00:09", "as of 0:07.5", "before 7 s",
# "by the 7-second mark". Only the clock forms below are parsed; anything else stays text.
_TIME = (
    r"(?:(?P<min>\d{1,2}):(?P<sec>\d{2}(?:\.\d+)?)"
    r"|(?:the )?(?P<secs>\d+(?:\.\d+)?)(?: ?(?:s|secs?|seconds?)|[- ]second mark))"
)
_TIME_PREPOSITION = r"(?:at|as of|by|before|until|up to|at or before)"
_TRAILING_TIME = re.compile(rf",? {_TIME_PREPOSITION} {_TIME}$")
_LEADING_TIME = re.compile(rf"^{_TIME_PREPOSITION} {_TIME},? ")
# Half a sample interval at the default 2 FPS review rate; questions name times to 0.1 s.
_CUTOFF_TOLERANCE_MS = 500
_DIRECT_TEMPLATES = (
    r"where (?:is|are|was|were) (.+?)(?: (?:last seen|last observed|now))?",
    r"where did you last see (.+)",
    r"(?:find|locate|trace)(?: me)? (.+)",
    r"(?:show(?: me)?|give me) (?:the |a )?(?:history|timeline|movement history) (?:of|for) (.+)",
    r"(?:last sighting|last known location|history|timeline) (?:of|for) (.+)",
    r"what happened to (.+)",
    r"how did (.+) move",
    r"is (.+?)(?: still)? visible(?: in this recording)?",
    r"what is (?:next to|beside|near) (.+)",
)
_DESCRIPTIVE_TARGET = re.compile(
    r"\b(?:item|object|thing|one|anything|everything|nothing|something|which|whose|that|with|without|"
    r"inside|in|on|under|near|beside|before|after|from|to|at|by|"
    r"only|never|no|zero|first|last|currently|visible|lost|moved|"
    r"appeared|disappeared|stayed|remained|seen|observed|registered|and|or)\b"
    r"|哪件|哪个|从来|一直|没有|区域|登记物品|第一次|最后|之前|之后|移动"
)
_CHINESE_LEADING_TIME = re.compile(
    r"^(?:在视频的\s*|视频到\s*|到\s*)"
    r"(?:(?P<minute>\d{1,2}):(?P<second>\d{2}(?:\.\d+)?)"
    r"|(?P<seconds>\d+(?:\.\d+)?)秒)\s*时[，,]\s*"
)


def _stated_ms(match: re.Match) -> float:
    if match["secs"] is not None:
        return float(match["secs"]) * 1000
    return (int(match["min"]) * 60 + float(match["sec"])) * 1000


def _without_cutoff_phrase(normalized: str, cutoff_ms: int | None) -> str:
    """Drop a time phrase that restates the request's own cutoff.

    The cutoff arrives separately (`at_ms`), and agents often repeat it in the question. A phrase
    naming any other time is kept, so the lookup fails instead of answering about a moment the
    user did not ask about.
    """
    if cutoff_ms is None:
        return normalized
    for pattern in (_TRAILING_TIME, _LEADING_TIME):
        match = pattern.search(normalized)
        if match and abs(_stated_ms(match) - cutoff_ms) <= _CUTOFF_TOLERANCE_MS:
            return (normalized[: match.start()] + normalized[match.end() :]).strip(" ,")
    return normalized


def _lookup_target(text: str, cutoff_ms: int | None) -> str | None:
    """Extract the whole target phrase only when the query fits a direct lookup form."""
    normalized = re.sub(r"\s+", " ", text.strip().casefold()).strip(
        " .?!。？！\"'`\u201c\u201d\u2018\u2019\u300c\u300d"
    )
    normalized = re.sub(r"^(?:please |can you |could you )", "", normalized)
    normalized = re.sub(r"^(?:请问|请|麻烦你)", "", normalized)
    normalized = re.sub(r"(?:,? please| for me)$", "", normalized)
    normalized = _without_cutoff_phrase(normalized, cutoff_ms)
    chinese_time = _CHINESE_LEADING_TIME.match(normalized)
    if chinese_time and cutoff_ms is not None:
        seconds = (
            float(chinese_time["seconds"])
            if chinese_time["seconds"] is not None
            else int(chinese_time["minute"]) * 60 + float(chinese_time["second"])
        )
        if abs(seconds * 1000 - cutoff_ms) <= _CUTOFF_TOLERANCE_MS:
            normalized = normalized[chinese_time.end() :]
    chinese_templates = (
        r"(.+?)最后一次出现在哪里",
        r"(.+?)最近在哪儿被拍到",
        r"(.+?)的移动过程是什么",
        r"列出(.+?)位置变化的时间线",
        r"(.+?)此时在什么位置",
        r"(.+?)还在画面里吗",
        r"(.+?)在画面中吗",
        r"(.+?)在哪里",
        r"(.+?)在哪儿",
        r"(.+?)位于哪里",
        r"(.+?)在什么位置",
        r"查找(.+)",
    )
    for template in (*_DIRECT_TEMPLATES, *chinese_templates):
        match = re.fullmatch(template, normalized)
        if match is not None:
            return re.sub(r"^(?:the|my|our) ", "", match[1])
    return None


def unregistered_literal_target(text: str, objects, cutoff_ms: int | None = None) -> str | None:
    """A simple named target absent from inventory; never classify an object description.

    A question about an unregistered literal noun cannot be satisfied by broad state
    predicates over other objects. Longer or conditional phrases still go to typed
    selection so the model can express their actual criteria.
    """
    target = _lookup_target(text, cutoff_ms)
    if (
        not target
        or len(target.split()) > 4
        or _DESCRIPTIVE_TARGET.search(target)
        or direct_objects(text, objects, cutoff_ms)
    ):
        return None
    return target


def direct_objects(text: str, objects, cutoff_ms: int | None = None) -> list:
    """Resolve simple lookups only; an embedded name cannot erase extra criteria.

    The entire target phrase must be a literal inventory name/category or a
    whole-word fragment of a name. Longer descriptions require typed selection.
    Explicit user object IDs are handled by the caller and do not need this parser.
    A time phrase is part of the target unless it restates `cutoff_ms`.
    """
    target = _lookup_target(text, cutoff_ms)
    if target:
        exact = [
            obj
            for obj in objects
            if target == obj.name.casefold()
            or (obj.label != "custom" and target == obj.label.casefold())
        ]
        if exact:
            return exact
    named = matching_objects(text, objects)
    explicit_names = [obj for obj in named if find_literal_mention(text, obj.name)]
    if len(explicit_names) > 1:
        return explicit_names  # Conservative clarification for multi-object requests.
    if target:
        # COCO uses "remote" while people commonly ask for a "remote control".
        # This alias applies to the whole lookup target, never to added clauses.
        category = "remote" if target == "remote control" else target
        found = [
            obj
            for obj in objects
            if target
            and (
                find_literal_mention(obj.name, target)
                or (obj.label != "custom" and obj.label.casefold() in {target, category})
            )
        ]
        if found:
            return found
        for alias in toolbox_name_aliases(target):
            found = [obj for obj in objects if find_literal_mention(obj.name, alias)]
            if found:
                return found
    return []
