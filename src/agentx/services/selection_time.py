"""Conservative checks for explicit English video-time constraints.

This recognizes common point/range expressions, not arbitrary natural language.
It rejects dropped or invented bounds; it never supplies replacement filters.
"""

import re
from dataclasses import dataclass
from decimal import Decimal

CARDINAL = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
ORDINAL = "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth".split()
WORDS = {word: index for words in (CARDINAL, ORDINAL) for index, word in enumerate(words)}
NUMBER = (
    r"(?:\d+(?:\.\d+)?(?:st|nd|rd|th)?|" + "|".join(sorted(WORDS, key=len, reverse=True)) + r")"
)
UNIT = r"(?:milliseconds?|msecs?|ms|seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)"
STAMP = r"(?:\d+:)?\d+:\d+(?:\.\d+)?"
# A spelled-out number needs a separator before its unit ("ones" is not one + s, "seconds" is not
# second + s), and a one-letter unit only counts written against digits ("5s", but not "a 2 m cable").
DIGITS = r"\d+(?:\.\d+)?(?:st|nd|rd|th)?"
WORD_NUMBER = "(?:" + "|".join(sorted(WORDS, key=len, reverse=True)) + ")"
LONG_UNIT = r"(?:milliseconds?|msecs?|ms|seconds?|secs?|minutes?|mins?|hours?|hrs?)"
POINT = rf"(?:{DIGITS}[\s-]*{LONG_UNIT}|{DIGITS}(?:ms|s|m|h)|{WORD_NUMBER}[\s-]+{LONG_UNIT})"
VALUE = rf"(?:{STAMP}|{NUMBER}[\s-]+{UNIT})"
RECORDING_START = re.compile(
    r"\b(?:at\s+(?:the\s+)?(?:very\s+)?(?:start|beginning)"
    r"(?:\s+of\s+(?:the\s+)?(?:recording|video|clip))?(?!\s+of\b)"
    r"|(?:in|on|at)\s+(?:the\s+)?(?:very\s+)?(?:opening|first|initial)\s+frame"
    r"(?:\s+of\s+(?:the\s+)?(?:recording|video|clip))?(?!\s+(?:after|following|of)\b)"
    r"|when\s+(?:the\s+)?(?:recording|video|clip)\s+(?:began|started))\b"
)


@dataclass(frozen=True)
class TimeConstraint:
    relation: str
    start: int | None = None
    end: int | None = None


def milliseconds(text: str) -> int:
    text = text.strip().casefold()
    if ":" in text:
        value = Decimal(0)
        for part in text.split(":"):
            value = value * 60 + Decimal(part)
        return int(value * 1000)
    match = re.fullmatch(rf"({NUMBER})[\s-]+({UNIT})", text)
    if match is None:
        raise ValueError("Unrecognized video time")
    quantity = match[1]
    number = (
        Decimal(WORDS[quantity])
        if quantity in WORDS
        else Decimal(re.sub(r"(?:st|nd|rd|th)$", "", quantity))
    )
    unit = match[2]
    scale = (
        1
        if unit.startswith(("ms", "millis"))
        else 3_600_000
        if unit.startswith("h")
        else 60_000
        if unit.startswith("m")
        else 1000
    )
    return int(number * scale)


def explicit_times(text: str) -> list[TimeConstraint]:
    question = text.casefold()
    constraints = []
    covered = []
    # Shared-unit ranges: "between two and four seconds", "from 2s to 4s".
    ranges = rf"\b(?:between|from)\s+({NUMBER})(?:[\s-]*({UNIT}))?\s+(?:and|to)\s+({NUMBER})[\s-]*({UNIT})\b"
    for match in re.finditer(ranges, question):
        start = milliseconds(match[1] + " " + (match[2] or match[4]))
        end = milliseconds(match[3] + " " + match[4])
        constraints.append(TimeConstraint("range", start, end))
        covered.append(match.span())
    # Canonicalize adjacent units without changing the user's stored question.
    point_pattern = rf"(?<!\w)({STAMP}|{POINT})\b"
    for match in re.finditer(point_pattern, question):
        if any(start <= match.start() < end for start, end in covered):
            continue
        raw = re.sub(r"(?<=\d)(?=[a-z])", " ", match[1]) if ":" not in match[1] else match[1]
        raw = re.sub(r"(\d+)\s+(st|nd|rd|th)\b", r"\1\2", raw)
        value = milliseconds(raw)
        prefix = question[max(0, match.start() - 32) : match.start()]
        if re.search(r"\b(?:before|earlier than)\s+(?:the\s+)?$", prefix):
            constraints.append(TimeConstraint("before", end=value - 1))
        elif re.search(r"\b(?:after|later than|since)\s+(?:the\s+)?$", prefix):
            constraints.append(TimeConstraint("after", start=value + 1))
        elif re.search(r"\b(?:first|through|until|up to|by)\s+(?:the\s+)?$", prefix) or (
            # "in the first minute" is the span up to one minute, not the instant at one minute.
            match[1].split()[0].split("-")[0] == "first"
            and re.search(r"\b(?:in|during|within)\s+the\s+$", prefix)
        ):
            constraints.append(TimeConstraint("through", end=value))
        else:
            constraints.append(TimeConstraint("at", start=value, end=value))
    if RECORDING_START.search(question):
        anchor = TimeConstraint("at", 0, 0)
        if anchor not in constraints:
            constraints.append(anchor)
    return constraints


def validate_time_filters(filters, text: str, cutoff: int) -> None:
    if not filters:
        return
    constraints = explicit_times(text)
    temporal = [p for p in filters if p.kind != "name"]
    for bound in constraints:
        if bound.start is not None and bound.start > cutoff:
            raise ValueError(
                "The requested time starts after the cutoff; return no filters, never an earlier substitute."
            )
        end = min(cutoff, bound.end) if bound.end is not None else None
        start = bound.start

        def matches(p, relation=bound.relation, start=start, end=end):
            point = getattr(p, "at_ms", None)
            if p.kind in {"state", "other_state"} and point is None:
                point = cutoff
            lower, upper = getattr(p, "start_ms", None), getattr(p, "end_ms", None)
            if hasattr(p, "end_ms") and upper is None:
                upper = cutoff
            if relation == "at":
                return point == start or (lower == start and upper == end)
            if relation == "range":
                return lower == start and upper == end
            if relation == "after":
                return lower == start or (point is not None and point >= start)
            return upper == end or (point is not None and point <= end)

        if not any(matches(p) for p in temporal):
            hint = ""
            if bound.relation == "at" and start == 0:
                if any(p.kind == "registered" for p in temporal):
                    hint = (
                        " Registration at the recording start requires kind=registered with "
                        "start_ms=0 AND end_ms=0. A lower bound alone allows later registrations. "
                        "Do not replace registration with state or at_ms. Keep the full plan envelope."
                    )
                else:
                    hint = (
                        " An opening/first/initial recording frame is the state at at_ms=0. "
                        "Use kind=state with at_ms=0 and the requested region/status. "
                        "first_seen with start_ms=0 alone permits a later first sighting."
                    )
            elif bound.relation == "after":
                hint = (
                    f" AFTER sets the LOWER bound start_ms={start}. It does not set end_ms; "
                    "do not translate after into before or through. Apply this to the described "
                    "event/history predicate."
                )
            elif bound.relation == "before":
                hint = (
                    f" BEFORE excludes the requested instant. Use the upper bound end_ms={end} "
                    "on the same event/history predicate, not an inclusive through-time bound. "
                    "Keep the full plan envelope."
                )
            raise ValueError(
                f"The plan dropped or changed an explicit time constraint ({bound.relation}, "
                f"start_ms={start}, end_ms={end}). Preserve it in a temporal filter or return no filters."
                + hint
            )
    if constraints:
        return
    for predicate in temporal:
        for key in ("at_ms", "start_ms", "end_ms"):
            value = getattr(predicate, key, None)
            allowed = value is None or value == cutoff or (key == "start_ms" and value == 0)
            if key == "at_ms" and value == 0:
                allowed = bool(
                    re.search(r"\b(start|beginning|initially|initial|began)\b", text.casefold())
                )
            if not allowed:
                raise ValueError(
                    "The question gives no numeric time. Omit invented bounds; history defaults from start through the cutoff."
                )
