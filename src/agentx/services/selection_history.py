"""Narrow omission checks for explicit region-history quantifiers.

This is repair feedback for the planner, not a natural-language parser. It never
chooses an object or rewrites a predicate. Unsupported descriptions may abstain.
"""

import re

from agentx.domain.text import find_literal_mention

_ONLY = re.compile(
    r"\b(?:never\s+left|never\s+(?:(?:been|seen|observed|recorded)\s+)*outside|"
    r"(?:only|always)\s+(?:ever\s+)?(?:(?:been|seen|observed|recorded)\s+)?"
    r"(?:in|inside|within))\s+(?:the\s+)?$"
    r"|(?:从未离开过?|从来没有离开过?|只在|仅在|始终在|一直在)\s*$"
)
_NEVER = re.compile(
    r"\bnever\s+(?:(?:been|seen|observed|recorded)\s+)*(?:in|inside|within)\s+"
    r"(?:the\s+)?$"
    r"|(?:从未在|从来没有在)\s*$"
)
_SINGLE_ITEM = re.compile(
    r"\bonly\s+(?:registered\s+)?(?:item|object|thing|target)\s+"
    r"(?:currently\s+)?(?:in|inside|within)\s+(?:the\s+)?$"
)


def validate_history_quantifiers(filters, text: str, regions: list[dict]) -> None:
    if not filters:
        return
    for region in regions:
        aliases = [region["id"], region["name"]]
        if region["id"] == "center":
            aliases += ["middle", "centre"]
        for alias in aliases:
            # Check each occurrence: a question may mention the same region twice.
            remaining, offset = text, 0
            while mention := find_literal_mention(remaining, alias):
                prefix = text[: offset + mention.start()]
                suffix = text[offset + mention.end() :]
                if (
                    _SINGLE_ITEM.search(prefix)
                    and re.fullmatch(r"(?:\s+(?:area|region))?\s*[.?!]*", suffix)
                    and any(p.kind == "only_seen_in" and p.zone == region["id"] for p in filters)
                ):
                    raise ValueError(
                        "Here 'only' modifies the number of matching items, not their history. "
                        "Use state with the requested zone at the cutoff. Do not invent "
                        "only_seen_in; code must retain ambiguity if several objects are there."
                    )
                for kind, pattern in (("only_seen_in", _ONLY), ("never_seen_in", _NEVER)):
                    match = pattern.search(prefix)
                    if not match:
                        continue
                    # "Not always in ..." is not a universal positive claim.
                    if re.search(r"\b(?:not|never)\s+$|n['’]t\s+$|不是$", prefix[: match.start()]):
                        continue
                    if not any(p.kind == kind and p.zone == region["id"] for p in filters):
                        raise ValueError(
                            f"The explicit region-history condition requires {kind} for region "
                            f"{region['id']!r}. A current state or one sighting drops the "
                            "whole-history quantifier; never outside is not never inside. "
                            "Preserve the requested time window, or return no filters if "
                            "the description cannot be represented."
                        )
                offset += mention.end()
                remaining = text[offset:]
