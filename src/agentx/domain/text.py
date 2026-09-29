"""Literal mention matching for both spaced words and unsegmented Han text."""

import re

_HAN = re.compile(r"[\u3400-\u9fff]")


def find_literal_mention(text: str, term: str) -> re.Match[str] | None:
    """Match an exact term without requiring spaces around Han characters.

    Latin edges still use word boundaries. Unsegmented Han names may be substrings
    of longer names; callers must keep multiple inventory matches ambiguous.
    """
    needle = term.strip().casefold()
    if not needle:
        return None
    prefix = "" if _HAN.fullmatch(needle[0]) else r"(?<!\w)"
    suffix = "" if _HAN.fullmatch(needle[-1]) else r"(?!\w)"
    return re.search(prefix + re.escape(needle) + suffix, text.casefold())


def toolbox_name_aliases(text: str) -> tuple[str, ...]:
    """Prefer spelling variants before the broader toolbox/toolkit synonym."""
    original = text.casefold()
    spelling = re.sub(r"\btool[ -]box\b", "toolbox", original)
    synonym = re.sub(r"\btool[ -]?box\b", "toolkit", original)
    return tuple(dict.fromkeys(alias for alias in (spelling, synonym) if alias != original))
