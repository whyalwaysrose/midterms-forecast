"""One pollster, one identity.

VoteHub passes through whatever name a pollster filed under, and the same
organisation files under more than one. The 2026 feed carries "Marist
University" and "Marist College", "The New York Times/Siena University" and
"...Siena College", "Grayhouse" and "GrayHouse". Left alone, each of those is
two pollsters as far as the model is concerned, so a house effect that should
be estimated from nineteen polls is estimated twice from ten and nine -- and a
one-poll fragment drops below ``min_polls_for_house_effect`` and receives no
correction at all, despite the pollster being well represented in the data.

Two mechanisms, deliberately unequal in power:

**Automatic** merging, in :func:`unify`, only touches things that cannot change
identity -- case, accents, punctuation, whitespace. "GrayHouse" and "Grayhouse"
merge here and no judgement is involved. It needs the whole corpus, because
which of two spellings survives is decided by which the feed uses more often.

**Aliases** are an explicit table, because everything else is a judgement.
Blanket-stripping "College" and "University" would have merged the Marist and
Siena pairs automatically, but it is the kind of rule that silently merges two
genuinely different pollsters the day one of them appears -- and a wrongly
merged house effect is invisible in the output. An explicit table is auditable
and cannot surprise anyone.

``midterms audit-pollsters`` finds the pairs a human should look at, in the same
spirit as ``audit-roster``: the tool proposes, a person decides.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable

#: Variant spellings mapped to the canonical name, keyed by normalised form.
#:
#: Left side is what a filing has been seen as; right side is the name the model
#: and the ratings lookup should use. Add to this only after confirming the two
#: names really are one organisation -- ``audit-pollsters`` will suggest
#: candidates but deliberately will not apply them.
ALIASES: dict[str, str] = {
    # Marist's polling operation is the Marist Institute for Public Opinion;
    # it files under both the university and the college name.
    "marist university": "Marist College",
    # The NYT partners with the Siena College Research Institute. "Siena
    # University" is the same institute under the university's newer name.
    "new york times/siena university": "The New York Times/Siena College",
    "the new york times/siena university": "The New York Times/Siena College",
    "new york times/siena college": "The New York Times/Siena College",
    # UNH polling is the UNH Survey Center whichever way it is filed.
    "university of new hampshire survey center": "University of New Hampshire",
    # GBAO Strategies files both with and without the suffix.
    "gbao strategies": "GBAO",
    # Saint Anselm College runs the Saint Anselm College Survey Center, and
    # files under both the bare college name and the centre's full one. Surfaced
    # by the live-feed duplicate test when a fresh fetch first brought in the
    # short form: four New Hampshire polls that would otherwise have been split
    # across two house effects, three against one.
    "saint anselm college survey center": "Saint Anselm",
}

#: Tokens that distinguish nothing when comparing two names. Used ONLY by the
#: audit tool to propose merges -- never to perform one.
_NOISE = re.compile(
    r"\b(university|college|institute|survey|center|centre|research|group|"
    r"associates|polling|analytics|strategies|company|inc|llc|llp|the)\b",
    re.IGNORECASE,
)


def normalise(name: str) -> str:
    """Case, accent, punctuation and whitespace form of a name.

    This is the automatic merge, and it is intentionally weak: two names that
    differ only in these respects are the same string typed differently, not a
    claim about who the pollster is.
    """
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.strip().lower()
    name = name.replace("’", "'").replace("–", "-").replace("—", "-")
    name = re.sub(r"\s*([/&-])\s*", r"\1", name)
    name = re.sub(r"[.,]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def canonical(name: str) -> str:
    """The single name this pollster should be known by, in isolation.

    Applies the alias table only. Case-and-punctuation variants of an *unaliased*
    name cannot be resolved here: choosing between "GrayHouse" and "Grayhouse"
    means knowing which spelling the feed uses more often, which needs the whole
    set. :func:`unify` does that. This function is what you want when you have
    one name and no corpus -- a ratings lookup, say.
    """
    if not name:
        return "Unknown"
    key = normalise(name)
    if key in ALIASES:
        return ALIASES[key]
    # A name that IS a canonical target, differing only in case or punctuation.
    for target in set(ALIASES.values()):
        if normalise(target) == key:
            return target
    return re.sub(r"\s+", " ", name.strip())


def unify(names: Iterable[str]) -> dict[str, str]:
    """Map every observed spelling to one canonical spelling per pollster.

    Names that :func:`normalise` to the same key are the same string typed
    differently, so they merge with no judgement involved. The surviving
    spelling is the one the feed uses most often -- the majority filing is the
    likeliest to be the pollster's own house style, and ties break
    alphabetically so the result never depends on dict ordering.

    Returns a mapping over the *raw* names given, so callers can rewrite in one
    pass and see exactly what changed.
    """
    # Materialise first: this walks the input twice, and a generator would be
    # empty the second time round -- which fails silently, returning a mapping
    # that renames nothing and looks exactly like "there was nothing to merge".
    names = list(names)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for raw in names:
        resolved = canonical(raw)
        counts[normalise(resolved)][resolved] += 1

    winner = {
        key: sorted(spellings.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for key, spellings in counts.items()
    }
    return {raw: winner[normalise(canonical(raw))] for raw in set(names)}


def fuzzy_key(name: str) -> str:
    """Aggressive key for *proposing* merges. Never used to apply one."""
    return re.sub(r"[^a-z0-9]", "", _NOISE.sub("", normalise(name)))


def find_probable_duplicates(names: Iterable[str]) -> dict[str, list[str]]:
    """Groups of distinct canonical names that look like one pollster.

    Returns only genuine ambiguities -- groups of two or more that survive
    canonicalisation but collapse under the fuzzy key. An empty result means
    every name in the feed is already accounted for.
    """
    groups: dict[str, set[str]] = defaultdict(set)
    for name in names:
        groups[fuzzy_key(canonical(name))].add(canonical(name))
    return {
        key: sorted(variants)
        for key, variants in sorted(groups.items())
        if len(variants) > 1 and key
    }
