"""
ATTRIBUTE-AGREEMENT GATE for recall (round auto/round-20260925T0823-fix-7).

One shared predicate, owned here, so every recall site agrees BY CONSTRUCTION
instead of each keeping its own private copy of the rule (the recurring
lesson from the pet-slot rename: N hand-kept copies of one rule drift apart
and the drift surfaces as an unrelated-looking CI failure).

THE RULE. A query asks for an ATTRIBUTE of an entity ("what is my cat's
name"). A stored fact answers with a VALUE. The answer is admissible only if
the value actually IS that attribute: a name is the short single token the
user gave as the entity's name; a predicate phrase ("diagnosed with a chronic
illness", "weaves baskets") is the entity's STATE. Answering a name query
with the state is the right animal and the wrong attribute, stated
confidently — the documented confabulation shape.

The rule is grammatical (attribute agreement), NOT a per-topic table: it
holds for any entity, any attribute, any language-level noun.

TWO PIECES, both needed:

1. `asks_name_only(q)` — is this query a PURE name query? A COMPOUND
   question asks for more than one thing ("what's my grandmother's name AND
   what does she make?"). Restricting a compound query to name-shaped values
   would delete the very fact the second half asked for, so the gate must
   stand down and let the entity render every attribute it holds.

2. `is_name_shaped(v)` — does this stored value actually answer a name
   query? A name is a short single alphabetic token. A predicate phrase is
   not.

Neither function authors a reply: callers keep their own frames and read the
store. The content still comes from what the user actually said.
"""
import re

# The attribute-agreement trigger: the query names the ATTRIBUTE it wants.
# "name" is the universal one; "nickname" is the same attribute said
# differently. Grammatical category, not a topic list.
_NAME_ATTR = re.compile(r"\b(?:name|named|called|nickname)\b", re.IGNORECASE)

# A COORDINATED second question. English builds a compound question by
# joining two interrogative clauses with a coordinator, so "and" (or
# "also"/"plus") followed by another question element is the grammatical
# signal that this query asks for MORE than the name. Requiring the
# question element after the coordinator is what keeps "my cat's name and
# age" style possessives and ordinary declaratives out of the gate.
_COORDINATED_Q = re.compile(
    r"\b(?:and|also|plus|as well as)\b[^.?!]{0,60}?"
    r"\b(?:what|which|who|whom|whose|where|when|why|how|"
    r"do|does|did|is|are|was|were|can|could|will|would|should|"
    r"has|have|had)\b",
    re.IGNORECASE,
)


def asks_name_only(q: str) -> bool:
    """True when `q` asks for a name and NOTHING else.

    A compound question ("what's my grandmother's name and what does she
    make?") is not a name-only query: the second clause asks for the
    entity's activity, and the gate must not strip the fact that answers it.
    """
    t = q or ""
    if not _NAME_ATTR.search(t):
        return False
    return not _COORDINATED_Q.search(t)


def is_name_shaped(value) -> bool:
    """True when a stored value is name-shaped: a short single token.

    A name is one word the user gave as the entity's name. A multi-word
    phrase is a predicate, i.e. the entity's state or activity, and must
    not be offered as an answer to a name query.
    """
    v = (value or "").strip()
    return bool(v) and len(v.split()) == 1 and v.isalpha() and len(v) > 1
