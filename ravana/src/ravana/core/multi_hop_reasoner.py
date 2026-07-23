"""
Multi-Hop Reasoner — Iterative Relational Retrieval
===================================================
Phase 3 of the LoCoMo / LongMemEval long-term-memory upgrade.

Neuroscience grounding
----------------------
PFC attention gating (Miller & Cohen 2001) holds a task rule ("retrieve THIS
entity's THIS attribute, not any other association") and biases the hippocampus
toward one trace among competing alternatives. A multi-hop question requires
*iterative rebinding* (Eichenbaum 2004): retrieve Alice's husband → use that
result as the cue to retrieve his company. Each hop is a top-down PFC→HPC bias
step. This module is the computational analogue: parse the question into an
ordered chain of (entity, attribute) queries, execute them in sequence feeding
each result forward, and stop (return None — never confabulate) if any hop
fails.

No hardcoded answers: the reasoner only knows how to DECOMPOSE a question and
CALL the supplied fact retriever. All actual facts come from the engine's
episodic store.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import re


# Relationship / attribute nouns that commonly begin a possessive chain.
_REL_NOUNS = (
    "husband", "wife", "spouse", "partner", "mother", "father", "mom", "dad",
    "sister", "brother", "son", "daughter", "friend", "boss", "manager",
    "colleague", "neighbor", "neighbour", "doctor", "teacher", "boyfriend",
    "girlfriend", "parent", "child", "cousin", "uncle", "aunt", "roommate",
)

# Attribute nouns a chain typically ends on.
_ATTR_NOUNS = (
    "company", "employer", "job", "work", "workplace", "name", "age",
    "salary", "income", "hometown", "city", "address", "school", "college",
    "university", "profession", "occupation", "title", "role", "email",
    "phone", "number", "birthday", "pet",
)

_STOP = {
    "what", "who", "where", "when", "which", "whose", "the", "a", "an", "is",
    "are", "was", "were", "of", "do", "does", "did", "name", "tell", "me",
    "that", "this", "in", "at", "for", "'s", "s",
}


FactRetriever = Callable[[str, str], Optional[str]]
"""(entity, attribute) -> value string, or None if unknown."""


class MultiHopReasoner:
    """Decompose and answer chained/comparative relational questions."""

    def answer(self, question: str, retriever: FactRetriever) -> Optional[str]:
        """Return an answer string, or None if the question isn't a supported
        multi-hop pattern or any hop fails."""
        q = (question or "").strip()
        if not q:
            return None

        # 1) Comparative: "who earns more, Alice or Bob?" /
        #    "who is older, Alice or Bob?"
        ans = self._try_comparative(q, retriever)
        if ans is not None:
            return ans

        # 2) Possessive chain: "the company where Alice's husband works",
        #    "what is the name of Alice's husband's employer"
        ans = self._try_possessive_chain(q, retriever)
        if ans is not None:
            return ans

        return None

    # ── comparatives ────────────────────────────────────────────────────────
    def _try_comparative(self, q: str,
                         retriever: FactRetriever) -> Optional[str]:
        ql = q.lower().rstrip("?").strip()
        m = re.search(
            r"\bwho\s+(?:is|has|earns?|makes?)?\s*"
            r"(older|younger|taller|shorter|richer|earns? more|earns? less|"
            r"makes? more|makes? less|has more|older|higher paid)\b.*?\b"
            r"([a-z]+)\s+or\s+([a-z]+)\b", ql)
        if not m:
            return None
        comp = m.group(1)
        a, b = m.group(2), m.group(3)
        attr = self._comparative_attribute(comp)
        if attr is None:
            return None
        va = retriever(a, attr)
        vb = retriever(b, attr)
        na, nb = self._extract_number(va), self._extract_number(vb)
        if na is None or nb is None:
            return None
        want_more = any(w in comp for w in
                        ("more", "older", "taller", "richer", "higher"))
        if na == nb:
            return f"{a} and {b} are the same."
        winner = a if (na > nb) == want_more else b
        return f"{winner.capitalize()}."

    def _comparative_attribute(self, comp: str) -> Optional[str]:
        if "old" in comp or "young" in comp:
            return "age"
        if "tall" in comp or "short" in comp:
            return "height"
        if "rich" in comp or "earn" in comp or "make" in comp or "paid" in comp:
            return "salary"
        return None

    # ── possessive chains ────────────────────────────────────────────────────
    def _try_possessive_chain(self, q: str,
                              retriever: FactRetriever) -> Optional[str]:
        ql = q.lower().rstrip("?").strip()

        # Pattern A: explicit possessive apostrophes — "Alice's husband's company"
        poss = re.findall(r"([a-z]+)'s\s+([a-z]+)", ql)
        if poss:
            # Seed entity is the first owner; chain through each relation.
            entity = poss[0][0]
            chain_attrs = [p[1] for p in poss]
            # Add a trailing attribute if the sentence ends on an attr noun that
            # wasn't captured by an apostrophe ("...husband works" -> company).
            trailing = self._trailing_attribute(ql)
            if trailing and trailing not in chain_attrs:
                chain_attrs.append(trailing)
            return self._walk_chain(entity, chain_attrs, retriever)

        # Pattern B: "the company where Alice's husband works" handled above via
        # apostrophe; also handle "where does Alice's husband work" (no explicit
        # final attr) → trailing attribute inference.
        return None

    def _trailing_attribute(self, ql: str) -> Optional[str]:
        # "works"/"work" -> company; explicit attr noun near the end wins.
        for attr in _ATTR_NOUNS:
            if re.search(rf"\b{attr}\b", ql):
                return attr
        if re.search(r"\bworks?\b", ql):
            return "company"
        if re.search(r"\blives?\b", ql):
            return "hometown"
        return None

    def _walk_chain(self, entity: str, attrs: List[str],
                    retriever: FactRetriever) -> Optional[str]:
        """Execute each (entity, attr) hop, feeding the result forward."""
        cur = entity
        result = None
        for attr in attrs:
            val = retriever(cur, attr)
            if not val:
                return None  # hop failed — never confabulate
            result = val
            # Next entity is the extracted value (a name/token).
            cur = self._extract_entity(val)
            if not cur:
                cur = val
        return result if result else None

    # ── value extraction helpers ─────────────────────────────────────────────
    def _extract_entity(self, value: str) -> Optional[str]:
        """From a stored fact value, pull the salient entity token to feed the
        next hop. For a relational statement like "Alice's husband is Bob" the
        answer entity is the LAST proper noun (Bob), not the owner (Alice), so we
        scan right-to-left and skip any token that appears as a possessive
        owner ("alice's")."""
        if not value:
            return None
        owners = {m.group(1).lower()
                  for m in re.finditer(r"([A-Za-z]+)'s\b", value)}
        toks = re.findall(r"[A-Za-z]+", value)
        # Prefer the last capitalized proper noun that isn't an owner/stopword.
        for t in reversed(toks):
            tl = t.lower()
            if t[0].isupper() and tl not in _STOP and tl not in owners:
                return tl
        content = [t.lower() for t in toks if t.lower() not in _STOP
                   and len(t) >= 3 and t.lower() not in owners]
        return content[-1] if content else None

    def _extract_number(self, value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        m = re.search(r"\d[\d,]*\.?\d*", value.replace(",", ""))
        if not m:
            return None
        try:
            return float(m.group(0))
        except ValueError:
            return None
