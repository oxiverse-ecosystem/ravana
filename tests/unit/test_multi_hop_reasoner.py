"""Phase 3 unit tests: multi-hop reasoner (chain + comparative)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))

from ravana.core.multi_hop_reasoner import MultiHopReasoner


def _check(name, cond):
    assert cond, name


# A tiny mock fact store: (entity, attribute) -> value
FACTS = {
    ("alice", "husband"): "Alice's husband is Bob",
    ("bob", "company"): "Bob works at Google",
    ("alice", "salary"): "Alice earns 90000 a year",
    ("bob", "salary"): "Bob earns 120000 a year",
    ("alice", "age"): "Alice is 30 years old",
    ("bob", "age"): "Bob is 45 years old",
}


def retriever(entity, attr):
    # loose match: try exact, then any fact whose entity matches and attr word
    # appears in the value
    entity = entity.lower()
    attr = attr.lower()
    if (entity, attr) in FACTS:
        return FACTS[(entity, attr)]
    for (e, a), v in FACTS.items():
        if e == entity and (attr in a or attr in v.lower()):
            return v
    # attr synonyms
    syn = {"company": ("work", "google", "employer"),
           "salary": ("earn", "income", "make")}
    for (e, a), v in FACTS.items():
        if e == entity and attr in syn and any(s in v.lower() for s in syn[attr]):
            return v
    return None


def test_possessive_chain():
    r = MultiHopReasoner()
    ans = r.answer("What is the name of the company where Alice's husband works?",
                   retriever)
    ok = ans is not None and "google" in ans.lower()
    _check("Alice's husband's company -> Google", ok)


def test_comparative_salary():
    r = MultiHopReasoner()
    ans = r.answer("Who earns more, Alice or Bob?", retriever)
    ok = ans is not None and "bob" in ans.lower()
    _check("who earns more Alice/Bob -> Bob", ok)


def test_comparative_age():
    r = MultiHopReasoner()
    ans = r.answer("Who is older, Alice or Bob?", retriever)
    ok = ans is not None and "bob" in ans.lower()
    _check("who is older Alice/Bob -> Bob", ok)


def test_comparative_younger():
    r = MultiHopReasoner()
    ans = r.answer("Who is younger, Alice or Bob?", retriever)
    ok = ans is not None and "alice" in ans.lower()
    _check("who is younger Alice/Bob -> Alice", ok)


def test_failed_hop_returns_none():
    r = MultiHopReasoner()
    ans = r.answer("What is the name of the company where Carol's husband works?",
                   retriever)
    ok = ans is None
    _check("unknown entity chain -> None (no confabulation)", ok)


def test_non_multihop_returns_none():
    r = MultiHopReasoner()
    ans = r.answer("What is the weather today?", retriever)
    ok = ans is None
    _check("non-multihop question -> None", ok)


if __name__ == "__main__":
    tests = [
        test_possessive_chain, test_comparative_salary, test_comparative_age,
        test_comparative_younger, test_failed_hop_returns_none,
        test_non_multihop_returns_none,
    ]
    print("Phase 3 — multi-hop reasoner tests")
    results = [t() for t in tests]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    if passed != len(results):
        sys.exit(1)
    print("ALL PASS")
