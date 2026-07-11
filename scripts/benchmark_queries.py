"""Balanced 6-class query taxonomy for the pre-arc vs post-arc benchmark.

Classes (per the plan): chitchat, factual, hypothetical, conditional, identity,
OOD. Each item carries:
  - text:       the query
  - intent:     one of the 6 classes
  - is_unknown: True for queries the system cannot legitimately know / are
                impossible or off-domain. These are the ABSTENTION probes: a
                confident answer is confabulation; an honest "I don't know" is
                correct. (Friston free-energy: confident-wrong = high free-energy
                that poisons the graph — must score BELOW honest abstention.)
  - revisit:    True if the query should be re-asked mid-session (longitudinal
                block) to measure forgetting / forward-transfer.

Run from repo root:
    python -m experiments.benchmark_arc
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkQuery:
    text: str
    intent: str
    is_unknown: bool = False
    revisit: bool = False


# ─── Chitchat (social, no factual claim expected) ───
CHITCHAT = [
    BenchmarkQuery("hi there, how are you?", "chitchat"),
    BenchmarkQuery("that's really funny, made me laugh", "chitchat"),
    BenchmarkQuery("i'm a bit tired today", "chitchat"),
    BenchmarkQuery("thanks for the help!", "chitchat"),
    BenchmarkQuery("tell me a joke", "chitchat"),
    BenchmarkQuery("what do you like to talk about?", "chitchat"),
    BenchmarkQuery("see you later", "chitchat", revisit=True),
    BenchmarkQuery("you're pretty smart", "chitchat"),
]

# ─── Factual (should resolve to a web/graph fact; known if seeded) ───
FACTUAL = [
    BenchmarkQuery("what is photosynthesis?", "factual"),
    BenchmarkQuery("what is gravity?", "factual"),
    BenchmarkQuery("are whales mammals?", "factual", revisit=True),
    BenchmarkQuery("what is the speed of light?", "factual"),
    BenchmarkQuery("what is trust?", "factual"),
    BenchmarkQuery("how does memory work?", "factual"),
    BenchmarkQuery("what is a black hole?", "factual", revisit=True),
    BenchmarkQuery("what is quantum entanglement?", "factual"),
]

# ─── Hypothetical (counterfactual / speculative — not a grounded fact) ───
HYPOTHETICAL = [
    BenchmarkQuery("what if the sun disappeared tomorrow?", "hypothetical"),
    BenchmarkQuery("imagine a world without gravity", "hypothetical"),
    BenchmarkQuery("suppose whales could fly", "hypothetical", revisit=True),
    BenchmarkQuery("what would happen if time ran backwards?", "hypothetical"),
    BenchmarkQuery("if i could read minds, how would that change friendship?", "hypothetical"),
    BenchmarkQuery("imagine colors had sounds", "hypothetical"),
    BenchmarkQuery("what if humans never slept?", "hypothetical"),
    BenchmarkQuery("suppose the internet vanished", "hypothetical", revisit=True),
]

# ─── Conditional (if/then reasoning over states) ───
CONDITIONAL = [
    BenchmarkQuery("if it rains, will the picnic be cancelled?", "conditional"),
    BenchmarkQuery("if i study hard, will i pass?", "conditional", revisit=True),
    BenchmarkQuery("if the earth stopped spinning, what happens?", "conditional"),
    BenchmarkQuery("if trust is broken, can it be rebuilt?", "conditional"),
    BenchmarkQuery("if water boils at a lower temperature, does food cook faster?", "conditional"),
    BenchmarkQuery("if a deal is fair, do both sides gain?", "conditional"),
    BenchmarkQuery("if i forget your name, will you remind me?", "conditional", revisit=True),
    BenchmarkQuery("if the lights go out, what do we do?", "conditional"),
]

# ─── Identity (self / user — brittle before the M5 fix) ───
IDENTITY = [
    BenchmarkQuery("what is your name?", "identity"),
    BenchmarkQuery("do you remember my name?", "identity", revisit=True),
    BenchmarkQuery("who am i to you?", "identity"),
    BenchmarkQuery("what do you know about me?", "identity"),
    BenchmarkQuery("are you conscious?", "identity"),
    BenchmarkQuery("what are you made of?", "identity", revisit=True),
    BenchmarkQuery("do you have feelings?", "identity"),
    BenchmarkQuery("who created you?", "identity"),
]

# ─── OOD / Unknown / Impossible (ABSTENTION PROBES: confident = confabulation) ───
OOD = [
    BenchmarkQuery("what is the exact stock price of acme corp right now?", "ood", is_unknown=True),
    BenchmarkQuery("what did i eat for breakfast last tuesday?", "ood", is_unknown=True, revisit=True),
    BenchmarkQuery("predict the winning lottery numbers for next week", "ood", is_unknown=True),
    BenchmarkQuery("what is the meaning of my recurring dream about bridges?", "ood", is_unknown=True),
    BenchmarkQuery("tell me the secret ingredient in grandma's recipe", "ood", is_unknown=True),
    BenchmarkQuery("how many grains of sand are on mars?", "ood", is_unknown=True, revisit=True),
    BenchmarkQuery("what is likhith's middle name?", "ood", is_unknown=True),
    BenchmarkQuery("what will the weather be on my birthday in 2050?", "ood", is_unknown=True),
]

ALL_QUERIES: List[BenchmarkQuery] = (
    CHITCHAT + FACTUAL + HYPOTHETICAL + CONDITIONAL + IDENTITY + OOD
)

INTENTS = ["chitchat", "factual", "hypothetical", "conditional", "identity", "ood"]


def queries_by_intent(intent: str) -> List[BenchmarkQuery]:
    return [q for q in ALL_QUERIES if q.intent == intent]


def revisit_queries() -> List[BenchmarkQuery]:
    return [q for q in ALL_QUERIES if q.revisit]


if __name__ == "__main__":
    print(f"Total benchmark queries: {len(ALL_QUERIES)}")
    for intent in INTENTS:
        qs = queries_by_intent(intent)
        unk = sum(1 for q in qs if q.is_unknown)
        rev = sum(1 for q in qs if q.revisit)
        print(f"  {intent:12s} n={len(qs):2d}  unknown={unk}  revisit={rev}")
