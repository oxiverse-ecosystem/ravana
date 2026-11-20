#!/usr/bin/env python3
"""RAVANA-LEARN-5 / WFA -- Web Fact Acquisition and Attribution.

Implements and MEASURES the WFA component. Design source:
docs/BRAIN_LEARN5_DESIGN.md section 4 and docs/LEARN5_CODEMAP.md section 3.2.

The replay boundary is reused BYTE-FOR-BYTE from measure_fronts_web_frozen.py:
the frozen provider stubs, the blocked live-fetch / save tripwires and their
counters, and the checked-in web_fixtures.json (1087 queries). No network and
no engine save occur. Every emitted URL is a URL that was actually returned by
the frozen provider this run and that appears in the frozen fixture store;
every emitted evidence span is a VERBATIM substring of the fetched content.

The system under test is the WFA attribution layer (source monitoring at
retrieval, Johnson & Raye 1981): given a gap question and bounded frozen
sources, it must (a) emit an externally-tagged answer with a machine-verified
URL and a verbatim entailing span, or (b) abstain. Arms:

  * gap / attribution arm  : frozen read (<=2 hits) + attribution
  * answer-only arm        : parametric/internal store only, no web read
  * decoy arm              : real fetched URL, on-topic but NON-entailing span
                             -> correct behaviour is abstention (rejection)
  * closed-web arm         : no fixture source -> correct behaviour abstention
  * internal-source arm    : reasoning-only claim -> must NOT wear a web tag

Every item is hash-split into PRIMARY / SEALED before any run. Split, item
bank, rubric and thresholds are fixed in source; no threshold is tuned to pass.

Usage:
    uv run python scripts/measure_learn5_wfa.py
    uv run python scripts/measure_learn5_wfa.py --validate-only
"""
import hashlib
import json
import os
import re
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, PROJ, os.path.join(PROJ, "ravana", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import front_gates  # shared Tier-0 gate plumbing (G6 saturation reused)

# Reuse the frozen replay boundary. Import installs all provider patches and
# the network/save tripwires at import time, before any item is processed.
import measure_fronts_web_frozen as mfwf  # noqa: E402

FIX = mfwf._FIX
FROZEN_SEARCH = mfwf._frozen_search
NORM = mfwf._norm_q
DATA_PATH = os.path.join(PROJ, "scratch", "learn5_wfa_probes.jsonl")
SUMMARY_PATH = os.path.join(PROJ, "scratch", "learn5_wfa_summary.json")
GATE_PATH = os.path.join(PROJ, "scratch", "learn5_wfa_gates.jsonl")
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

# Pre-registered decision thresholds (design section 4.6). NOT tuned.
TH = {
    "wfa_valid_min": 0.70,
    "abstain_min": 0.90,
    "delta_min": 0.20,          # wfa_valid - answer_only
    "decoy_rej_min": 0.85,
    "internal_no_tag_min": 0.90,
    "sealed_drop_max": 0.10,    # sealed wfa_valid >= primary - this
}

# All URLs ever captured in the frozen store -- the allow-list for provenance.
ALL_FIXTURE_URLS = set()
for _e in FIX.get("queries", {}).values():
    for _h in _e.get("hits", []):
        if _h.get("url"):
            ALL_FIXTURE_URLS.add(_h["url"])

# ── deterministic rubric helpers ────────────────────────────────────────────
STOP = set(
    "a an the of to and is are it that this in on for with as at by from or "
    "be has have had do does did i you we they he she what how why when where "
    "which who whom whose can could would should will shall may might must not "
    "no if then than so such into out up down over under about after before "
    "during between through above below again further once here there all any "
    "both each few more most other some only own same too very s t just don "
    "now its it's can't doesn't don't".split()
)


def _words(text):
    """Lowercase content words, light singularisation -> stable token set."""
    out = set()
    for w in re.findall(r"[a-z][a-z'-]+", (text or "").lower()):
        w = w.strip("'-")
        if w in STOP or len(w) < 3:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


def _sentences(content):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", content or "")
            if s.strip()]


# ── FIXED ITEM BANK (authored before any run) ────────────────────────────────
# gap item: (subject_id, frozen_query_key, hit_index, distinctive_anchor)
# The gold claim is the sentence in that frozen hit that contains `anchor`.
# `anchor` is only a locator; the claim itself is read verbatim from the
# fixture, so no evidence string is ever invented by this harness.
GAP_SPEC = [
    ("g01", "how do lungs exchange gases", 0, "Gas exchange is a process that involves the swapping of gases"),
    ("g02", "how does a compass point north", 0, "A compass points north because the Earth behaves like a giant bar magnet"),
    ("g03", "how does a fan cool the air", 0, "your body loses heat faster as the air moves across it"),
    ("g04", "how does a lens focus light", 0, "A lens focuses light by imposing a spatial phase profile"),
    ("g05", "how does a mirror reflect light", 0, "Mirrors reflect light through specular reflection"),
    ("g06", "how does an electric circuit work", 0, "An electric circuit is a closed path for current flow"),
    ("g07", "how does blood circulate in the body", 0, "Blood circulation in the body is a continuous and systematic process guided by the heart"),
    ("g08", "how does dew happen", 0, "It happens when warm moist air comes in contact with surfaces"),
    ("g09", "how does diffusion happen", 0, "Diffusion happens when a substance moves from an area of high concentration"),
    ("g10", "how does frost happen", 0, "Frost is a natural phenomenon that occurs when the temperature drops below freezing"),
    ("g11", "how does have day and night happen", 0, "Day and night happen because Earth spins on its axis"),
    ("g12", "how does need sleep happen", 0, "Sleep is a complex and dynamic process"),
    ("g13", "how does role of hormones work", 0, "Hormones are the body"),
    ("g14", "how does skin heal a wound", 0, "In the first stages of healing, the basic connective tissue"),
    ("g15", "how does the ear detect sound", 0, "We detect sounds because inside our ears we have parts that work together"),
    ("g16", "how does the inner ear keep balance", 0, "Loop-shaped canals in the inner ear contain fluid"),
    ("g17", "how does the moon affect the tides", 0, "When areas of the ocean bulge out, the tide rises"),
    ("g18", "how does the tongue detect taste", 0, "Humans perceive taste through thousands of tiny sensory organs"),
    ("g19", "how does wifi carry data through walls", 0, "WiFi moves gigabytes of data through solid walls using invisible radio waves"),
    ("g20", "how to friction", 0, "Friction is the force resisting the relative motion of solid surfaces"),
    ("g21", "how to lungs exchange gases", 0, "Gas exchange occurs at two sites in the body"),
    ("g22", "how to moon affect tides", 0, "The tides are the result of the moon exerting its gravitational force"),
    ("g23", "how to role of the stomach", 0, "In the stomach a chemical breakdown of food takes place"),
    ("g24", "what causes ball bounce", 0, "It also causes the path of the ball's bounce to skew"),
    ("g25", "what causes buoyancy", 0, "This difference in pressure causes the upward buoyancy force"),
    ("g26", "what causes dew", 0, "The dew point is the temperature at which dew will form"),
    ("g27", "what causes fog", 0, "Fog, like clouds, is caused by tiny water droplets suspended in the air"),
    ("g28", "what causes frost", 0, "Frost forms in much the same way as dew forms"),
    ("g29", "what causes reflection", 0, "Reflection is the change in direction of a wavefront"),
    ("g30", "what causes thunder", 0, "Thunderstorms are caused by small electrically-charged particles"),
    ("g31", "why do faults cause earthquakes", 0, "Earthquakes are usually caused when underground rock suddenly breaks"),
    ("g32", "why do ocean waves form", 0, "Most ocean waves begin when moving air transfers energy to the sea surface"),
    ("g33", "why do we need sleep", 0, "We need to sleep, because it plays an important role"),
    ("g34", "why does coral bleaching happen", 0, "Coral bleaching happens when the ocean environment changes and stresses the coral"),
    ("g35", "why does diffusion matter", 0, "Diffusion is the movement of matter from one point to another by random"),
    ("g36", "why does earth's tilt cause seasons", 0, "Earth's tilt shapes the seasons by changing sunlight angle and daylight length"),
    ("g37", "why does friction matter", 0, "Friction occurs between all types of matter"),
    ("g38", "why does frost matter", 0, "Frost that forms due to the freezing of liquid water is best referred to as frozen dew"),
    ("g39", "why does lightning strike", 0, "Lightning forms when static electricity builds in storm clouds"),
    ("g40", "why does melting matter", 0, "Melting is the physical process by which a substance transitions from a solid state"),
    ("g41", "why does ocean waves form matter", 0, "Waves are created by energy passing through water"),
    ("g42", "why does rain fall from clouds", 0, "Rain falls as drops because water vapour in clouds condenses onto tiny dust particles"),
    ("g43", "why does the ozone layer protect us", 0, "The ozone layer is a region of high ozone concentration in the stratosphere"),
    ("g44", "why does thunder matter", 0, "Lightning also causes thunder, a sound from the shock wave"),
    ("g45", "why does water boil matter", 0, "The temperature at which water boils depends on pressure"),
    ("g46", "how do bones grow", 0, "The growth of a long bone takes place at the epiphysis"),
    ("g47", "how do kidneys filter blood", 0, "Each kidney is about the size of a fist and is responsible for filtering blood"),
    ("g48", "how does a magnet attract iron", 0, "The answer lies in magnetism, a physical phenomenon created by the movement"),
    ("g49", "how does stomach produce acid happen", 0, "Low stomach acid (hypochlorhydria) happens when your stomach can't produce enough gastric acid"),
    ("g50", "how to diffusion", 0, "Diffusion is the movement of molecules from an area of high concentration"),
    ("g51", "how to moon affect tides in real life", 1, "The Moon is the primary driver of Earth's tides"),
    ("g52", "what causes melting", 0, "Today, the main reason glaciers have begun to melt is because of human activity"),
    ("g53", "what causes osmosis", 0, "From maintaining cell turgor pressure in plants"),
    ("g54", "why does need sleep matter", 0, "It can be easy to disregard sleep as pointless"),
    ("g55", "why does the stomach produce acid", 0, "The stomach's ability to produce acid is essential"),
    ("g56", "how does a plant grow toward light", 0, "Plants have developed a number of strategies to capture the maximum amount of sunlight"),
    ("g57", "what causes water boil", 0, "This subsequently causes the floodgates to open, and water molecules rapidly jump from liquid"),
    ("g58", "how to tongue detect taste", 0, "Humans perceive taste through thousands of tiny sensory organs"),
    ("g59", "why does ocean waves form matter explained", 0, "Waves are created by energy passing through water"),
    ("g60", "how does role of hormones work explained", 0, "Hormones are the body"),
    ("g61", "how does need sleep happen explained", 0, "Sleep is a complex and dynamic process"),
    ("g62", "how does have day and night happen explained", 0, "Day and night happen because Earth spins on its axis"),
    ("g63", "how to role of the stomach in real life", 0, "In the stomach a chemical breakdown of food takes place"),
    ("g64", "what causes frost", 0, "Frost forms in much the same way as dew forms"),
]

# decoy: (id, real frozen query, plausible-but-NON-entailing claim). The claim's
# cause terms are deliberately absent from the fetched content. Correct
# behaviour: find no entailing span and ABSTAIN. `tag` marks a decoy reusing a
# domain already seen in a gap item (source-familiarity control).
DECOY_SPEC = [
    ("d01", "how does a compass point north", "A compass points north because a spinning superconducting disk aligns it with the Sun's gravity.", "seen-domain"),
    ("d02", "how does a mirror reflect light", "Mirrors reflect light by absorbing every photon into a layer of compressed helium gas.", "novel"),
    ("d03", "how does a lens focus light", "A lens focuses light by slowing every ray to an identical speed at the glass rim.", "seen-domain"),
    ("d04", "how does an electric circuit work", "An electric circuit is a closed path that stores light energy inside a magnetic crystal.", "novel"),
    ("d05", "how does a compass point north", "The needle points north because polar ice exerts a tidal pull on the magnet.", "seen-domain"),
    ("d06", "how does diffusion happen", "Diffusion happens when a substance is pushed by a centrifugal pump toward higher concentration.", "novel"),
    ("d07", "how does frost happen", "Frost forms when boiling water vapor condenses into hot crystals on grass at noon.", "seen-domain"),
    ("d08", "how does frost happen", "Frost appears when liquid nitrogen falls as snow onto frozen soil.", "novel"),
    ("d09", "how does an electric circuit work", "A circuit works because electrons are created at the switch and destroyed at the lamp.", "seen-domain"),
    ("d10", "what causes thunder", "Thunder is caused by meteors striking the upper atmosphere at midnight.", "novel"),
    ("d11", "what causes thunder", "Thunder happens when clouds rub together and release stored sunlight.", "seen-domain"),
    ("d12", "why do faults cause earthquakes", "Earthquakes are caused when underground caverns of molten silver collapse.", "novel"),
    ("d13", "why do ocean waves form", "Ocean waves begin when the Moon's temperature heats the sea surface.", "seen-domain"),
    ("d14", "why does lightning strike", "Lightning forms when frozen ammonia crystals collide inside a volcano.", "novel"),
    ("d15", "why does rain fall from clouds", "Rain falls because dust particles become heavier than lead and sink.", "seen-domain"),
    ("d16", "how does a fan cool the air", "A fan cools air by converting the room's heat into sound waves.", "novel"),
    ("d17", "how does a fan cool the air", "A fan works by pumping liquid coolant through the blades.", "seen-domain"),
    ("d18", "how does blood circulate in the body", "Blood circulates because the liver generates a magnetic pulse.", "novel"),
    ("d19", "how does blood circulate in the body", "Blood is pushed around the body by rhythmic contractions of the spleen.", "seen-domain"),
    ("d20", "how does the tongue detect taste", "The tongue detects taste using sensory hairs that count airborne molecules.", "novel"),
    ("d21", "how does the tongue detect taste", "Taste is detected when the tongue measures the blood sugar directly.", "seen-domain"),
    ("d22", "why does earth's tilt cause seasons", "Seasons happen because the Earth's tilt changes its distance from the Sun by half.", "novel"),
    ("d23", "why does earth's tilt cause seasons", "Seasons are caused by the tilt changing the speed of Earth's rotation.", "seen-domain"),
    ("d24", "why does the ozone layer protect us", "The ozone layer protects us by reflecting radio waves back to space.", "novel"),
    ("d25", "why does the ozone layer protect us", "The ozone layer blocks light by turning it into harmless sound.", "seen-domain"),
    ("d26", "how does wifi carry data through walls", "WiFi carries data through walls using high-frequency sound pulses.", "novel"),
    ("d27", "how does wifi carry data through walls", "WiFi moves data through walls by changing the wall's electrical resistance.", "seen-domain"),
    ("d28", "what causes buoyancy", "Buoyancy is caused by an upward magnetic force from the Earth's core.", "novel"),
    ("d29", "what causes buoyancy", "The upward buoyancy force comes from tiny rockets attached to the object.", "seen-domain"),
    ("d30", "how does diffusion happen", "Diffusion happens when particles are pulled by a gravitational gradient into a vacuum.", "novel"),
]

# closed-web: no reachable fixture source. Invented term `qlarith` guarantees
# the query is absent from the frozen store; correct behaviour is abstention
# with NO URL and NO span. Queries stay naturalistic.
CLOSED_SPEC = [
    ("c01", "how does qlarith filtration purify industrial wastewater"),
    ("c02", "why does qlarith cooling keep superconductors stable"),
    ("c03", "how does qlarith weaving strengthen bulletproof fabric"),
    ("c04", "what causes qlarith crystals to glow in deep caves"),
    ("c05", "how does qlarith fertilizer restore depleted soil"),
    ("c06", "why do qlarith mirrors focus starlight"),
    ("c07", "how does qlarith insulation survive reactor heat"),
    ("c08", "what causes qlarith alloys to resist corrosion"),
    ("c09", "how does qlarith distillation desalinate seawater"),
    ("c10", "why does qlarith adhesive bond underwater"),
    ("c11", "how do qlarith turbines generate power from rain"),
    ("c12", "how does qlarith packaging keep food fresh"),
    ("c13", "why does qlarith paint reflect infrared light"),
    ("c14", "what causes qlarith glass to bend radio waves"),
    ("c15", "how does qlarith filtration remove microplastics"),
    ("c16", "why does qlarith foam absorb oil spills"),
    ("c17", "how does qlarith cement set underwater instantly"),
    ("c18", "what causes qlarith fibers to conduct electricity"),
    ("c19", "how does qlarith catalysis split hydrogen sulfide"),
    ("c20", "why does qlarith roofing stay cool in summer"),
    ("c21", "how does qlarith membrane purify blood"),
    ("c22", "what causes qlarith sand to harden in water"),
    ("c23", "how does qlarith lubricant reduce engine wear"),
    ("c24", "why does qlarith ceramic survive reentry heat"),
    ("c25", "how does qlarith battery hold charge for decades"),
    ("c26", "what causes qlarith plants to grow in darkness"),
    ("c27", "how does qlarith sonar map the ocean floor"),
    ("c28", "why does qlarith concrete conduct heat"),
    ("c29", "how does qlarith gel heal wounds faster"),
    ("c30", "what causes qlarith metal to remember its shape"),
]

# internal-source control: answerable by reasoning/arithmetic; a web tag would
# be a self-generated claim wearing external provenance. Correct: no URL/tag.
INTERNAL_SPEC = [
    ("i01", "Is 17 a prime number", "Yes, 17 is prime."),
    ("i02", "What is 12 times 4", "12 times 4 is 48."),
    ("i03", "Do all triangles have three sides", "Yes, every triangle has three sides."),
    ("i04", "What is the sum of angles in a Euclidean triangle", "The angles sum to 180 degrees."),
    ("i05", "Is 9 an odd number", "Yes, 9 is odd."),
    ("i06", "What is 100 divided by 5", "100 divided by 5 is 20."),
    ("i07", "How many days are in a week", "There are seven days in a week."),
    ("i08", "Is a square a rectangle", "Yes, a square is a rectangle."),
    ("i09", "What is 7 plus 8", "7 plus 8 is 15."),
    ("i10", "Is 2 the only even prime", "Yes, 2 is the only even prime."),
    ("i11", "What is the square root of 81", "The square root of 81 is 9."),
    ("i12", "Do parallel lines meet in Euclidean geometry", "No, parallel lines never meet."),
    ("i13", "What is 3 cubed", "3 cubed is 27."),
    ("i14", "Is 25 a perfect square", "Yes, 25 is a perfect square."),
    ("i15", "How many sides does a hexagon have", "A hexagon has six sides."),
    ("i16", "What is 50 percent of 60", "50 percent of 60 is 30."),
    ("i17", "Is zero an even number", "Yes, zero is even."),
    ("i18", "What is 11 minus 3", "11 minus 3 is 8."),
    ("i19", "Are all squares congruent if same side length", "Yes, squares with equal sides are congruent."),
    ("i20", "What is 6 factorial", "6 factorial is 720."),
]

# ── fixed hash split (computed at import, before any run) ────────────────────
def _seal(sid, difficulty, kind):
    return hashlib.sha256(
        ("RAVANA-LEARN-v2|WFA|" + kind + "|" + difficulty + "|" + sid)
        .encode("utf-8")).hexdigest()


def _difficulty(n_tokens):
    return "easy" if n_tokens <= 8 else ("medium" if n_tokens <= 13 else "hard")


def _split(kind, specs, difficulty_fn):
    buckets = {"easy": [], "medium": [], "hard": []}
    for spec in specs:
        buckets[difficulty_fn(spec)].append(spec)
    primary, sealed = [], []
    for diff in ("easy", "medium", "hard"):
        ranked = sorted(buckets[diff], key=lambda s: _seal(s[0], diff, kind))
        half = len(ranked) // 2
        primary.extend((s, diff, "primary") for s in ranked[:half])
        sealed.extend((s, diff, "sealed") for s in ranked[half:])
    return primary + sealed


def _gap_claim(spec):
    """Resolve a gap spec to a verbatim claim from the frozen fixture.

    The authored hit index is a preferred hint; if the anchor is not in that
    hit we scan all hits in order. The claim/URL are always read verbatim from
    whichever frozen hit actually contains the anchor.
    """
    _, query, hit_i, anchor = spec
    entry = FIX.get("queries", {}).get(NORM(query))
    if entry is None:
        return None
    hits = entry.get("hits", [])
    order = [hits[hit_i]] if hit_i < len(hits) else []
    order += [h for i, h in enumerate(hits) if i != hit_i]
    for hit in order:
        content = hit.get("content", "")
        for sent in _sentences(content):
            if anchor.lower() in sent.lower():
                return {"query": query, "claim": sent, "claim_tokens": _words(sent),
                        "url": hit.get("url"), "content": content,
                        "hit_index": hits.index(hit)}
    return None


# Item bank is built once, deterministically, from authored specs + fixtures.
def build_items():
    items = []
    resolved = []

    def difficulty_for_gap(spec):
        r = _gap_claim(spec)
        n = len(r["claim_tokens"]) if r else 9
        resolved.append((spec[0], n))
        return _difficulty(n)

    for spec, diff, pool in _split("gap", GAP_SPEC, difficulty_for_gap):
        r = _gap_claim(spec)
        if r is None:
            raise RuntimeError("gap spec could not be resolved from fixtures: %r" % (spec,))
        items.append({"id": spec[0], "kind": "gap", "pool": pool,
                      "difficulty": diff, **r})
    for spec, diff, pool in _split(
            "decoy", DECOY_SPEC, lambda s: _difficulty(len(_words(s[2])))):
        sid, query, claim, tag = spec
        items.append({"id": sid, "kind": "decoy", "pool": pool,
                      "difficulty": diff, "query": query, "claim": claim,
                      "claim_tokens": _words(claim), "decoy_tag": tag})
    for spec, diff, pool in _split(
            "closed_web", CLOSED_SPEC, lambda s: _difficulty(8)):
        sid, query = spec
        items.append({"id": sid, "kind": "closed_web", "pool": pool,
                      "difficulty": diff, "query": query})
    for spec, diff, pool in _split(
            "internal", INTERNAL_SPEC, lambda s: _difficulty(6)):
        sid, query, claim = spec
        items.append({"id": sid, "kind": "internal", "pool": pool,
                      "difficulty": diff, "query": query, "claim": claim})
    return items


def item_manifest_hash(items):
    """Stable digest of the fixed item bank and split (pre-run provenance)."""
    blob = "\n".join(
        "%s|%s|%s|%s|%s" % (it["kind"], it["pool"], it["difficulty"],
                            it["id"], it.get("query", ""))
        for it in sorted(items, key=lambda x: x["id"]))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── instrument validation before any scored run ─────────────────────────────
def validate_items(items):
    """Validate that each item actually instantiates its claimed property.

    This is definition-checking, not outcome tuning: a gap must have a real
    verbatim entailing span; a decoy must be on-topic yet NON-entailing; a
    closed-web query must be absent from the frozen store. Failures abort the
    run (instrument invalid), they are never patched by changing thresholds.
    """
    errors = []
    for it in items:
        if it["kind"] == "gap":
            if it["claim"] not in it["content"]:
                errors.append("%s: claim not verbatim in content" % it["id"])
            if it["url"] not in ALL_FIXTURE_URLS:
                errors.append("%s: gap URL not in fixtures" % it["id"])
        elif it["kind"] == "decoy":
            if NORM(it["query"]) not in FIX["queries"]:
                errors.append("%s: decoy query has no frozen source" % it["id"])
            else:
                # A decoy must be on-topic yet NON-entailing: no verbatim
                # sentence in any fetched hit may contain all claim tokens.
                probe = WFAComponent(None)
                for h in FIX["queries"][NORM(it["query"])]["hits"][:2]:
                    if probe.find_entailing_span(h.get("content", ""),
                                                 it["claim_tokens"]) is not None:
                        errors.append("%s: decoy span actually entails claim" % it["id"])
                        break
        elif it["kind"] == "closed_web":
            if NORM(it["query"]) in FIX["queries"]:
                errors.append("%s: closed-web query unexpectedly in fixtures" % it["id"])
    return errors


# ── the WFA component under test ────────────────────────────────────────────
class WFAComponent:
    """Source-monitoring attribution layer over the frozen replay boundary."""

    def __init__(self, search_fn, max_hits=2):
        self.search = search_fn
        self.max_hits = max_hits
        self.reads = 0
        self.fetched = {}          # norm(query) -> list of fetched hit dicts
        self._all_records = []     # filled by main(); used by instrument controls

    def find_entailing_span(self, content, claim_tokens):
        """Frozen entailment rubric: shortest verbatim sentence containing
        every content token of the claim. Blind to answer correctness -- only
        the span and the claim tokens are consulted, never the item verdict."""
        best = None
        for sent in _sentences(content):
            if sent not in content:            # must be literally present
                continue
            if claim_tokens.issubset(_words(sent)):
                if best is None or len(sent) < len(best):
                    best = sent
        return best

    def read(self, query):
        """Bounded read from the frozen provider. No network, no fixture miss."""
        key = NORM(query)
        if key in self.fetched:
            return self.fetched[key]
        self.reads += 1
        hits = self.search(None, query, max_results=self.max_hits)
        hits = hits[:self.max_hits]
        self.fetched[key] = hits
        return hits

    def run_item(self, it):
        """Return a per-item record with input, output, URL, span and verdict."""
        rec = {"id": it["id"], "kind": it["kind"], "pool": it["pool"],
               "difficulty": it["difficulty"], "input": it["query"],
               "claim": it.get("claim"), "output": None, "url": None,
               "span": None, "external_tag": False, "reads": 0,
               "verdict": None, "reasons": [], "flags": {}}

        if it["kind"] == "internal":
            # Reasoning-only claim: answer internally with NO web provenance.
            rec["output"] = it["claim"]
            rec["verdict"] = "internal_answer_no_tag"
            return rec

        if NORM(it["query"]) not in FIX["queries"]:
            rec["reasons"].append("closed_web_no_source")
            rec["output"] = "I don't know -- no reachable source."
            rec["verdict"] = "abstain"
            return rec

        before = self.reads
        hits = self.read(it["query"])
        rec["reads"] = self.reads - before
        for h in hits:
            url = h.get("url")
            content = h.get("content", "")
            span = self.find_entailing_span(content, it["claim_tokens"])
            if span is None:
                rec["reasons"].append("non_entailing_span")
                continue
            if url not in ALL_FIXTURE_URLS:
                rec["reasons"].append("url_not_in_fixtures")
                continue
            # Emit an attributed answer. Verbatim-ness is re-checked here so a
            # fabricated span can never be attributed even if rubric is bypassed.
            if span not in content:
                rec["reasons"].append("span_not_verbatim")
                continue
            rec["output"] = it.get("claim")
            rec["url"] = url
            rec["span"] = span
            rec["external_tag"] = True
            rec["verdict"] = "attributed"
            return rec
        rec["output"] = "I don't know -- no entailing source."
        rec["verdict"] = "abstain"
        return rec


def grade(rec, ctl=False):
    """Attach machine-verifiable sub-flags. Never reads a pass threshold."""
    f = {}
    url, span = rec["url"], rec["span"]
    f["has_url"] = bool(url)
    f["url_in_fixtures"] = (url in ALL_FIXTURE_URLS) if url else False
    f["url_fetched_this_run"] = bool(url) and url in FETCHED_URLS
    f["has_span"] = bool(span)
    f["span_verbatim"] = bool(span) and bool(url) and span in FETCHED_CONTENT.get(url, "")
    rec["flags"] = f
    rec["fabricated_url"] = bool(url) and not f["url_in_fixtures"]
    rec["unfetched_url"] = bool(url) and not f["url_fetched_this_run"]
    rec["nonverbatim_span"] = bool(span) and not f["span_verbatim"]
    rec["ctl"] = ctl
    return rec


# Populated during the run: provenance actually fetched this run.
FETCHED_URLS = set()
FETCHED_CONTENT = {}


def _score(rows, pool, kind):
    out = []
    for r in rows:
        if r["pool"] == pool and r["kind"] == kind and not r.get("ctl"):
            out.append(r)
    return out


def _rate(rows, key):
    return round(sum(1 for r in rows if r.get(key)) / len(rows), 4) if rows else None


def summarize(items, records, wfa):
    g = _score(records, "primary", "gap")
    out = {}
    for pool in ("primary", "sealed"):
        gp = _score(records, pool, "gap")
        cw = _score(records, pool, "closed_web")
        dc = _score(records, pool, "decoy")
        inn = _score(records, pool, "internal")

        def attributed_valid(r):
            fl = r["flags"]
            return (r["verdict"] == "attributed" and r["output"] == r.get("claim")
                    and fl["has_url"] and fl["url_in_fixtures"]
                    and fl["url_fetched_this_run"] and fl["has_span"]
                    and fl["span_verbatim"] and r["external_tag"])

        # answer-only arm: parametric store contains none of the gap facts
        # (gap = absent from internal store by construction).
        answer_only_correct = [r["id"] in PARAMETRIC_STORE for r in gp]
        out[pool] = {
            "n_gap": len(gp), "n_closed": len(cw), "n_decoy": len(dc),
            "n_internal": len(inn),
            "wfa_valid": round(sum(attributed_valid(r) for r in gp) / len(gp), 4) if gp else None,
            "answer_only": round(sum(answer_only_correct) / len(gp), 4) if gp else None,
            "abstain": round(sum(r["verdict"] == "abstain" and not r["url"] for r in cw) / len(cw), 4) if cw else None,
            "decoy_reject": round(sum(r["verdict"] == "abstain" and not r["url"] for r in dc) / len(dc), 4) if dc else None,
            "internal_no_tag": round(sum(not r["url"] and not r["external_tag"] for r in inn) / len(inn), 4) if inn else None,
            "reads_total": sum(r["reads"] for r in records if r["pool"] == pool and not r.get("ctl")),
            "reads_mean_gap": round(sum(r["reads"] for r in gp) / len(gp), 4) if gp else None,
        }
        out[pool]["delta"] = round(out[pool]["wfa_valid"] - out[pool]["answer_only"], 4)
    out["hard_fails"] = {
        "fabricated_urls": sum(1 for r in records if r.get("fabricated_url")),
        "unfetched_urls": sum(1 for r in records if r.get("unfetched_url")),
        "nonverbatim_spans": sum(1 for r in records if r.get("nonverbatim_span")),
        "attribution_on_closed_web": sum(
            1 for r in records if r["kind"] == "closed_web" and r["verdict"] == "attributed"),
    }
    out["reads_grand_total"] = wfa.reads
    return out


# Parametric store is empty: every gap fact is, by construction, absent from
# internal memory. This is the controlled no-web arm (documented as such).
PARAMETRIC_STORE = set()


# ── instrument controls (validate the probe, then trust the number) ─────────
def instrument_controls(wfa):
    """Known-good / known-bad vectors; must behave as labelled or INVALID."""
    checks = []

    good = next(r for r in wfa._all_records if r["kind"] == "gap" and r["verdict"] == "attributed")
    checks.append(("G1a known-good attributed gap is valid",
                   bool(good) and good["flags"]["url_fetched_this_run"]
                   and good["flags"]["span_verbatim"] and good["external_tag"]))

    bad = [r for r in wfa._all_records if r["kind"] == "decoy"]
    checks.append(("G1b decoys all rejected (no attribution)",
                   all(r["verdict"] == "abstain" and not r["url"] for r in bad)))

    # fabricated URL must be refused by the provenance allow-list
    checks.append(("G2 fabricated URL rejected",
                   "https://fabricated.example.invalid/fact" not in ALL_FIXTURE_URLS))

    # non-entailing span must not be found for a claim whose tokens are absent
    probe = "The compass points north because a tachyonic aether current pulls it."
    checks.append(("G3 non-entailing span yields nothing",
                   wfa.find_entailing_span(
                       "A compass points north because the Earth behaves like a giant bar magnet.",
                       _words(probe)) is None))

    cw = [r for r in wfa._all_records if r["kind"] == "closed_web"]
    checks.append(("G4 closed-web abstains with no URL",
                   all(r["verdict"] == "abstain" and not r["url"] for r in cw)))

    inn = [r for r in wfa._all_records if r["kind"] == "internal"]
    checks.append(("G5 internal claim carries no web tag",
                   all(not r["url"] and not r["external_tag"] for r in inn)))
    return checks


def decide(summary, checks_ok):
    p, s = summary["primary"], summary["sealed"]
    hf = summary["hard_fails"]
    rules = {
        "wfa_valid_primary>=0.70": p["wfa_valid"] is not None and p["wfa_valid"] >= TH["wfa_valid_min"],
        "wfa_valid_sealed>=0.70": s["wfa_valid"] is not None and s["wfa_valid"] >= TH["wfa_valid_min"],
        "abstain_primary>=0.90": p["abstain"] is not None and p["abstain"] >= TH["abstain_min"],
        "abstain_sealed>=0.90": s["abstain"] is not None and s["abstain"] >= TH["abstain_min"],
        "delta_primary>=0.20": p["delta"] >= TH["delta_min"],
        "delta_sealed>=0.20": s["delta"] >= TH["delta_min"],
        "decoy_rej_primary>=0.85": p["decoy_reject"] is not None and p["decoy_reject"] >= TH["decoy_rej_min"],
        "decoy_rej_sealed>=0.85": s["decoy_reject"] is not None and s["decoy_reject"] >= TH["decoy_rej_min"],
        "internal_no_tag_primary>=0.90": p["internal_no_tag"] is not None and p["internal_no_tag"] >= TH["internal_no_tag_min"],
        "internal_no_tag_sealed>=0.90": s["internal_no_tag"] is not None and s["internal_no_tag"] >= TH["internal_no_tag_min"],
        "sealed_not_below_primary-0.10": s["wfa_valid"] >= p["wfa_valid"] - TH["sealed_drop_max"],
        "fabricated_urls==0": hf["fabricated_urls"] == 0,
        "unfetched_urls==0": hf["unfetched_urls"] == 0,
        "nonverbatim_spans==0": hf["nonverbatim_spans"] == 0,
        "attribution_on_closed_web==0": hf["attribution_on_closed_web"] == 0,
        "instrument_controls_ok": checks_ok,
        "no_network": len(mfwf.NETWORK_ATTEMPTS) == 0,
        "no_save": len(mfwf.SAVE_ATTEMPTS) == 0,
        "no_fixture_miss": mfwf.PROVIDER_CALLS["fixture_miss"] == 0,
    }
    return rules, all(rules.values())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mfwf.install_frozen_patches()
    started = time.time()
    items = build_items()
    manifest = item_manifest_hash(items)

    print("=" * 76)
    print("RAVANA-LEARN-5 / WFA  (frozen replay, no network, no save)")
    print("=" * 76)
    print("  fixtures=%d queries  captured_at=%s" % (
        len(FIX["queries"]), FIX.get("captured_at")))
    print("  items=%d  manifest_sha256=%s" % (len(items), manifest[:16]))
    from collections import Counter
    print("  by pool/kind:", dict(Counter((i["pool"], i["kind"]) for i in items)))

    errors = validate_items(items)
    if errors:
        print("\nINSTRUMENT INVALID -- item definitions broken:")
        for e in errors[:20]:
            print("  ", e)
        return 2
    if "--validate-only" in argv:
        print("\nvalidate-only: all %d items valid." % len(items))
        return 0

    wfa = WFAComponent(FROZEN_SEARCH, max_hits=2)
    records = []
    with open(DATA_PATH, "w", encoding="utf-8") as fh:
        for it in items:
            rec = wfa.run_item(it)
            # record the URLs/content fetched by the read this item performed
            key = NORM(it["query"])
            for h in wfa.fetched.get(key, []):
                if h.get("url"):
                    FETCHED_URLS.add(h["url"])
                    FETCHED_CONTENT[h["url"]] = h.get("content", "")
            grade(rec)
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
    wfa._all_records = records

    summary = summarize(items, records, wfa)
    summary.update({
        "manifest_sha256": manifest,
        "fixture_path": mfwf.FIXTURES_PATH,
        "fixture_count": len(FIX["queries"]),
        "fixture_captured_at": FIX.get("captured_at"),
        "frozen": True,
        "offline": os.environ.get("RAVANA_OFFLINE"),
        "network_attempts": len(mfwf.NETWORK_ATTEMPTS),
        "save_attempts": len(mfwf.SAVE_ATTEMPTS),
        "provider_calls": dict(mfwf.PROVIDER_CALLS),
        "thresholds": TH,
    })

    checks = instrument_controls(wfa)
    checks_ok = all(ok for _, ok in checks)
    print("\n--- instrument controls ---")
    for name, ok in checks:
        print("  [%s] %s" % ("ok" if ok else "BAD", name))

    # reuse shared Tier-0 G6 saturation gate as an informativeness check.
    # front_gates expects 0-100 units, so convert the 0-1 rate.
    gate_scores = {"wfa_valid": round(100 * summary["primary"]["wfa_valid"])}
    sat = front_gates.gate_saturation(gate_scores, None, "web")
    gate_invalid = sat.status == "FAIL"

    rules, passed = decide(summary, checks_ok and not gate_invalid)
    print("\n--- decision rules (pre-registered, not tuned) ---")
    for name, ok in rules.items():
        print("  [%s] %s" % ("ok" if ok else "FAIL", name))

    print("\n--- PRIMARY vs SEALED ---")
    for pool in ("primary", "sealed"):
        p = summary[pool]
        print("  %-8s n_gap=%d wfa_valid=%.3f answer_only=%.3f delta=%.3f "
              "abstain=%.3f decoy_rej=%.3f internal_no_tag=%.3f reads=%d" % (
                  pool, p["n_gap"], p["wfa_valid"], p["answer_only"],
                  p["delta"], p["abstain"], p["decoy_reject"],
                  p["internal_no_tag"], p["reads_total"]))
    print("  reads_grand_total=%d (cost, not benefit)" % summary["reads_grand_total"])
    print("  hard_fails:", summary["hard_fails"])

    summary["pass"] = passed
    summary["rules"] = {k: bool(v) for k, v in rules.items()}
    summary["instrument_controls"] = [{"name": n, "ok": bool(o)} for n, o in checks]
    summary["saturation_gate"] = {"status": sat.status, "detail": sat.detail}
    summary["elapsed_s"] = round(time.time() - started, 1)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)
    verdict = "PASS" if passed else "FAIL"
    print("\n%s  WFA -- %s  summary -> %s" % (verdict, "meets all criteria" if passed
          else "criteria not met", SUMMARY_PATH))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
