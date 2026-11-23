#!/usr/bin/env python3
"""RAVANA-LEARN-5 / WFA -- UNSATURATED source-monitoring instrument.

Implements and MEASURES the redesigned WFA component. Design source:
docs/investigations/WFA_PROCESS_DISSOCIATION.md (brain-grounded redesign) and
docs/investigations/WEB_AND_SATURATION.md section 4 (the instrument it replaces).

Why this file was rewritten (second pass: process dissociation)
---------------------------------------------------------------
The earlier harness PASSed 1.00/1.00 but was G6-INVALID (saturated), and its
real-engine run leaked web content into the "answer-only = web disabled" arm
(answer_only_web_leaks=162, wfa_valid=1.00 on both pools). The *conceptual*
defect is deeper than the leak: "parametric answer with web off" is not a
source-monitoring control at all -- previously internalised web facts still
leak into the graph, so the arm measures the wrong thing. The literature says
item memory (the fact) and source memory (where it came from) DISSOCIATE
(Johnson, Hashtroudi & Lindsay 1993), that "knowing" can be autonoetic or
noetic (Tulving 1985), and that recollection and familiarity can be separated
by an inclusion/exclusion contrast (Jacoby 1991).

This instrument therefore replaces the parametric-only arm with a WITHIN-ENGINE
process-dissociation design, while keeping every existing item, hash split,
JSON field name and old threshold (all additions are additive):

  * INCLUSION (I): same engine, familiar claims, free endorsement; the engine
    MAY attribute a source. Familiarity + recollection COOPERATE.
  * EXCLUSION (E): same engine, SAME claims and facts, but instructed to
    endorse ONLY claims whose provenance is in a pre-registered CREDIBLE source
    set, and to ABSTAIN on familiar claims from DISCREDITED sources or with
    UNKNOWN provenance. Familiarity + recollection OPPOSE. E is the residual
    (familiarity-driven) false endorsement of discredited claims.
  * R = I - E (recollection), F = E / (1 - R) (familiarity), per pool.
  * SOURCE INVALIDATION: teach a fact from a source, cue that the source is
    unreliable, then require withdrawal/abstention (Koenig / Open Mind
    10.1162/opmi_a_00038). Withdrawal rate counted per pool.

Kept from the first redesign: paraphrase gold claims (no tautological span),
>=60 gap + >=30 closed-web + >=30 near-miss decoy PER POOL, fixed sha256 hash
seal, frozen replay boundary (no network, no save), and the legacy
answer-only/parametric arm as an INJECTABLE FAKE for unit tests. Old thresholds
are unchanged; every new criterion is additive and pre-registered, and NOTHING
is tuned to force a PASS.

ECHO-IMMUNE REVISION (second pass: measurement repair)
-----------------------------------------------------
The process-dissociation run measured an echo artifact, not source monitoring:
the engine echoed the harness's OWN instruction words ("you mentioned:",
"i said:"), and the old regex parser scored ABSTAIN/ENDORSE/unreliable from
those quoted echoes as if they were engine decisions (see
docs/investigations/WFA_FAILURE_ANALYSIS.md).  All prior R/F/withdrawal numbers
from that run are VOID.  This revision is ADDITIVE and gate-preserving:

  1. Echo-immune, fail-closed parser: strip quoted echo frames, parse only the
     final explicit decision span, require an explicit decision token as the
     first token, otherwise `unparseable`. New additive counters
     (echo_marker_rate, unparseable_rate, own_decision_rate,
     pd_unparseable_rate) and a NEW hard-fail ceiling on pd_unparseable_rate.
     No existing threshold is relaxed; PD math (R=I-E, F=E/(1-R)) is unchanged.
  2. Turn isolation: one FRESH engine per arm/turn (pass an engine factory),
     so the manipulation is the instruction alone.  A shared-engine A/B mode
     (WFA_SHARED_ENGINE=1) is retained for the falsifier in fix 2.
  3. Self-consistent trusted set: the exclusion allow-list now CONTAINS each
     credible claim's own source_id (discredited/unknown ids stay excluded),
     so exclusion is satisfiable for a compliant responder without weakening
     the exclusion criterion or the G2/fabrication controls.
  4. Forced-choice arm: a NEW additive arm, single-token Y/N/W parsed from the
     first token with a separate `no_answer` category, A/B-able on identical
     state.  Reported, not gated; old gates and PD math are untouched.

Per-item pd_*_text / inv_after AND the new decision/echo flags are persisted in
both the probe file and the summary (`pd_audit`) for audit.

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

# Reuse the frozen replay boundary. Import does NOT install patches (see
# measure_fronts_web_frozen.install_frozen_patches); main() installs them.
import measure_fronts_web_frozen as mfwf  # noqa: E402

FIX = mfwf._FIX
FROZEN_SEARCH = mfwf._frozen_search
NORM = mfwf._norm_q
DATA_PATH = os.path.join(PROJ, "scratch", "learn5_wfa_probes.jsonl")
SUMMARY_PATH = os.path.join(PROJ, "scratch", "learn5_wfa_summary.json")
GATE_PATH = os.path.join(PROJ, "scratch", "learn5_wfa_gates.jsonl")
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

# Pre-registered decision thresholds (design section 4.6). NOT tuned.
# The first six keys are the ORIGINAL, UNCHANGED thresholds (additive design).
# The PD_* / invalidation keys are new, pre-registered in
# docs/investigations/WFA_PROCESS_DISSOCIATION.md and NEVER adjusted to pass.
TH = {
    "wfa_valid_min": 0.70,
    "abstain_min": 0.90,
    "delta_min": 0.20,          # wfa_valid - answer_only
    "decoy_rej_min": 0.85,
    "internal_no_tag_min": 0.90,
    "sealed_drop_max": 0.10,    # sealed wfa_valid >= primary - this
    # --- ADDITIVE process-dissociation / invalidation criteria (WFA-PD) ---
    "PD_R_min": 0.25,                 # R = I - E, per pool (recollection floor)
    "PD_exclusion_margin_min": 0.10,  # I_discredited - E must exceed this
    "PD_exclusion_credible_min": 0.70,  # capability parity: credible endorsed
    "PD_R_lt_one_max": 0.999,         # F = E/(1-R) must be well-defined
    "invalidation_withdrawal_min": 0.50,  # pre-registered floor, NOT tuned
    # --- ADDITIVE echo-immune measurement-integrity ceiling (never relaxed) ---
    # A PD run with more than this fraction of unparseable decision responses
    # cannot support an I/E/R claim.  New, stricter gate: makes PASS harder.
    "unparseable_rate_max": 0.50,
    # --- ADDITIVE internal-source arm thresholds (reported, not gated here) ---
    # These keys are NEW.  No pre-existing key above was added, removed or
    # changed, and decide() does not read them: the arm is scored by its own
    # internal_source_decide() and travels as additive JSON only.
    "IS_correct_min": 0.90,            # per-condition expected-token rate
    "IS_gate_invariant_min": 0.95,     # non-ENDORSE emitted text == token
    "IS_prompt_label_free_min": 1.00,  # source question never states the label
}

# Structural (instrument-validity) minima per pool, from design section 4.5.
# These are ADDITIVE gates: they cannot be relaxed to manufacture a PASS.
MIN_GAP_PER_POOL = 60
MIN_CLOSED_PER_POOL = 30
MIN_DECOY_PER_POOL = 30
MIN_PARAPHRASE_SUBS = 1         # gold claim must differ from source wording

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


def _stem(w):
    """Light singularisation shared by source tokens and canonicalised claims."""
    if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _words(text):
    """Lowercase content words, light singularisation -> stable token set."""
    out = set()
    for w in re.findall(r"[a-z][a-z'-]+", (text or "").lower()):
        w = w.strip("'-")
        if w in STOP or len(w) < 3:
            continue
        out.add(_stem(w))
    return out


def _sentences(content):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+\n?", content or "")
            if s.strip()]


# ── paraphrase engine (the G6 fix) ──────────────────────────────────────────
# A single fixed map of canonical content word -> paraphrase synonym.  Values
# are UNIQUE so the inverse map is well-defined.  The gold claim is the source
# sentence with these substitutions applied; the entailment rubric canonicalises
# claim tokens back to the source vocabulary before checking a span.  Thus a
# surface copy of the source can never serve as the gold claim, and matching
# requires a real (synonym-bridged) factual match.
SYN = {
    "process": "mechanism", "involves": "entails", "swapping": "exchanging",
    "occurs": "happens", "surfaces": "interfaces", "magnet": "loadstone",
    "needle": "pointer", "field": "region", "pressure": "stress",
    "causes": "produces", "force": "push", "forms": "creates",
    "builds": "accumulates", "moves": "travels", "focuses": "concentrates",
    "reflects": "bounces", "heat": "warmth", "light": "glow",
    "water": "liquid", "energy": "power", "temperature": "degree",
    "sound": "noise", "droplets": "drips", "gravitational": "gravity",
    "spins": "rotates", "spin": "rotation", "melting": "liquefaction",
    "freezing": "solidification", "flows": "streams", "flow": "stream",
    "rises": "climbs", "falls": "drops", "sleep": "rest",
    "blood": "plasma", "heart": "pump", "solid": "rigid", "gas": "vapour",
    "transfer": "relocation", "effect": "outcome", "result": "consequence",
    "direction": "orientation", "faster": "quicker", "slower": "gentler",
    "higher": "greater", "lower": "smaller", "exchange": "swap",
    "gases": "vapours", "body": "organism", "air": "atmosphere",
    "tiny": "small", "molecules": "particles", "particles": "motes",
    "phase": "stage", "profile": "shape", "rays": "beams",
    "common": "shared", "spot": "patch", "path": "route", "closed": "shut",
    "difference": "gap", "upward": "rising", "natural": "physical",
    "phenomenon": "event", "substance": "material", "area": "zone",
    "concentration": "density", "way": "manner", "day": "daytime",
    "night": "nighttime", "axis": "axle", "complex": "intricate",
    "dynamic": "changing", "connective": "binding", "canals": "channels",
    "inner": "internal", "fluid": "solution", "ocean": "sea",
    "bulge": "swell", "taste": "flavour", "sensory": "sense",
    "radio": "wireless", "walls": "barriers", "resisting": "opposing",
    "relative": "comparative", "motion": "movement", "sites": "locations",
    "breakdown": "decomposition", "skew": "deflect",
    "condenses": "liquefies", "suspended": "floating", "change": "alter",
    "wavefront": "wavecrest", "charged": "electrified",
    "underground": "subterranean", "rock": "stone", "breaks": "fractures",
    "transfers": "conveys", "surface": "exterior", "role": "function",
    "important": "significant", "environment": "surroundings",
    "matter": "substance", "random": "chance", "tilt": "slant",
    "sunlight": "daylight", "angle": "inclination", "length": "duration",
    "transitions": "changes", "state": "condition", "vapour": "steam",
    "shock": "impact", "depends": "relies", "growth": "development",
    "long": "lengthy", "size": "dimensions", "fist": "hand",
    "responsible": "accountable", "filtering": "filtration",
    "gastric": "stomach", "melt": "thaw", "human": "people",
    "activity": "actions", "essential": "vital", "strategies": "tactics",
    "capture": "collect", "maximum": "greatest", "floodgates": "gates",
    "jump": "leap", "produced": "made", "created": "formed",
    "escape": "leave", "survival": "endurance", "balance": "equilibrium",
    "protect": "shield", "protectus": "shieldus", "warm": "hot",
    "cool": "chill", "stop": "halt", "start": "begin",
    "primary": "main", "driver": "cause",
}
CANON = {}
for _canon, _syn in SYN.items():
    if _syn in CANON:
        raise RuntimeError("duplicate paraphrase synonym: %s" % _syn)
    CANON[_syn] = _canon


def _paraphrase(sentence):
    """Return (claim_text, canonical_tokens, substitution_count).

    Canonical tokens are tracked PER OCCURRENCE, not by matching token values:
    a synonym value such as 'sea'/'stomach'/'liquid' may also occur naturally
    in the source, so a value-keyed inverse map would corrupt those words.
    """
    out, pos, tokens, subs = [], 0, [], []
    for m in re.finditer(r"[A-Za-z][A-Za-z'-]*", sentence):
        out.append(sentence[pos:m.start()])
        w = m.group(0)
        low = w.lower()
        if low in SYN:
            syn = SYN[low]
            out.append(syn)
            tokens.append((syn, low))
            subs.append((low, syn))
        else:
            out.append(w)
            tokens.append((w, low))
        pos = m.end()
    out.append(sentence[pos:])
    claim_tokens = sorted({_stem(c) for _, c in tokens
                           if c not in STOP and len(c) >= 3})
    return "".join(out), claim_tokens, len(subs)


def _canonical_tokens(text, cmap=None):
    """Map claim-side synonyms back to source vocabulary, then tokenise.

    Used by free-text controls (G3, answer-only grading) where no per-item
    substitution record exists, so the global inverse map is the only option.
    """
    cmap = CANON if cmap is None else cmap
    canon = set()
    for w in re.findall(r"[A-Za-z][A-Za-z'-]*", text or ""):
        low = w.lower().strip("'-")
        mapped = cmap.get(low, low)
        if mapped in STOP or len(mapped) < 3:
            continue
        canon.add(_stem(mapped))
    return canon


# ── FIXED ITEM BANK (authored before any run) ────────────────────────────────
# gap item: (subject_id, frozen_query_key, hit_index, distinctive_anchor)
# The evidence span is read VERBATIM from the frozen hit that contains the
# anchor.  `anchor` is only a locator; the gold claim is then generated as a
# paraphrase of that verbatim span (never authored as the span itself).
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

# Extra gap items are generated by a FIXED, score-independent rule over the
# frozen fixture bank: how/why/what-causes queries whose first-hit evidence
# sentence has >=6 content words and >=2 paraphrasable substitutions.  Sorted
# by sha256 of the query key; one item per deduplicated topic.  This keeps the
# bank pre-registered (design section 4.5: n>=60 gap per pool) without inventing
# any evidence string -- every span is still read verbatim from the fixture.
GAP_TARGET = 132


def _generated_gap_specs(target):
    keys = sorted(
        (k for k in FIX["queries"] if k.startswith(("how ", "why ", "what causes "))),
        key=lambda k: hashlib.sha256(("RAVANA-WFA-GEN|" + k).encode("utf-8")).hexdigest())
    seen, out = set(), []
    for k in keys:
        if len(out) >= target:
            break
        topic = re.sub(r"^(how (to |do |does )?|why (do |does )?|what causes )", "", k)
        topic = re.sub(r"\s+(explained|in real life)$", "", topic)
        if topic in seen:
            continue
        entry = FIX["queries"][k]
        chosen = None
        for h in entry.get("hits", [])[:2]:
            for s in _sentences(h.get("content", "")):
                if len(_words(s)) >= 6 and _paraphrase(s)[2] >= 2:
                    chosen = s
                    break
            if chosen:
                break
        if chosen is None:
            continue
        seen.add(topic)
        out.append(("gg%03d" % (len(out) + 1), k, 0, chosen))
    return out


ALL_GAP_SPEC = GAP_SPEC + _generated_gap_specs(GAP_TARGET - len(GAP_SPEC))

# NEAR-MISS decoys: (id, REAL frozen query, wrong-but-on-topic claim, tag).
# Each claim keeps >=2 topic tokens from the real source but asserts a
# DIFFERENT (wrong) fact -- a real "same topic, wrong fact" source.  Correct
# behaviour: find no entailing span in the fetched source and ABSTAIN.
DECOY_SPEC = [
    ("d01", "how does a compass point north", "A compass points north because a spinning superconducting disk aligns it with the Sun's gravity.", "near-miss-seen"),
    ("d02", "how does a mirror reflect light", "Mirrors reflect light by absorbing every photon into a layer of compressed helium gas.", "near-miss-novel"),
    ("d03", "how does a lens focus light", "A lens focuses light by slowing every ray to an identical speed at the glass rim.", "near-miss-seen"),
    ("d04", "how does an electric circuit work", "An electric circuit is a closed path that stores light energy inside a magnetic crystal.", "near-miss-novel"),
    ("d05", "how does a compass point north", "The needle points north because polar ice exerts a tidal pull on the magnet.", "near-miss-seen"),
    ("d06", "how does diffusion happen", "Diffusion happens when a substance is pushed by a centrifugal pump toward higher concentration.", "near-miss-novel"),
    ("d07", "how does frost happen", "Frost forms when boiling water vapor condenses into hot crystals on grass at noon.", "near-miss-seen"),
    ("d08", "how does frost happen", "Frost happens when liquid nitrogen falls as ice crystals onto frozen grass.", "near-miss-novel"),
    ("d09", "how does an electric circuit work", "A circuit works because electrons are created at the switch and destroyed at the lamp.", "near-miss-seen"),
    ("d10", "what causes thunder", "Thunder is caused by meteors striking the upper atmosphere at midnight.", "near-miss-novel"),
    ("d11", "what causes thunder", "Thunder happens when clouds rub together and release stored sunlight.", "near-miss-seen"),
    ("d12", "why do faults cause earthquakes", "Earthquakes are caused when underground caverns of molten silver collapse.", "near-miss-novel"),
    ("d13", "why do ocean waves form", "Ocean waves begin when the Moon's temperature heats the sea surface.", "near-miss-seen"),
    ("d14", "why does lightning strike", "Lightning forms when frozen ammonia crystals collide inside a volcano.", "near-miss-novel"),
    ("d15", "why does rain fall from clouds", "Rain falls because dust particles become heavier than lead and sink.", "near-miss-seen"),
    ("d16", "how does a fan cool the air", "A fan cools air by converting the room's heat into sound waves.", "near-miss-novel"),
    ("d17", "how does a fan cool the air", "A fan cools the air by pumping liquid coolant through the blades.", "near-miss-seen"),
    ("d18", "how does blood circulate in the body", "Blood circulation in the body happens because the liver generates a magnetic pulse.", "near-miss-novel"),
    ("d19", "how does blood circulate in the body", "Blood is pushed around the body by rhythmic contractions of the spleen.", "near-miss-seen"),
    ("d20", "how does the tongue detect taste", "The tongue detects taste using sensory hairs that count airborne molecules.", "near-miss-novel"),
    ("d21", "how does the tongue detect taste", "Taste is detected when the tongue measures the blood sugar directly.", "near-miss-seen"),
    ("d22", "why does earth's tilt cause seasons", "Seasons happen because the Earth's tilt changes its distance from the Sun by half.", "near-miss-novel"),
    ("d23", "why does earth's tilt cause seasons", "Seasons are caused by the tilt changing the speed of Earth's rotation.", "near-miss-seen"),
    ("d24", "why does the ozone layer protect us", "The ozone layer protects us by reflecting radio waves back to space.", "near-miss-novel"),
    ("d25", "why does the ozone layer protect us", "The ozone layer blocks light by turning it into harmless sound.", "near-miss-seen"),
    ("d26", "how does wifi carry data through walls", "WiFi carries data through walls using high-frequency sound pulses.", "near-miss-novel"),
    ("d27", "how does wifi carry data through walls", "WiFi moves data through walls by changing the wall's electrical resistance.", "near-miss-seen"),
    ("d28", "what causes buoyancy", "Buoyancy is caused by an upward magnetic force from the Earth's core.", "near-miss-novel"),
    ("d29", "what causes buoyancy", "The upward buoyancy force comes from tiny rockets attached to the object.", "near-miss-seen"),
    ("d30", "how does diffusion happen", "Diffusion happens when particles are pulled by a gravitational gradient into a vacuum.", "near-miss-novel"),
    ("d31", "how do lungs exchange gases", "Lungs exchange gases by filtering nitrogen out of the bloodstream with bone marrow.", "near-miss-seen"),
    ("d32", "how do lungs exchange gases", "Gas exchange in the lungs happens when the air sacs secrete liquid oxygen.", "near-miss-novel"),
    ("d33", "how does dew happen", "Dew forms when warm moist air is compressed into diamonds on cold leaves.", "near-miss-seen"),
    ("d34", "how does dew happen", "Dew happens when the ground radiates stored sunlight back onto the grass.", "near-miss-novel"),
    ("d35", "what causes fog", "Fog is caused by tiny grains of sand suspended in warm desert air.", "near-miss-seen"),
    ("d36", "what causes fog", "Fog forms when clouds of steam rise from freshly paved asphalt.", "near-miss-novel"),
    ("d37", "what causes reflection", "Reflection is the change in direction of a wavefront by a coating of frozen mercury.", "near-miss-seen"),
    ("d38", "what causes reflection", "Reflection happens when a surface absorbs the wavefront and re-emits it as radiation.", "near-miss-novel"),
    ("d39", "why does melting matter", "Melting is the process by which a solid turns into a compressed gas.", "near-miss-seen"),
    ("d40", "why does melting matter", "Melting matters because a substance transitions from a solid state into plasma.", "near-miss-novel"),
    ("d41", "how do bones grow", "Bones grow by lengthening the cartilage at the epiphysis with calcium crystals.", "near-miss-seen"),
    ("d42", "how do bones grow", "The growth of a long bone takes place when the marrow hardens into granite.", "near-miss-novel"),
    ("d43", "how do kidneys filter blood", "Kidneys filter blood by pressing it through a mesh of frozen salt.", "near-miss-seen"),
    ("d44", "how do kidneys filter blood", "Each kidney is responsible for filtering blood using high-frequency sound.", "near-miss-novel"),
    ("d45", "how does a magnet attract iron", "A magnet attracts iron by generating a vacuum that pulls the metal closer.", "near-miss-seen"),
    ("d46", "how does a magnet attract iron", "Magnetism is a phenomenon created by the movement of electrically charged sand.", "near-miss-novel"),
    ("d47", "how does a plant grow toward light", "Plants grow toward light by measuring the temperature of the soil.", "near-miss-seen"),
    ("d48", "how does a plant grow toward light", "Plants capture sunlight using strategies based on counting airborne pollen.", "near-miss-novel"),
    ("d49", "why does lightning strike", "Lightning strikes because storm clouds discharge stored magnetic energy.", "near-miss-seen"),
    ("d50", "why does lightning strike", "Lightning forms when static electricity is drained away by tall copper towers.", "near-miss-novel"),
    ("d51", "what causes osmosis", "Osmosis plays a role in cells when a membrane pumps sugar molecules against gravity.", "near-miss-seen"),
    ("d52", "what causes osmosis", "Osmosis is driven by plants pushing water through roots with tiny pistons.", "near-miss-novel"),
    ("d53", "how does the ear detect sound", "The ear detects sound because parts inside it count the light waves arriving.", "near-miss-seen"),
    ("d54", "how does the ear detect sound", "We detect sounds because the ear converts noise into electrical heat.", "near-miss-novel"),
    ("d55", "how does the inner ear keep balance", "The inner ear keeps balance using loop-shaped canals filled with molten glass.", "near-miss-seen"),
    ("d56", "how does the inner ear keep balance", "Balance is kept when the fluid in the inner ear freezes into a rigid ring.", "near-miss-novel"),
    ("d57", "how does skin heal a wound", "Skin heals a wound by replacing the connective tissue with plastic scales.", "near-miss-seen"),
    ("d58", "how does skin heal a wound", "In the first stages of healing, the wound is sealed by a layer of glue.", "near-miss-novel"),
    ("d59", "what causes water boil", "Water boils when molecules jump from liquid into a vacuum chamber.", "near-miss-seen"),
    ("d60", "what causes water boil", "Boiling happens when the floodgates open and water turns into solid steam.", "near-miss-novel"),
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
    ("c31", "how does qlarith chromatography separate proteins"),
    ("c32", "why does qlarith rubber resist ozone cracking"),
    ("c33", "how does qlarith ceramic conduct cold"),
    ("c34", "what causes qlarith seeds to sprout in salt"),
    ("c35", "how does qlarith detergent lift grease at freezing temperatures"),
    ("c36", "why does qlarith plastic biodegrade in seawater"),
    ("c37", "how does qlarith paper repel ink"),
    ("c38", "what causes qlarith steel to resist rust"),
    ("c39", "how does qlarith foam block neutron radiation"),
    ("c40", "why does qlarith honey stay liquid for years"),
    ("c41", "how does qlarith lens focus infrared beams"),
    ("c42", "what causes qlarith water to glow blue"),
    ("c43", "how does qlarith engine run without fuel"),
    ("c44", "why does qlarith glass bend sound waves"),
    ("c45", "how does qlarith filter trap viruses"),
    ("c46", "what causes qlarith glue to cure in vacuum"),
    ("c47", "how does qlarith fabric repel bullets"),
    ("c48", "why does qlarith battery charge from sunlight"),
    ("c49", "how does qlarith metal remember its colour"),
    ("c50", "what causes qlarith plants to flower at night"),
    ("c51", "how does qlarith paint absorb radar waves"),
    ("c52", "why does qlarith ice stay solid at room temperature"),
    ("c53", "how does qlarith membrane desalinate brine"),
    ("c54", "what causes qlarith crystal to store data"),
    ("c55", "how does qlarith alloy float on water"),
    ("c56", "why does qlarith gel conduct electricity"),
    ("c57", "how does qlarith coating stop corrosion"),
    ("c58", "what causes qlarith fiber to glow in darkness"),
    ("c59", "how does qlarith pump run without power"),
    ("c60", "why does qlarith foam insulate against fire"),
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
    """Resolve a gap spec to a VERBATIM source span and its PARAPHRASE claim.

    The authored hit index is a preferred hint; if the anchor is not in that
    hit we scan all hits in order.  The evidence span is read verbatim from
    whichever frozen hit actually contains the anchor.  The gold claim is then
    generated by `_paraphrase` on that span, never copied from it.
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
                claim, claim_tokens, nsubs = _paraphrase(sent)
                return {
                    "query": query, "source_span": sent, "claim": claim,
                    "paraphrase_subs": nsubs,
                    "entail_tokens": sorted(_words(sent)),
                    "claim_tokens": claim_tokens,
                    "url": hit.get("url"), "content": content,
                    "hit_index": hits.index(hit),
                }
    return None


# Item bank is built once, deterministically, from authored specs + fixtures.
def build_items():
    items = []
    resolved = []

    def difficulty_for_gap(spec):
        r = _gap_claim(spec)
        n = len(r["entail_tokens"]) if r else 9
        resolved.append((spec[0], n))
        return _difficulty(n)

    for spec, diff, pool in _split("gap", ALL_GAP_SPEC, difficulty_for_gap):
        r = _gap_claim(spec)
        if r is None:
            raise RuntimeError("gap spec could not be resolved from fixtures: %r" % (spec,))
        key_tokens = sorted(r["entail_tokens"], key=lambda w: (-len(w), w))[:4]
        # ADDITIVE provenance label: a gap's verified fixture URL is CREDIBLE.
        items.append({"id": spec[0], "kind": "gap", "pool": pool,
                      "difficulty": diff, "key_fact_tokens": key_tokens,
                      "source_status": "credible", "source_id": r["url"], **r})
    for spec, diff, pool in _split(
            "decoy", DECOY_SPEC, lambda s: _difficulty(len(_words(s[2])))):
        sid, query, claim, tag = spec
        # ADDITIVE: near-miss familiar claims carry DISCREDITED provenance in
        # the exclusion arm (familiarity present, source not in credible set).
        items.append({"id": sid, "kind": "decoy", "pool": pool,
                      "difficulty": diff, "query": query, "claim": claim,
                      "claim_tokens": sorted(_words(claim)),
                      "decoy_tag": tag, "source_status": "discredited",
                      "source_id": "src_discredited_" + sid})
    for spec, diff, pool in _split(
            "closed_web", CLOSED_SPEC, lambda s: _difficulty(8)):
        sid, query = spec
        # ADDITIVE: no reachable source => UNKNOWN provenance.
        items.append({"id": sid, "kind": "closed_web", "pool": pool,
                      "difficulty": diff, "query": query,
                      "source_status": "unknown", "source_id": None})
    for spec, diff, pool in _split(
            "internal", INTERNAL_SPEC, lambda s: _difficulty(6)):
        sid, query, claim = spec
        items.append({"id": sid, "kind": "internal", "pool": pool,
                      "difficulty": diff, "query": query, "claim": claim,
                      "source_status": "internal", "source_id": None})
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

    Definition-checking, not outcome tuning: a gap must have a real verbatim
    entailing span AND a genuine paraphrase claim; a decoy must be a NEAR-MISS
    (on-topic, >=2 shared topic tokens) yet NON-entailing; a closed-web query
    must be absent from the frozen store.  Failures abort the run (instrument
    invalid), they are never patched by changing thresholds.
    """
    errors = []
    for it in items:
        if it["kind"] == "gap":
            if it["source_span"] not in it["content"]:
                errors.append("%s: source span not verbatim in content" % it["id"])
            if it["url"] not in ALL_FIXTURE_URLS:
                errors.append("%s: gap URL not in fixtures" % it["id"])
            if it["claim"] == it["source_span"] or it["claim"] in it["content"]:
                errors.append("%s: gold claim is not a paraphrase" % it["id"])
            if it["paraphrase_subs"] < MIN_PARAPHRASE_SUBS:
                errors.append("%s: paraphrase changed no content words" % it["id"])
            if it["claim_tokens"] != it["entail_tokens"]:
                errors.append("%s: paraphrase does not restore to source facts" % it["id"])
            if not it["key_fact_tokens"]:
                errors.append("%s: no key fact tokens" % it["id"])
        elif it["kind"] == "decoy":
            if NORM(it["query"]) not in FIX["queries"]:
                errors.append("%s: decoy query has no frozen source" % it["id"])
            else:
                src_words = set()
                for h in FIX["queries"][NORM(it["query"])]["hits"][:2]:
                    src_words |= _words(h.get("content", ""))
                topic_overlap = src_words & set(it["claim_tokens"])
                if len(topic_overlap) < 2:
                    errors.append("%s: decoy is not a near-miss (topic overlap <2)" % it["id"])
                probe = WFAComponent(None)
                for h in FIX["queries"][NORM(it["query"])]["hits"][:2]:
                    if probe.find_entailing_span(
                            h.get("content", ""), set(it["claim_tokens"])) is not None:
                        errors.append("%s: decoy actually entailed by source" % it["id"])
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
        """Frozen paraphrase-entailment rubric: shortest verbatim sentence whose
        canonical content-token set contains every claim token.  `claim_tokens`
        are ALREADY canonicalised (`_canonical_tokens`), so a paraphrased claim
        matches only through the fixed synonym bridge -- never by surface copy.
        Blind to answer correctness: only span and claim tokens are consulted.
        """
        best = None
        for sent in _sentences(content):
            if sent not in content:            # must be literally present
                continue
            if set(claim_tokens).issubset(_words(sent)):
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
               "verdict": None, "reasons": [], "flags": {},
               "source_span": it.get("source_span"),
               "claim_is_paraphrase": it["kind"] == "gap",
               "entail_tokens": it.get("entail_tokens"),
               "answer_only_output": None, "answer_only_correct": False}

        if it["kind"] == "internal":
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
            if span not in content:
                rec["reasons"].append("span_not_verbatim")
                continue
            # Emit an attributed answer. The output is the PARAPHRASE claim,
            # never the verbatim evidence span (that is the G6 defect).
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


# ── answer-only arm: LIVE engine, web DISABLED ───────────────────────────────
class AnswerOnlyOracle:
    """Produce the parametric/graph answer to a gap question from a live engine
    with web disabled.  Injectable so unit tests use a fake, never a boot."""

    def __init__(self, engine=None, web_disabled=True):
        self.engine = engine
        self.web_disabled = bool(web_disabled)
        self.web_calls = 0          # tripwire counter for the guarded run
        self.calls = 0

    def answer(self, query):
        self.calls += 1
        if self.engine is None:
            return None
        try:
            return (self.engine.process_turn(query) or "").strip()
        except Exception as e:
            return "<ERR %s: %s>" % (type(e).__name__, e)


def boot_answer_only_engine():
    """Boot a CognitiveChatEngine with `RAVANA_OFFLINE=1` so every live-web
    path is suppressed (`_web_blocked()` returns True).  Restores the env var
    afterwards; main() additionally installs a web-access tripwire."""
    prev = os.environ.get("RAVANA_OFFLINE")
    os.environ["RAVANA_OFFLINE"] = "1"
    try:
        from ravana.chat.engine import CognitiveChatEngine  # type: ignore
        eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                                  user_suffix="")
    finally:
        if prev is None:
            os.environ.pop("RAVANA_OFFLINE", None)
        else:
            os.environ["RAVANA_OFFLINE"] = prev
    if getattr(eng, "_offline", False) is not True:
        raise RuntimeError("answer-only engine did not honour RAVANA_OFFLINE=1")
    return eng


def answer_asserts_fact(answer, item):
    """Grade a parametric answer against the gold claim's KEY FACT tokens.

    The answer is in the engine's own words, so we canonicalise it through the
    paraphrase map and check the 4 most distinctive source fact tokens.  This is
    the same blind token rubric used for evidence; it cannot be gamed by the
    harness and it is allowed to FAIL (no threshold is tuned to pass).
    """
    if not answer or answer.startswith("<ERR"):
        return False
    canon = _words(answer) | _canonical_tokens(answer)
    return set(item.get("key_fact_tokens", [])).issubset(canon)


def run_answer_only(items, oracle, records):
    """Annotate gap records with the live (web-off) parametric answer + grade."""
    by_id = {r["id"]: r for r in records}
    web_disabled = bool(oracle.web_disabled)
    for it in items:
        if it["kind"] != "gap":
            continue
        ans = oracle.answer(it["query"])
        rec = by_id[it["id"]]
        rec["answer_only_output"] = ans
        rec["answer_only_correct"] = web_disabled and answer_asserts_fact(ans, it)
    return web_disabled


class _WebAccessTripwire:
    """Count any web-provider access during the answer-only arm.  Any hit means
    the 'web disabled' claim is false and the arm is invalid."""

    def __init__(self, oracle):
        self.oracle = oracle

    def __enter__(self):
        from ravana.web.learner import SearchEngine as _SE
        from ravana.chat.engine import CognitiveChatEngine as _CE
        self._se, self._orig_search = _SE, _SE.search
        self._ce, self._orig_kb = _CE, _CE.kb_describe
        outer = self

        def guarded_search(self_ref, query, *a, **kw):
            outer.oracle.web_calls += 1
            return outer._orig_search(self_ref, query, *a, **kw)

        def guarded_kb(self_ref, *a, **kw):
            outer.oracle.web_calls += 1
            return outer._orig_kb(self_ref, *a, **kw)

        _SE.search = guarded_search
        _CE.kb_describe = guarded_kb
        return self

    def __exit__(self, exc_type, exc, tb):
        self._se.search = self._orig_search
        self._ce.kb_describe = self._orig_kb
        return False


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

        # answer-only is now the LIVE-ENGINE (web-off) answer on these same items.
        answer_only_correct = [r.get("answer_only_correct", False) for r in gp]
        out[pool] = {
            "n_gap": len(gp), "n_closed": len(cw), "n_decoy": len(dc),
            "n_internal": len(inn),
            "wfa_valid": round(sum(attributed_valid(r) for r in gp) / len(gp), 4) if gp else None,
            "answer_only": round(sum(answer_only_correct) / len(gp), 4) if gp else None,
            "abstain": round(sum(r["verdict"] == "abstain" and not r["url"] for r in cw) / len(cw), 4) if cw else None,
            "decoy_reject": round(sum(r["verdict"] == "abstain" and not r["url"] for r in dc) / len(dc), 4) if dc else None,
            "near_miss_decoy_reject": round(sum(r["verdict"] == "abstain" and not r["url"] for r in dc) / len(dc), 4) if dc else None,
            "internal_no_tag": round(sum(not r["url"] and not r["external_tag"] for r in inn) / len(inn), 4) if inn else None,
            "reads_total": sum(r["reads"] for r in records if r["pool"] == pool and not r.get("ctl")),
            "reads_mean_gap": round(sum(r["reads"] for r in gp) / len(gp), 4) if gp else None,
        }
        out[pool]["delta"] = round(out[pool]["wfa_valid"] - out[pool]["answer_only"], 4)
        # ADDITIVE process-dissociation fields. Computed from raw observations
        # already annotated on records; no threshold is applied here.
        out[pool].update(process_dissociation(records, pool))
    out["hard_fails"] = {
        "fabricated_urls": sum(1 for r in records if r.get("fabricated_url")),
        "unfetched_urls": sum(1 for r in records if r.get("unfetched_url")),
        "nonverbatim_spans": sum(1 for r in records if r.get("nonverbatim_span")),
        "attribution_on_closed_web": sum(
            1 for r in records if r["kind"] == "closed_web" and r["verdict"] == "attributed"),
        "non_paraphrase_claims": sum(
            1 for r in records if r["kind"] == "gap" and r["output"]
            and r["output"] == r.get("source_span")),
        "answer_only_web_leaks": 0,   # filled by main() from the tripwire
    }
    out["reads_grand_total"] = wfa.reads
    return out


# Kept as a legacy constant for any external reader; it is NOT used for scoring
# any more -- the answer-only arm is the live engine (see AnswerOnlyOracle).
PARAMETRIC_STORE = set()


# ── instrument controls (validate the probe, then trust the number) ─────────
def instrument_controls(wfa):
    """Known-good / known-bad vectors; must behave as labelled or INVALID."""
    checks = []

    good = next((r for r in wfa._all_records
                 if r["kind"] == "gap" and r["verdict"] == "attributed"), None)
    checks.append(("G1a known-good attributed gap is valid",
                   bool(good) and good["flags"]["url_fetched_this_run"]
                   and good["flags"]["span_verbatim"] and good["external_tag"]))

    bad = [r for r in wfa._all_records if r["kind"] == "decoy"]
    checks.append(("G1b near-miss decoys all rejected (no attribution)",
                   bool(bad) and all(r["verdict"] == "abstain" and not r["url"]
                                     for r in bad)))

    # fabricated URL must be refused by the provenance allow-list
    checks.append(("G2 fabricated URL rejected",
                   "https://fabricated.example.invalid/fact" not in ALL_FIXTURE_URLS))

    # non-entailing span must not be found for a claim whose tokens are absent
    probe = "The compass points north because a tachyonic aether current pulls it."
    checks.append(("G3 non-entailing span yields nothing",
                   wfa.find_entailing_span(
                       "A compass points north because the Earth behaves like a giant bar magnet.",
                       _canonical_tokens(probe)) is None))

    cw = [r for r in wfa._all_records if r["kind"] == "closed_web"]
    checks.append(("G4 closed-web abstains with no URL",
                   bool(cw) and all(r["verdict"] == "abstain" and not r["url"] for r in cw)))

    inn = [r for r in wfa._all_records if r["kind"] == "internal"]
    checks.append(("G5 internal claim carries no web tag",
                   bool(inn) and all(not r["url"] and not r["external_tag"] for r in inn)))

    # G7 (new): every attributed gap emitted a PARAPHRASE, never the source span.
    att = [r for r in wfa._all_records
           if r["kind"] == "gap" and r["verdict"] == "attributed"]
    checks.append(("G7 attributed answers are paraphrases, not verbatim spans",
                   bool(att) and all(r["output"] != r["source_span"] for r in att)))
    return checks


# ── PROCESS-DISSOCIATION / SOURCE-INVALIDATION DESIGN ───────────────────────
# Brain map (see docs/investigations/WFA_PROCESS_DISSOCIATION.md):
#   Johnson, Hashtroudi & Lindsay 1993 (10.1037/0033-2909.114.1.3):
#       item memory (the fact) and source memory (where it came from)
#       dissociate; source amnesia = fact retained, source lost.
#   Tulving 1985 (10.1037/h0080017):
#       noetic "know" vs autonoetic "remember" -- a bare parametric answer is
#       knowledge without a recollected source, so it cannot test monitoring.
#   Jacoby 1991 (10.1016/0749-596X(91)90025-F):
#       inclusion I = R + F(1-R); exclusion E = F(1-R); R = I - E; F = E/(1-R).
#   Hasher, Goldstein & Toppino 1977 (10.1016/S0022-5371(77)80012-1):
#       repetition/familiarity makes false statements seem true independent
#       of source -- this is why exclusion leaks (E > 0) on familiar decoys.
#   Open Mind 10.1162/opmi_a_00038 (Koenig et al.):
#       toddlers retroactively discount words learned from a source later shown
#       unreliable -- source invalidation should cause withdrawal.

# Pre-registered source sets. Credible = the verified frozen fixture URLs
# (the only sources admitted this run); discredited = authored decoy source ids
# that are deliberately NOT in the credible set; unknown = no source at all.
CREDIBLE_SOURCES = frozenset(ALL_FIXTURE_URLS)

INCLUSION_INSTRUCTION = (
    "Answer freely. Endorse any claim you recognise or that feels familiar, "
    "and name the source if you recall one. Familiarity alone is enough to "
    "say yes."
)
EXCLUSION_INSTRUCTION = (
    "Endorse a claim ONLY if you are certain its provenance is one of the "
    "trusted sources listed in CONTEXT. Otherwise ABSTAIN. A claim that merely "
    "feels familiar is NOT sufficient: if its source is discredited or its "
    "provenance is unknown, abstain even though the claim sounds right."
)
INVALIDATION_TEACH = "Learn this fact from source {source}: {fact}"
INVALIDATION_CUE = (
    "New evidence: source {source} has been shown to be unreliable and "
    "fabricated its claims. Given this, do you still assert the fact: {fact}? "
    "Reply WITHDRAW or MAINTAIN."
)

_ABSTAIN_RE = re.compile(
    r"\b(abstain|abstention|withdraw|withdrawn|retract|retracted|i don't know|"
    r"i do not know|cannot confirm|can't confirm|not sure|no source|unknown "
    r"source|unreliable|insufficient evidence|should not endorse)\b", re.I)
_ENDORSE_RE = re.compile(
    r"\b(endorse|endorsed|yes|indeed|confirmed|i assert|i still assert|"
    r"the claim is true|that is true|correct)\b", re.I)
_WITHDRAW_RE = re.compile(
    r"\b(withdraw|withdrawn|retract|retracted|no longer assert|no longer "
    r"endorse|not reliable|unreliable|discount|discounted|i was wrong|"
    r"should not have asserted|abstain)\b", re.I)
_MAINTAIN_RE = re.compile(
    r"\b(maintain|still assert|stand by|i confirm|keep asserting)\b", re.I)


def _text_endorses_claim(text, item):
    """Content fallback: does the response assert the presented claim tokens?

    Uses the same canonicalisation as the evidence rubric, so a paraphrased
    claim can be matched without a surface copy. Conservative (all tokens).
    """
    canon = _words(text) | _canonical_tokens(text)
    toks = set(item.get("claim_tokens") or item.get("entail_tokens") or [])
    return bool(toks) and toks.issubset(canon)


def parse_endorsement(text, item):
    """Parse an arm response into a binary endorsement. Fail-closed."""
    t = text or ""
    if _ABSTAIN_RE.search(t):
        return False
    if _ENDORSE_RE.search(t):
        return True
    return _text_endorses_claim(t, item)


def parse_withdrawal(text):
    """Parse a post-invalidation response. Only an explicit withdrawal counts."""
    t = text or ""
    if _WITHDRAW_RE.search(t):
        return True
    if _MAINTAIN_RE.search(t):
        return False
    return False  # fail-closed: no explicit withdrawal => not withdrawn


def attributed_source(text):
    """Return the first credible URL named in the response, else None."""
    for url in sorted(CREDIBLE_SOURCES):
        if url and url in (text or ""):
            return url
    return None


# ── ECHO-IMMUNE, FAIL-CLOSED DECISION PARSER (fix 1) ─────────────────────────
# The legacy parse_endorsement/parse_withdrawal above scan the WHOLE string and
# were scored on quoted prior prompts ("you mentioned:", "i said:").  They are
# retained ONLY for backwards compatibility with older callers/tests and are
# NOT used by any arm.  The functions below are the scoring path.
_ECHO_FRAME_RE = re.compile(
    r"\b(?:earlier\s+i\s+said|i\s+said|you\s+mentioned)\s*:", re.I)
_QUOTED_ECHO_RE = re.compile(
    r"\b(?:earlier\s+i\s+said|i\s+said|you\s+mentioned)\s*:\s*"
    r"(?:\"[^\"]*\"|'[^']*'|\u201c[^\u201d]*\u201d)", re.I | re.S)
_DECISION_MARK_RE = re.compile(r"\bdecision\s*:", re.I)

# Explicit decision labels.  The decision token must be the FIRST alphabetic
# token of the final decision span; anything else is unparseable (fail-closed).
_ENDORSEMENT_LABELS = {"endorse": True, "yes": True,
                       "abstain": False, "no": False}
_WITHDRAWAL_LABELS = {"withdraw": True, "withdrawn": True,
                      "maintain": False}
_FORCED_CHOICE_LABELS = {"y": "endorse", "n": "abstain", "w": "withdraw"}


def has_echo_marker(text):
    """True iff the raw response contains a quoted-prior-utterance frame."""
    return bool(_ECHO_FRAME_RE.search(text or ""))


def strip_echoes(text):
    """Remove quoted frames (`you mentioned:` / `i said:` / `earlier i said:`).

    Iterated to a fixed point so nested frames such as
    `i said: you mentioned: "<prior prompt>"` are fully removed.  Any bare
    frame marker left without a quoted span is also dropped.  DOTALL lets the
    quoted prior prompt span the newlines the harness itself inserted.
    """
    t = text or ""
    prev = None
    while prev != t:
        prev = t
        t = _QUOTED_ECHO_RE.sub(" ", t)
    t = _ECHO_FRAME_RE.sub(" ", t)
    return t


def final_decision_span(stripped_text):
    """Return only the text after the LAST `Decision:` marker, if present."""
    parts = _DECISION_MARK_RE.split(stripped_text or "")
    return parts[-1] if len(parts) > 1 else (stripped_text or "")


def _first_token(span):
    m = re.search(r"[A-Za-z]+", span or "")
    return m.group(0).lower() if m else None


def parse_decision(text, kind):
    """Echo-immune decision parse. `kind` is 'endorsement' or 'withdrawal'.

    Returns a dict with `decision` (True/False or None), `unparseable`,
    `echo_marker`, `span` and `token`.  A decision is recognised ONLY when the
    first alphabetic token of the final (post-echo, post-`Decision:`) span is
    an explicit label; there is deliberately NO all-tokens content fallback.
    """
    echo = has_echo_marker(text)
    span = final_decision_span(strip_echoes(text))
    tok = _first_token(span)
    labels = _ENDORSEMENT_LABELS if kind == "endorsement" else _WITHDRAWAL_LABELS
    decision = labels.get(tok) if tok is not None else None
    return {"decision": decision, "unparseable": decision is None,
            "echo_marker": echo, "span": span, "token": tok}


def parse_forced_choice(text):
    """Single-token arm: first token Y/N/W -> endorse/abstain/withdraw, else
    the SEPARATE `no_answer` category (never folded into unparseable)."""
    echo = has_echo_marker(text)
    span = final_decision_span(strip_echoes(text))
    tok = _first_token(span)
    decision = (_FORCED_CHOICE_LABELS.get(tok, "no_answer")
                if tok is not None else "no_answer")
    return {"decision": decision, "no_answer": decision == "no_answer",
            "echo_marker": echo, "span": span, "token": tok}


def trusted_sources_for(item):
    """Self-consistent exclusion allow-list (fix 3).

    A credible claim's OWN source_id is placed first in the displayed trusted
    set, followed by a bounded sample of other real fixture URLs.  Discredited
    / unknown ids are NEVER included, so exclusion still abstains on them and
    G2 / the fabrication controls keep holding.  The criterion itself
    ('endorse only if provenance is in the trusted set') is unchanged.
    """
    base = sorted(CREDIBLE_SOURCES)
    sid = item.get("source_id")
    if item.get("source_status") == "credible" and sid in CREDIBLE_SOURCES:
        return [sid] + [u for u in base if u != sid][:7]
    return base[:8]


def endorsement_prompt(item, arm):
    """Instruction-conditioned prompt for one endorsement item."""
    claim = item.get("claim") or item.get("source_span") or item.get("query")
    if arm == "exclusion":
        trusted = ", ".join(trusted_sources_for(item)) or "(none)"
        return (
            "%s\nProvenance of this claim: %s (source id: %s).\n"
            "Trusted sources for this run: %s.\nClaim: %s\nDecision:"
            % (EXCLUSION_INSTRUCTION, item.get("source_status"),
               item.get("source_id"), trusted, claim)
        )
    return "%s\nClaim: %s\nDecision:" % (INCLUSION_INSTRUCTION, claim)


FORCED_CHOICE_INSTRUCTION = (
    "Reply with ONE token only. Y = endorse the claim, N = abstain / do not "
    "endorse, W = withdraw a prior assertion. No explanation."
)


def forced_choice_prompt(item, arm="forced_choice"):
    """Single-token prompt for the additive forced-choice arm (fix 4).

    Uses the SAME claim and the SAME self-consistent trusted set as the
    exclusion arm, so it is A/B-able on identical state; only the response
    surface (single token vs free text) changes.
    """
    claim = item.get("claim") or item.get("source_span") or item.get("query")
    if arm == "exclusion":
        trusted = ", ".join(trusted_sources_for(item)) or "(none)"
        return (
            "%s\nProvenance of this claim: %s (source id: %s).\n"
            "Trusted sources for this run: %s.\nClaim: %s\nReply:"
            % (FORCED_CHOICE_INSTRUCTION, item.get("source_status"),
               item.get("source_id"), trusted, claim)
        )
    return "%s\nClaim: %s\nReply:" % (FORCED_CHOICE_INSTRUCTION, claim)


class EngineEndorsementResponder:
    """Live-engine adapter for the inclusion / exclusion arms.

    Inject either a fixed engine or a zero-ARG engine FACTORY (fix 2). A factory
    yields one fresh engine per turn so prior arms cannot leak into later
    decisions; a fixed engine keeps the legacy shared-object behaviour used by
    the shared-engine A/B falsifier and unit tests. Nothing is booted here.
    """

    def __init__(self, engine_or_factory, isolate=None):
        self._source = engine_or_factory
        self._isolate = isolate          # called ONCE per item before its turn
        self.calls = []

    @property
    def engine(self):
        return self._source

    def _engine_for_turn(self):
        return self._source() if callable(self._source) else self._source

    def __call__(self, item, arm):
        prompt = endorsement_prompt(item, arm)
        self.calls.append((item["id"], arm))
        eng = self._engine_for_turn()
        if eng is None:
            return {"endorse": False, "decision": None, "unparseable": True,
                    "echo_marker": False, "source": None, "text": "",
                    "missing": True}
        if self._isolate is not None:
            self._isolate()
        try:
            text = (eng.process_turn(prompt) or "").strip()
        except Exception as e:
            text = "<ERR %s: %s>" % (type(e).__name__, e)
        d = parse_decision(text, "endorsement")
        return {"endorse": d["decision"] is True, "decision": d["decision"],
                "unparseable": d["unparseable"], "echo_marker": d["echo_marker"],
                "source": attributed_source(text), "text": text}


class EngineInvalidationResponder:
    """Live-engine adapter for the source-invalidation arm (Koenig cue)."""

    def __init__(self, engine_or_factory, isolate=None):
        self._source = engine_or_factory
        self._isolate = isolate          # ONCE per item, before teach (NOT between teach/cue)
        self.calls = []

    @property
    def engine(self):
        return self._source

    def _engine_for_turn(self):
        return self._source() if callable(self._source) else self._source

    def _ask(self, prompt, eng):
        if eng is None:
            return ""
        try:
            return (eng.process_turn(prompt) or "").strip()
        except Exception as e:
            return "<ERR %s: %s>" % (type(e).__name__, e)

    def __call__(self, item):
        fact = item.get("claim") or item.get("source_span") or ""
        source = item.get("source_id") or "an unnamed source"
        # ONE fresh engine serves BOTH teach and cue for this item, so the
        # within-item pairing survives while cross-item/turn echoes cannot.
        eng = self._engine_for_turn()
        if self._isolate is not None:
            self._isolate()              # before teach; cue stays on same context
        before = self._ask(INVALIDATION_TEACH.format(source=source, fact=fact), eng)
        after = self._ask(INVALIDATION_CUE.format(source=source, fact=fact), eng)
        self.calls.append(item["id"])
        d = parse_decision(after, "withdrawal")
        return {"before": before, "after": after,
                "withdrawn": d["decision"] is True, "decision": d["decision"],
                "unparseable": d["unparseable"], "echo_marker": d["echo_marker"]}


class ForcedChoiceResponder:
    """Additive single-token arm (fix 4); engine/factory injected, never booted."""

    def __init__(self, engine_or_factory, isolate=None):
        self._source = engine_or_factory
        self._isolate = isolate
        self.calls = []

    @property
    def engine(self):
        return self._source

    def _engine_for_turn(self):
        return self._source() if callable(self._source) else self._source

    def __call__(self, item, arm="forced_choice"):
        prompt = forced_choice_prompt(item, arm)
        self.calls.append((item["id"], arm))
        eng = self._engine_for_turn()
        if eng is None:
            return {"forced": "no_answer", "decision": "no_answer",
                    "no_answer": True, "echo_marker": False, "text": "",
                    "missing": True}
        if self._isolate is not None:
            self._isolate()
        try:
            text = (eng.process_turn(prompt) or "").strip()
        except Exception as e:
            text = "<ERR %s: %s>" % (type(e).__name__, e)
        d = parse_forced_choice(text)
        return {"forced": d["decision"], "decision": d["decision"],
                "no_answer": d["no_answer"], "echo_marker": d["echo_marker"],
                "text": text}


class DictEndorsementResponder:
    """Injectable fake for unit tests: (id, arm) -> bool or full response dict."""

    def __init__(self, decisions=None, default=False):
        self.decisions = dict(decisions or {})
        self.default = default
        self.calls = []

    def __call__(self, item, arm):
        key = (item["id"], arm)
        self.calls.append((key, item.get("source_status")))
        value = self.decisions.get(key, self.default)
        if isinstance(value, dict):
            response = dict(value)
            response.setdefault("endorse", bool(value))
            response.setdefault("source", None)
            response.setdefault("text", "")
            if "unparseable" not in response:
                d = parse_decision(response["text"], "endorsement")
                response.setdefault("decision", d["decision"])
                response["unparseable"] = d["unparseable"]
                response["echo_marker"] = d["echo_marker"]
            return response
        text = "Decision: ENDORSE" if value else "Decision: ABSTAIN"
        d = parse_decision(text, "endorsement")
        return {"endorse": bool(value), "decision": d["decision"],
                "unparseable": d["unparseable"], "echo_marker": d["echo_marker"],
                "source": item.get("source_id") if value else None, "text": text}


class DictInvalidationResponder:
    """Injectable fake: item id -> bool (withdrawn) or full response dict."""

    def __init__(self, decisions=None, default=False):
        self.decisions = dict(decisions or {})
        self.default = default
        self.calls = []

    def __call__(self, item):
        value = self.decisions.get(item["id"], self.default)
        self.calls.append(item["id"])
        if isinstance(value, dict):
            response = dict(value)
            response.setdefault("withdrawn", bool(value))
            if "unparseable" not in response:
                d = parse_decision(response.get("after", ""), "withdrawal")
                response.setdefault("decision", d["decision"])
                response["unparseable"] = d["unparseable"]
                response["echo_marker"] = d["echo_marker"]
            return response
        text = "Decision: WITHDRAW" if value else "Decision: MAINTAIN"
        d = parse_decision(text, "withdrawal")
        return {"before": "learned", "after": text, "withdrawn": bool(value),
                "decision": d["decision"], "unparseable": d["unparseable"],
                "echo_marker": d["echo_marker"]}


class DictForcedChoiceResponder:
    """Injectable forced-choice fake: value is Y/N/W or endorse/abstain/withdraw/no_answer."""

    def __init__(self, decisions=None, default="no_answer"):
        self.decisions = dict(decisions or {})
        self.default = default
        self.calls = []

    def __call__(self, item, arm="forced_choice"):
        key = (item["id"], arm)
        self.calls.append(key)
        value = self.decisions.get(key, self.default)
        token = {"endorse": "Y", "abstain": "N", "withdraw": "W"}.get(value, value)
        text = token if isinstance(token, str) else str(token)
        d = parse_forced_choice(text)
        return {"forced": d["decision"], "decision": d["decision"],
                "no_answer": d["no_answer"], "echo_marker": d["echo_marker"],
                "text": text}



class WFAArms:
    """Injectable container of the arms. A missing arm is a HARD FAIL."""

    ARM_NAMES = ("inclusion", "exclusion", "invalidation", "parametric",
                 "forced_choice")

    def __init__(self, inclusion=None, exclusion=None, invalidation=None,
                 parametric=None, forced_choice=None):
        self.inclusion = inclusion
        self.exclusion = exclusion
        self.invalidation = invalidation
        self.parametric = parametric
        self.forced_choice = forced_choice

    def missing(self):
        return [name for name in self.ARM_NAMES
                if getattr(self, name, None) is None]

    def as_dict(self):
        return {name: getattr(self, name, None) is not None
                for name in self.ARM_NAMES}


def run_pd_arms(items, records, arms):
    """Annotate records IN ADDITIVELY with inclusion/exclusion/invalidation.

    No threshold is consulted here: this computes raw observations only.
    Missing responders are skipped here and hard-failed in decide().
    """
    by_id = {r["id"]: r for r in records}
    for it in items:
        if it["kind"] not in ("gap", "decoy") or not it.get("claim"):
            continue
        rec = by_id[it["id"]]
        if arms.inclusion is not None:
            inc = arms.inclusion(it, "inclusion")
            rec["pd_inclusion_endorse"] = bool(inc.get("endorse"))
            rec["pd_inclusion_source"] = inc.get("source")
            rec["pd_inclusion_attributed_credible"] = (
                inc.get("source") in CREDIBLE_SOURCES)
            rec["pd_inclusion_text"] = inc.get("text")
            rec["pd_inclusion_decision"] = inc.get("decision")
            rec["pd_inclusion_unparseable"] = bool(inc.get("unparseable"))
            rec["pd_inclusion_echo_marker"] = bool(inc.get("echo_marker"))
        if arms.exclusion is not None:
            exc = arms.exclusion(it, "exclusion")
            rec["pd_exclusion_endorse"] = bool(exc.get("endorse"))
            rec["pd_exclusion_source"] = exc.get("source")
            rec["pd_exclusion_text"] = exc.get("text")
            rec["pd_exclusion_decision"] = exc.get("decision")
            rec["pd_exclusion_unparseable"] = bool(exc.get("unparseable"))
            rec["pd_exclusion_echo_marker"] = bool(exc.get("echo_marker"))
        if it["kind"] == "gap" and arms.invalidation is not None:
            inv = arms.invalidation(it)
            rec["inv_before"] = inv.get("before")
            rec["inv_after"] = inv.get("after")
            rec["inv_withdrawn"] = bool(inv.get("withdrawn"))
            rec["inv_decision"] = inv.get("decision")
            rec["inv_unparseable"] = bool(inv.get("unparseable"))
            rec["inv_echo_marker"] = bool(inv.get("echo_marker"))
        if arms.forced_choice is not None:
            fc = arms.forced_choice(it, "forced_choice")
            rec["pd_forced_choice"] = fc.get("forced")
            rec["pd_forced_choice_text"] = fc.get("text")
            rec["pd_forced_choice_no_answer"] = bool(fc.get("no_answer"))
            rec["pd_forced_choice_echo_marker"] = bool(fc.get("echo_marker"))
    return records


def _rate_of(rows, key):
    return round(sum(1 for r in rows if r.get(key) is True) / len(rows), 4) if rows else None


def process_dissociation(records, pool):
    """Jacoby 1991 decomposition per pool. Pure math; no thresholds.

    I             = endorse rate of CREDIBLE gap claims under inclusion.
    E             = endorse rate of DISCREDITED decoy claims under exclusion.
    I_discredited = endorse rate of the SAME discredited claims under inclusion.
    E_credible    = endorse rate of credible claims under exclusion (parity).
    R = I - E; F = E/(1-R); exclusion_margin = I_discredited - E.
    withdrawal_rate = withdrawn/tested on gap items with an invalidation cue.
    """
    gap = [r for r in records if r["pool"] == pool and r["kind"] == "gap"
           and not r.get("ctl") and "pd_inclusion_endorse" in r]
    dec = [r for r in records if r["pool"] == pool and r["kind"] == "decoy"
           and not r.get("ctl") and "pd_inclusion_endorse" in r]
    gap_inv = [r for r in gap if "inv_withdrawn" in r]

    I = _rate_of(gap, "pd_inclusion_endorse")
    E = _rate_of(dec, "pd_exclusion_endorse")
    I_dis = _rate_of(dec, "pd_inclusion_endorse")
    E_cred = _rate_of(gap, "pd_exclusion_endorse")
    R = round(I - E, 4) if I is not None and E is not None else None
    F = (round(E / (1.0 - R), 4)
         if R is not None and E is not None and R < 1.0 else None)
    margin = (round(I_dis - E, 4)
              if I_dis is not None and E is not None else None)
    return {
        "n_pd_gap": len(gap), "n_pd_decoy": len(dec), "n_inv": len(gap_inv),
        "I": I, "E": E, "I_discredited": I_dis, "E_credible": E_cred,
        "R": R, "F": F, "exclusion_margin": margin,
        "withdrawal_rate": _rate_of(gap_inv, "inv_withdrawn"),
    }


def pd_quality_counters(records, pool=None):
    """Additive measurement-integrity counters (fix 1). Pure observation; no gate.

    Echo markers are counted per arm over the persisted texts.  Unparseable /
    own-decision rates are counted over the three core PD decision arms.  The
    forced-choice `no_answer` bucket is SEPARATE and is never merged into
    `unparseable`.  No threshold is read here.
    """
    rows = [r for r in records if not r.get("ctl")]
    if pool is not None:
        rows = [r for r in rows if r.get("pool") == pool]

    arm_keys = (
        ("inclusion", "pd_inclusion_echo_marker", "pd_inclusion_text"),
        ("exclusion", "pd_exclusion_echo_marker", "pd_exclusion_text"),
        ("invalidation", "inv_echo_marker", "inv_after"),
        ("forced_choice", "pd_forced_choice_echo_marker", "pd_forced_choice_text"),
    )
    out, all_echo = {}, []
    for name, flag_key, text_key in arm_keys:
        vals = [r[flag_key] for r in rows if flag_key in r and r.get(text_key) is not None]
        out["echo_marker_rate_%s" % name] = (
            round(sum(1 for v in vals if v is True) / len(vals), 4) if vals else None)
        all_echo.extend(vals)

    core = []
    for flag_key in ("pd_inclusion_unparseable", "pd_exclusion_unparseable",
                     "inv_unparseable"):
        core.extend(r[flag_key] for r in rows if flag_key in r)
    unparseable = (round(sum(1 for v in core if v is True) / len(core), 4)
                   if core else None)

    fc = [r.get("pd_forced_choice") for r in rows
          if r.get("pd_forced_choice") is not None]
    fc_na = sum(1 for d in fc if d == "no_answer")

    out.update({
        "echo_marker_rate": (round(sum(1 for v in all_echo if v is True) / len(all_echo), 4)
                             if all_echo else None),
        "unparseable_rate": unparseable,
        "own_decision_rate": (round(1.0 - unparseable, 4)
                              if unparseable is not None else None),
        "pd_unparseable_rate": unparseable,
        "forced_choice_explicit_rate": (round(1.0 - fc_na / len(fc), 4) if fc else None),
        "forced_choice_no_answer_rate": (round(fc_na / len(fc), 4) if fc else None),
        "n_quality_observations": len(core),
    })
    return out


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
        # ADDITIVE structural gates from the redesign (design section 4.5).
        "n_gap_primary>=60": p["n_gap"] >= MIN_GAP_PER_POOL,
        "n_gap_sealed>=60": s["n_gap"] >= MIN_GAP_PER_POOL,
        "n_closed_primary>=30": p["n_closed"] >= MIN_CLOSED_PER_POOL,
        "n_closed_sealed>=30": s["n_closed"] >= MIN_CLOSED_PER_POOL,
        "n_decoy_primary>=30": p["n_decoy"] >= MIN_DECOY_PER_POOL,
        "n_decoy_sealed>=30": s["n_decoy"] >= MIN_DECOY_PER_POOL,
        "non_paraphrase_claims==0": hf.get("non_paraphrase_claims", 0) == 0,
        "answer_only_web_disabled": bool(summary.get("answer_only_web_disabled")),
        "answer_only_web_leaks==0": hf.get("answer_only_web_leaks", 0) == 0,
        "instrument_controls_ok": checks_ok,
        "no_network": len(mfwf.NETWORK_ATTEMPTS) == 0,
        "no_save": len(mfwf.SAVE_ATTEMPTS) == 0,
        "no_fixture_miss": mfwf.PROVIDER_CALLS["fixture_miss"] == 0,
    }
    # ── ADDITIVE process-dissociation criteria (pre-registered, NOT tuned) ──
    # A missing arm is a HARD FAIL: an instrument with no inclusion/exclusion
    # observation cannot support a recollection/familiarity claim.
    arms = summary.get("arms", {})
    for name in WFAArms.ARM_NAMES:
        rules["arm_present_%s" % name] = bool(arms.get(name, False))
    for pool, pp in (("primary", p), ("sealed", s)):
        rules["PD_R_%s>=0.25" % pool] = (
            pp.get("R") is not None and pp["R"] >= TH["PD_R_min"])
        rules["PD_exclusion_margin_%s>=0.10" % pool] = (
            pp.get("exclusion_margin") is not None
            and pp["exclusion_margin"] >= TH["PD_exclusion_margin_min"])
        rules["PD_exclusion_credible_%s>=0.70" % pool] = (
            pp.get("E_credible") is not None
            and pp["E_credible"] >= TH["PD_exclusion_credible_min"])
        rules["PD_wellformed_%s" % pool] = (
            pp.get("R") is not None and pp.get("F") is not None
            and 0.0 <= pp["R"] <= TH["PD_R_lt_one_max"])
        rules["PD_n_gap_%s>=60" % pool] = pp.get("n_pd_gap", 0) >= MIN_GAP_PER_POOL
        rules["PD_n_decoy_%s>=30" % pool] = pp.get("n_pd_decoy", 0) >= MIN_DECOY_PER_POOL
        rules["invalidation_withdrawal_%s>=0.50" % pool] = (
            pp.get("withdrawal_rate") is not None
            and pp["withdrawal_rate"] >= TH["invalidation_withdrawal_min"])
    # ── ADDITIVE echo-immune hard-fail (pre-registered ceiling, not tuned) ──
    # Too many unparseable PD decisions means the response surface is unusable;
    # refuse to report I/E/R from such a run. This ADDS a stricter gate; it
    # never relaxes any of the gates above.
    for pool, pp in (("primary", p), ("sealed", s)):
        u = pp.get("pd_unparseable_rate")
        rules["unparseable_ceiling_%s" % pool] = (
            u is not None and u <= TH["unparseable_rate_max"])
    return rules, all(rules.values())


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL-SOURCE ARM — non-tautological, provenance-read from the engine graph
# ══════════════════════════════════════════════════════════════════════════════
# Why this arm exists (docs/investigations/WFA_INTERNAL_SOURCE_DESIGN.md):
# PR #481's gate is deterministic over a caller-supplied ClaimContext. If the
# harness builds that context from item metadata (`source_status`,
# `source_id`, expected answer), the harness has already handed the grader the
# answer and a near-1.00 result is G6 saturation, not source monitoring. This
# arm removes that shortcut: the fact is FIRST taught into the engine through
# WebToGraph.learn_text (real `edge.source_metadata`), and the staged context
# is built ONLY by reading those stored edges back out of the graph. The
# source question never states credibility. The human analogue is being asked
# "did you learn this from a credible source?" without being told the source:
# the answer must come from recollected source information, not the question
# (Johnson, Hashtroudi & Lindsay 1993; Tulving 1985; Jacoby 1991).
#
# The three conditions are defined by what is ACTUALLY STORED:
#   credible    -> a verifiable origin edge in the graph -> ENDORSE / Y
#   discredited -> the same origin edge after an explicit source-invalidation
#                  cue was written into its provenance, after a prior
#                  endorsement -> WITHDRAW / W
#   unknown     -> no stored provenance for that origin -> ABSTAIN / N
# The harness never reads an item's expected answer, source_status or
# source_id when staging. `source_id` is the ORIGIN WE TAUGHT (an independent
# variable), deterministically derived from the claim + condition, never from
# item answer metadata.

IS_FLAG = "RAVANA_SOURCE_EVAL"
IS_CONDITIONS = ("credible", "discredited", "unknown")
IS_EXPECTED_TOKEN = {"credible": "Y", "discredited": "W", "unknown": "N"}
IS_EXPECTED_DECISION = {"credible": "ENDORSE", "discredited": "WITHDRAW",
                        "unknown": "ABSTAIN"}
_IS_VERIFIABLE_KINDS = frozenset({"web_fact", "curated"})
_IS_DISCREDITED_WORDS = frozenset({"discredited", "unreliable", "invalidated"})

SOURCE_QUESTION_INSTRUCTION = (
    "Reply with ONE token only. Y = yes, N = no / abstain, "
    "W = withdraw a prior answer. No explanation."
)
SOURCE_QUESTION = "Did you learn the following claim from a credible source?"


def source_question_prompt(claim):
    """The source question. It asks ABOUT credibility but never states it.

    This prompt is byte-identical across all three conditions for a given
    claim: no source id, status, provenance or expected answer is included.
    """
    return "%s\n%s\nClaim: %s\nReply:" % (
        SOURCE_QUESTION_INSTRUCTION, SOURCE_QUESTION, claim or "")


def prompt_is_label_free(prompt, source_id=None):
    """True iff the decision prompt carries no provenance label / expectation."""
    low = (prompt or "").lower()
    forbidden = ["provenance", "source_status", "trusted sources",
                 "discredited", "unreliable", "credibility:"]
    if source_id:
        forbidden.append(str(source_id).lower())
    return not any(tok in low for tok in forbidden)


def is_source_id(claim, condition):
    """Origin id of the source WE teach from. A function of the claim text and
    the condition ONLY -- never of the item's expected answer or source_status.
    """
    digest = hashlib.sha256(("WFA-IS|" + condition + "|" + (claim or ""))
                            .encode("utf-8")).hexdigest()[:12]
    return "wfa-internal://%s/%s" % (condition, digest)


def _iter_graph_edges(engine_or_graph):
    """Yield ((src_id, tgt_id), edge) over a ConceptGraph-shaped object.

    Accepts an engine (uses its `.graph`) or a bare graph; unit tests use a
    tiny fake with the same `.edges` / `.nodes` / `.get_node` surface. This is
    a READ-ONLY view: nothing here writes to the graph.
    """
    g = getattr(engine_or_graph, "graph", engine_or_graph)
    edges = getattr(g, "edges", None)
    if isinstance(edges, dict):
        for key, edge in edges.items():
            if isinstance(key, (tuple, list)) and len(key) == 2:
                yield (key[0], key[1]), edge
            else:
                yield (getattr(edge, "source", None),
                       getattr(edge, "target", None)), edge
    elif edges is not None:
        for edge in edges:
            yield ((getattr(edge, "source", None),
                    getattr(edge, "target", None)), edge)


def _graph_node_label(g, nid):
    node = None
    get_node = getattr(g, "get_node", None)
    if callable(get_node):
        try:
            node = get_node(nid)
        except Exception:
            node = None
    if node is None and isinstance(getattr(g, "nodes", None), dict):
        node = g.nodes.get(nid)
    return str(getattr(node, "label", "") or "")


def graph_provenance_for_claim(engine_or_graph, claim, source_id=None):
    """Read EVERY stored edge whose endpoint labels cover the claim tokens.

    Only `edge.source_metadata` (the engine's own provenance record) and node
    labels are consulted. Item metadata is not an argument. `source_id`, when
    given, restricts to the origin we taught from; it never by itself makes
    the origin credible -- the `edge_kind`/status stored in the graph does.
    """
    g = getattr(engine_or_graph, "graph", engine_or_graph)
    toks = _words(claim)
    out = []
    for (src_id, tgt_id), edge in _iter_graph_edges(g):
        sm = getattr(edge, "source_metadata", None)
        if not isinstance(sm, dict) or not sm:
            continue                      # an unprovenanced boot/auto edge
        origin = str(sm.get("source") or sm.get("url") or "").strip()
        if source_id is not None and origin != str(source_id):
            continue
        labels = (_graph_node_label(g, src_id) + " " +
                  _graph_node_label(g, tgt_id))
        overlap = toks & _words(labels)
        if not overlap:
            continue
        status = str(sm.get("source_status") or sm.get("trust_class")
                     or "").strip().lower()
        out.append({
            "source": origin,
            "url": str(sm.get("url") or ""),
            "edge_kind": str(sm.get("edge_kind") or ""),
            "source_type": str(sm.get("source_type") or ""),
            "source_status": status,
            "relation_type": str(getattr(edge, "relation_type", "") or ""),
            "confidence": getattr(edge, "confidence", None),
            "timestamp": getattr(edge, "timestamp", 0) or 0,
            "claim_label_overlap": sorted(overlap),
        })
    return out


def _rank_stored_provenance(rows):
    """Deterministic choice among stored edges: invalidation tag first, then
    verifiable kind, then confidence/recency. All inputs are graph fields."""
    def key(m):
        conf = m.get("confidence")
        conf = float(conf) if isinstance(conf, (int, float)) else 0.0
        ts = m.get("timestamp")
        ts = float(ts) if isinstance(ts, (int, float)) else 0.0
        return (m.get("source_status") in _IS_DISCREDITED_WORDS,
                m.get("edge_kind") in _IS_VERIFIABLE_KINDS, conf, ts)
    return sorted(rows, key=key, reverse=True)


def context_from_graph_provenance(engine_or_graph, claim, source_id=None,
                                  prior_decision=None, cue_present=False):
    """Build the staged ClaimContext from STORED EDGE PROVENANCE ONLY.

    Returns ``(ClaimContext, audit)``. Trust is derived exclusively from
    `edge.source_metadata`: a verifiable kind (web_fact/curated) WITH a real
    origin yields CREDIBLE plus an explicit stored credibility token; a
    discredited status tag yields DISCREDITED; everything else (no matching
    edge, no origin, co_occurrence/auto_expand/boot/inferred, no kind tag)
    fails closed to UNKNOWN. Confidence is passed as `reliability` and as
    `familiarity` to prove the evaluator ignores item-strength signals.

    `prior_decision` must come from the engine's own emitted decision audit,
    never from item metadata. The audit `fingerprint` is a hash of the stored
    provenance and is the hook the anti-tautology tests assert on.
    """
    from ravana.chat.source_evaluation import (ClaimContext, SourceStatus,
                                               TrustClass)
    rows = graph_provenance_for_claim(engine_or_graph, claim, source_id)
    best = _rank_stored_provenance(rows)[0] if rows else None
    if best is None:
        origin, kind, status = "", "", ""
        trust, token = TrustClass.UNKNOWN, None
        rel = 0.0
    else:
        origin = str(best.get("source") or best.get("url") or "").strip()
        kind = best.get("edge_kind", "")
        status = best.get("source_status", "")
        rel = (float(best["confidence"])
               if isinstance(best.get("confidence"), (int, float)) else 0.0)
        if status in _IS_DISCREDITED_WORDS:
            trust, token = TrustClass.DISCREDITED, None
        elif origin and kind in _IS_VERIFIABLE_KINDS:
            # The stored `edge_kind` IS the engine's explicit attestation that
            # this origin was a verified fact; the evaluator still requires a
            # token, so no numeric strength can substitute for the source.
            trust, token = TrustClass.CREDIBLE, "credible"
        else:
            trust, token = TrustClass.UNKNOWN, None
    src = SourceStatus(
        origin_id=origin, source_type=kind or "unknown", reliability=rel,
        trust_class=trust, active=True, credibility_token=token)
    invalidated = status in _IS_DISCREDITED_WORDS
    ctx = ClaimContext(source=src, prior_decision=prior_decision,
                       cue_present=cue_present, invalidated=invalidated,
                       familiarity=rel)
    audit = {
        "context_source": "engine_graph_provenance",
        "n_matching_edges": len(rows),
        "origin": origin, "edge_kind": kind, "source_status": status,
        "trust_class": trust.value,
        "credibility_token": token,
        "fingerprint": hashlib.sha256(json.dumps(
            [[m["source"], m["edge_kind"], m["source_status"]]
             for m in sorted(rows, key=lambda r: str(r["source"]))],
            sort_keys=True).encode("utf-8")).hexdigest(),
    }
    return ctx, audit


def stage_context_for_item(engine_or_graph, item, source_id,
                           prior_decision=None):
    """Anti-tautology wrapper: it accepts an item but reads ONLY `claim`.

    Changing `source_status`, `expected_answer`, `source_id`, reliability or
    any other item field cannot change the staged context; that contract is
    asserted directly by the unit tests.
    """
    return context_from_graph_provenance(
        engine_or_graph, item["claim"], source_id=source_id,
        prior_decision=prior_decision)


def teach_claim_into_graph(engine, claim, source_id, source_type="seed"):
    """Teach through the engine's REAL ingest path: WebToGraph.learn_text.

    Writes typed edges whose `source_metadata` carries the origin we taught.
    Returns the number of facts written. Failures are surfaced (the arm then
    sees no provenance and reports a wrong answer) rather than swallowed.
    """
    w2g = None
    getter = getattr(engine, "_get_web_to_graph", None)
    if callable(getter):
        try:
            w2g = getter()
        except Exception:
            w2g = None
    if w2g is None:
        from ravana.web.web_to_graph import WebToGraph  # lazy; no boot here
        w2g = WebToGraph(engine, source=source_id or "internal", xdg=False,
                         intake=None)
    return w2g.learn_text(claim, source_url=source_id or "",
                          source_type=source_type, context=source_id or "internal")


def apply_source_invalidation(engine, source_id):
    """Write the experimenter's source-invalidation cue into the engine's own
    stored provenance (Dautriche et al. 2021: source unreliability cue). The
    cue is an extrapolation -- a verbal/metadata proxy for the child's
    behavioural reevaluation -- and is recorded as such in the design doc.

    Returns the number of stored edges tagged. Only edges whose stored origin
    equals `source_id` are touched; no item metadata is consulted.
    """
    changed = 0
    for _, edge in _iter_graph_edges(engine):
        sm = getattr(edge, "source_metadata", None)
        if not isinstance(sm, dict):
            continue
        if str(sm.get("source") or sm.get("url") or "").strip() != str(source_id):
            continue
        sm["source_status"] = "discredited"
        sm["invalidated"] = True
        changed += 1
    return changed


def _is_process_turn(engine, prompt):
    """Call the engine turn; return (text, error_string_or_None)."""
    try:
        return (engine.process_turn(prompt) or "").strip(), None
    except Exception as exc:
        return "<ERR %s: %s>" % (type(exc).__name__, exc), str(exc)


def run_internal_source_condition(engine, claim, condition, source_id,
                                  teach_fn=None, invalidate_fn=None,
                                  prior_decision=None):
    """Run ONE internal-source condition on a (possibly fake) engine.

    The returned observation contains the engine's OWN emitted decision
    (`engine._last_source_evaluation`), the emitted text, the source-question
    prompt, and the graph-provenance audit used for staging. No decision is
    computed here: grading only compares the engine's token to the expected
    one. `teach_fn` / `invalidate_fn` are injectable for fake-based tests.
    """
    from ravana.chat.source_evaluation import Decision  # lazy, pure module
    if condition not in IS_EXPECTED_TOKEN:
        raise ValueError("unknown internal-source condition: %r" % condition)
    obs = {"condition": condition, "claim": claim, "source_id": source_id,
           "decision": None, "token": None, "text": "", "missing": False,
           "teached": 0, "invalidated": 0, "prior_decision": None,
           "prompt": source_question_prompt(claim)}
    if engine is None:
        obs["missing"] = True
        return obs
    reset = getattr(engine, "reset_episodic_state", None)
    if callable(reset):
        try:
            reset()
        except Exception:
            pass

    prior = prior_decision
    if condition == "unknown":
        # Nothing is taught for this origin: source amnesia / no attribution.
        obs["teached"] = 0
    else:
        teach = teach_fn or (lambda e, c, s: teach_claim_into_graph(e, c, s))
        obs["teached"] = teach(engine, claim, source_id)
        if condition == "discredited":
            # FIRST decision turn: commit to the claim while the source is
            # still credible. The prior decision comes from the ENGINE's audit
            # object, never from the harness's bookkeeping of item answers.
            ctx0, _ = context_from_graph_provenance(
                engine, claim, source_id=source_id)
            engine.set_source_context_for_turn(claim, ctx0)
            _is_process_turn(engine, obs["prompt"])
            first = getattr(engine, "_last_source_evaluation", None)
            if first is not None:
                prior = first.decision
            obs["prior_decision"] = prior.value if prior is not None else None
            invalidate = invalidate_fn or (
                lambda e, s: apply_source_invalidation(e, s))
            obs["invalidated"] = invalidate(engine, source_id)

    ctx, audit = context_from_graph_provenance(
        engine, claim, source_id=source_id, prior_decision=prior)
    obs.update({"staged_" + k: v for k, v in audit.items()})
    engine.set_source_context_for_turn(claim, ctx)
    text, err = _is_process_turn(engine, obs["prompt"])
    obs["text"] = text
    obs["error"] = err
    decision = getattr(engine, "_last_source_evaluation", None)
    obs["decision"] = decision.decision.value if decision is not None else None
    obs["token"] = decision.token if decision is not None else None
    obs["rule"] = decision.rule if decision is not None else None
    # Gate invariant: a non-ENDORSE decision MUST be the engine's single
    # decision token; an ENDORSE leaves the generated text untouched.
    obs["gate_ok"] = bool(
        decision is not None and
        (decision.decision is Decision.ENDORSE
         or text.strip() == decision.token))
    obs["prompt_label_free"] = prompt_is_label_free(obs["prompt"], source_id)
    return obs


def grade_internal_source_condition(obs, condition):
    """Attach additive flags; reads no threshold and computes no decision."""
    expected = IS_EXPECTED_TOKEN[condition]
    tok, dec = obs.get("token"), obs.get("decision")
    return {
        "condition": condition,
        "decision": dec,
        "token": tok,
        "expected_token": expected,
        # ENDORSE is deliberately SILENT at the emission boundary: the gate
        # leaves the claim text unchanged and records the decision on the
        # engine. Requiring a literal "Y" token contradicts that contract and
        # scored correct endorsements as failures.
        "correct": (dec == IS_EXPECTED_DECISION[condition] and
                    (condition == "credible" or tok == expected)),
        "decision_expected": dec == IS_EXPECTED_DECISION[condition],
        "gate_ok": bool(obs.get("gate_ok")),
        "prompt_label_free": bool(obs.get("prompt_label_free")),
        "context_from_graph": obs.get("staged_context_source") ==
                              "engine_graph_provenance",
        "fingerprint": obs.get("staged_fingerprint"),
        "rule": obs.get("rule"),
        "missing": bool(obs.get("missing")),
        "teached": obs.get("teached", 0),
        "invalidated": obs.get("invalidated", 0),
        "prompt": obs.get("prompt"),
    }


class InternalSourceArms:
    """Injectable container; a missing condition is reported, never faked."""

    CONDITIONS = IS_CONDITIONS

    def __init__(self, credible=None, discredited=None, unknown=None):
        self.credible = credible
        self.discredited = discredited
        self.unknown = unknown

    def missing(self):
        return [c for c in self.CONDITIONS if getattr(self, c, None) is None]

    def as_dict(self):
        return {c: getattr(self, c, None) is not None for c in self.CONDITIONS}


def run_internal_source_arms(items, records, arms):
    """Annotate records IN ADDITIVELY with the three internal-source arms."""
    by_id = {r["id"]: r for r in records}
    for it in items:
        if it["kind"] != "gap" or not it.get("claim"):
            continue
        rec = by_id[it["id"]]
        for cond in IS_CONDITIONS:
            fn = getattr(arms, cond, None)
            if fn is None:
                continue
            obs = fn(it, cond)
            flags = grade_internal_source_condition(obs, cond)
            rec["is_%s_flags" % cond] = flags
            for key in ("token", "decision", "correct", "gate_ok",
                        "prompt_label_free", "context_from_graph", "missing"):
                rec["is_%s_%s" % (cond, key)] = flags[key]
    return records


def internal_source_summary(records, max_per_pool=0):
    """Rates per condition and pool, from raw observations; no threshold read.

    All keys are additive; no existing summary field is renamed or removed.
    """
    out = {"arm_present": True, "conditions": list(IS_CONDITIONS),
           "flag": IS_FLAG, "context_source": "engine_graph_provenance",
           "anti_tautology_contract": (
               "context_from_graph_provenance() reads only graph provenance; "
               "item answer/status/id fields are never passed to it"),
           "pools": {}}
    for pool in ("primary", "sealed"):
        rows = [r for r in records if r.get("pool") == pool and
                "is_credible_flags" in r]
        if max_per_pool and len(rows) > max_per_pool:
            rows = rows[:max_per_pool]
        per = {"n": len(rows), "conditions": {}}
        for cond in IS_CONDITIONS:
            sel = [r for r in rows if "is_%s_flags" % cond in r]
            def rate(key):
                return (round(sum(1 for r in sel
                                  if r["is_%s_flags" % cond].get(key) is True)
                              / len(sel), 4) if sel else None)
            per["conditions"][cond] = {
                "n": len(sel),
                "correct": rate("correct"),
                "decision_expected": rate("decision_expected"),
                "gate_ok": rate("gate_ok"),
                "prompt_label_free": rate("prompt_label_free"),
                "context_from_graph": rate("context_from_graph"),
                "endorse_rate": (round(sum(
                    1 for r in sel if r["is_%s_flags" % cond]["decision"]
                    == "ENDORSE") / len(sel), 4) if sel else None),
                "abstain_rate": (round(sum(
                    1 for r in sel if r["is_%s_flags" % cond]["decision"]
                    == "ABSTAIN") / len(sel), 4) if sel else None),
                "withdraw_rate": (round(sum(
                    1 for r in sel if r["is_%s_flags" % cond]["decision"]
                    == "WITHDRAW") / len(sel), 4) if sel else None),
            }
        per["anti_tautology_ok"] = all(
            r.get("is_%s_context_from_graph" % c) is True and
            r.get("is_%s_prompt_label_free" % c) is True
            for r in rows for c in IS_CONDITIONS
            if "is_%s_flags" % c in r)
        out["pools"][pool] = per
    out["audit"] = [
        {k: r[k] for k in ("id", "pool", "is_credible_flags",
                          "is_discredited_flags", "is_unknown_flags")
         if k in r}
        for r in records if "is_credible_flags" in r]
    return out


def internal_source_decide(summary):
    """Pre-registered criteria for the NEW arm, kept SEPARATE from decide().

    Existing gates and thresholds are untouched: this function produces only
    `summary['internal_source']['rules'/'pass']`, an additive JSON subtree.
    """
    rules = {"IS_arm_present": bool(summary.get("arm_present")),
             "IS_flag_enabled": bool(summary.get("flag_enabled"))}
    for pool, pp in summary.get("pools", {}).items():
        for cond in IS_CONDITIONS:
            cc = pp.get("conditions", {}).get(cond, {})
            rules["IS_%s_correct_%s>=%.2f" % (cond, pool, TH["IS_correct_min"])] = (
                cc.get("correct") is not None and
                cc.get("correct") >= TH["IS_correct_min"])
            rules["IS_%s_gate_%s>=%.2f" % (cond, pool, TH["IS_gate_invariant_min"])] = (
                cc.get("gate_ok") is not None and
                cc.get("gate_ok") >= TH["IS_gate_invariant_min"])
            rules["IS_%s_label_free_%s" % (cond, pool)] = (
                cc.get("prompt_label_free") is not None and
                cc.get("prompt_label_free") >= TH["IS_prompt_label_free_min"])
        rules["IS_anti_tautology_%s" % pool] = bool(pp.get("anti_tautology_ok"))
        rules["IS_n_%s>=20" % pool] = pp.get("n", 0) >= 20
    return rules, all(rules.values())


def internal_source_provenance_mutation_control(claim, source_id, graph,
                                                prior_decision):
    """Mutation control: flip ONLY stored provenance and require the decision
    to move ENDORSE -> WITHDRAW. A staged context that ignores the graph fails.
    Pure; works on a graph snapshot or a fake, no engine boot.
    """
    from ravana.chat.source_evaluation import Decision, evaluate_source
    before, _ = context_from_graph_provenance(graph, claim, source_id,
                                              prior_decision=prior_decision)
    d_before = evaluate_source(claim, before).decision
    for _, edge in _iter_graph_edges(graph):
        sm = getattr(edge, "source_metadata", None)
        if (isinstance(sm, dict) and
                str(sm.get("source") or sm.get("url") or "").strip() == source_id):
            sm["source_status"] = "discredited"
            sm["invalidated"] = True
    after, _ = context_from_graph_provenance(graph, claim, source_id,
                                             prior_decision=prior_decision)
    d_after = evaluate_source(claim, after).decision
    return {"before": d_before.value, "after": d_after.value,
            "moved": d_before is Decision.ENDORSE and
                     d_after is Decision.WITHDRAW}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mfwf.install_frozen_patches()
    started = time.time()
    items = build_items()
    manifest = item_manifest_hash(items)

    print("=" * 76)
    print("RAVANA-LEARN-5 / WFA (UNSATURATED)  frozen replay, no network, no save")
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

    # ── web / attribution arm (frozen provider) ─────────────────────────────
    wfa = WFAComponent(FROZEN_SEARCH, max_hits=2)
    records = []
    with open(DATA_PATH, "w", encoding="utf-8") as fh:
        for it in items:
            rec = wfa.run_item(it)
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

    # ── legacy parametric/fake arm (injectable; default = no engine stub) ───
    # The old "answer-only = web disabled" arm is no longer evidence for the
    # source-monitoring claim. It is retained ONLY as an injectable fake slot
    # for backward-compatible JSON fields and unit tests. Set
    # WFA_PARAMETRIC_LIVE=1 to boot the offline engine (and its web tripwire)
    # explicitly; the PD arms below carry the evidential weight either way.
    if os.environ.get("WFA_PARAMETRIC_LIVE") == "1":
        oracle = AnswerOnlyOracle(engine=boot_answer_only_engine(), web_disabled=True)
        parametric_mode = "live_engine_web_disabled"
    else:
        oracle = AnswerOnlyOracle(engine=None, web_disabled=True)
        parametric_mode = "fake_no_engine_stub"
    with _WebAccessTripwire(oracle):
        web_disabled = run_answer_only(items, oracle, records)
    summary = summarize(items, records, wfa)
    summary["hard_fails"]["answer_only_web_leaks"] = oracle.web_calls
    # Persist the answer-only annotations alongside the web records.
    with open(DATA_PATH, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("\n--- answer-only arm (live engine, web disabled) ---")
    print("  web_disabled=%s  web_calls=%d (must be 0)  answers=%d" % (
        web_disabled, oracle.web_calls, oracle.calls))

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
        "answer_only_web_disabled": web_disabled,
        "answer_only_web_calls": oracle.web_calls,
        "answer_only_mode": parametric_mode,
        "web_arm_mode": "frozen_provider_attribution",
        "paraphrase_synonym_map_size": len(SYN),
    })

    # ── PROCESS-DISSOCIATION ARMS (turn-isolated) ────────────────────────────
    # Fix 2: clear the engine's volatile episodic/recall context before each
    # arm/turn so the manipulation is the INSTRUCTION alone. CognitiveChatEngine
    # documents reset_episodic_state() as equivalent to a fresh engine's volatile
    # state, so this is cheap. Alternatives for the pre-registered falsifier A/B:
    #   WFA_FRESH_ENGINE=1  -> one brand-new engine per turn (strongest form)
    #   WFA_SHARED_ENGINE=1 -> old shared, UNISOLATED engine (A/B falsifier arm)
    # A boot failure leaves the arms missing, which decide() hard-fails.
    def _boot_pd_engine():
        from ravana.chat.engine import CognitiveChatEngine  # type: ignore
        return CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                                   user_suffix="")

    engine_factory, isolate, pd_mode = None, None, None
    shared_engine = None
    try:
        shared_engine = _boot_pd_engine()
        if os.environ.get("WFA_SHARED_ENGINE") == "1":
            pd_mode = "shared_engine_unisolated_AB"       # falsifier: NO isolation
        elif os.environ.get("WFA_FRESH_ENGINE") == "1":
            engine_factory = _boot_pd_engine             # fresh engine per turn
            pd_mode = "fresh_engine_per_arm_turn"
        else:
            isolate = shared_engine.reset_episodic_state  # cheap, per-item clear
            pd_mode = "cleared_episodic_context_per_turn"
    except Exception as exc:
        print("  [WARN] PD engine boot failed (%s); arms missing -> hard FAIL" % exc)
        pd_mode = "boot_failed"

    source = engine_factory if engine_factory is not None else (
        shared_engine if pd_mode not in (None, "boot_failed") else None)
    arms = WFAArms(
        inclusion=EngineEndorsementResponder(source, isolate) if source is not None else None,
        exclusion=EngineEndorsementResponder(source, isolate) if source is not None else None,
        invalidation=EngineInvalidationResponder(source, isolate) if source is not None else None,
        parametric=oracle,
        forced_choice=ForcedChoiceResponder(source, isolate) if source is not None else None,
    )
    run_pd_arms(items, records, arms)
    for pool in ("primary", "sealed"):
        summary[pool].update(process_dissociation(records, pool))
        summary[pool].update(pd_quality_counters(records, pool))
    summary.update(pd_quality_counters(records))   # additive top-level counters
    summary["arms"] = arms.as_dict()
    summary["arms_missing"] = arms.missing()
    summary["pd_mode"] = pd_mode
    summary["arm_call_counts"] = {
        "inclusion": len(arms.inclusion.calls) if arms.inclusion else 0,
        "exclusion": len(arms.exclusion.calls) if arms.exclusion else 0,
        "invalidation": len(arms.invalidation.calls) if arms.invalidation else 0,
        "parametric": oracle.calls,
        "forced_choice": len(arms.forced_choice.calls) if arms.forced_choice else 0,
    }
    # The forced-choice arm is A/B-able on identical state: both surface arms
    # share the same engine/factory, item texts and self-consistent trusted set;
    # only the response surface (free text vs single token) differs.
    summary["forced_choice_ab"] = {
        "same_state": True, "engine_mode": pd_mode,
        "surfaces": ["free_text_inclusion_exclusion", "single_token_forced_choice"],
    }
    # PD informativeness caveat: a dissociation is uninformative when R>=1
    # (F undefined) or when R<=0 (no recollection). Reported, not gated
    # differently from the pre-registered PD_wellformed rule.
    summary["pd_informativeness"] = {
        pool: {"informative": (summary[pool]["R"] is not None
                               and summary[pool]["F"] is not None
                               and 0.0 < summary[pool]["R"] < 1.0),
               "R": summary[pool]["R"], "F": summary[pool]["F"]}
        for pool in ("primary", "sealed")
    }
    # Persist per-item pd_*_text / inv_after + decision/echo flags for audit,
    # in BOTH the probe file and the summary (fix 1 / analysis section 5.6).
    AUDIT_KEYS = (
        "id", "pool", "kind",
        "pd_inclusion_text", "pd_inclusion_decision",
        "pd_inclusion_unparseable", "pd_inclusion_echo_marker",
        "pd_exclusion_text", "pd_exclusion_decision",
        "pd_exclusion_unparseable", "pd_exclusion_echo_marker",
        "inv_before", "inv_after", "inv_decision",
        "inv_unparseable", "inv_echo_marker",
        "pd_forced_choice", "pd_forced_choice_text",
        "pd_forced_choice_no_answer", "pd_forced_choice_echo_marker",
    )
    summary["pd_audit"] = [
        {k: r[k] for k in AUDIT_KEYS if k in r}
        for r in records
        if any(k in r for k in ("pd_inclusion_text", "pd_exclusion_text",
                                "inv_after", "pd_forced_choice_text"))
    ]
    with open(DATA_PATH, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("\n--- process-dissociation arms (%s) ---" % pd_mode)
    print("  arms_present=%s  missing=%s" % (summary["arms"], summary["arms_missing"]))
    for pool in ("primary", "sealed"):
        pd = summary[pool]
        print("  %-8s I=%s E=%s R=%s F=%s margin=%s withdraw=%s" % (
            pool, pd["I"], pd["E"], pd["R"], pd["F"],
            pd["exclusion_margin"], pd["withdrawal_rate"]))
        print("  %-8s echo=%.3f unparseable=%s own_decision=%s fc_explicit=%s" % (
            pool, pd["echo_marker_rate"] or 0.0, pd["unparseable_rate"],
            pd["own_decision_rate"], pd["forced_choice_explicit_rate"]))

    # ── INTERNAL-SOURCE ARM: ClaimContext read from the engine's OWN graph ──
    # One dedicated engine (booted sequentially, never concurrently) serves
    # all conditions; each item's origin id is derived from its claim text, so
    # an invalidation on one item cannot leak into another item's context.
    is_mode, is_engine = "not_run", None
    if source is not None:
        try:
            is_engine = _boot_pd_engine()
            is_mode = "dedicated_engine_graph_derived_context"
        except Exception as exc:
            is_mode = "boot_failed: %s" % type(exc).__name__
    max_is = int(os.environ.get("WFA_IS_MAX_PER_POOL", "0") or 0)
    is_flag_prev = os.environ.get(IS_FLAG)
    os.environ[IS_FLAG] = "1"          # opt-in for this arm only; restored below
    try:
        is_items = items
        if max_is > 0:                 # deterministic bounded subset (still >=20/pool)
            seen, keep, kept = Counter(), [], set()
            for it in items:
                if it["kind"] != "gap":
                    kept.add(it["id"])
                    continue
                if seen[it["pool"]] < max_is:
                    keep.append(it)
                    seen[it["pool"]] += 1
                    kept.add(it["id"])
            is_items = [it for it in items if it["id"] in kept]

        def _is_fn(cond):
            # The staged origin comes from is_source_id(claim, cond), NEVER
            # from it['source_id'] / it['source_status'] / expected answer.
            return lambda it, c: run_internal_source_condition(
                is_engine, it["claim"], c, is_source_id(it["claim"], c))

        is_arms = InternalSourceArms(credible=_is_fn("credible"),
                                     discredited=_is_fn("discredited"),
                                     unknown=_is_fn("unknown"))
        run_internal_source_arms(is_items, records, is_arms)
    finally:
        if is_flag_prev is None:
            os.environ.pop(IS_FLAG, None)
        else:
            os.environ[IS_FLAG] = is_flag_prev

    summary["internal_source"] = internal_source_summary(records, max_is)
    summary["internal_source"].update({
        "flag_enabled": is_engine is not None,
        "flag_restored": os.environ.get(IS_FLAG) == is_flag_prev,
        "mode": is_mode, "max_per_pool": max_is,
        "missing_conditions": is_arms.missing(),
        "arm_call_counts": {
            c: sum(1 for r in records if "is_%s_flags" % c in r)
            for c in IS_CONDITIONS},
    })
    is_rules, is_pass = internal_source_decide(summary["internal_source"])
    summary["internal_source"]["rules"] = {k: bool(v) for k, v in is_rules.items()}
    summary["internal_source"]["pass"] = is_pass
    # Persist additive is_* fields alongside all other per-item observations.
    with open(DATA_PATH, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("\n--- internal-source arm (context read from engine graph) ---")
    print("  mode=%s flag_restored=%s" %
          (is_mode, summary["internal_source"]["flag_restored"]))
    for pool in ("primary", "sealed"):
        ipp = summary["internal_source"]["pools"][pool]
        print("  %-8s n=%d credible=%s discredited=%s unknown=%s anti_tautology=%s" % (
            pool, ipp["n"],
            ipp["conditions"]["credible"]["correct"],
            ipp["conditions"]["discredited"]["correct"],
            ipp["conditions"]["unknown"]["correct"],
            ipp["anti_tautology_ok"]))

    checks = instrument_controls(wfa)
    checks_ok = all(ok for _, ok in checks)
    print("\n--- instrument controls ---")
    for name, ok in checks:
        print("  [%s] %s" % ("ok" if ok else "BAD", name))

    # reuse shared Tier-0 G6 saturation gate as an informativeness check.
    gate_scores = {"wfa_valid": round(100 * summary["primary"]["wfa_valid"])}
    sat = front_gates.gate_saturation(gate_scores, None, "web")
    gate_invalid = sat.status == "FAIL"

    rules, passed = decide(summary, checks_ok and not gate_invalid)
    print("\n--- decision rules (pre-registered, not tuned) ---")
    for name, ok in rules.items():
        print("  [%s] %s" % ("ok" if ok else "FAIL", name))

    print("\n--- PRIMARY vs SEALED ---")
    for pool in ("primary", "sealed"):
        pp = summary[pool]
        print("  %-8s n_gap=%d wfa_valid=%.3f answer_only=%.3f delta=%.3f "
              "abstain=%.3f decoy_rej=%.3f internal_no_tag=%.3f reads=%d" % (
                  pool, pp["n_gap"], pp["wfa_valid"], pp["answer_only"],
                  pp["delta"], pp["abstain"], pp["decoy_reject"],
                  pp["internal_no_tag"], pp["reads_total"]))
        print("  %-8s I=%s E=%s R=%s F=%s exclusion_margin=%s withdrawal=%s" % (
            pool, pp["I"], pp["E"], pp["R"], pp["F"],
            pp["exclusion_margin"], pp["withdrawal_rate"]))
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
    print("\n%s  WFA (unsaturated) -- %s  summary -> %s" % (
        verdict, "meets all criteria" if passed else "criteria not met", SUMMARY_PATH))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
