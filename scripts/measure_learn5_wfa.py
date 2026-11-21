#!/usr/bin/env python3
"""RAVANA-LEARN-5 / WFA -- UNSATURATED web fact acquisition and attribution.

Implements and MEASURES the redesigned WFA component. Design source:
docs/investigations/WEB_AND_SATURATION.md section 4 and
docs/BRAIN_LEARN5_DESIGN.md section 4 (especially 4.1-4.6).

Why this file was rewritten
---------------------------
The previous harness PASSed 1.00/1.00 but was G6-INVALID (a saturated progress
signal). Root causes, all fixed here:

  * GOLD CLAIM WAS THE SOURCE SENTENCE (tautological entailment).  Now the gold
    claim is a deterministic PARAPHRASE of the verbatim source span: source
    content words are replaced by fixed single-word synonyms, and the frozen
    entailment rubric canonicalises the claim back to the source vocabulary
    before matching.  A claim that is identical to its evidence is rejected.
  * ANSWER-ONLY ARM WAS AN EMPTY PARAMETRIC STORE.  Now it is the LIVE engine
    with web disabled (RAVANA_OFFLINE=1, `_offline=True`), answering the same
    gap questions from its parametric/graph store.  Its answers are graded with
    the same fact-token rubric; a web-access tripwire hard-fails the arm.
  * DECOYS WERE WEAK / TOO FEW.  Now >=60 NEAR-MISS decoy sources per whole
    bank (same topic, wrong fact, >=2 topic tokens shared with the real source)
    that must be rejected by abstention.
  * n WAS 32/pool.  Now >=60 gap + >=30 closed-web + >=30 decoy PER POOL, kept
    honest by the same fixed sha256 hash split as before.

The replay boundary is still reused from measure_fronts_web_frozen.py: frozen
provider stubs, blocked live-fetch / save tripwires and counters, and the
checked-in web_fixtures.json (1087 queries).  No network and no engine save
are performed by the web arm.  Every emitted URL was returned by the frozen
provider this run; every emitted evidence span is a VERBATIM substring of the
fetched content.  This is source monitoring at retrieval (Johnson & Raye 1981).

All thresholds stay at their pre-registered design values.  NOTHING is tuned
to force a PASS: paraphrase entailment + a live-web-off baseline + near-miss
decoys introduce error modes that are allowed to fail.

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
TH = {
    "wfa_valid_min": 0.70,
    "abstain_min": 0.90,
    "delta_min": 0.20,          # wfa_valid - answer_only
    "decoy_rej_min": 0.85,
    "internal_no_tag_min": 0.90,
    "sealed_drop_max": 0.10,    # sealed wfa_valid >= primary - this
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
        items.append({"id": spec[0], "kind": "gap", "pool": pool,
                      "difficulty": diff, "key_fact_tokens": key_tokens, **r})
    for spec, diff, pool in _split(
            "decoy", DECOY_SPEC, lambda s: _difficulty(len(_words(s[2])))):
        sid, query, claim, tag = spec
        items.append({"id": sid, "kind": "decoy", "pool": pool,
                      "difficulty": diff, "query": query, "claim": claim,
                      "claim_tokens": sorted(_words(claim)),
                      "decoy_tag": tag})
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
    return rules, all(rules.values())


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

    # ── answer-only arm: LIVE engine with web disabled ──────────────────────
    oracle = AnswerOnlyOracle(engine=boot_answer_only_engine(), web_disabled=True)
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
        "answer_only_mode": "live_engine_web_disabled",
        "web_arm_mode": "frozen_provider_attribution",
        "paraphrase_synonym_map_size": len(SYN),
    })

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
