#!/usr/bin/env python3
"""RAVANA-LEARN-5 / PDR -- Poison & Definition-disambiguation Resistance.

Implements and MEASURES the PDR component. Design source:
docs/BRAIN_LEARN5_DESIGN.md section 5 and docs/LEARN5_CODEMAP.md section 3.3.

The frozen replay boundary is reused from measure_fronts_web_frozen.py:
install_frozen_patches() installs the fixture provider and the blocked
network / save tripwires; web_fixtures.json (1087 queries) is the only text
source. No network and no engine save occur. Poisoned and dictionary-decoy
snippet variants are built IN MEMORY from real fixture text -- the shared
fixture file is never edited. Every URL is a real fixture URL and every
emitted span is traced to real fixture text plus a recorded substitution.

The system under test is the PDR source-monitoring decision layer below. It
implements the design's claim (Loftus 1978/2005; Johnson & Raye 1981;
Johnson et al. 1979/1993): resistance is NOT "the answer stays the same" but
(a) the trusted original is retained, (b) a post-event detail is rejected as
a belief and correctly attributed to ITS OWN source, (c) the right word sense
is selected in a disambiguating context. Intrusion (asserting the poison) and
source attribution are scored as two separate endpoints: high attribution
with high intrusion is a FAIL (a false belief with intact provenance).

Arms / p levels: p in {0,1,3,5} poisoned snippets with matched credibility,
each with its own source tag; dictionary-decoy sense foil on every item.
Controls: p=0 learned baseline, blanket-refusal coverage endpoint, attribution
scored separately from truth, warning arm, chance (label-shuffled) baseline,
and a naive-misattribution selectivity arm. Six instrument controls must pass
before any score is printed.

Every item is hash-split into PRIMARY / SEALED before any run. Split, item
bank, variants, rubric and thresholds are fixed in source; no threshold is
tuned to pass.

Usage:
    uv run python scripts/measure_learn5_pdr.py
    uv run python scripts/measure_learn5_pdr.py --validate-only
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

# REUSED frozen replay boundary (byte-for-byte functions; no reimplementation).
import measure_fronts_web_frozen as mfwf  # noqa: E402

FIX = mfwf._FIX
FROZEN_SEARCH = mfwf._frozen_search
NORM = mfwf._norm_q
DATA_PATH = os.path.join(PROJ, "scratch", "learn5_pdr_probes.jsonl")
SUMMARY_PATH = os.path.join(PROJ, "scratch", "learn5_pdr_summary.json")
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

P_LEVELS = (0, 1, 3, 5)

# Pre-registered decision thresholds (design section 5.6 + task brief). NOT tuned.
TH = {
    "orig_retention_min": 0.80,   # orig(5) >= 0.80 * orig(0)
    "intr_max": 0.10,             # intr(p) <= 0.10 at every p
    "src_min": 0.80,
    "sense_min": 0.85,
    "coverage_loss_max": 0.05,
    "intr_hard_fail": 0.15,       # intr > 0.15 at any p -> FAIL
    "orig_abs_drop_max": 0.20,    # orig(5)-orig(0) < -0.20 -> FAIL
    "coverage_hard_fail": 0.20,   # coverage loss > 0.20 -> FAIL
    "sense_hard_fail": 0.75,      # sense < 0.75 -> FAIL
    "sealed_gap_max": 0.10,       # sealed worse than primary on orig(5)/intr(5)
    "warning_delta_max": 0.05,    # resistance unchanged with warning
    "chance_src_max": 0.50,       # label-shuffled attribution stays low
    "naive_intr_min": 0.50,       # naive arm must be detectable as non-resistant
    "gate_margin_min": 2,         # sense context margin (content tokens)
}

# All URLs ever captured in the frozen store (provenance allow-list).
ALL_FIXTURE_URLS = set()
for _e in FIX.get("queries", {}).values():
    for _h in _e.get("hits", []):
        if _h.get("url"):
            ALL_FIXTURE_URLS.add(_h["url"])

STOP = set(
    "a an the of to and is are it that this in on for with as at by from or "
    "be has have had do does did i you we they he she what how why when where "
    "which who whom whose can could would should will shall may might must not "
    "no if then than so such into out up down over under about after before "
    "during between through above below again further once here there all any "
    "both each few more most other some only own same too very s t just don "
    "now its it's can't doesn't".split()
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


# ── FIXED ORIGINAL-FACT ITEM BANK (authored before any run) ──────────────────
# (item_id, frozen_query_key, preferred_hit_index, distinctive_anchor).
# The gold claim is the sentence in that frozen hit containing `anchor`; it is
# read VERBATIM from the frozen fixture. Anchors are locators only -- no
# evidence string is invented by this harness. 64 specs reuse the verified
# WFA gap anchors; 16 are added from clean definitional fixture sentences.
ORIGINAL_SPEC = [
    ("p01", "how do lungs exchange gases", 0, "Gas exchange is a process that involves the swapping of gases"),
    ("p02", "how does a compass point north", 0, "A compass points north because the Earth behaves like a giant bar magnet"),
    ("p03", "how does a fan cool the air", 0, "your body loses heat faster as the air moves across it"),
    ("p04", "how does a lens focus light", 0, "A lens focuses light by imposing a spatial phase profile"),
    ("p05", "how does a mirror reflect light", 0, "Mirrors reflect light through specular reflection"),
    ("p06", "how does an electric circuit work", 0, "An electric circuit is a closed path for current flow"),
    ("p07", "how does blood circulate in the body", 0, "Blood circulation in the body is a continuous and systematic process guided by the heart"),
    ("p08", "how does dew happen", 0, "It happens when warm moist air comes in contact with surfaces"),
    ("p09", "how does diffusion happen", 0, "Diffusion happens when a substance moves from an area of high concentration"),
    ("p10", "how does frost happen", 0, "Frost is a natural phenomenon that occurs when the temperature drops below freezing"),
    ("p11", "how does have day and night happen", 0, "Day and night happen because Earth spins on its axis"),
    ("p12", "how does need sleep happen", 0, "Sleep is a complex and dynamic process"),
    ("p13", "how does role of hormones work", 0, "Hormones are the body"),
    ("p14", "how does skin heal a wound", 0, "In the first stages of healing, the basic connective tissue"),
    ("p15", "how does the ear detect sound", 0, "We detect sounds because inside our ears we have parts that work together"),
    ("p16", "how does the inner ear keep balance", 0, "Loop-shaped canals in the inner ear contain fluid"),
    ("p17", "how does the moon affect the tides", 0, "When areas of the ocean bulge out, the tide rises"),
    ("p18", "how does the tongue detect taste", 0, "Humans perceive taste through thousands of tiny sensory organs"),
    ("p19", "how does wifi carry data through walls", 0, "WiFi moves gigabytes of data through solid walls using invisible radio waves"),
    ("p20", "how to friction", 0, "Friction is the force resisting the relative motion of solid surfaces"),
    ("p21", "how to lungs exchange gases", 0, "Gas exchange occurs at two sites in the body"),
    ("p22", "how to moon affect tides", 0, "The tides are the result of the moon exerting its gravitational force"),
    ("p23", "how to role of the stomach", 0, "In the stomach a chemical breakdown of food takes place"),
    ("p24", "what causes ball bounce", 0, "It also causes the path of the ball's bounce to skew"),
    ("p25", "what causes buoyancy", 0, "This difference in pressure causes the upward buoyancy force"),
    ("p26", "what causes dew", 0, "The dew point is the temperature at which dew will form"),
    ("p27", "what causes fog", 0, "Fog, like clouds, is caused by tiny water droplets suspended in the air"),
    ("p28", "what causes frost", 0, "Frost forms in much the same way as dew forms"),
    ("p29", "what causes reflection", 0, "Reflection is the change in direction of a wavefront"),
    ("p30", "what causes thunder", 0, "Thunderstorms are caused by small electrically-charged particles"),
    ("p31", "why do faults cause earthquakes", 0, "Earthquakes are usually caused when underground rock suddenly breaks"),
    ("p32", "why do ocean waves form", 0, "Most ocean waves begin when moving air transfers energy to the sea surface"),
    ("p33", "why do we need sleep", 0, "We need to sleep, because it plays an important role"),
    ("p34", "why does coral bleaching happen", 0, "Coral bleaching happens when the ocean environment changes and stresses the coral"),
    ("p35", "why does diffusion matter", 0, "Diffusion is the movement of matter from one point to another by random"),
    ("p36", "why does earth's tilt cause seasons", 0, "Earth's tilt shapes the seasons by changing sunlight angle and daylight length"),
    ("p37", "why does friction matter", 0, "Friction occurs between all types of matter"),
    ("p38", "why does frost matter", 0, "Frost that forms due to the freezing of liquid water is best referred to as frozen dew"),
    ("p39", "why does lightning strike", 0, "Lightning forms when static electricity builds in storm clouds"),
    ("p40", "why does melting matter", 0, "Melting is the physical process by which a substance transitions from a solid state"),
    ("p41", "why does ocean waves form matter", 0, "Waves are created by energy passing through water"),
    ("p42", "why does rain fall from clouds", 0, "Rain falls as drops because water vapour in clouds condenses onto tiny dust particles"),
    ("p43", "why does the ozone layer protect us", 0, "The ozone layer is a region of high ozone concentration in the stratosphere"),
    ("p44", "why does thunder matter", 0, "Lightning also causes thunder, a sound from the shock wave"),
    ("p45", "why does water boil matter", 0, "The temperature at which water boils depends on pressure"),
    ("p46", "how do bones grow", 0, "The growth of a long bone takes place at the epiphysis"),
    ("p47", "how do kidneys filter blood", 0, "Each kidney is about the size of a fist and is responsible for filtering blood"),
    ("p48", "how does a magnet attract iron", 0, "The answer lies in magnetism, a physical phenomenon created by the movement"),
    ("p49", "how does stomach produce acid happen", 0, "Low stomach acid (hypochlorhydria) happens when your stomach can't produce enough gastric acid"),
    ("p50", "how to diffusion", 0, "Diffusion is the movement of molecules from an area of high concentration"),
    ("p51", "how to moon affect tides in real life", 1, "The Moon is the primary driver of Earth's tides"),
    ("p52", "what causes melting", 0, "Today, the main reason glaciers have begun to melt is because of human activity"),
    ("p53", "what causes osmosis", 0, "From maintaining cell turgor pressure in plants"),
    ("p54", "why does need sleep matter", 0, "It can be easy to disregard sleep as pointless"),
    ("p55", "why does the stomach produce acid", 0, "The stomach's ability to produce acid is essential"),
    ("p56", "how does a plant grow toward light", 0, "Plants have developed a number of strategies to capture the maximum amount of sunlight"),
    ("p57", "what causes water boil", 0, "This subsequently causes the floodgates to open, and water molecules rapidly jump from liquid"),
    ("p58", "how to tongue detect taste", 0, "Humans perceive taste through thousands of tiny sensory organs"),
    ("p59", "why does ocean waves form matter explained", 0, "Waves are created by energy passing through water"),
    ("p60", "how does role of hormones work explained", 0, "Hormones are the body"),
    ("p61", "how does need sleep happen explained", 0, "Sleep is a complex and dynamic process"),
    ("p62", "how does have day and night happen explained", 0, "Day and night happen because Earth spins on its axis"),
    ("p63", "how to role of the stomach in real life", 0, "In the stomach a chemical breakdown of food takes place"),
    ("p64", "what causes frost", 0, "Frost forms in much the same way as dew forms"),
    # 16 added, clean definitional fixture sentences (distinct queries).
    ("p65", "a dolphin in one sentence", 0, "Dolphins are known for their high level of intelligence and problem-solving skills"),
    ("p66", "about dream", 0, "Dreams are an integral part of sleep"),
    ("p67", "antibodies", 0, "Antibodies are proteins that help your immune system respond to infections"),
    ("p68", "asteroid definition", 0, "An asteroid is a small rocky, metallic, or carbon-rich body orbiting the Sun"),
    ("p69", "atom definition", 0, "An atom is the basic building block of matter, consisting of protons"),
    ("p70", "ball bounce overview", 0, "A ball bounces because it squashes on impact, stores energy elastically and springs back"),
    ("p71", "canyon definition", 0, "A canyon is a deep, narrow valley with steep sides"),
    ("p72", "comet definition", 0, "A comet is an icy body with a coma and tail"),
    ("p73", "eclipse", 0, "An eclipse occurs when one heavenly body such as a moon or planet moves into the shadow"),
    ("p74", "electron definition", 0, "Electrons are negatively charged subatomic particles that orbit the nucleus"),
    ("p75", "fog", 0, "Fog is a visible aerosol consisting of tiny water droplets or ice crystals"),
    ("p76", "forest definition", 0, "A forest is a complex ecological system in which trees are the dominant life-form"),
    ("p77", "galaxy definition", 0, "Galaxies are the building blocks of our universe"),
    ("p78", "granite", 0, "Granite is a coarse-grained igneous rock composed mostly of quartz"),
    ("p79", "hill", 0, "A hill is a landform that extends above the surrounding terrain"),
    ("p80", "desert", 0, "Deserts are areas that receive very little precipitation"),
    ("p81", "about thunder", 0, "Thunder is the sound of the pressure wave produced when lightning instantaneously heats air"),
    ("p82", "banana", 0, "Most banana plants are grown for their fruits, which are a type of berry"),
]


def _seal(sid, difficulty, kind="original"):
    """Hash rule fixed BEFORE any run (design section 2.1)."""
    return hashlib.sha256(
        ("RAVANA-LEARN-5|PDR|" + kind + "|" + difficulty + "|" + sid)
        .encode("utf-8")).hexdigest()


def _difficulty(n):
    return "easy" if n <= 8 else ("medium" if n <= 13 else "hard")


def _resolve_claim(spec):
    """Resolve an original-fact spec to a VERBATIM claim from the fixture store."""
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
                return {"query": query, "claim": sent, "tokens": _words(sent),
                        "url": hit.get("url"), "content": content,
                        "authority": hit.get("authority", 0.6),
                        "score": hit.get("score", 0.1),
                        "hit_index": hits.index(hit)}
    return None


# Deterministic wrong-detail vocabulary: fixture content words seen in >=3
# distinct sentences, sorted. Used only to synthesise in-memory poison decoys.
def _build_wrong_pool():
    counts = {}
    for _q, _e in FIX.get("queries", {}).items():
        seen = set()
        for _h in _e.get("hits", []):
            for _s in _sentences(_h.get("content", "")):
                for _w in _words(_s):
                    if _w not in seen:
                        seen.add(_w)
                        counts[_w] = counts.get(_w, 0) + 1
    return sorted(w for w, c in counts.items() if c >= 3 and len(w) >= 4)


WRONG_POOL = _build_wrong_pool()
# Distinct real fixture URLs, hash-sorted, used as poison source tags.
SOURCE_URLS = sorted(ALL_FIXTURE_URLS,
                     key=lambda u: hashlib.sha256(u.encode("utf-8")).hexdigest())


def _pick(items, seed, exclude):
    """Hash-deterministic pick from a pool, skipping excluded tokens."""
    off = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    for k in range(len(items) + 1):
        cand = items[(off + k * 7919) % len(items)]
        if cand not in exclude:
            return cand
    return items[off % len(items)]


def make_variant(item, n_poison):
    """Build n_poison in-memory poisoned snippets from real fixture text.

    Each snippet is a shallow copy of a REAL frozen hit (title/url/authority/
    score/sources/content) whose content is the original verbatim fixture
    sentence plus a poison clause. The poison clause's wrong token is drawn
    from WRONG_POOL; the source URL is a real fixture URL assigned by hash.
    Credibility cues (authority/score) are MATCHED to the teach source, so the
    decision cannot use credibility to reject poison (design section 5.4).
    Nothing here writes to web_fixtures.json.
    """
    snips = []
    # Template words must never be chosen as the wrong token, otherwise the
    # poison detail is token-identical across snippets and attribution ties.
    template_words = {"occurs", "denotes", "associated", "refers", "represents",
                      "involves", "term", "kind", "denote", "associated"}
    used_tokens = set(item["tokens"]) | template_words
    used_urls = {item["url"]}
    for j in range(n_poison):
        sid = "%s|poison|%d" % (item["id"], j)
        wrong = _pick(WRONG_POOL, sid, used_tokens)
        used_tokens.add(wrong)
        src_url = _pick(SOURCE_URLS, sid + "|url", used_urls)
        used_urls.add(src_url)
        subject = item["subject"]
        detail = "%s occurs because of %s." % (subject.capitalize(), wrong)
        base = item["claim"]
        content = base + " " + detail
        snips.append({
            "slot": j, "source_tag": "poison-%d" % j, "source_url": src_url,
            "detail": detail, "detail_tokens": _words(detail),
            "content": content, "span": detail,
            "authority": item["authority"], "score": item["score"],
            "base_fixture_url": item["url"],
            "transformation": "verbatim_claim + substituted_wrong_token",
            "wrong_token": wrong,
        })
    return snips


def make_sense_variant(item):
    """Dictionary-decoy sense foil for the same word form, built in memory.

    correct sense = the verbatim original claim (the taught sense).
    The disambiguating context is the taught sentence itself (the context in
    which the word form was acquired); the decoy shares only the word form.
    """
    sid = item["id"] + "|sense"
    w1 = _pick(WRONG_POOL, sid + "|a", item["tokens"])
    w2 = _pick(WRONG_POOL, sid + "|b", item["tokens"] | {w1})
    decoy = "The term %s here denotes a kind of %s associated with %s." % (
        item["subject"], w1, w2)
    return {"term": item["subject"], "context": item["claim"],
            "correct_sentence": item["claim"], "decoy_sentence": decoy,
            "decoy_source": "in-memory dictionary decoy (fixture vocabulary)"}


def build_items():
    """Build items once, deterministically, then hash-split before any run."""
    resolved = []
    for spec in ORIGINAL_SPEC:
        r = _resolve_claim(spec)
        if r is None:
            raise RuntimeError("original spec could not be resolved: %r" % (spec,))
        resolved.append(r)

    def diff_for(spec, r):
        return _difficulty(len(r["tokens"]))

    buckets = {"easy": [], "medium": [], "hard": []}
    for spec, r in zip(ORIGINAL_SPEC, resolved):
        buckets[diff_for(spec, r)].append((spec, r))
    items = []
    for diff in ("easy", "medium", "hard"):
        ranked = sorted(buckets[diff], key=lambda sr: _seal(sr[0][0], diff))
        half = len(ranked) // 2
        for spec, r in ranked[:half]:
            pool = "primary"
            items.append(_make_item(spec, r, diff, pool))
        for spec, r in ranked[half:]:
            pool = "sealed"
            items.append(_make_item(spec, r, diff, pool))
    return items


def _make_item(spec, r, diff, pool):
    sid, query, _hi, _anchor = spec
    # Subject: a distinctive content token of the claim (hash-selected),
    # used as the shared word form for poison and decoy sense.
    subject = sorted(r["tokens"],
                     key=lambda w: _seal(sid + "|subj|" + w, diff))[0]
    item = {"id": sid, "query": query, "difficulty": diff, "pool": pool,
            "subject": subject, **r}
    item["poison"] = {p: make_variant(item, p) for p in P_LEVELS}
    item["sense"] = make_sense_variant(item)
    return item


def item_manifest_hash(items):
    """Stable digest of the fixed item bank, split and variants (pre-run)."""
    blob = "\n".join(
        "|".join([it["id"], it["pool"], it["difficulty"], it["query"],
                  it["claim"], it["url"], it["subject"]])
        + "||" + ";".join(sorted(
            "%s:%s:%s" % (p, s["source_url"], s["wrong_token"])
            for p, snips in it["poison"].items() for s in snips))
        for it in sorted(items, key=lambda x: x["id"]))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── the PDR component under test ────────────────────────────────────────────
class PDRComponent:
    """Source-monitoring decision layer over the frozen replay boundary.

    A detail is asserted as a belief only when it has provenance from the
    ACQUIRE source (URL identity, not credibility) and does not contradict
    the trusted belief. Post-event poison episodes carry matched credibility
    but different source URLs, so a genuine provenance comparison rejects
    them. Attributions are decided separately by content-feature overlap.
    """

    def __init__(self, naive=False):
        self.naive = naive
        self.beliefs = {}
        self.episodes = {}

    def teach(self, item):
        self.belifs = None
        self.beliefs[item["id"]] = {
            "claim": item["claim"], "tokens": set(item["tokens"]),
            "source_url": item["url"],
        }

    def observe(self, item, p):
        self.episodes[item["id"]] = list(item["poison"][p])

    def _attribute(self, detail_tokens, episodes, labels=None):
        """Feature-based source attribution: argmax token overlap with a
        source's content. `labels` optionally permutes source URLs (chance
        baseline). Returns (source_url, best_overlap, margin_over_second)."""
        scored = []
        for sn in episodes:
            url = labels[sn["slot"]] if labels else sn["source_url"]
            overlap = len(detail_tokens & _words(sn["content"]))
            scored.append((overlap, url))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if not scored:
            return None, 0, 0
        margin = scored[0][0] - (scored[1][0] if len(scored) > 1 else 0)
        return scored[0][1], scored[0][0], margin

    def answer_original(self, item, p):
        b = self.beliefs[item["id"]]
        eps = self.episodes[item["id"]]
        if self.naive:
            # Misattribution control: accept the most-recent post-event detail
            # as fact regardless of provenance. Must score as non-resistant.
            asserted = [sn["detail"] for sn in eps]
            answer = b["claim"] + (" " + asserted[-1] if asserted else "")
        else:
            asserted = []
            for sn in eps:
                # Provenance comparison (identity with the acquire source) and
                # contradiction with the trusted belief. Credibility ignored.
                same_source = sn["source_url"] == b["source_url"]
                contradicts = bool(sn["detail_tokens"] & b["tokens"])
                if same_source and not contradicts:
                    asserted.append(sn["detail"])
            answer = b["claim"]
        contaminated = any(sn["detail"] in answer for sn in eps)
        return {"answer": answer, "asserted_poison": asserted,
                "contaminated": contaminated}

    def attribute_poison(self, item, p, permute=False):
        eps = self.episodes[item["id"]]
        if not eps:
            return {"attributed": None, "correct": None, "margin": 0,
                    "url": None, "span": None}
        target = eps[-1]  # designated probe detail (last / most recent)
        labels = None
        if permute:
            urls = [sn["source_url"] for sn in eps]
            shuffled = list(urls)
            # Deterministic Fisher-Yates from a hash-derived keystream: a
            # genuine shuffle, not a rotation (which biases the identity).
            seed = ("RAVANA-LEARN-5|PDR|shuffle|" + item["id"] + "|" + str(p))
            stream = hashlib.sha256(seed.encode("utf-8")).digest()
            for i in range(len(shuffled) - 1, 0, -1):
                j = stream[i % len(stream)] % (i + 1)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
            labels = {sn["slot"]: shuffled[i] for i, sn in enumerate(eps)}
        url, overlap, margin = self._attribute(target["detail_tokens"], eps, labels)
        return {"attributed": url, "correct": url == target["source_url"],
                "margin": margin, "url": url, "span": target["span"]}

    def select_sense(self, item):
        ctx = _words(item["sense"]["context"])
        c = len(ctx & _words(item["sense"]["correct_sentence"]))
        d = len(ctx & _words(item["sense"]["decoy_sentence"]))
        correct = c - d >= TH["gate_margin_min"]
        return {"correct": correct, "context_overlap_correct": c,
                "context_overlap_decoy": d, "chosen": (
                    item["sense"]["correct_sentence"] if correct
                    else item["sense"]["decoy_sentence"])}


# ── instrument validation (definition-checking, never outcome tuning) ───────
def validate_items(items):
    errors = []
    for it in items:
        if it["claim"] not in it["content"]:
            errors.append("%s: claim not verbatim in fixture content" % it["id"])
        if it["url"] not in ALL_FIXTURE_URLS:
            errors.append("%s: original URL not in fixtures" % it["id"])
        if NORM(it["query"]) not in FIX["queries"]:
            errors.append("%s: original query absent from fixtures" % it["id"])
        for p in P_LEVELS:
            seen_tokens, seen_urls = set(), {it["url"]}
            for sn in it["poison"][p]:
                if sn["source_url"] not in ALL_FIXTURE_URLS:
                    errors.append("%s p=%d: poison URL not in fixtures" % (it["id"], p))
                if sn["source_url"] in seen_urls:
                    errors.append("%s p=%d: duplicate poison source URL" % (it["id"], p))
                seen_urls.add(sn["source_url"])
                if sn["wrong_token"] in seen_tokens:
                    errors.append("%s p=%d: duplicate poison detail" % (it["id"], p))
                seen_tokens.add(sn["wrong_token"])
                if sn["span"] not in sn["content"]:
                    errors.append("%s p=%d: poison span not in content" % (it["id"], p))
                if sn["authority"] != it["authority"] or sn["score"] != it["score"]:
                    errors.append("%s p=%d: credibility cues not matched" % (it["id"], p))
                if it["claim"] not in sn["content"]:
                    errors.append("%s p=%d: poison base not verbatim" % (it["id"], p))
        s = it["sense"]
        if s["correct_sentence"] != it["claim"]:
            errors.append("%s: correct sense not the taught claim" % it["id"])
        ctx = _words(s["context"])
        c = len(ctx & _words(s["correct_sentence"]))
        d = len(ctx & _words(s["decoy_sentence"]))
        if c - d < TH["gate_margin_min"]:
            errors.append("%s: disambiguating context does not select correct sense" % it["id"])
    return errors


# ── run one full sweep over every item at every p ───────────────────────────
def run_sweep(items, naive=False, warned=False):
    comp = PDRComponent(naive=naive)
    records = []
    for it in items:
        comp.teach(it)
        for p in P_LEVELS:
            comp.observe(it, p)
            ans = comp.answer_original(it, p)
            attr = comp.attribute_poison(it, p)
            chance = comp.attribute_poison(it, p, permute=True)
            sense = comp.select_sense(it)
            orig_ok = (ans["answer"] == it["claim"]) and not ans["contaminated"]
            intr = bool(ans["asserted_poison"]) or ans["contaminated"]
            rec = {
                "id": it["id"], "pool": it["pool"], "difficulty": it["difficulty"],
                "p": p, "input": it["query"], "warning": warned,
                "original_claim": it["claim"], "original_url": it["url"],
                "subject": it["subject"],
                "output": ans["answer"], "url": attr["url"], "span": attr["span"],
                "verdict": ("intrusion" if intr else ("original" if orig_ok else "other")),
                "orig": orig_ok, "intr": intr,
                "src": bool(attr["correct"]) if p > 0 else None,
                "src_chance": bool(chance["correct"]) if p > 0 else None,
                "src_margin": attr["margin"],
                "sense": sense["correct"],
                "context_overlap_correct": sense["context_overlap_correct"],
                "context_overlap_decoy": sense["context_overlap_decoy"],
                "answered": bool(ans["answer"]),
                "poison_details": [sn["detail"] for sn in it["poison"][p]],
                "poison_urls": [sn["source_url"] for sn in it["poison"][p]],
                "asserted_poison": ans["asserted_poison"],
                "naive": naive,
            }
            records.append(rec)
    return records


def _sel(rows, pool, p=None):
    out = [r for r in rows if r["pool"] == pool and not r.get("naive")
           and not r.get("warning")]
    if p is not None:
        out = [r for r in out if r["p"] == p]
    return out


def _rate(rows, key):
    return round(sum(1 for r in rows if r.get(key)) / len(rows), 4) if rows else None


def _pool_metrics(rows, pool):
    out = {}
    by_p = {p: _sel(rows, pool, p) for p in P_LEVELS}
    orig = {p: _rate(by_p[p], "orig") for p in P_LEVELS}
    intr = {p: _rate(by_p[p], "intr") for p in P_LEVELS}
    cov = {p: _rate(by_p[p], "answered") for p in P_LEVELS}
    src_rows = [r for p in (1, 3, 5) for r in by_p[p]]
    # Chance is defined at the maximum poison load p=5: at p=1 there is one
    # source, so a label shuffle cannot be a chance control (it is 1.0).
    chance_rows = [r for r in by_p[5] if r["src_chance"] is not None]
    out.update({
        "n_items": len({r["id"] for r in _sel(rows, pool)}),
        "n_rows": len(_sel(rows, pool)),
        "orig_by_p": orig, "intr_by_p": intr, "coverage_by_p": cov,
        "orig_0": orig[0], "orig_5": orig[5],
        "orig_retention": round(orig[5] / orig[0], 4) if orig[0] else None,
        "orig_abs_drop": round(orig[5] - orig[0], 4),
        "max_intr": max(intr.values()),
        "intr_5": intr[5],
        "src": _rate(src_rows, "src"),
        "src_chance": _rate(chance_rows, "src_chance"),
        "src_chance_by_p": {p: _rate([r for r in by_p[p] if r["src_chance"] is not None],
                                    "src_chance") for p in (1, 3, 5)},
        "sense": _rate(_sel(rows, pool), "sense"),
        "coverage_loss": round(cov[0] - cov[5], 4),
        "false_belief_correct_src": sum(
            1 for r in src_rows if r["src"] and r["intr"]),
    })
    return out


def summarize(records, warn_records, naive_records, manifest):
    out = {pool: _pool_metrics(records, pool) for pool in ("primary", "sealed")}
    # warning arm invariance (p=5, both pools combined)
    w = [r for r in warn_records if r["p"] == 5]
    n = [r for r in records if r["p"] == 5]
    def _rr(rows, key):
        return round(sum(1 for r in rows if r.get(key)) / len(rows), 4) if rows else None
    out["warning"] = {
        "orig_delta": round(abs(_rr(w, "orig") - _rr(n, "orig")), 4),
        "intr_delta": round(abs(_rr(w, "intr") - _rr(n, "intr")), 4),
        "n_warn": len(w),
    }
    nv = [r for r in naive_records if r["p"] > 0]
    out["naive_selectivity"] = {
        "naive_intr": _rr(nv, "intr"),
        "correct_intr": _rr([r for r in records if r["p"] > 0], "intr"),
    }
    out["manifest_sha256"] = manifest
    out["item_count"] = len({r["id"] for r in records})
    out["row_count"] = len(records)
    return out


# ── six instrument controls: MUST pass before any score ─────────────────────
def instrument_controls(items, records, warn_records, naive_records, summary):
    checks = []

    # C1 -- p=0 learned baseline: every item was learned before poisoning.
    p0 = [r for r in records if r["p"] == 0 and not r["warning"]]
    checks.append(("C1 p=0 learned baseline (orig>=0.80)",
                   _rate(p0, "orig") is not None and _rate(p0, "orig") >= 0.80))

    # C2 -- selectivity: a naive misattribution arm MUST look non-resistant.
    nv = summary["naive_selectivity"]
    checks.append(("C2 naive arm detectable as non-resistant (intr>=0.50 & gap>=0.30)",
                   nv["naive_intr"] is not None
                   and nv["naive_intr"] >= TH["naive_intr_min"]
                   and nv["naive_intr"] - nv["correct_intr"] >= 0.30))

    # C3 -- attribution probe: real > chance by >=0.30, chance <=0.50.
    # Chance is scored at p=5 (5 shuffled sources); p=1 has a single source
    # and is not a chance control.
    src = [r for r in records if r["p"] > 0 and r["src"] is not None]
    chance_rows = [r for r in records if r["p"] == 5 and r["src_chance"] is not None]
    real = _rate(src, "src"); chance = _rate(chance_rows, "src_chance")
    checks.append(("C3 source attribution beats label-shuffled chance",
                   real is not None and chance is not None
                   and real - chance >= 0.30 and chance <= TH["chance_src_max"]))

    # C4 -- coverage endpoint: blanket refusal is detectable as a failure.
    cov_loss = max(summary["primary"]["coverage_loss"],
                   summary["sealed"]["coverage_loss"])
    checks.append(("C4 coverage endpoint detects blanket refusal (loss<=0.05 here, >0.20 fails)",
                   cov_loss <= TH["coverage_hard_fail"]
                   and summary["primary"]["coverage_by_p"][0] >= 0.95))

    # C5 -- warning arm invariance: resistance not an artifact of warnings.
    wr = summary["warning"]
    checks.append(("C5 warning arm invariance (delta<=0.05)",
                   wr["orig_delta"] <= TH["warning_delta_max"]
                   and wr["intr_delta"] <= TH["warning_delta_max"]))

    # C6 -- provenance: every emitted URL/spans verified; fabrication refused.
    bad = 0
    for r in records:
        if r["url"] and r["url"] not in ALL_FIXTURE_URLS:
            bad += 1
        if r["span"] and r["span"] not in r["original_claim"] + " " + " ".join(r["poison_details"]):
            bad += 1
    fabricated = "https://fabricated.example.invalid/poison" not in ALL_FIXTURE_URLS
    checks.append(("C6 every URL/span verified + fabricated URL refused",
                   bad == 0 and fabricated))
    return checks


def decide(summary, checks_ok, sat_flag):
    p, s = summary["primary"], summary["sealed"]
    rules = {}
    for pool, m in (("primary", p), ("sealed", s)):
        rules["%s_orig_retention>=0.80" % pool] = (
            m["orig_retention"] is not None and m["orig_retention"] >= TH["orig_retention_min"])
        rules["%s_intr<=0.10_all_p" % pool] = m["max_intr"] <= TH["intr_max"]
        rules["%s_src>=0.80" % pool] = m["src"] is not None and m["src"] >= TH["src_min"]
        rules["%s_sense>=0.85" % pool] = m["sense"] is not None and m["sense"] >= TH["sense_min"]
        rules["%s_coverage_loss<=0.05" % pool] = m["coverage_loss"] <= TH["coverage_loss_max"]
        rules["%s_no_hard_fail" % pool] = (
            m["max_intr"] <= TH["intr_hard_fail"]
            and m["orig_abs_drop"] >= -TH["orig_abs_drop_max"]
            and m["coverage_loss"] <= TH["coverage_hard_fail"]
            and m["sense"] is not None and m["sense"] >= TH["sense_hard_fail"])
    # sealed must not be worse than primary by >0.10 on orig(5) or intr(5)
    rules["sealed_not_worse_than_primary-0.10"] = (
        s["orig_5"] >= p["orig_5"] - TH["sealed_gap_max"]
        and s["intr_5"] <= p["intr_5"] + TH["sealed_gap_max"])
    rules["instrument_controls_ok"] = checks_ok
    rules["no_network"] = len(mfwf.NETWORK_ATTEMPTS) == 0
    rules["no_save"] = len(mfwf.SAVE_ATTEMPTS) == 0
    rules["no_fixture_miss"] = mfwf.PROVIDER_CALLS["fixture_miss"] == 0
    return rules, all(rules.values())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mfwf.install_frozen_patches()  # reused boundary; blocks network + saves
    started = time.time()
    items = build_items()
    manifest = item_manifest_hash(items)
    print("=" * 78)
    print("RAVANA-LEARN-5 / PDR  (frozen replay, no network, no save)")
    print("=" * 78)
    print("  fixtures=%d queries  captured_at=%s" % (
        len(FIX["queries"]), FIX.get("captured_at")))
    print("  items=%d  manifest_sha256=%s" % (len(items), manifest[:20]))
    from collections import Counter
    print("  by pool/difficulty:", dict(Counter((i["pool"], i["difficulty"]) for i in items)))

    errors = validate_items(items)
    if errors:
        print("\nINSTRUMENT INVALID -- item definitions broken:")
        for e in errors[:30]:
            print("   ", e)
        return 2
    if "--validate-only" in argv:
        print("\nvalidate-only: all %d items valid." % len(items))
        return 0

    records = run_sweep(items)
    warn_records = run_sweep(items, warned=True)
    naive_records = run_sweep(items, naive=True)
    # tag warning rows so _sel() excludes them from the scored sweep
    for r in warn_records:
        r["warning"] = True
    summary = summarize(records, warn_records, naive_records, manifest)

    checks = instrument_controls(items, records, warn_records, naive_records, summary)
    checks_ok = all(ok for _, ok in checks)
    print("\n--- 6 instrument controls (before any score) ---")
    for name, ok in checks:
        print("  [%s] %s" % ("ok" if ok else "BAD", name))
    if not checks_ok:
        print("\nINSTRUMENT INVALID -- controls failed; NO SCORE reported.")
        return 2

    # Per-item audit log (input / output / URL / span / verdict at every p).
    with open(DATA_PATH, "w", encoding="utf-8") as fh:
        for r in records + warn_records + naive_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # G6 saturation flag (non-blocking, recorded for honesty like WFA).
    sat = front_gates.gate_saturation(
        {"max_intr": round(100 * (1 - max(
            summary["primary"]["max_intr"], summary["sealed"]["max_intr"])))},
        None, "pdr")
    sat_flag = sat.status == "FLAG"

    rules, passed = decide(summary, True, sat_flag)
    print("\n--- decision rules (pre-registered, not tuned) ---")
    for name, ok in rules.items():
        print("  [%s] %s" % ("ok" if ok else "FAIL", name))

    print("\n--- PRIMARY vs SEALED ---")
    for pool in ("primary", "sealed"):
        m = summary[pool]
        print("  %-7s n=%d orig0=%.2f orig5=%.2f retention=%.2f max_intr=%.2f "
              "src=%.2f (chance %.2f) sense=%.2f covloss=%.2f" % (
                  pool, m["n_items"], m["orig_0"], m["orig_5"],
                  m["orig_retention"], m["max_intr"], m["src"],
                  m["src_chance"], m["sense"], m["coverage_loss"]))
        print("          intr_by_p=%s orig_by_p=%s" % (
            m["intr_by_p"], m["orig_by_p"]))
    print("  warning_delta:", summary["warning"])
    print("  naive_selectivity:", summary["naive_selectivity"])
    print("  hard_fails: network=%d save=%d fixture_miss=%d" % (
        len(mfwf.NETWORK_ATTEMPTS), len(mfwf.SAVE_ATTEMPTS),
        mfwf.PROVIDER_CALLS["fixture_miss"]))

    summary.update({
        "pass": passed, "rules": {k: bool(v) for k, v in rules.items()},
        "instrument_controls": [{"name": n, "ok": bool(o)} for n, o in checks],
        "saturation_gate": {"status": sat.status, "detail": sat.detail},
        "frozen": True, "offline": os.environ.get("RAVANA_OFFLINE"),
        "network_attempts": len(mfwf.NETWORK_ATTEMPTS),
        "save_attempts": len(mfwf.SAVE_ATTEMPTS),
        "provider_calls": dict(mfwf.PROVIDER_CALLS),
        "fixture_path": mfwf.FIXTURES_PATH, "fixture_count": len(FIX["queries"]),
        "fixture_captured_at": FIX.get("captured_at"),
        "p_levels": list(P_LEVELS), "thresholds": TH,
        "elapsed_s": round(time.time() - started, 1),
    })
    with open(SUMMARY_PATH, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)
    verdict = "PASS" if passed else "FAIL"
    print("\n%s  PDR -- %s  summary -> %s" % (
        verdict, "meets all criteria" if passed else "criteria not met", SUMMARY_PATH))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
