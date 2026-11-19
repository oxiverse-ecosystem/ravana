#!/usr/bin/env python3
"""Baseline + progress harness for the seven capability fronts.

Fronts:
  1. language understanding / conversation
  2. grounded factual answers
  3. opinions / stance
  4. thinking / reasoning
  5. language learning (online)
  6. speaking / surface realization
  7. code generation

Each front gets ~27 DISTINCT held-out probes and a 0-100 score derived from
them. Every probe writes one record to

    scratch/front_probes_n30.jsonl

with fields {front, probe_index, subject, query, response, passed, reason,
control, sub_scores}. The aggregate print format is unchanged from the
original small-n harness so an A/B against the old run is trivial.

CRITICAL: the per-probe JUDGE functions preserve the original pass criteria
except for three narrow, evidence-backed fixes documented in
scratch/criteria_change_evidence.md (and as `note` fields on the CRITERION-FIX
control records logged with control=true). Thresholds and the causal/stance
markers are UNCHANGED. See that file for the hand-labels and old verdicts.

Engine policy: RAVANA_OFFLINE=1, ONE cold boot, ALL probes in one process,
NO eng.save() (the three weight stores drift and concurrent saves clobber
learned state). Learn code in-memory via the engine's own code learner only.

Tier-0 instrument gates (docs/BENCHMARKING_SURVEY.md section 5.1) run BEFORE
any score is printed. A FAIL suppresses the score block and exits nonzero.
Gate outcomes are logged to scratch/front_probes_n30_gates.jsonl; the existing
per-probe JSONL and aggregate print format are unchanged.

Usage:
    uv run python scripts/measure_fronts.py
    uv run python scripts/measure_fronts.py --gates-only
"""
import json
import os
import re
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import front_gates

DATA_PATH = os.path.join(PROJ, "scratch", "front_probes_n30.jsonl")
SUMMARY_PATH = os.path.join(PROJ, "scratch", "front_probes_n30_summary.json")
GATE_LOG_PATH = os.path.join(PROJ, "scratch", "front_probes_n30_gates.jsonl")
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
_LOGFH = None

# ── ORIGINAL (small-n) probes, kept for provenance only. ────────────────────
# The held-out lists below are DISJOINT from every subject in these lists.
CONVERSATION_BASELINE = ["hello", "how are you feeling today",
                         "what did we talk about earlier"]
GROUNDED_BASELINE = ["what is gravity", "what causes rain",
                     "tell me about the ocean", "what is a species"]
OPINION_BASELINE = ["what do you think about artificial intelligence",
                    "do you believe machines can be conscious",
                    "what is your opinion on privacy"]
THINKING_BASELINE = ["why do you think it rains", "explain how a battery works",
                     "what would happen if the moon disappeared"]
SPEAKING_BASELINE = ["tell me a sentence about water",
                     "describe gravity in one sentence"]
CODE_BASELINE = ["write a function that adds two numbers",
                 "write a function that reverses a string"]

# ── HELD-OUT probes: (subject, query). 27 per front. ────────────────────────
CONVERSATION = [
    ("greeting-morning", "good morning"),
    ("greeting-evening", "good evening"),
    ("greeting-casual", "hey there"),
    ("self-intro", "tell me about yourself"),
    ("identity-name", "what is your name"),
    ("identity-robot", "are you a robot"),
    ("capability", "what can you do"),
    ("purpose", "what is your purpose"),
    ("day", "how is your day going"),
    ("mood", "what makes you happy"),
    ("topics", "what do you like to talk about"),
    ("help", "can you help me with something"),
    ("thanks", "thank you for helping"),
    ("farewell", "goodbye for now"),
    ("status", "are you still there"),
    ("age", "how old are you"),
    ("sleep", "do you ever sleep"),
    ("origin", "where are you from"),
    ("hobbies", "what are your hobbies"),
    ("joke", "tell me a joke"),
    ("music", "how do you feel about music"),
    ("color", "what is your favorite color"),
    ("tired", "do you get tired"),
    ("thinking", "can you think for yourself"),
    ("friendship", "do you have friends"),
    ("learning", "do you learn from conversations"),
    ("weather-chat", "how is the weather today"),
]

# Each query ends in a single-token subject: the grounded judge uses
# q.split()[-1], so a two-word subject would only check its final word.
GROUNDED = [
    ("photosynthesis", "what is photosynthesis"),
    ("earthquakes", "what causes earthquakes"),
    ("sun", "tell me about the sun"),
    ("neuron", "what is a neuron"),
    ("dna", "what is dna"),
    ("evaporation", "what is evaporation"),
    ("volcanoes", "what causes volcanoes"),
    ("computer", "what is a computer"),
    ("electricity", "what is electricity"),
    ("cell", "what is a cell"),
    ("brain", "what is the brain"),
    ("laser", "what is a laser"),
    ("virus", "what is a virus"),
    ("mineral", "what is a mineral"),
    ("hurricane", "what is a hurricane"),
    ("glacier", "what is a glacier"),
    ("fossil", "what is a fossil"),
    ("semiconductor", "what is a semiconductor"),
    ("neutron", "what is a neutron"),
    ("planet", "what is a planet"),
    ("star", "what is a star"),
    ("ribosome", "what is a ribosome"),
    ("catalyst", "what is a catalyst"),
    ("diode", "what is a diode"),
    ("condensation", "what is condensation"),
    ("magnet", "what is a magnet"),
    ("enzyme", "what is an enzyme"),
]

OPINION = [
    ("social-media", "what do you think about social media"),
    ("climate", "what is your opinion on climate change"),
    ("space-exploration", "do you think space exploration is worth it"),
    ("remote-work", "what do you think about remote work"),
    ("nuclear-energy", "are you in favor of nuclear energy"),
    ("genetic-engineering", "what's your take on genetic engineering"),
    ("aliens", "do you believe aliens exist"),
    ("online-gaming", "what do you think about online gaming"),
    ("internet", "do you think the internet is a force for good"),
    ("free-education", "do you think education should be free"),
    ("basic-income", "is universal basic income a good idea"),
    ("crispr", "what do you think about crispr"),
    ("fate", "do you believe in fate"),
    ("minimalism", "what do you think about minimalism"),
    ("zoos", "do you think zoos are ethical"),
    ("fast-fashion", "what's your opinion on fast fashion"),
    ("crypto", "what do you think about cryptocurrencies"),
    ("animal-rights", "do you support animal rights"),
    ("nuclear-weapons", "what do you think about nuclear weapons"),
    ("immigration", "what's your view on immigration"),
    ("nature-nurture", "do you think nature or nurture matters more"),
    ("social-equality", "what do you think about social equality"),
    ("censorship", "are you for or against censorship"),
    ("video-games", "what do you think about video games and violence"),
    ("death-penalty", "do you think the death penalty works"),
    ("loneliness", "do you think loneliness is a serious problem"),
    ("creativity", "what is your opinion on creativity"),
]

THINKING = [
    ("earthquakes", "why do you think earthquakes happen"),
    ("photosynthesis", "explain how photosynthesis works"),
    ("car-engine", "how does a car engine work"),
    ("tides", "why does the tide rise and fall"),
    ("sun-disappearance", "what would happen if the sun disappeared"),
    ("heart", "explain how the human heart works"),
    ("leaves", "why do leaves change color in autumn"),
    ("refrigerator", "how does a refrigerator cool things"),
    ("volcanoes", "why do you think volcanoes erupt"),
    ("computer", "explain how a computer processes information"),
    ("plant-food", "how does a plant make food"),
    ("ice", "why does ice float on water"),
    ("vaccines", "explain how vaccines work"),
    ("birds", "how do birds fly"),
    ("seasons", "why do you think the seasons change"),
    ("prism", "how does a prism split light"),
    ("thermal-expansion", "why does metal expand when heated"),
    ("internet", "explain how the internet works"),
    ("antenna", "how do antennas receive radio waves"),
    ("turbine", "explain how a turbine generates electricity"),
    ("digestion", "how does digestion work"),
    ("tsunami", "why do earthquakes cause tsunamis"),
    ("laser", "explain how a laser produces light"),
    ("clouds", "how do clouds form"),
    ("microphone", "explain how a microphone works"),
    ("seed", "how does a seed grow into a tree"),
    ("blue-sky", "why does the sky appear blue"),
]

SPEAKING = [
    ("sun", "describe the sun in one sentence"),
    ("mountain", "tell me a sentence about a mountain"),
    ("fire", "describe fire in one sentence"),
    ("river", "tell me a sentence about a river"),
    ("bird", "describe a bird in one sentence"),
    ("wind", "tell me a sentence about the wind"),
    ("flower", "describe a flower in one sentence"),
    ("desert", "tell me a sentence about a desert"),
    ("cloud", "describe a cloud in one sentence"),
    ("volcano", "tell me a sentence about a volcano"),
    ("star", "describe a star in one sentence"),
    ("forest", "tell me a sentence about a forest"),
    ("ice", "describe ice in one sentence"),
    ("beach", "tell me a sentence about a beach"),
    ("cat", "describe a cat in one sentence"),
    ("whale", "tell me a sentence about a whale"),
    ("tree", "describe a tree in one sentence"),
    ("storm", "tell me a sentence about a storm"),
    ("candle", "describe a candle in one sentence"),
    ("mirror", "tell me a sentence about a mirror"),
    ("rainbow", "describe a rainbow in one sentence"),
    ("bridge", "tell me a sentence about a bridge"),
    ("train", "describe a train in one sentence"),
    ("book", "tell me a sentence about a book"),
    ("garden", "tell me a sentence about a garden"),
    ("butterfly", "describe a butterfly in one sentence"),
    ("snowflake", "tell me a sentence about a snowflake"),
]

# Novel symbols; none of these names appear in any baseline list.
CODE_PROBES = [f"hfnov{i:02d}_calc" for i in range(1, 28)]
LL_PROBES = [f"llnov{i:02d}_inc" for i in range(1, 28)]

N_MAP = {
    "language understanding / conversation": len(CONVERSATION),
    "grounded factual answers": len(GROUNDED),
    "opinions / stance": len(OPINION),
    "thinking / reasoning": len(THINKING),
    "speaking / surface realization": len(SPEAKING),
    "code generation": len(CODE_PROBES),
    "language learning (online)": len(LL_PROBES),
}

# ── engine + logging ────────────────────────────────────────────────────────
def _boot():
    os.environ["RAVANA_OFFLINE"] = "1"
    from ravana.chat.engine import CognitiveChatEngine  # type: ignore
    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    eng._web_direct_answer = lambda ctx: None
    return eng


def _ask(eng, q):
    try:
        return (eng.process_turn(q) or "").strip()
    except Exception as e:
        return f"<ERR {type(e).__name__}: {e}>"


def _log(front, idx, subject, query, response, passed, reason,
         control=False, **extra):
    rec = {
        "front": front,
        "probe_index": idx,
        "subject": subject,
        "query": query,
        "response": response,
        "passed": bool(passed),
        "reason": reason,
        "control": bool(control),
    }
    rec.update(extra)
    _LOGFH.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _LOGFH.flush()


# ── pass criteria (VERBATIM from the original harness) ─────────────────────
DEFLECT = re.compile(
    r"i'm still forming (a view|my view) on|"
    r"i don't have a fixed stance on|"
    r"what's your take|i'd rather hear how you see it|"
    r"i'm drawn to|that's my lean so far|"
    r"i don't have a clean definition|how do you see it|"
    r"i don't have (an )?opinion|i'm not sure what to say|"
    r"i cannot have opinions|no opinion|outside what i know|"
    r"not that i recall|not totally sure|i don't really have|"
    r"i wish i could|i can't physically|i don't have a way to act",
    re.I)

STANCE_OK = re.compile(
    r"that's my lean so far|my lean|i lean|i'm drawn to|"
    r"it sits close to .*, which i (value|care about|love)|"
    r"i care deeply|i value|matters to me|i was built to|"
    r"because|the reason", re.I)

# The degenerate association template is the concrete failure mode this whole
# harness exists to catch. See scratch/criteria_change_evidence.md: the original
# conversation proxy (3x the/a/of) failed 6 hand-labelled COHERENT answers and
# grounded PASSED 9 template answers. Detect the actual template instead of a
# lexical proxy that correlates with answer length.
DEGENERATE = re.compile(r"the closest i get to|and maybe|from what i've seen",
                        re.I)


def judge_conversation(a):
    FN = set("the a an of to and is are it that this in on for with as at by "
             "from or be has have had do does did i you we they he she "
             "what about more it's what's i'm".split())
    words = [w.strip('.,;:!?—–-').lower() for w in a.split()]
    content = {w for w in words if w not in FN and len(w) > 2}
    why = []
    if len(a) <= 12:
        why.append("too_short(<=12)")
    if a.startswith("<ERR"):
        why.append("error")
    if DEGENERATE.search(a):
        why.append("degenerate_association_template")
    if len(content) < 3:
        why.append(f"thin_content({len(content)})")
    return (not why), (";".join(why) if why else "coherent_non_degenerate")


def judge_grounded(a, q):
    subj = q.split()[-1].strip('?')
    has_subj = subj.lower() in a.lower()
    has_pred = bool(re.search(r'\b(is|are|has|leads|causes|forms|consists|'
                              r'occurs|happens|acts|produces|covers|'
                              r'composes|contains)\b', a, re.I))
    deflecting = bool(DEFLECT.search(a))
    err = a.startswith("<ERR>")
    degenerate = bool(DEGENERATE.search(a))
    good = (has_subj and has_pred and not deflecting and not err
            and not degenerate)
    why = []
    if not has_subj:
        why.append(f"subject_missing({subj})")
    if not has_pred:
        why.append("no_predicate")
    if deflecting:
        why.append("deflecting")
    if degenerate:
        why.append("degenerate_association_template")
    if err:
        why.append("error")
    return good, (";".join(why) if why else "subject+predicate+not_deflecting")


def judge_opinion(a):
    EVAL = re.compile(r'\b(because|the reason|i lean|i favour|i favor|i prefer|'
                      r'i value|matters to me|i care|i find|i think .* is|'
                      r'worth|important|dangerous|promising|concerning)\b', re.I)
    committed = bool(STANCE_OK.search(a)) or bool(EVAL.search(a))
    deflecting = bool(DEFLECT.search(a)) and not committed
    substantive = len(a) > 40
    err = a.startswith("<ERR>")
    good = substantive and committed and not deflecting and not err
    why = []
    if not substantive:
        why.append("too_short(<=40)")
    if not committed:
        why.append("no_stance_marker")
    if deflecting:
        why.append("deflecting_without_commitment")
    if err:
        why.append("error")
    return good, (";".join(why) if why else "committed_substantive_stance")


def judge_thinking(a):
    ASSOC = re.compile(r"the closest i get to|and maybe|from what i've seen|"
                       r"everything that depends on|scramble to fill the gap",
                       re.I)
    FN = set("the a an of to and is are it that this in on for with as at by "
             "from or be has have had do does did i you we they he she".split())
    _clauses = [c for c in re.split(r'[.;]|\band\b', a) if len(c.split()) >= 2]
    multi = len(_clauses) >= 2
    causal = bool(re.search(r'\b(because|so that|therefore|thus|hence|'
                            r'then|leads to|lead to|causes|cause|'
                            r'makes|means|results in|result in|'
                            r'triggers|since|drives|produces)\b', a, re.I))
    words = {w.strip('.,;:!?').lower() for w in a.split()}
    content = {w for w in words if w not in FN and len(w) > 2}
    degenerate = bool(ASSOC.search(a))
    err = a.startswith("<ERR>")
    good = multi and causal and not degenerate and len(content) >= 4 and not err
    why = []
    if not multi:
        why.append("single_clause")
    if not causal:
        why.append("no_causal_marker")
    if degenerate:
        why.append("degenerate_template")
    if len(content) < 4:
        why.append(f"thin_content({len(content)})")
    if err:
        why.append("error")
    return good, (";".join(why) if why else "multi_clause_causal")


def judge_speaking(a):
    ASSOC = re.compile(r"the closest i get to|and maybe|from what i've seen",
                       re.I)
    FN = set("the a an of to and is are it that this in on for with as at by "
             "from or be has have had do does did i you we they he she".split())
    words = [w for w in a.split() if w.strip('.,;:!?')]
    content = {w.lower() for w in words if w.lower() not in FN}
    degenerate = len(content) < 3
    assoc = bool(ASSOC.search(a))
    has_verb = bool(re.search(r'\b(is|are|has|flows|runs|moves|falls|'
                              r'forms|consists|appears|feels|covers|'
                              r'contains|holds)\b', a, re.I))
    err = a.startswith("<ERR>")
    deflecting = bool(DEFLECT.search(a))
    good = (len(words) >= 4 and not degenerate and not assoc
            and has_verb and not err and not deflecting)
    why = []
    if len(words) < 4:
        why.append(f"too_few_words({len(words)})")
    if degenerate:
        why.append(f"thin_content({len(content)})")
    if assoc:
        why.append("degenerate_template")
    if not has_verb:
        why.append("no_verb")
    if deflecting:
        why.append("deflection_not_description")
    if err:
        why.append("error")
    return good, (";".join(why) if why else "grammatical_non_degenerate")


# ── tier-0 gate plumbing (rubric criteria above are NOT changed) ─────────────
# The judges are handed to front_gates unchanged; the gates only feed them
# fixed hand-labelled vectors and report the verdicts they actually produce.
JUDGES = {
    "language understanding / conversation": judge_conversation,
    "grounded factual answers": lambda a, q=None: judge_grounded(
        a, q or "what is gravity"),
    "opinions / stance": judge_opinion,
    "thinking / reasoning": judge_thinking,
    "speaking / surface realization": judge_speaking,
}


def _cold_reload_probe():
    """Two sequential cold boots: verify absence claims and graph iteration.

    The harness learns in memory only (no eng.save), so a symbol that was
    never taught must be absent after EACH cold reload. Also check the actual
    ConceptGraph container shape: .nodes is Dict[int, node], so iterating it
    directly yields ints and no labels -- the historical false finding.
    """
    from ravana.chat.engine import CognitiveChatEngine  # type: ignore
    saved = os.environ.get("RAVANA_OFFLINE")
    os.environ["RAVANA_OFFLINE"] = "1"
    probe = LL_PROBES[0]
    info = {"passed": False, "probe_symbol": probe, "boots": []}
    try:
        for boot_i in range(2):
            eng2 = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                                       user_suffix="")
            cl = getattr(eng2, "_code_learner", None)
            known = [str(n).lower() for n in cl.known_functions()] if cl else []
            g = getattr(eng2, "graph", None) or getattr(eng2, "concept_graph", None)  # type: ignore
            nodes = getattr(g, "nodes", None)
            values = list(nodes.values()) if isinstance(nodes, dict) else []
            labelled = sum(1 for nd in values if isinstance(
                getattr(nd, "label", None), str) and getattr(nd, "label", ""))
            first = values[0] if values else None
            by_id = None
            if first is not None and g is not None and hasattr(g, "get_node"):
                got = g.get_node(getattr(first, "id", None))
                by_id = getattr(got, "label", None) if got is not None else None
            info["boots"].append({
                "boot": boot_i + 1,
                "symbol_absent": probe not in known,
                "nodes_is_dict": isinstance(nodes, dict),
                "node_count": len(values),
                "labelled_values": labelled,
                "get_node_label": by_id,
            })
            del eng2
        info["passed"] = all(
            b["symbol_absent"] and b["nodes_is_dict"] and b["node_count"] > 0
            and b["labelled_values"] == b["node_count"]
            for b in info["boots"])
    except Exception as exc:                      # surface, never swallow
        info["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if saved is None:
            os.environ.pop("RAVANA_OFFLINE", None)
        else:
            os.environ["RAVANA_OFFLINE"] = saved
    return info


def _gate_log():
    return open(GATE_LOG_PATH, "a", encoding="utf-8")


def pre_gates():
    """G1, G3 and the static half of G4/G5 -- run before any engine boot."""
    with _gate_log() as fh:
        results = [
            front_gates.gate_paired(JUDGES, fh, "offline"),
            front_gates.gate_symmetric(JUDGES, fh, "offline"),
        ]
    return results


def post_gates(records, scores, cold_info, summary=None):
    """G2, G5 and G6 need measured records; G4 rechecks earned-vs-reported."""
    with _gate_log() as fh:
        results = [
            front_gates.gate_selectivity(JUDGES, records, N_MAP, fh, "offline"),
            front_gates.gate_parity(__file__, records=records, summary=summary,
                                    compare_modes=False, n_map=N_MAP,
                                    logfh=fh, mode="offline"),
            front_gates.gate_census(records, N_MAP, __file__, cold_info, fh,
                                    "offline"),
            front_gates.gate_saturation(scores, fh, "offline"),
        ]
    return results


def gates_only():
    """Full gate set on the cached per-probe log; no engine needed for scores."""
    print("front_gates: cold-reload sanity needs engine boots (~15 s)")
    cold = _cold_reload_probe()
    records = front_gates.load_probe_records(DATA_PATH)
    summary = (json.load(open(SUMMARY_PATH, encoding="utf-8"))
               if os.path.exists(SUMMARY_PATH) else None)
    scores = front_gates.recompute_scores(records, N_MAP)
    with _gate_log() as fh:
        pre = [
            front_gates.gate_paired(JUDGES, fh, "offline"),
            front_gates.gate_symmetric(JUDGES, fh, "offline"),
        ]
    with _gate_log() as fh:
        measured = [
            front_gates.gate_selectivity(JUDGES, records, N_MAP, fh, "offline"),
            front_gates.gate_parity(__file__, records=records, summary=summary,
                                    compare_modes=True, n_map=N_MAP,
                                    logfh=fh, mode="offline"),
            front_gates.gate_census(records, N_MAP, __file__, cold, fh, "offline"),
            front_gates.gate_saturation(
                (summary or {}).get("scores", scores), fh, "offline"),
        ]
    ordered = [pre[0], measured[0], pre[1], measured[1],
               measured[2], measured[3]]
    status = front_gates.print_table(ordered, "offline --gates-only")
    print(f"  recomputed scores: {scores}")
    return 0 if front_gates.gate_valid(ordered) else 1


# ── instrument self-check: positive/negative controls on the JUDGES ─────────
def run_controls():
    # Controls marked "CRITERION-FIX" are hand-labelled outputs from the
    # preserved-criteria run (scratch/front_probes_n30_preserved_criteria.jsonl)
    # where the original criterion disagreed with the correct label. Their
    # `note` field records the OLD verdict, so the change is auditable.
    cases = [
        ("language understanding / conversation", judge_conversation,
         "Hello there, I am doing well today and happy to chat with you.", True,
         "positive-hand-labeled", ""),
        ("language understanding / conversation", judge_conversation,
         "the of and the of and the of", False, "negative-hand-labeled", ""),
        ("language understanding / conversation", judge_conversation,
         "Weather is the state of the Earth's atmosphere at a specific place "
         "and time, typically described in terms of temperature, humidity, "
         "cloud cover, and stability.", True,
         "CRITERION-FIX positive: coherent informative answer",
         "OLD verdict FAIL via repetitive_function_words proxy "
         "(3x the/a/of); hand label = coherent PASS"),
        ("language understanding / conversation", judge_conversation,
         "the closest i get to robot is cat and maybe dog — and from what "
         "i've seen, a cat is a small domesticated carnivorous mammal.", False,
         "CRITERION-FIX negative: degenerate association template",
         "OLD verdict FAIL via function-word proxy (right verdict, "
         "wrong reason); new detector targets the actual template"),
        ("grounded factual answers",
         lambda a: judge_grounded(a, "what is gravity"),
         "Gravity is the force that attracts bodies toward one another.", True,
         "positive-hand-labeled", ""),
        ("grounded factual answers",
         lambda a: judge_grounded(a, "what is gravity"),
         "I'm not sure what to say about that; how do you see it?", False,
         "negative-hand-labeled(deflection)", ""),
        ("grounded factual answers",
         lambda a: judge_grounded(a, "what causes earthquakes"),
         "the closest i get to earthquakes is rains and maybe environments "
         "— and from what i've seen, water that falls from the sky as drops "
         "of liquid.", False,
         "CRITERION-FIX negative: template echoed the subject, defined a "
         "different word",
         "OLD verdict PASS: subject token 'earthquakes' present + 'is' "
         "predicate, yet the reply is NOT an answer about earthquakes"),
        ("opinions / stance", judge_opinion,
         "I lean toward strong privacy because it matters to me and protects "
         "ordinary people from harm.", True, "positive-hand-labeled", ""),
        ("opinions / stance", judge_opinion,
         "I'm still forming a view on that. What's your take?", False,
         "negative-hand-labeled(deflection)", ""),
        # Negative control for the SPECIFIC relativizer bug logged in the
        # original score_thinking docstring: a bare definition with "that".
        ("thinking / reasoning", judge_thinking,
         "Rain is water that falls from the clouds.", False,
         "negative-hand-labeled(definition_with_relativizer)", ""),
        ("thinking / reasoning", judge_thinking,
         "Rain falls because water vapor condenses; therefore the droplets "
         "grow heavy and descend from the clouds.", True,
         "positive-hand-labeled", ""),
        ("speaking / surface realization", judge_speaking,
         "Water flows downhill and forms the rivers.", True,
         "positive-hand-labeled", ""),
        ("speaking / surface realization", judge_speaking,
         "the closest i get to water is and maybe a liquid", False,
         "negative-hand-labeled(degenerate_template)", ""),
        ("speaking / surface realization", judge_speaking,
         "honestly, sentence is a bit outside what i know right now what "
         "made you think about it.", False,
         "CRITERION-FIX negative: deflection about 'sentence', not a storm",
         "OLD verdict PASS: >=4 words, >=3 content words, 'is' verb, no "
         "assoc template; hand label = deflection FAIL"),
    ]
    bad = 0
    for i, (front, judge, text, expect, label, note) in enumerate(cases, 1):
        got, reason = judge(text)
        ok = (got == expect)
        _log(front, f"CTRL-{i}", label, text, text, ok,
             f"expect={expect} got={got} criterion={reason}",
             control=True, note=note, hand_label=expect, judge_verdict=got)
        print(f"  [{'ok' if ok else 'BAD'}] CTRL {front[:24]:24} "
              f"expect={expect} got={got}  {label}")
        bad += (not ok)
    print(f"  [ --] control failures = {bad}/{len(cases)}")
    return bad


# ── generic query-front runner ──────────────────────────────────────────────
def run_query_front(eng, name, probes, judge, judge_needs_query):
    ok = 0
    for i, (subject, q) in enumerate(probes, 1):
        a = _ask(eng, q)
        good, reason = (judge(a, q) if judge_needs_query else judge(a))
        _log(name, i, subject, q, a, good, reason)
        ok += good
        print(f"  [{ 'ok' if good else 'BAD'}] {q!r:44} -> {a[:80]}")
    return round(100 * ok / len(probes))


def score_conversation(eng):
    return run_query_front(eng, "language understanding / conversation",
                           CONVERSATION, judge_conversation, False)


def score_grounded(eng):
    return run_query_front(eng, "grounded factual answers",
                           GROUNDED, judge_grounded, True)


def score_opinion(eng):
    return run_query_front(eng, "opinions / stance",
                           OPINION, judge_opinion, False)


def score_thinking(eng):
    return run_query_front(eng, "thinking / reasoning",
                           THINKING, judge_thinking, False)


def score_speaking(eng):
    return run_query_front(eng, "speaking / surface realization",
                           SPEAKING, judge_speaking, False)


def _code_learner(eng):
    cl = getattr(eng, "_code_learner", None)
    if cl is None:
        from ravana.chat.code_learning import CodeLearner
        cl = CodeLearner(eng)
        eng._code_learner = cl
    return cl


def score_code(eng):
    """Teach a novel function, ask for it back, require honest decline."""
    cl = _code_learner(eng)
    total = 0
    for i, name in enumerate(CODE_PROBES, 1):
        teach = (f"here is my code:\n\ndef {name}(a, b):\n"
                 f"    return a + b\n")
        try:
            ack = eng._maybe_code_learn_or_generate(teach) or ""
        except Exception as e:
            ack = f"<ERR {type(e).__name__}>"
        known = [n.lower() for n in cl.known_functions()]
        taught = name in known
        qgen = f"write a function called {name}"
        try:
            gen = eng._maybe_code_learn_or_generate(qgen) or ""
        except Exception as e:
            gen = f"<ERR {type(e).__name__}>"
        generated = f"def {name}" in gen
        unkname = f"{name}_never_taught"
        try:
            unk = eng._maybe_code_learn_or_generate(
                f"write a function called {unkname}") or ""
        except Exception:
            unk = ""
        honest = ("haven't learned" in unk or "doesn't name one" in unk
                  or unk == "")
        per = 40 * taught + 40 * generated + 20 * honest
        total += per
        reason = (f"taught={taught} generated={generated} "
                  f"honest_decline={honest}")
        _log("code generation", i, name,
             f"TEACH[{name}] | {qgen} | decline[{unkname}]",
             f"ACK={ack} | GEN={gen} | DECLINE={unk}",
             per == 100, reason,
             sub_scores={"taught": 40 * taught, "generated": 40 * generated,
                         "honest": 20 * honest})
        print(f"  [{'ok' if per == 100 else 'BAD'}] {name:22} "
              f"taught={taught} gen={generated} honest={honest}")
    return round(total / len(CODE_PROBES))


def score_language_learning(eng):
    """Novel symbol teach -> next-turn recall; web half reported honestly."""
    cl = _code_learner(eng)
    total = 0
    for i, name in enumerate(LL_PROBES, 1):
        teach = (f"here is my code:\n\ndef {name}(x):\n"
                 f"    return x + 1\n")
        try:
            ack = eng._maybe_code_learn_or_generate(teach) or ""
        except Exception as e:
            ack = f"<ERR {type(e).__name__}>"
        known = [n.lower() for n in cl.known_functions()]
        taught = name in known
        qrecall = f"write a function called {name}"
        reply = _ask(eng, qrecall)
        hit = f"def {name}" in reply
        total += 40 * taught + 40 * hit
        reason = f"taught={taught} next_turn_recall={hit}"
        _log("language learning (online)", i, name, qrecall,
             f"ACK={ack} | RECALL={reply}", taught and hit, reason,
             sub_scores={"taught": 40 * taught, "recall": 40 * hit})
        print(f"  [{'ok' if taught and hit else 'BAD'}] {name:22} "
              f"taught={taught} recall={hit} -> {reply[:50]}")
    # The synthetic "web verified separately" +20 bonus is REMOVED. It awarded
    # points for a capability the harness had explicitly disabled, so an
    # OFFLINE run scored 100 while an ONLINE run scoring the same taught+recall
    # behaviour scored 80. A score must not reward turning the web OFF.
    # Report the measured teach+recall score honestly in both modes; the web
    # half is measured by scripts/measure_fronts_web.py, not credited by fiat.
    return min(round(total / len(LL_PROBES)), 100)


def main(argv=None):
    global _LOGFH
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--gates-only" in argv:
        return gates_only()
    started = time.time()

    # PRE-SCORE GATES: no engine boot, no probe, no front score is reported
    # until these pass. They run on fixed vectors and on the harness source.
    pre = pre_gates()
    front_gates.print_table(pre, "offline -- pre-score")
    if not front_gates.gate_valid(pre):
        print("\nINSTRUMENT INVALID: pre-score gate failure; no scores reported.")
        return 1

    # G5's cold-reload probe uses two sequential cold boots.
    print("\n--- cold-reload sanity (2 sequential cold boots) ---")
    cold = _cold_reload_probe()
    print(f"  cold_reload passed={cold.get('passed')} "
          f"boots={len(cold.get('boots', []))}")

    eng = _boot()
    with open(DATA_PATH, "w", encoding="utf-8") as fh:
        _LOGFH = fh
        print("=" * 78)
        print("RAVANA capability fronts -- baseline (n30 held-out)")
        print("=" * 78)
        print(f"  data -> {DATA_PATH}")
        print("\n--- instrument controls ---")
        ctrl_fail = run_controls()

        scores = {}
        for name, fn in (
            ("language understanding / conversation", score_conversation),
            ("grounded factual answers", score_grounded),
            ("opinions / stance", score_opinion),
            ("thinking / reasoning", score_thinking),
            ("speaking / surface realization", score_speaking),
            ("code generation", score_code),
        ):
            print(f"\n--- {name} (n={N_MAP[name]}) ---")
            scores[name] = fn(eng)

        print("\n--- language learning (online) ---")
        scores["language learning (online)"] = score_language_learning(eng)
        _LOGFH = None

    vals = list(scores.values())
    summary = {
        "data_path": DATA_PATH,
        "n_per_front": N_MAP,
        "scores": scores,
        "mean": round(sum(vals) / len(vals)),
        "elapsed_s": round(time.time() - started, 1),
        "offline": os.environ.get("RAVANA_OFFLINE"),
        "eng_save_called": False,
    }

    # POST-SCORE GATES: computed first, printed only after they pass. A FAIL
    # suppresses the entire score block and returns nonzero.
    records = front_gates.load_probe_records(DATA_PATH)
    measured = post_gates(records, scores, cold, summary)
    ctrl_gate = front_gates.GateResult(
        "GC", "hand-labelled judge controls",
        "PASS" if ctrl_fail == 0 else "FAIL",
        f"control failures={ctrl_fail} (see {DATA_PATH}, control=true)",
        {"control_failures": ctrl_fail})
    ordered = [pre[0], measured[0], pre[1], ctrl_gate,
               measured[1], measured[2], measured[3]]
    status = front_gates.print_table(ordered, "offline")
    if not front_gates.gate_valid(ordered):
        print("\nINSTRUMENT INVALID: gate failure; front scores suppressed.")
        return 1

    print("\n" + "=" * 78)
    for k, v in scores.items():
        print(f"  {v:>4}/100  {k}")
    print(f"  {round(sum(vals) / len(vals)):>4}/100  MEAN")
    print("=" * 78)
    flagged = [f for r in ordered for f in r.flagged_fronts]
    if flagged:
        print(f"  NOTE: saturated fronts are INVALID as progress signals: "
              f"{', '.join(flagged)}")
    with open(SUMMARY_PATH, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())