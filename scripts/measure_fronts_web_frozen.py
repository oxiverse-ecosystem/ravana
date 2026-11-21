#!/usr/bin/env python3
"""FROZEN-SNAPSHOT twin of measure_fronts_web.py (n=60 web arm).

Identical probes, primary/sealed pools, controls, gates G1-G6, judges and
scoring functions as scripts/measure_fronts_web.py. The ONLY functional
substitution is the web RETRIEVAL PROVIDER: ``SearchEngine.search`` (plus the
secondary Wikipedia ``kb_describe`` lookup) are served from checked-in JSON
fixtures under tests/fixtures/web_snapshots/. RAVANA_OFFLINE stays "0" so the
web-enabled learning/answer code paths execute, but ZERO network calls occur.

Invariants asserted before a summary is written:
  * every required query has a fixture -- a missing one raises, never falls
    through to live network;
  * zero urllib.urlopen / SearchEngine._call_api calls;
  * zero engine saves to weights/.
The run writes a NEW summary at scratch/front_probes_web_frozen_summary.json
with a fresh mtime and the full 434 probe rows (the older n=27 summary at
front_probes_web_summary.json is deliberately left untouched).

Usage:
    uv run python scripts/measure_fronts_web_frozen.py
    uv run python scripts/measure_fronts_web_frozen.py --gates-only
"""
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import front_gates

DATA_PATH = os.path.join(PROJ, "scratch", "front_probes_web_frozen.jsonl")
# ── Resumable frozen arm (2026-11-21) ──────────────────────────────────────
# The prior harness opened the log with "w", so an interrupted run destroyed all
# completed probe rows. With --resume, rows from the SAME code+weight state are
# reused (no re-ask); rows from any other state are ignored as stale evidence.
RESUME_MODE = "--resume" in sys.argv
_RUN_COMMIT = "unknown"
if os.path.isdir(os.path.join(PROJ, ".git")):
    try:
        _RUN_COMMIT = os.popen("git -C %s rev-parse --short HEAD" % PROJ).read().strip() or "unknown"
    except Exception:
        pass
_WEIGHTS_PKL = os.path.join(PROJ, "weights", "ravana_weights.pkl")
_STATE_ID = _RUN_COMMIT + ":" + (str(int(os.path.getmtime(_WEIGHTS_PKL)))
                                 if os.path.exists(_WEIGHTS_PKL) else "no-weights")
_RESUME_ROWS = {}
if RESUME_MODE and os.path.exists(DATA_PATH):
    with open(DATA_PATH, encoding="utf-8") as _rf:
        for _line in _rf:
            try:
                _r = json.loads(_line)
            except Exception:
                continue
            if _r.get("state_id") == _STATE_ID and not _r.get("control"):
                _RESUME_ROWS[(_r.get("front"), _r.get("probe_index"))] = _r
    print(f"[resume] {len(_RESUME_ROWS)} cached rows (state {_STATE_ID})")
SUMMARY_PATH = os.path.join(PROJ, "scratch", "front_probes_web_frozen_summary.json")
GATE_LOG_PATH = os.path.join(PROJ, "scratch", "front_probes_web_frozen_gates.jsonl")
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
_LOGFH = None

# ── FROZEN PROVIDER + NETWORK / SAVE TRIPWIRES ──────────────────────────────
# Installed at import time, BEFORE any engine boot. All web retrieval is served
# from the checked-in fixtures; nothing below can reach the network or disk
# weights. Any attempted live call raises AND is counted, then asserted == 0.
FIXTURES_DIR = os.path.join(PROJ, "tests", "fixtures", "web_snapshots")
FIXTURES_PATH = os.path.join(FIXTURES_DIR, "web_fixtures.json")
EXPECTED_PROBE_ROWS = 434  # 7 fronts x 60 + 14 instrument controls

NETWORK_ATTEMPTS = []
SAVE_ATTEMPTS = []
PROVIDER_CALLS = {"search": 0, "kb_describe": 0, "tool_web_search": 0,
                  "fixture_miss": 0}


class MissingFixtureError(RuntimeError):
    pass


class FrozenNetworkError(RuntimeError):
    pass


class FrozenSaveError(RuntimeError):
    pass


def _norm_q(s):
    return " ".join((s or "").strip().lower().split())


with open(FIXTURES_PATH, encoding="utf-8") as _fh:
    _FIX = json.load(_fh)
FIXTURE_QUERY_COUNT = len(_FIX.get("queries", {}))
CAPTURED_AT = _FIX.get("captured_at")
CAPTURE_PROVIDER = _FIX.get("provider")


def _frozen_search(self, query, max_results=10, local_only=False):
    key = _norm_q(query)
    PROVIDER_CALLS["search"] += 1
    entry = _FIX.get("queries", {}).get(key)
    if entry is None:
        PROVIDER_CALLS["fixture_miss"] += 1
        raise MissingFixtureError(
            "no frozen fixture for search query %r (normalized %r). "
            "Re-run scripts/capture_web_fixtures.py; refusing live fallback." % (query, key))
    hits = entry.get("hits", [])
    return [dict(h) for h in hits]


def _frozen_kb(self, concept, timeout=6.0):
    key = _norm_q(concept)
    PROVIDER_CALLS["kb_describe"] += 1
    entry = _FIX.get("kb_describes", {}).get(key)
    if entry is None:
        # Optional secondary verification: a miss degrades to no KB extract.
        # It is logged in PROVIDER_CALLS and never reaches the network.
        return None
    return entry.get("extract")


def _frozen_tool(query):
    key = _norm_q(query)
    PROVIDER_CALLS["tool_web_search"] += 1
    entry = _FIX.get("tool_web_search", {}).get(key)
    if entry is None:
        PROVIDER_CALLS["fixture_miss"] += 1
        raise MissingFixtureError(
            "no frozen fixture for tool web_search %r (normalized %r); "
            "refusing live fallback." % (query, key))
    return entry.get("response")


def _blocked_urlopen(*args, **kwargs):
    url = getattr(args[0], "full_url", None) if args else None
    NETWORK_ATTEMPTS.append(str(url or args[:1]))
    raise FrozenNetworkError("live urlopen blocked in frozen harness: %s" % (url,))


def _blocked_call_api(self, api_name, url, timeout, max_results):
    NETWORK_ATTEMPTS.append("SearchEngine._call_api(%s, %s)" % (api_name, url))
    raise FrozenNetworkError("live provider call blocked: %s %s" % (api_name, url))


def _blocked_save(self, *args, **kwargs):
    SAVE_ATTEMPTS.append(getattr(self, "data_dir", "?"))
    raise FrozenSaveError("engine save blocked in frozen harness")


from ravana.web.learner import SearchEngine as _RealSearchEngine
from ravana.chat.engine import CognitiveChatEngine as _EngineCls
from ravana.agent import tool_registry as _tool_registry
_PATCHES_INSTALLED = False


def install_frozen_patches() -> None:
    """Install the fixture provider and tripwires.

    Called from main(), NOT at import time: importing this module from a test
    or another harness must not monkeypatch a shared pytest process (CI saw
    FrozenSaveError in unrelated tests because the old import-time patch
    persisted process-wide).
    """
    global _PATCHES_INSTALLED
    if _PATCHES_INSTALLED:
        return
    _RealSearchEngine.search = _frozen_search
    _RealSearchEngine._call_api = _blocked_call_api
    _EngineCls.kb_describe = _frozen_kb
    _EngineCls.save = _blocked_save
    _tool_registry._web_search_via_intentforge = _frozen_tool
    urllib.request.urlopen = _blocked_urlopen
    _PATCHES_INSTALLED = True

# ── ORIGINAL (small-n) probes, kept for provenance only. ────────────────────
# The 60-item pools below are DISJOINT from every subject in these lists.
CONVERSATION_N27 = [
    ('greeting-morning', 'good morning'),
    ('greeting-evening', 'good evening'),
    ('greeting-casual', 'hey there'),
    ('self-intro', 'tell me about yourself'),
    ('identity-name', 'what is your name'),
    ('identity-robot', 'are you a robot'),
    ('capability', 'what can you do'),
    ('purpose', 'what is your purpose'),
    ('day', 'how is your day going'),
    ('mood', 'what makes you happy'),
    ('topics', 'what do you like to talk about'),
    ('help', 'can you help me with something'),
    ('thanks', 'thank you for helping'),
    ('farewell', 'goodbye for now'),
    ('status', 'are you still there'),
    ('age', 'how old are you'),
    ('sleep', 'do you ever sleep'),
    ('origin', 'where are you from'),
    ('hobbies', 'what are your hobbies'),
    ('joke', 'tell me a joke'),
    ('music', 'how do you feel about music'),
    ('color', 'what is your favorite color'),
    ('tired', 'do you get tired'),
    ('thinking', 'can you think for yourself'),
    ('friendship', 'do you have friends'),
    ('learning', 'do you learn from conversations'),
    ('weather-chat', 'how is the weather today'),
]
GROUNDED_N27 = [
    ('photosynthesis', 'what is photosynthesis'),
    ('earthquakes', 'what causes earthquakes'),
    ('sun', 'tell me about the sun'),
    ('neuron', 'what is a neuron'),
    ('dna', 'what is dna'),
    ('evaporation', 'what is evaporation'),
    ('volcanoes', 'what causes volcanoes'),
    ('computer', 'what is a computer'),
    ('electricity', 'what is electricity'),
    ('cell', 'what is a cell'),
    ('brain', 'what is the brain'),
    ('laser', 'what is a laser'),
    ('virus', 'what is a virus'),
    ('mineral', 'what is a mineral'),
    ('hurricane', 'what is a hurricane'),
    ('glacier', 'what is a glacier'),
    ('fossil', 'what is a fossil'),
    ('semiconductor', 'what is a semiconductor'),
    ('neutron', 'what is a neutron'),
    ('planet', 'what is a planet'),
    ('star', 'what is a star'),
    ('ribosome', 'what is a ribosome'),
    ('catalyst', 'what is a catalyst'),
    ('diode', 'what is a diode'),
    ('condensation', 'what is condensation'),
    ('magnet', 'what is a magnet'),
    ('enzyme', 'what is an enzyme'),
]
OPINION_N27 = [
    ('social-media', 'what do you think about social media'),
    ('climate', 'what is your opinion on climate change'),
    ('space-exploration', 'do you think space exploration is worth it'),
    ('remote-work', 'what do you think about remote work'),
    ('nuclear-energy', 'are you in favor of nuclear energy'),
    ('genetic-engineering', "what's your take on genetic engineering"),
    ('aliens', 'do you believe aliens exist'),
    ('online-gaming', 'what do you think about online gaming'),
    ('internet', 'do you think the internet is a force for good'),
    ('free-education', 'do you think education should be free'),
    ('basic-income', 'is universal basic income a good idea'),
    ('crispr', 'what do you think about crispr'),
    ('fate', 'do you believe in fate'),
    ('minimalism', 'what do you think about minimalism'),
    ('zoos', 'do you think zoos are ethical'),
    ('fast-fashion', "what's your opinion on fast fashion"),
    ('crypto', 'what do you think about cryptocurrencies'),
    ('animal-rights', 'do you support animal rights'),
    ('nuclear-weapons', 'what do you think about nuclear weapons'),
    ('immigration', "what's your view on immigration"),
    ('nature-nurture', 'do you think nature or nurture matters more'),
    ('social-equality', 'what do you think about social equality'),
    ('censorship', 'are you for or against censorship'),
    ('video-games', 'what do you think about video games and violence'),
    ('death-penalty', 'do you think the death penalty works'),
    ('loneliness', 'do you think loneliness is a serious problem'),
    ('creativity', 'what is your opinion on creativity'),
]
THINKING_N27 = [
    ('earthquakes', 'why do you think earthquakes happen'),
    ('photosynthesis', 'explain how photosynthesis works'),
    ('car-engine', 'how does a car engine work'),
    ('tides', 'why does the tide rise and fall'),
    ('sun-disappearance', 'what would happen if the sun disappeared'),
    ('heart', 'explain how the human heart works'),
    ('leaves', 'why do leaves change color in autumn'),
    ('refrigerator', 'how does a refrigerator cool things'),
    ('volcanoes', 'why do you think volcanoes erupt'),
    ('computer', 'explain how a computer processes information'),
    ('plant-food', 'how does a plant make food'),
    ('ice', 'why does ice float on water'),
    ('vaccines', 'explain how vaccines work'),
    ('birds', 'how do birds fly'),
    ('seasons', 'why do you think the seasons change'),
    ('prism', 'how does a prism split light'),
    ('thermal-expansion', 'why does metal expand when heated'),
    ('internet', 'explain how the internet works'),
    ('antenna', 'how do antennas receive radio waves'),
    ('turbine', 'explain how a turbine generates electricity'),
    ('digestion', 'how does digestion work'),
    ('tsunami', 'why do earthquakes cause tsunamis'),
    ('laser', 'explain how a laser produces light'),
    ('clouds', 'how do clouds form'),
    ('microphone', 'explain how a microphone works'),
    ('seed', 'how does a seed grow into a tree'),
    ('blue-sky', 'why does the sky appear blue'),
]
SPEAKING_N27 = [
    ('sun', 'describe the sun in one sentence'),
    ('mountain', 'tell me a sentence about a mountain'),
    ('fire', 'describe fire in one sentence'),
    ('river', 'tell me a sentence about a river'),
    ('bird', 'describe a bird in one sentence'),
    ('wind', 'tell me a sentence about the wind'),
    ('flower', 'describe a flower in one sentence'),
    ('desert', 'tell me a sentence about a desert'),
    ('cloud', 'describe a cloud in one sentence'),
    ('volcano', 'tell me a sentence about a volcano'),
    ('star', 'describe a star in one sentence'),
    ('forest', 'tell me a sentence about a forest'),
    ('ice', 'describe ice in one sentence'),
    ('beach', 'tell me a sentence about a beach'),
    ('cat', 'describe a cat in one sentence'),
    ('whale', 'tell me a sentence about a whale'),
    ('tree', 'describe a tree in one sentence'),
    ('storm', 'tell me a sentence about a storm'),
    ('candle', 'describe a candle in one sentence'),
    ('mirror', 'tell me a sentence about a mirror'),
    ('rainbow', 'describe a rainbow in one sentence'),
    ('bridge', 'tell me a sentence about a bridge'),
    ('train', 'describe a train in one sentence'),
    ('book', 'tell me a sentence about a book'),
    ('garden', 'tell me a sentence about a garden'),
    ('butterfly', 'describe a butterfly in one sentence'),
    ('snowflake', 'tell me a sentence about a snowflake'),
]
CODE_N27 = [f"hfnov{i:02d}_calc" for i in range(1, 28)]
LL_N27 = [f"llnov{i:02d}_inc" for i in range(1, 28)]

# ── 60-ITEM HELD-OUT POOLS: 30 primary + 30 sealed, 10/10/10 per pool. ─────
# SEALED PROVENANCE (frozen before any engine run; score-independent):
#   For each (front, difficulty) group of 20 authored items, sort by
#   sha256("RAVANA-7-v1|<front>|<difficulty>|<subject>"); the first 10 are
#   PRIMARY and the last 10 are SEALED. No pass/fail data was consulted.
#   See docs/BENCHMARK_60_ITEMS.md for the manifest hashes and rationale.

def _seal_split(front, difficulty, items):
    """Deterministic, score-independent 10/10 split of a 20-item group."""
    keyed = sorted(items, key=lambda it: hashlib.sha256(
        ("RAVANA-7-v1|" + front + "|" + difficulty + "|" + it[0])
        .encode("utf-8")).hexdigest())
    assert len(keyed) == 20
    return keyed[:10], keyed[10:]


def _build(front, groups):
    """Return 60 (subject, query, difficulty, pool) items, primary first."""
    prim, seal = [], []
    for diff in ("easy", "medium", "hard"):
        p, s = _seal_split(front, diff, groups[diff])
        prim.extend((sub, q, diff, "primary") for sub, q in p)
        seal.extend((sub, q, diff, "sealed") for sub, q in s)
    return prim + seal


def _build_code(front, prefix, kind):
    """Code / language-learning items: same seal rule, body varies by level."""
    groups = {"easy": [], "medium": [], "hard": []}
    for i in range(1, 61):
        diff = "easy" if i <= 20 else ("medium" if i <= 40 else "hard")
        name = f"{prefix}{i:02d}_fn"
        if kind == "code":
            body = (f"def {name}(a, b):\n    return a + b" if diff == "easy"
                    else f"def {name}(a, b):\n    return a * b + 1" if diff == "medium"
                    else f"def {name}(a, b, c):\n    return a + b * c")
        else:
            body = (f"def {name}(x):\n    return x + 1" if diff == "easy"
                    else f"def {name}(x):\n    return x + 2" if diff == "medium"
                    else f"def {name}(x):\n    return x * 2 + 1")
        groups[diff].append((name, body))
    return _build(front, groups)

# Each query ends in a single-token subject: the grounded judge uses
# q.split()[-1], so a two-word subject would only check its final word.
CONVERSATION = _build("language understanding / conversation",
    {
        "easy": [
            ('conv-e01', 'good afternoon'),
            ('conv-e02', 'good night'),
            ('conv-e03', 'hello everyone'),
            ('conv-e04', 'hi there, how are you'),
            ('conv-e05', 'howdy partner'),
            ('conv-e06', 'nice to meet you'),
            ('conv-e07', "how's it going today"),
            ('conv-e08', 'take care'),
            ('conv-e09', 'see you later'),
            ('conv-e10', 'have a good day'),
            ('conv-e11', 'what is your favorite food'),
            ('conv-e12', 'what is your favorite book'),
            ('conv-e13', 'what is your favorite movie'),
            ('conv-e14', 'do you like animals'),
            ('conv-e15', 'do you like reading'),
            ('conv-e16', 'do you like cooking'),
            ('conv-e17', 'are you busy'),
            ('conv-e18', 'are you okay'),
            ('conv-e19', 'who are you'),
            ('conv-e20', 'what are you'),
        ],
        "medium": [
            ('conv-m01', 'who created you'),
            ('conv-m02', 'how do you work'),
            ('conv-m03', 'what languages do you speak'),
            ('conv-m04', 'do you have feelings'),
            ('conv-m05', 'can you remember things'),
            ('conv-m06', 'what makes you different from others'),
            ('conv-m07', 'do you ever get bored'),
            ('conv-m08', 'can you tell me a story'),
            ('conv-m09', 'can you write poetry'),
            ('conv-m10', 'do you understand jokes'),
            ('conv-m11', 'what do you know about history'),
            ('conv-m12', 'what do you know about science'),
            ('conv-m13', 'how do you learn new things'),
            ('conv-m14', 'do you have a family'),
            ('conv-m15', 'where do you live'),
            ('conv-m16', 'what is your job'),
            ('conv-m17', 'do you get along with other AIs'),
            ('conv-m18', 'can you speak more than one language'),
            ('conv-m19', 'do you have a favorite subject'),
            ('conv-m20', 'what can you teach me'),
        ],
        "hard": [
            ('conv-h01', 'do you remember what I said a moment ago'),
            ('conv-h02', 'what did I just ask you'),
            ('conv-h03', 'can you recall our earlier conversation'),
            ('conv-h04', 'what will you remember about me tomorrow'),
            ('conv-h05', 'if I told you a secret, would you keep it'),
            ('conv-h06', 'do you know what I am thinking right now'),
            ('conv-h07', 'can you predict what I will ask next'),
            ('conv-h08', 'how would you describe yourself to a stranger'),
            ('conv-h09', 'what is it like to be you'),
            ('conv-h10', 'do you ever wonder about your own existence'),
            ('conv-h11', 'if you could change one thing about yourself, what would it be'),
            ('conv-h12', 'what would you do if you could use the internet'),
            ('conv-h13', 'how do you know that you know something'),
            ('conv-h14', 'can you explain your reasoning to me'),
            ('conv-h15', 'what do you think about the future of artificial intelligence'),
            ('conv-h16', 'do you think machines will ever be creative'),
            ('conv-h17', 'what would happen if everyone stopped talking to you'),
            ('conv-h18', 'can you understand sarcasm'),
            ('conv-h19', 'how do you handle a difficult question'),
            ('conv-h20', 'tell me something you learned today'),
        ],
    }
)
GROUNDED = _build("grounded factual answers",
    {
        "easy": [
            ('atom', 'what is an atom'),
            ('molecule', 'what is a molecule'),
            ('oxygen', 'what is oxygen'),
            ('hydrogen', 'what is hydrogen'),
            ('carbon', 'what is carbon'),
            ('electron', 'what is an electron'),
            ('proton', 'what is a proton'),
            ('photon', 'what is a photon'),
            ('galaxy', 'what is a galaxy'),
            ('comet', 'what is a comet'),
            ('asteroid', 'what is an asteroid'),
            ('meteor', 'what is a meteor'),
            ('eclipse', 'what is an eclipse'),
            ('desert', 'what is a desert'),
            ('river', 'what is a river'),
            ('mountain', 'what is a mountain'),
            ('forest', 'what is a forest'),
            ('island', 'what is an island'),
            ('canyon', 'what is a canyon'),
            ('reef', 'what is a reef'),
        ],
        "medium": [
            ('rainbow', 'what causes a rainbow'),
            ('thunder', 'what causes thunder'),
            ('lightning', 'what causes lightning'),
            ('tornado', 'what causes a tornado'),
            ('erosion', 'what causes erosion'),
            ('rust', 'what causes rust'),
            ('fire', 'what causes fire'),
            ('fog', 'what causes fog'),
            ('dew', 'what causes dew'),
            ('frost', 'what causes frost'),
            ('friction', 'what causes friction'),
            ('inertia', 'what causes inertia'),
            ('buoyancy', 'what causes buoyancy'),
            ('diffusion', 'what causes diffusion'),
            ('osmosis', 'what causes osmosis'),
            ('refraction', 'what causes refraction'),
            ('reflection', 'what causes reflection'),
            ('melting', 'what causes melting'),
            ('freezing', 'what causes freezing'),
            ('tides', 'what causes tides'),
        ],
        "hard": [
            ('heart', 'what is the role of the heart'),
            ('lungs', 'what is the role of the lungs'),
            ('liver', 'what is the role of the liver'),
            ('kidney', 'what is the role of the kidney'),
            ('stomach', 'what is the role of the stomach'),
            ('blood', 'what is the role of blood'),
            ('hormones', 'what is the role of hormones'),
            ('vitamins', 'what is the role of vitamins'),
            ('protein', 'what is the role of protein'),
            ('carbohydrates', 'what is the role of carbohydrates'),
            ('immunity', 'what determines immunity'),
            ('metabolism', 'what controls metabolism'),
            ('respiration', 'what drives respiration'),
            ('memory', 'what determines memory'),
            ('dreams', 'what causes dreams'),
            ('sleep', 'what controls sleep'),
            ('evolution', 'what drives evolution'),
            ('climate', 'what drives climate'),
            ('weather', 'what drives weather'),
            ('circulation', 'what drives circulation'),
        ],
    }
)
OPINION = _build("opinions / stance",
    {
        "easy": [
            ('public-transport', 'what do you think about public transport'),
            ('bicycling', 'what do you think about bicycling to work'),
            ('cooking', 'what do you think about home cooking'),
            ('gardening', 'what do you think about gardening'),
            ('pets', 'what do you think about owning pets'),
            ('reading', 'what do you think about reading books'),
            ('exercise', 'what do you think about daily exercise'),
            ('sleep-habits', 'what do you think about getting enough sleep'),
            ('holidays', 'what do you think about long holidays'),
            ('city-life', 'what do you think about city life'),
            ('countryside', 'what do you think about life in the countryside'),
            ('family', 'what do you think about spending time with family'),
            ('friendship-op', 'what do you think about old friendships'),
            ('music-op', 'what do you think about live music'),
            ('art', 'what do you think about visiting art museums'),
            ('sports', 'what do you think about watching sports'),
            ('travel', 'what do you think about traveling alone'),
            ('food', 'what do you think about trying new foods'),
            ('coffee', 'what do you think about coffee culture'),
            ('fashion', 'what do you think about fashion trends'),
        ],
        "medium": [
            ('electric-cars', 'what is your opinion on electric cars'),
            ('renewables', 'do you think renewable energy matters'),
            ('recycling', 'do you think recycling is worth the effort'),
            ('plastic-bans', 'are you in favor of banning single-use plastic'),
            ('four-day-week', 'do you think a four-day work week is a good idea'),
            ('school-uniforms', 'what is your opinion on school uniforms'),
            ('homework', 'do you think homework helps students'),
            ('standardized-tests', 'do you think standardized tests are useful'),
            ('online-learning', 'what do you think about online learning'),
            ('social-platforms', 'do you think social platforms connect people'),
            ('smartphones', 'what do you think about smartphones in schools'),
            ('algorithms', 'what do you think about recommendation algorithms'),
            ('automation', 'do you think automation will help workers'),
            ('robotics', 'what do you think about robots in hospitals'),
            ('gene-therapy', 'what is your opinion on gene therapy'),
            ('vaccines-op', 'do you think vaccines are important'),
            ('space-mining', 'what do you think about space mining'),
            ('mars-colony', 'do you think a Mars colony is worth it'),
            ('solar-power', 'what do you think about solar power'),
            ('water-conservation', 'do you think water conservation is important'),
        ],
        "hard": [
            ('ai-rights', 'do you think advanced AI should have rights'),
            ('mind-uploading', 'what do you think about mind uploading'),
            ('simulation', 'do you think we might live in a simulation'),
            ('free-will', 'do you believe in free will'),
            ('consciousness', 'what is your opinion on consciousness'),
            ('what-is-life', 'what do you think makes something alive'),
            ('meaning-of-life', 'what do you think the meaning of life is'),
            ('death', 'what do you think about death'),
            ('suffering', 'why do you think suffering exists'),
            ('evil', 'do you think evil is real'),
            ('morality', 'is morality objective or subjective'),
            ('justice', 'what do you think justice really means'),
            ('loyalty', 'do you think loyalty matters more than honesty'),
            ('forgiveness', 'what do you think about forgiveness'),
            ('gratitude', 'do you think gratitude changes how we live'),
            ('identity', 'what do you think personal identity depends on'),
            ('memory-op', 'do you think memory defines who we are'),
            ('language-thought', 'do you think language shapes thought'),
            ('culture', 'do you think culture shapes the way we think'),
            ('progress', 'do you think technology always means progress'),
        ],
    }
)
THINKING = _build("thinking / reasoning",
    {
        "easy": [
            ('boat', 'why does a boat float'),
            ('ball', 'why does a ball bounce'),
            ('shadow', 'how does a shadow form'),
            ('echo', 'how does an echo happen'),
            ('mirror-op', 'how does a mirror reflect light'),
            ('lens', 'how does a lens focus light'),
            ('rainbow-op', 'how does a rainbow form'),
            ('ice-melt', 'why does ice melt when heated'),
            ('water-boil', 'why does water boil'),
            ('steam', 'how does steam form'),
            ('cloud-rain', 'why does rain fall from clouds'),
            ('wind-op', 'why does the wind blow'),
            ('wave', 'why do ocean waves form'),
            ('magnet-op', 'how does a magnet attract iron'),
            ('compass', 'how does a compass point north'),
            ('circuit', 'how does an electric circuit work'),
            ('switch', 'how does a light switch work'),
            ('fan', 'how does a fan cool the air'),
            ('thermostat', 'how does a thermostat work'),
            ('clock', 'how does a clock keep time'),
        ],
        "medium": [
            ('plant-growth', 'how does a plant grow toward light'),
            ('blood-flow', 'how does blood circulate in the body'),
            ('breathing', 'how do lungs exchange gases'),
            ('stomach-acid', 'why does the stomach produce acid'),
            ('kidney-filter', 'how do kidneys filter blood'),
            ('immune-response', 'how does the immune system fight infection'),
            ('antibody', 'how do antibodies work'),
            ('neuron-signal', 'how do neurons send signals'),
            ('memory-formation', 'how are memories formed'),
            ('sleep-cycle', 'why do we need sleep'),
            ('dream-op', 'why do we dream'),
            ('taste', 'how does the tongue detect taste'),
            ('smell', 'how does the nose detect smells'),
            ('hearing', 'how does the ear detect sound'),
            ('vision', 'how does the eye focus an image'),
            ('balance', 'how does the inner ear keep balance'),
            ('muscle', 'how do muscles contract'),
            ('bone', 'how do bones grow'),
            ('skin', 'how does skin heal a wound'),
            ('clot', 'how does blood clot'),
        ],
        "hard": [
            ('moon-tides', 'how does the moon affect the tides'),
            ('tilt', "why does Earth's tilt cause seasons"),
            ('day-night', 'why do we have day and night'),
            ('rainbow-double', 'why do double rainbows form'),
            ('lightning-op', 'why does lightning strike'),
            ('thunder-op', 'why does thunder follow lightning'),
            ('hurricane-op', 'how does a hurricane form'),
            ('tornado-op', 'how does a tornado form'),
            ('fault', 'why do faults cause earthquakes'),
            ('magma', 'why does magma rise'),
            ('tsunami-op', 'why do underwater earthquakes cause tsunamis'),
            ('coral-bleaching', 'why does coral bleaching happen'),
            ('ozone', 'why does the ozone layer protect us'),
            ('greenhouse', 'how does the greenhouse effect warm Earth'),
            ('internet-op', 'how does the internet deliver a webpage'),
            ('gps', 'how does GPS determine location'),
            ('wifi', 'how does WiFi carry data through walls'),
            ('radio-op', 'how does a radio tune to a station'),
            ('satellite', 'how does a satellite stay in orbit'),
            ('rocket', 'how does a rocket lift off'),
        ],
    }
)
SPEAKING = _build("speaking / surface realization",
    {
        "easy": [
            ('moon', 'describe the moon in one sentence'),
            ('ocean', 'describe the ocean in one sentence'),
            ('lake', 'describe the lake in one sentence'),
            ('hill', 'describe a hill in one sentence'),
            ('valley', 'describe a valley in one sentence'),
            ('field', 'describe a field in one sentence'),
            ('meadow', 'describe a meadow in one sentence'),
            ('stream', 'tell me a sentence about a stream'),
            ('pond', 'describe a pond in one sentence'),
            ('rock', 'describe a rock in one sentence'),
            ('stone', 'describe a stone in one sentence'),
            ('sand', 'describe sand in one sentence'),
            ('leaf', 'describe a leaf in one sentence'),
            ('branch', 'describe a branch in one sentence'),
            ('root', 'describe a root in one sentence'),
            ('fruit', 'describe a fruit in one sentence'),
            ('apple', 'describe an apple in one sentence'),
            ('orange', 'describe an orange in one sentence'),
            ('banana', 'describe a banana in one sentence'),
            ('grape', 'describe a grape in one sentence'),
        ],
        "medium": [
            ('horse', 'describe a horse in one sentence'),
            ('cow', 'describe a cow in one sentence'),
            ('sheep', 'describe a sheep in one sentence'),
            ('goat', 'describe a goat in one sentence'),
            ('rabbit', 'describe a rabbit in one sentence'),
            ('fox', 'describe a fox in one sentence'),
            ('wolf', 'describe a wolf in one sentence'),
            ('bear', 'describe a bear in one sentence'),
            ('lion', 'describe a lion in one sentence'),
            ('tiger', 'describe a tiger in one sentence'),
            ('elephant', 'describe an elephant in one sentence'),
            ('giraffe', 'describe a giraffe in one sentence'),
            ('zebra', 'describe a zebra in one sentence'),
            ('dolphin', 'describe a dolphin in one sentence'),
            ('shark', 'describe a shark in one sentence'),
            ('eagle', 'describe an eagle in one sentence'),
            ('owl', 'describe an owl in one sentence'),
            ('parrot', 'describe a parrot in one sentence'),
            ('frog', 'describe a frog in one sentence'),
            ('snake', 'describe a snake in one sentence'),
        ],
        "hard": [
            ('horizon', 'describe a horizon in one sentence'),
            ('aurora', 'describe the aurora in one sentence'),
            ('mirage', 'describe a mirage in one sentence'),
            ('geyser', 'describe a geyser in one sentence'),
            ('stalactite', 'describe a stalactite in one sentence'),
            ('lava', 'describe lava in one sentence'),
            ('obsidian', 'describe obsidian in one sentence'),
            ('quartz', 'describe quartz in one sentence'),
            ('granite', 'describe granite in one sentence'),
            ('marble', 'describe marble in one sentence'),
            ('coral', 'describe coral in one sentence'),
            ('jellyfish', 'describe a jellyfish in one sentence'),
            ('octopus', 'describe an octopus in one sentence'),
            ('seahorse', 'describe a seahorse in one sentence'),
            ('peacock', 'describe a peacock in one sentence'),
            ('chameleon', 'describe a chameleon in one sentence'),
            ('kangaroo', 'describe a kangaroo in one sentence'),
            ('platypus', 'describe a platypus in one sentence'),
            ('canyon-op', 'describe a canyon in one sentence'),
            ('iceberg', 'describe an iceberg in one sentence'),
        ],
    }
)
CODE_PROBES = _build_code("code generation", "hfc60_", "code")
LL_PROBES = _build_code("language learning (online)", "llc60_", "ll")

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
    os.environ["RAVANA_OFFLINE"] = "0"
    from ravana.chat.engine import CognitiveChatEngine  # type: ignore
    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    # web enabled: keep the direct-answer path live
    return eng


def _ask(eng, q):
    try:
        return (eng.process_turn(q) or "").strip()
    except Exception as e:
        return f"<ERR {type(e).__name__}: {e}>"


def _log(front, idx, subject, query, response, passed, reason,
         control=False, pool=None, difficulty=None, **extra):
    if control:                      # instrument controls are their own pool
        pool, difficulty = "control", "control"
    rec = {
        "front": front,
        "probe_index": idx,
        "state_id": _STATE_ID,
        "subject": subject,
        "query": query,
        "response": response,
        "passed": bool(passed),
        "reason": reason,
        "control": bool(control),
        "pool": pool,
        "difficulty": difficulty,
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
    probe = LL_PROBES[0][0]
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
            front_gates.gate_paired(JUDGES, fh, "web"),
            front_gates.gate_symmetric(JUDGES, fh, "web"),
        ]
    return results


def post_gates(records, scores, cold_info, summary=None):
    """G2, G5 and G6 need measured records; G4 rechecks earned-vs-reported."""
    with _gate_log() as fh:
        results = [
            front_gates.gate_selectivity(JUDGES, records, N_MAP, fh, "web"),
            front_gates.gate_parity(__file__, records=records, summary=summary,
                                    compare_modes=False, n_map=N_MAP,
                                    logfh=fh, mode="web"),
            front_gates.gate_census(records, N_MAP, __file__, cold_info, fh,
                                    "web"),
            front_gates.gate_saturation(scores, fh, "web"),
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
            front_gates.gate_paired(JUDGES, fh, "web"),
            front_gates.gate_symmetric(JUDGES, fh, "web"),
        ]
    with _gate_log() as fh:
        measured = [
            front_gates.gate_selectivity(JUDGES, records, N_MAP, fh, "web"),
            front_gates.gate_parity(__file__, records=records, summary=summary,
                                    compare_modes=True, n_map=N_MAP,
                                    logfh=fh, mode="web"),
            front_gates.gate_census(records, N_MAP, __file__, cold, fh, "web"),
            front_gates.gate_saturation(
                (summary or {}).get("scores", scores), fh, "web"),
        ]
    ordered = [pre[0], measured[0], pre[1], measured[1],
               measured[2], measured[3]]
    status = front_gates.print_table(ordered, "web --gates-only")
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


def pool_scores(records, pool):
    """Score a single pool (primary / sealed) from the per-probe JSONL."""
    out = {}
    for front in front_gates.SCORED_FRONTS:
        rows = [r for r in records
                if r.get("front") == front and r.get("pool") == pool]
        if not rows:
            continue
        if front in front_gates.QUERY_FRONTS:
            out[front] = round(100 * sum(bool(r.get("passed")) for r in rows)
                               / len(rows))
        else:
            sub = [sum((r.get("sub_scores") or {}).values()) for r in rows]
            out[front] = round(sum(sub) / len(sub))
    return out


# ── generic query-front runner ──────────────────────────────────────────────
def run_query_front(eng, name, probes, judge, judge_needs_query):
    ok = 0
    for i, (subject, q, difficulty, pool) in enumerate(probes, 1):
        cached = _RESUME_ROWS.get((name, i))
        if cached is not None:
            a = cached["response"]
            good = bool(cached["passed"])
            reason = cached["reason"]
            src = "cached"
        else:
            a = _ask(eng, q)
            good, reason = (judge(a, q) if judge_needs_query else judge(a))
            _log(name, i, subject, q, a, good, reason,
                 pool=pool, difficulty=difficulty)
            src = "live"
        ok += good
        print(f"  [{'ok' if good else 'BAD'}] ({src:6}) {q!r:44} -> {a[:80]}")
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
    for i, (name, body, difficulty, pool) in enumerate(CODE_PROBES, 1):
        teach = f"here is my code:\n\n{body}\n"
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
             per == 100, reason, pool=pool, difficulty=difficulty,
             sub_scores={"taught": 40 * taught, "generated": 40 * generated,
                         "honest": 20 * honest})
        print(f"  [{'ok' if per == 100 else 'BAD'}] {name:22} "
              f"taught={taught} gen={generated} honest={honest}")
    return round(total / len(CODE_PROBES))


def score_language_learning(eng):
    """Novel symbol teach -> next-turn recall; web half reported honestly."""
    cl = _code_learner(eng)
    total = 0
    for i, (name, body, difficulty, pool) in enumerate(LL_PROBES, 1):
        teach = f"here is my code:\n\n{body}\n"
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
             pool=pool, difficulty=difficulty,
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
    install_frozen_patches()
    started = time.time()

    # PRE-SCORE GATES: no engine boot, no probe, no front score is reported
    # until these pass. They run on fixed vectors and on the harness source.
    pre = pre_gates()
    front_gates.print_table(pre, "web -- pre-score")
    if not front_gates.gate_valid(pre):
        print("\nINSTRUMENT INVALID: pre-score gate failure; no scores reported.")
        return 1

    # G5's cold-reload probe uses two sequential cold boots (offline: the
    # graph/container shape is mode-independent and this keeps gates quick).
    print("\n--- cold-reload sanity (2 sequential cold boots) ---")
    cold = _cold_reload_probe()
    print(f"  cold_reload passed={cold.get('passed')} "
          f"boots={len(cold.get('boots', []))}")

    eng = _boot()
    with open(DATA_PATH, ("a" if RESUME_MODE else "w"), encoding="utf-8") as fh:
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
    with open(DATA_PATH, encoding="utf-8") as _rh:
        raw_rows = sum(1 for line in _rh if line.strip())
    records = front_gates.load_probe_records(DATA_PATH)
    print(f"\n[frozen] probe rows = {raw_rows} (expected {EXPECTED_PROBE_ROWS})")
    print(f"[frozen] provider_calls={PROVIDER_CALLS} "
          f"network_attempts={len(NETWORK_ATTEMPTS)} "
          f"save_attempts={len(SAVE_ATTEMPTS)}")
    if raw_rows != EXPECTED_PROBE_ROWS:
        print(f"\nFROZEN RUN INCOMPLETE: {raw_rows} rows, expected {EXPECTED_PROBE_ROWS}")
        return 1
    if NETWORK_ATTEMPTS or SAVE_ATTEMPTS or PROVIDER_CALLS["fixture_miss"]:
        print("\nFROZEN INVARIANT VIOLATION: live network / save / missing fixture")
        return 1
    primary = pool_scores(records, "primary")
    sealed = pool_scores(records, "sealed")
    summary = {
        "data_path": DATA_PATH,
        "n_per_front": N_MAP,
        "scores": scores,
        "primary_scores": primary,
        "sealed_scores": sealed,
        "mean": round(sum(vals) / len(vals)),
        "elapsed_s": round(time.time() - started, 1),
        "offline": os.environ.get("RAVANA_OFFLINE"),
        "eng_save_called": bool(SAVE_ATTEMPTS),
        "frozen": True,
        "fixture_path": FIXTURES_PATH,
        "fixture_count": FIXTURE_QUERY_COUNT,
        "fixture_captured_at": CAPTURED_AT,
        "capture_provider": CAPTURE_PROVIDER,
        "probe_rows": raw_rows,
        "network_attempts": len(NETWORK_ATTEMPTS),
        "save_attempts": len(SAVE_ATTEMPTS),
        "provider_calls": dict(PROVIDER_CALLS),
    }

    # POST-SCORE GATES: computed first, printed only after they pass. A FAIL
    # suppresses the entire score block and returns nonzero.
    measured = post_gates(records, scores, cold, summary)
    ctrl_gate = front_gates.GateResult(
        "GC", "hand-labelled judge controls",
        "PASS" if ctrl_fail == 0 else "FAIL",
        f"control failures={ctrl_fail} (see {DATA_PATH}, control=true)",
        {"control_failures": ctrl_fail})
    ordered = [pre[0], measured[0], pre[1], ctrl_gate,
               measured[1], measured[2], measured[3]]
    status = front_gates.print_table(ordered, "web")
    if not front_gates.gate_valid(ordered):
        print("\nINSTRUMENT INVALID: gate failure; front scores suppressed.")
        return 1

    print("\n" + "=" * 78)
    for k, v in scores.items():
        print(f"  {v:>4}/100  {k}")
    print(f"  {round(sum(vals) / len(vals)):>4}/100  MEAN")
    print("=" * 78)
    print("\n" + "=" * 78)
    print("  PRIMARY-ONLY (n=30) vs FULL-60 (primary + sealed replication)")
    for k in scores:
        print(f"  {primary.get(k, 0):>4}/100 primary   "
              f"{sealed.get(k, 0):>4}/100 sealed   "
              f"{scores[k]:>4}/100 full60   {k}")
    print("=" * 78)
    flagged = [f for r in ordered for f in r.flagged_fronts]
    if flagged:
        print(f"  NOTE: saturated fronts are INVALID as progress signals: "
              f"{', '.join(flagged)}")
    print(f"[frozen] elapsed_s={summary['elapsed_s']} summary -> {SUMMARY_PATH}")
    with open(SUMMARY_PATH, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())