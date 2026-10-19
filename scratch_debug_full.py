import os, sys, re
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src', f'{PROJ}\\ravana-v2\\src'):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_topic_gr')
eng.process_turn('i love open source software')
eng.process_turn('what do you think about open source?')

q = "do you ever feel curious about open source?"
t = q.lower().strip()

# Check self-introspection gate
_self_introspect = re.search(
    r"\b(your|you)\b.*\b(read|line|take|view|mind|thinking|thought|"
    r"opinion|stance|self|who you are|what you (?:are|were)|how you "
    r"(?:see|feel|think)|want|wants|wanted|desire|desires|aim|aims|"
    r"goal|goals|hope|hopes|said|told|spoke|mentioned|recall|remember"
    r")\b", t)
print("SELF_INTROSPECT:", _self_introspect)

# Check if it matches the opinion guard
if _self_introspect and re.search(
        r"\b(your|you)\s+(?:honest\s+)?(take|view|opinion|stance|"
        r"read|thoughts?|think|thinking|thought)\s+"
        r"(on|about|of|regarding|toward)\b", t):
    print("OPINION GUARD: matches -> self_introspect = None")
    _self_introspect = None

# Check epistemic humility
if re.search(r"\b(do|does|did|would|could)\s+you\s+(know|think|believe|understand)\s+(everything|it\s+all|all\s+of\s+it)\b", t) \
   or re.search(r"\bwhat\s+(don'?t|do\s+not)\s+you\s+know\b", t) \
   or re.search(r"\bwhat\s+do\s+you\s+wish\s+you\s+(knew|knew\s+more\s+about)\b", t) \
   or re.search(r"\bhow\s+(much|well)\s+do\s+you\s+(know|understand)\b", t) \
   or re.search(r"\bare\s+you\s+(sure|certain|confident)\b", t) \
   or re.search(r"\bdo\s+you\s+(ever|sometimes)\s+(not\s+know|get\s+things\s+wrong)\b", t):
    print("EPISTEMIC_HUMILITY: matches")

# Check agent opinion
from ravana.chat.engine_self_query import _SELF_OPINION_SHAPE
_agent_opinion = re.search(_SELF_OPINION_SHAPE, t)
print("AGENT_OPINION:", _agent_opinion)

# Check self-experience
_self_exp = re.search(r"\b(what do you dream about|what are you afraid of|what makes you feel|do you ever feel|do you ever think|do you ever wonder)\b", t)
print("SELF_EXPERIENCE:", _self_exp)

# Check what _route_self_experience returns
result = eng._route_self_experience(q, t)
print("ROUTE_SELF_EXPERIENCE:", repr(result))

# Check what _route_self_query returns
result2 = eng._route_self_query(q)
print("ROUTE_SELF_QUERY:", repr(result2))

eng.stop_background_learning()
