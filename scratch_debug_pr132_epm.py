import os, sys, io, contextlib, tempfile, shutil
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

def _build():
    d = tempfile.mkdtemp(prefix="ravana_contradict_")
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d), d

eng, d = _build()
eng.process_turn("i really love street art, especially big murals on warehouse walls")

# Test _mine_new_opinion_polarity directly
query = "i saw a mural downtown that was just tagged over a local business's sign — changed my mind, not all street art is good"
result = eng._mine_new_opinion_polarity(query, "street art")
print("_mine_new_opinion_polarity result:", result)

# Test EPM directly
um = eng.user_model
epm = um._ensure_evaluative_polarity()
if epm:
    print("EPM score for 'good':", epm.score("good"))
    print("EPM score for 'street art':", epm.score("street art"))
else:
    print("EPM is None")

# Check what the EPM finds in the post-contradiction clause
import re
t = query.lower().strip()
_nc = re.search(
    r"\b(?:actually|on\s+second\s+thought|i\s+changed\s+my\s+mind|"
    r"(?:i\s+)?changed\s+my\s+mind|"
    r"i\s+was\s+wrong|i\s+take\s+it\s+back|never\s+mind|"
    r"i\s+don'?t\s+think\s+so\s+anymore|"
    r"i\s+don'?t\s+(?:really\s+)?(?:like|love|enjoy)\s+(?:it|that|this)\s+anymore|"
    r"i\s+(?:actually\s+)?(?:hate|dislike|loathe|detest|can'?t\s+stand)|"
    r"i\s+was\s+(?:too\s+)?(?:hasty|quick|wrong)|"
    r"i\s+think\s+differently\s+now|"
    r"i'?ve\s+(?:changed|reversed)\s+(?:my\s+)?(?:mind|position|stance)|"
    r"i\s+(?:take|am\s+taking)\s+it\s+all\s+back|"
    r"i\s+don'?t\s+feel\s+that\s+way\s+anymore)\b",
    t)
if _nc:
    _rest = t[_nc.end():].strip()
    _rest = re.sub(
        r"^(?:[—–\-:]|i\s+(?:think\s+|feel\s+|now\s+)?|"
        r"about\s+|on\s+|regarding\s+|that\s+|"
        r"i\s+(?:now\s+)?(?:think\s+|feel\s+|"
        r"am\s+(?:now\s+)?|have\s+)?|"
        r"i\s+changed\s+my\s+mind\s*|"
        r"i\s+take\s+(?:it|that)\s+back\s*|"
        r"i\s+was\s+wrong\s*)\s*",
        "", _rest).strip()
    print("Post-contradiction clause:", repr(_rest))
    
    # Check what the EPM regex finds
    for _m in re.finditer(
            r"\b([a-z][a-z'\-]{2,})\s+(?:is|are|was|were)\s+"
            r"(?:very\s+|really\s+|quite\s+|so\s+|too\s+|"
            r"pretty\s+|highly\s+|absolutely\s+|completely\s+|"
            r"rather\s+|fairly\s+|extremely\s+|actually\s+|"
            r"just\s+|simply\s+|totally\s+|utterly\s+|"
            r"incredibly\s+|somewhat\s+)?"
            r"([a-z][a-z'\-]{2,})\b", _rest):
        print(f"  EPM match: subject={_m.group(1)!r}, predicate={_m.group(2)!r}")
