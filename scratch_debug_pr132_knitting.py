import os, sys, io, contextlib, tempfile, shutil
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

def _new_engine(suffix):
    import glob, os as _os
    for f in (glob.glob(f"weights/ravana_weights{suffix}.pkl")
              + glob.glob(f"weights/ravana_usermodel{suffix}.pkl")):
        _os.remove(f)
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)

eng = _new_engine("test_dA_debug2")
with contextlib.redirect_stdout(io.StringIO()):
    eng.process_turn("i love knitting, it's the most calming thing in the world.")

# Test _mine_new_opinion_polarity for the knitting contradiction
query = "actually, you know what, i've come to hate knitting — it just makes my hands ache."
result = eng._mine_new_opinion_polarity(query, "knitting")
print("_mine_new_opinion_polarity for knitting:", result)

# Check what the post-contradiction clause looks like
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
