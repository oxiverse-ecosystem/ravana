import os, sys, re
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src', f'{PROJ}\\ravana-v2\\src'):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_rfix01')
eng.process_turn('i think board games are a great way to bond with friends')

# Check what's in _own_replies
store = getattr(eng, '_own_replies', {})
print("STORE KEYS:", list(store.keys()))
for k, entries in store.items():
    for e in entries:
        print(f"  KEY={k!r} src_tokens={e.get('src_tokens', [])} text={e.get('text', '')[:80]!r}")

# Check utterance log
log = getattr(eng, '_utterance_log', [])
print("\nUTTERANCE LOG:")
for turn_num, reply_text in log:
    print(f"  turn={turn_num} reply={reply_text[:80]!r}")

# Now trace the query
q = "remind me what you said about board games"
_q = q.lower()
_stop = {
    "what", "did", "do", "you", "your", "yourself", "say", "said", "says",
    "tell", "told", "telling", "me", "about", "earlier", "before", "again",
    "something", "anything", "the", "a", "an", "is", "are", "was", "were",
    "have", "has", "had", "i", "my", "we", "our", "it", "this", "that",
    "form", "formed", "opinion", "think", "feel", "feel", "mention",
    "mentioned", "remember", "recall", "remind", "answer", "answered",
    "reply", "replied", "state", "stated", "still", "now", "then", "how",
    "why", "who", "when", "where", "which", "any", "some", "thing",
    "things", "yes", "no", "ask", "asked", "wonder", "wondering",
    "tellme",
}
_TAIL_SCAFFOLD_REC = {
    "most", "behind", "would", "view", "start", "change", "first",
    "live", "does", "will", "been", "conversation", "good", "really",
    "something", "anything", "thing", "things", "everything",
    "nothing", "me", "myself", "yourself", "itself", "them", "they",
    "us", "one", "way", "else", "rather", "instead", "that", "this",
    "these", "those", "ever", "even", "also", "too", "back",
}
_cands = [w for w in re.findall(r"[a-z']+", _q)
          if len(w) >= 3 and w not in _stop and w not in _TAIL_SCAFFOLD_REC]
print("\nCANDS:", _cands)
_q_set = set(_cands)
print("Q_SET:", _q_set)

# Score against store
for k, entries in store.items():
    for e in entries:
        _src = set(e.get("src_tokens", []) or [])
        _ov = len(_q_set & _src)
        print(f"  STORE KEY={k!r} overlap={_ov} src={_src}")

# Score against utterance log
for turn_num, reply_text in log:
    _reply_tokens = set(
        w for w in re.findall(r"[a-z']+", reply_text.lower())
        if len(w) >= 3 and w not in _stop)
    _ov = len(_q_set & _reply_tokens)
    print(f"  LOG turn={turn_num} overlap={_ov} tokens={_reply_tokens}")

eng.stop_background_learning()
