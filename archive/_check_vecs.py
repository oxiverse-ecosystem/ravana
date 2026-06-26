"""Quick check which words have GloVe vectors."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ravana', 'src'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from ravana_chat import CognitiveChatEngine
import warnings
warnings.filterwarnings('ignore')
eng = CognitiveChatEngine(baby_mode=True, dim=64)
for w in ['causal', 'contrastive', 'hypothetical', 'semantic', 'analogical', 'temporal', 'cause', 'because', 'contrast', 'explain', 'describe']:
    v = eng._glove_vector(w)
    ok = "YES" if v is not None else "NO"
    dim = len(v) if v is not None else "N/A"
    print(f"{w}: {ok} (dim={dim})")
