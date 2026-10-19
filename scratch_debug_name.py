import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src'):
    sys.path.insert(0, p)
from ravana.chat import attribute_gate

# Test asks_name_only
for q in ["what is my dog called", "what is my dog's name", "what is my dog name", "what does my dog do"]:
    print(f"{q!r}: asks_name_only={attribute_gate.asks_name_only(q)}")
