import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src'):
    sys.path.insert(0, p)

from ravana.chat.user_model import UserModel
from ravana.chat.slot_naming import trailing_modifiers

um = UserModel()
tm = trailing_modifiers()
opinion_stop = um._OPINION_STOP

# Find trailing modifiers that are also in _OPINION_STOP
overlap = tm & opinion_stop
print("Trailing modifiers also in _OPINION_STOP:")
for w in sorted(overlap):
    print(f"  {w}")

# Test _opinion_topic for each
print("\n_opinion_topic results for these words:")
for w in sorted(overlap):
    result = um._opinion_topic(w)
    print(f"  _opinion_topic('{w}'): {result}")
