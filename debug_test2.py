import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)

from ravana.chat.relation_attrs import relation_of, learn_relation

# Check what relation_of returns for key words
print("relation_of('first'):", relation_of("first"))
print("relation_of('mentor'):", relation_of("mentor"))
print("relation_of('friend'):", relation_of("friend"))
print("relation_of('old'):", relation_of("old"))
print("relation_of('dear'):", relation_of("dear"))
print("relation_of('late'):", relation_of("late"))

# Check _KIN set
from ravana.chat.user_model import UserModel
um = UserModel()
print("_KIN:", hasattr(um, '_KIN'))
# Check if _KIN is a class attribute or instance attribute
import inspect
for name, val in inspect.getmembers(UserModel):
    if name == '_KIN':
        print("_KIN at class level:", val)
        break
