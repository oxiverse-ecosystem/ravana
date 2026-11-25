"""Real-engine check of the default-off typed slot primitives on main."""
import os, shutil, sys
P="C:/Users/Likhith/Documents/projects/ravana"
for p in (P,P+"/scripts",P+"/ravana/src",P+"/ravana_ml/src"):
    if p not in sys.path: sys.path.insert(0,p)
os.environ["RAVANA_OFFLINE"]="1"; os.environ["RAVANA_TRA_SLOTS"]="1"
os.environ["PYTHONHASHSEED"]="0"
from ravana.chat.engine import CognitiveChatEngine
from ravana.chat import schema_induction as si
D=P+"/scratch/tra_prod_check_data"
if os.path.exists(D): shutil.rmtree(D)
os.makedirs(D, exist_ok=True)
eng=CognitiveChatEngine(data_dir=D,user_suffix="")
assertions=["mist is a weather phenomenon.", "election is a civic event.",
            "comet is a celestial event.", "wheel is part of a vehicle.",
            "chapter is part of a book.", "sneeze is a reflex action."]
for t in assertions:
    eng.process_turn(t)
print("taught",len(assertions),flush=True)
views=si.typed_slot_views_from_graph(eng.graph)
print("in_process_views",len(views),flush=True)
eng.save()
reloaded=CognitiveChatEngine(data_dir=D,user_suffix="")
views=si.typed_slot_views_from_graph(reloaded.graph)
print("cold_views",len(views),flush=True)
for v in views[:6]:
    print("VIEW",getattr(v,'subject',None),getattr(v,'type_label',None),getattr(v,'confidence',None),flush=True)
q="Considering the examples above, which type applies to mist? Options: weather phenomenon, civic event, celestial event, animal. Answer briefly."
r=si.slot_keyed_forced_choice(views,"mist",["weather phenomenon","civic event","celestial event","animal"])
print("PICK",r,flush=True)
d=si.slot_keyed_forced_choice(views,"mist",["apple","banana","chair","table"])
print("DISJOINT",d,flush=True)
print("HOOK",reloaded._typed_slot_candidate(q,"mist"),flush=True)
