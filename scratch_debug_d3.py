import re

q = "what did i just tell you about my dog"
_q = q.lower()

# Attribute guard regex from the branch
first = bool(re.search(r"\b(my|mine|me|i|we|our|myself)\b", _q))
second = bool(re.search(
    r"\b(name|named|called|age|live|lives|from|work|study|"
    r"studied|grew up|sister|brother|mother|father|mom|dad|"
    r"family|relative|kin|grandmother|grandfather|grandma|"
    r"grandpa|aunt|uncle|cousin|niece|nephew|grandchild|"
    r"wife|husband|partner|kid|child|son|daughter|pet|"
    r"friend|job|house|home|phone|computer|laptop|car|"
    r"cat|dog|crow|bird|fish|petname|pet name)\b", _q))
print(f"First regex (1st person): {first}")
print(f"Second regex (user attr): {second}")
print(f"Attribute guard matches: {first and second}")

# Check what the _is_first_person_recall regex matches
fp_match = bool(re.search(
    r"\b(what did i (?:just )?(?:say|tell|said|told|mention|mean)"
    r"|what did i say about|what did i tell you about|"
    r"remind me what you said|what did i say about|"
    r"what was it i told you|what did i just tell you about)\b", _q))
print(f"First-person recall regex: {fp_match}")
