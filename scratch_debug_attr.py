import re

q = "remind me what you said about board games"
_q = q.lower()

# Attribute guard regex from the D3 fix
attr_match = bool(re.search(
    r"\b(my|mine|me|i|we|our|myself)\b", _q)
    and re.search(
        r"\b(name|named|called|age|live|lives|from|work|study|"
        r"studied|grew up|sister|brother|mother|father|mom|dad|"
        r"family|relative|kin|grandmother|grandfather|grandma|"
        r"grandpa|aunt|uncle|cousin|niece|nephew|grandchild|"
        r"wife|husband|partner|kid|child|son|daughter|pet|"
        r"friend|job|house|home|phone|computer|laptop|car|"
        r"cat|dog|crow|bird|fish|petname|pet name)\b", _q))
print(f"Attribute guard matches: {attr_match}")

# Check if "me" matches the first regex
first = bool(re.search(r"\b(my|mine|me|i|we|our|myself)\b", _q))
print(f"First regex (1st person): {first}")
# "me" is in the query!

# Check second regex
second = bool(re.search(
    r"\b(name|named|called|age|live|lives|from|work|study|"
    r"studied|grew up|sister|brother|mother|father|mom|dad|"
    r"family|relative|kin|grandmother|grandfather|grandma|"
    r"grandpa|aunt|uncle|cousin|niece|nephew|grandchild|"
    r"wife|husband|partner|kid|child|son|daughter|pet|"
    r"friend|job|house|home|phone|computer|laptop|car|"
    r"cat|dog|crow|bird|fish|petname|pet name)\b", _q))
print(f"Second regex (user attr): {second}")
# "board" is not in the list, "games" is not in the list
# So the attribute guard should NOT match
