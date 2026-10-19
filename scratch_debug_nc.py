import re

# Test if "ive started to hate it" matches the contradiction signal
q = "my neighbors dog barks every morning at 6am and ive started to hate it"
_nc_signal = bool(re.search(
    r"\b(actually|on\s+second\s+thought|i\s+changed\s+my\s+mind|"
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
    q.lower()))
print(f"Contradiction signal for 'ive started to hate it': {_nc_signal}")

# Check what part matches
m = re.search(
    r"\b(actually|on\s+second\s+thought|i\s+changed\s+my\s+mind|"
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
    q.lower())
if m:
    print(f"MATCHED: {m.group()!r} at position {m.start()}-{m.end()}")

# Also check "changed my mind — silence is actually wonderful"
q2 = "changed my mind — silence is actually wonderful"
_nc2 = bool(re.search(
    r"\b(actually|on\s+second\s+thought|i\s+changed\s+my\s+mind|"
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
    q2.lower()))
print(f"Contradiction signal for 'changed my mind — silence is actually wonderful': {_nc2}")
m2 = re.search(
    r"\b(actually|on\s+second\s+thought|i\s+changed\s+my\s+mind|"
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
    q2.lower())
if m2:
    print(f"MATCHED: {m2.group()!r}")
