"""Guard: PRONOUNS_FALLBACK identity map must never be restored.

The map was an identity map (every value == its own key) whose 7 keys are all
short-circuited by the guard at _resolve_pronoun, making the read site
unreachable for every key the map held. Deleted as dead code.
"""
import sys, os
os.environ["RAVANA_OFFLINE"] = "1"

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJ, os.path.join(PROJ, "ravana", "src"),
          os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

from ravana.language.surface_realizer import SurfaceRealizer, DiscourseState


def test_pronouns_fallback_deleted():
    sr = SurfaceRealizer.__new__(SurfaceRealizer)
    assert not hasattr(sr, "PRONOUNS_FALLBACK"), \
        "PRONOUNS_FALLBACK must not be restored"


def test_resolve_pronoun_still_returns_identity():
    """The guard at _resolve_pronoun must still return subject verbatim."""
    sr = SurfaceRealizer.__new__(SurfaceRealizer)
    ds = DiscourseState()
    for pron in ("i", "you", "we", "they", "he", "she", "it"):
        result = sr._resolve_pronoun(pron, pron, ds)
        assert result == pron, f"_resolve_pronoun({pron!r}) = {result!r}"


if __name__ == "__main__":
    test_pronouns_fallback_deleted()
    test_resolve_pronoun_still_returns_identity()
    print("PASS: PRONOUNS_FALLBACK guard tests")
