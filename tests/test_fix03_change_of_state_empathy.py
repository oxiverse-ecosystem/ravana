"""FIX-RV-03 test: Emotional disclosures in non-standard syntactic frames
(infinitive-after-change-of-state, "ive" contraction) route to empathy."""
import os, sys
os.environ["RAVANA_OFFLINE"] = "1"

PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
sys.path.insert(0, PROJ)
sys.path.insert(0, f"{PROJ}\ravana\src")
sys.path.insert(0, f"{PROJ}\ravana_ml\src")
sys.path.insert(0, f"{PROJ}\ravana-v2\src")

from ravana.chat.engine import CognitiveChatEngine


def test_fix03_infinitive_disclosure_routes_to_empathy():
    """Change-of-state + infinitive affect: 'started to hate', 'began to fear',
    'grown to love' — these are genuine emotional disclosures, not factual
    statements. They MUST route to empathy (not bare 'noted' ack)."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="fix03test")

    # Primary case from the card
    r = eng.process_turn("my neighbors dog barks every morning at 6am and ive started to hate it")
    assert "noted" not in r.lower(), f"Got bare ack: {r}"
    assert any(w in r.lower() for w in ("rough", "hard", "feeling", "here", "what", "hate")), \
        f"Expected empathy, got: {r}"

    # Additional change-of-state infinitive frames
    r2 = eng.process_turn("i've begun to fear the future")
    assert "noted" not in r2.lower(), f"Got bare ack: {r2}"

    r3 = eng.process_turn("she's grown to love the quiet mornings")
    assert "noted" not in r3.lower(), f"Got bare ack: {r3}"

    # Verify it still works for simple disclosures too
    r4 = eng.process_turn("i hate the noise")
    assert "noted" not in r4.lower(), f"Got bare ack: {r4}"

    eng.stop_background_learning()
    print("FIX-RV-03 test PASSED")


if __name__ == "__main__":
    test_fix03_infinitive_disclosure_routes_to_empathy()
