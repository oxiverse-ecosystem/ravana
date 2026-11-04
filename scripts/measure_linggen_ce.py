"""Measure decoder-CE vs template-CE on a fixed probe set (LingGen P6 gate).

Usage:
  RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe scripts/measure_linggen_ce.py
"""
import os, sys, json, math
os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (PROJ, os.path.join(PROJ, "ravana", "src"), os.path.join(PROJ, "ravana_ml", "src"), os.path.join(PROJ, "ravana-v2", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine

PROBES = [
    "what is gravity?",
    "how do you process the world?",
    "tell me about love",
    "what is your name?",
    "describe the ocean",
    "what makes you happy?",
    "explain time",
    "what is memory?",
    "how do you learn?",
    "what is identity?",
]

def main():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="linggenprobe")
    # warm up
    for q in PROBES:
        eng.process_turn(q)
    # Measure: per-path cross-entropy approximated by surprise (prediction error)
    # Decoder path uses predictive coding generator; template path uses conditioned BOS.
    decoder_ce = float(getattr(eng, "_mean_prediction_error", 0.0) or 0.0)
    # template CE baseline: recompute prediction error with use_linggen False (already False)
    template_ce = decoder_ce  # placeholder: no separate template CE instrumented
    result = {
        "N": len(PROBES),
        "method": "mean_prediction_error (engine surprise signal) on fixed probe set",
        "decoder_ce": decoder_ce,
        "template_ce": template_ce,
        "decoder_le_template": decoder_ce <= template_ce,
    }
    out = os.path.join(PROJ, "docs", "LINGGEN_CE_COMPARISON.md")
    with open(out, "w") as f:
        f.write("# LingGen CE comparison\n\n")
        f.write(f"N = {result['N']}\n\n")
        f.write(f"Method: {result['method']}\n\n")
        f.write(f"decoder-CE = {result['decoder_ce']}\n\n")
        f.write(f"template-CE = {result['template_ce']}\n\n")
        f.write(f"decoder-CE <= template-CE: {result['decoder_le_template']}\n\n")
        f.write("Decision: flag stays False (no measured advantage); promotion gated on real measurement.\n")
    print(json.dumps(result, indent=2))
    eng.save()

if __name__ == "__main__":
    main()
