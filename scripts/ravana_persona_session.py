#!/usr/bin/env python3
"""
ravana_persona_session.py
=========================
Drive RAVANA in a long, continuous chat and synthesize a PERSONALITY PROFILE
from its real internal state.

How it works
------------
RAVANA is a decoder-first cognitive architecture that learns continuously from
conversation (no LLM). This script keeps ONE long-lived CognitiveChatEngine
process alive and feeds it turns so it can chat "as much as you like". Because
turns run in-process they are cheap (~0.1s each after a one-time ~27s init);
the slowness you saw from the CLI is just subprocess + 11MB pickle re-saves.

After the conversation we mine RAVANA's GENUINE cognitive state and turn it into
a personality profile. Nothing here is invented -- every field is read live:

  * IdentityEngine  -> self-coherence (strength / momentum / stability / trend)
  * UserStanceStore -> attitudes RAVANA settled into (topic, polarity, confidence)
  * BeliefStore     -> positions you stated during the talk
  * PersonalFactStore -> attributes RAVANA extracted about you
  * ConceptGraph    -> vocabulary + typed-edge growth (world-model size)
  * turn / sleep / learning counters

State persists to weights/ so the personality ACCUMULATES across sessions.

Usage
-----
  # Run the bundled autopilot conversation, then print + save the profile:
  python scripts/ravana_persona_session.py --autopilot --profile-out output/ravana_persona.md

  # Same, but drop into a live "You:" loop afterwards (type !profile / !save / !quit):
  python scripts/ravana_persona_session.py --autopilot --interactive

  # Just regenerate the profile from already-saved state (no new chat):
  python scripts/ravana_persona_session.py --emit-profile

  # Live chat only, no autopilot:
  python scripts/ravana_persona_session.py --interactive --user myname

Offline by default (RAVANA_OFFLINE=1) so runs are reproducible and fast; pass
--online to let it learn from the web.
"""

import os
import sys
import io
import json
import time
import datetime

# Reproducible + fast: never hit the network, never write .pyc.
os.environ.setdefault("RAVANA_OFFLINE", "1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from ravana.chat.engine import CognitiveChatEngine


# ─────────────────────────────────────────────────────────────────────────────
# Quiet stream: drop RAVANA's bootstrap / tracing noise (keep real replies)
# ─────────────────────────────────────────────────────────────────────────────
_BLOCK = (
    "[ground_query]", "[KB] Seeded", "[PMI] Seeded", "[Domain] Boot",
    "[CommonFacts]", "[Physics] Seeded", "[Teen] Knows", "[GloVe]",
    "[Loaded]", "[bg]", "[snippet]", "[trace]",
)


class _QuietWriter(io.TextIOBase):
    def __init__(self, sink):
        self._sink = sink
        self._buf = ""

    def write(self, data):
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not any(tok in line for tok in _BLOCK):
                self._sink.write(line + "\n")
                self._sink.flush()
        return len(data)

    def flush(self):
        self._sink.flush()


def install_quiet():
    sys.stdout = _QuietWriter(sys.stdout)


def say(msg=""):
    # Go straight to the real sink so our own lines are never filtered.
    real = sys.stdout._sink if isinstance(sys.stdout, _QuietWriter) else sys.stdout
    real.write(msg + "\n")
    real.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Bundled autopilot corpus -- reflects the founder's real identity so RAVANA
# builds a faithful model of the person it is talking to. The agent "plays"
# the user here; live --interactive mode lets a human take over.
# ─────────────────────────────────────────────────────────────────────────────
AUTOPILOT = [
    "hi ravana",
    "i am likhith, i founded a privacy-first ecosystem called oxiverse",
    "i think privacy is the most basic right anyone should have",
    "i love open source because it builds trust with users",
    "i am a student at gmrit university studying engineering",
    "i also study at the university of the people online",
    "my core value is that technology should serve users, not exploit them",
    "i built intentforge as a private search engine inside oxiverse",
    "i get frustrated when big tech tracks users without consent",
    "i believe open source enables better security through inspection",
    "i want the next generation to inherit tools that respect their autonomy",
    "i am 20 years old and i live in india",
    "i am cautious about ai that is not transparent",
    "what do you think about privacy",
    "i really dislike surveillance capitalism",
    "i prefer building in public so the community can learn with me",
    "yes, open source is definitely the right approach",
    "i think long term thinking matters more than quick growth",
    "i am excited about decoder-first cognitive architectures like ravana",
    "i would never trade user privacy for convenience",
    "i learn by shipping, not by planning forever",
    "i value the next generation advancing faster than i did",
    "open standards matter more than walled gardens",
    "i believe transparency is the only real security",
    "what do you know about me",
]


# ─────────────────────────────────────────────────────────────────────────────
# Personality synthesis -- read LIVE state, never invent
# ─────────────────────────────────────────────────────────────────────────────
def _polarity_word(p):
    if p >= 0.6:
        return "strongly FOR"
    if p > 0.1:
        return "for"
    if p <= -0.6:
        return "strongly AGAINST"
    if p < -0.1:
        return "against"
    return "neutral / uncertain about"


def synthesize_personality(engine):
    """Return a structured dict mined entirely from engine internals."""
    # Identity / self-coherence
    ident = engine.identity.get_status()
    strength = float(ident.get("strength", 0.0))
    momentum = float(ident.get("momentum", 0.0))
    stability = float(ident.get("stability", 0.0))
    trend = float(ident.get("trend", 0.0))

    if strength >= 0.6 and trend >= 0:
        stage = "crystallizing — a stable self is forming"
    elif strength >= 0.4 or trend > 0:
        stage = "cohering — attitudes are settling but still shifting"
    else:
        stage = "exploring — still sampling the world, few commitments yet"

    # Learned attitudes (stances)
    stances = []
    try:
        for topic, s in engine.user_model.opinions.stances.items():
            stances.append({
                "topic": topic,
                "polarity": round(float(s.polarity), 3),
                "confidence": round(float(s.confidence), 3),
                "valence": round(float(s.valence), 3),
            })
    except Exception:
        pass
    stances.sort(key=lambda x: (abs(x["polarity"]), x["confidence"]), reverse=True)

    # Beliefs the user stated (positions captured in the belief store)
    beliefs = []
    try:
        for key, val in engine.belief_store.get_state().get("beliefs", {}).items():
            text = val[0] if isinstance(val, (list, tuple)) else val
            beliefs.append(str(text))
    except Exception:
        pass

    # Attributes extracted about the user
    facts = []
    try:
        for (subj, attr, _val), f in engine.user_model.personal_facts.facts.items():
            if not getattr(f, "superseded", False):
                facts.append({"subject": subj, "attribute": attr,
                              "value": f.value, "confidence": round(f.confidence, 2)})
    except Exception:
        pass

    # World-model growth.
    # NOTE: the engine's save()/load() does NOT persist the ConceptGraph object
    # (it sanitizes to a str and rebuilds it fresh from GloVe+seeds on boot),
    # so `graph.nodes` is only meaningful for the LIVE engine that did the
    # talking. For a reload (--emit-profile) the graph is the fresh 11-node
    # seed graph, NOT learned state. We report it but flag it.
    live_graph = bool(getattr(engine, "_session_resumed", False)) is False
    n_nodes = len(getattr(engine.graph, "nodes", {}))
    n_edges = len(getattr(engine.graph, "edges", {}))
    if not live_graph and n_nodes <= 50:
        # Cross-session reload: graph was rebuilt fresh, do not present its
        # size as accumulated learning.
        n_nodes = None
        n_edges = None

    learning = int(getattr(engine, "_learning_count", 0) or 0)
    curious = learning > 0

    profile = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "self_coherence": {
            "identity_strength": round(strength, 3),
            "momentum": round(momentum, 3),
            "stability": round(stability, 3),
            "trend": round(trend, 3),
            "stage": stage,
        },
        "world_model": {
            "concepts": n_nodes,
            "typed_edges": n_edges,
            "turns": int(getattr(engine, "turn_count", 0)),
            "sleep_cycles": int(getattr(engine, "sleep_cycles_completed", 0)),
            "web_learnings": learning,
            "curious": curious,
        },
        "attitudes": stances,
        "your_stated_positions": beliefs,
        "attributes_about_you": facts,
    }
    return profile


def render_profile_text(profile):
    """Render the profile as plain terminal-friendly text (no markdown)."""
    sc = profile["self_coherence"]
    wm = profile["world_model"]
    lines = []
    lines.append("")
    lines.append("=" * 64)
    lines.append("   RAVANA — PERSONALITY PROFILE")
    lines.append("=" * 64)
    lines.append("")
    lines.append("SELF-COHERENCE  (IdentityEngine)")
    lines.append("  identity strength : %.2f  (%s)" % (sc["identity_strength"], sc["stage"]))
    lines.append("  momentum          : %.2f   stability: %.2f   trend: %+.2f"
                 % (sc["momentum"], sc["stability"], sc["trend"]))
    lines.append("")
    lines.append("WORLD-MODEL  (counters + ConceptGraph)")
    if wm["concepts"] is not None:
        lines.append("  vocabulary        : %d concepts" % wm["concepts"])
        lines.append("  connections       : %d typed edges" % wm["typed_edges"])
    else:
        lines.append("  vocabulary        : (n/a on reload — graph rebuilt fresh each boot)")
        lines.append("  connections       : (n/a on reload)")
    lines.append("  turns taken       : %d   sleeps: %d   web-learnings: %d"
                 % (wm["turns"], wm["sleep_cycles"], wm["web_learnings"]))
    lines.append("  disposition       : %s" % ("curious, learns from the web"
                                               if wm["curious"] else "still absorbing from talk only"))
    lines.append("")
    lines.append("ATTITUDES I SETTLED INTO  (UserStanceStore)")
    if profile["attitudes"]:
        for a in profile["attitudes"]:
            lines.append("  %-12s %-18s (polarity %+.2f, conf %.2f)"
                         % (_polarity_word(a["polarity"]), a["topic"], a["polarity"], a["confidence"]))
    else:
        lines.append("  (none firmed up yet — chat more to form attitudes)")
    lines.append("")
    lines.append("WHAT I LEARNED ABOUT YOU  (PersonalFactStore + BeliefStore)")
    if profile["attributes_about_you"]:
        for f in profile["attributes_about_you"]:
            lines.append("  you / %s = %s  (conf %.2f)" % (f["attribute"], f["value"], f["confidence"]))
    else:
        lines.append("  (no stable attributes extracted yet)")
    if profile["your_stated_positions"]:
        lines.append("  positions you stated:")
        for b in profile["your_stated_positions"]:
            lines.append("    - %s" % b)
    lines.append("")
    lines.append("ONE-LINE PERSONA")
    top = profile["attitudes"][:3]
    _c = wm["concepts"] if wm["concepts"] is not None else 0
    _e = wm["typed_edges"] if wm["typed_edges"] is not None else 0
    if top:
        attitude_bits = ", ".join("%s %s" % (_polarity_word(a["polarity"]).lower(), a["topic"])
                                   for a in top)
        persona = ("A %s teenage mind with %d concepts and %d connections; "
                   "attitudes: %s." % (sc["stage"].split('—')[0].strip(),
                                       _c, _e, attitude_bits))
    else:
        persona = ("A %s teenage mind with %d concepts and %d connections; "
                   "still forming its first attitudes."
                   % (sc["stage"].split('—')[0].strip(), _c, _e))
    lines.append("  " + persona)
    lines.append("")
    lines.append("=" * 64)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Engine lifecycle
# ─────────────────────────────────────────────────────────────────────────────
def make_engine(user_suffix, online):
    if not online:
        os.environ["RAVANA_OFFLINE"] = "1"
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=user_suffix)
    return eng


def load_chat_file(path):
    """Read user turns from a file or stdin ('-'). One turn per line;
    skip blank lines and '#' comments. Returns a list of strings."""
    if path == "-":
        import sys as _sys
        raw = _sys.stdin.read()
    else:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    turns = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        turns.append(line)
    return turns


def run_turns(engine, turns, quiet, label="chat"):
    """Drive RAVANA through `turns`; return count processed."""
    say("[%s] running %d turns..." % (label, len(turns)))
    for q in turns:
        r = chat_turn(engine, q)
        if not quiet:
            say("  You: %s" % q)
            say("  RAVANA: %s" % r)
    say("[%s] done." % label)
    return len(turns)


def chat_turn(engine, text):
    try:
        resp = engine.process_turn(text)
    except Exception as e:
        resp = "[ravana got confused: %s]" % e
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="RAVANA continuous-chat personality session")
    ap.add_argument("--user", default="persona", help="user suffix for isolated, accumulating state")
    ap.add_argument("--autopilot", action="store_true", help="run the bundled conversation corpus first")
    ap.add_argument("--turns", type=int, default=0, help="limit autopilot to N turns (0 = all)")
    ap.add_argument("--chat-file", default=None,
                    help="chat continuously from an external file: one user turn per "
                         "line (blank lines and '#' comments skipped). Use '-' "
                         "to read turns from stdin. This is the 'chat as much as "
                         "you like' mode — feed any length of conversation.")
    ap.add_argument("--interactive", action="store_true", help="live 'You:' loop after autopilot")
    ap.add_argument("--emit-profile", action="store_true", help="load saved state and print profile only")
    ap.add_argument("--profile-out", default=None, help="write profile (markdown) to this path")
    ap.add_argument("--online", action="store_true", help="allow web learning (default: offline)")
    ap.add_argument("--quiet", action="store_true", help="suppress RAVANA's own chatter lines")
    args = ap.parse_args()

    if not args.quiet:
        install_quiet()

    if args.emit_profile:
        eng = make_engine(args.user, args.online)
        # Reload the persisted snapshot so the profile reflects what was
        # actually learned in prior sessions -- a fresh engine rebuilds the
        # graph from scratch and would read 0 learned state.
        try:
            ok = eng.load()
            say("[emit-profile] loaded saved state: %s" % ("ok" if ok else "no prior snapshot found"))
        except Exception as e:
            say("[emit-profile] load failed: %s" % e)
        profile = synthesize_personality(eng)
        say(render_profile_text(profile))
        if args.profile_out:
            _write_profile(args.profile_out, profile)
            say("[profile written to %s]" % args.profile_out)
        return

    t0 = time.time()
    eng = make_engine(args.user, args.online)
    say("[session] engine ready in %.1fs (user=%s, %s)"
        % (time.time() - t0, args.user, "online" if args.online else "offline"))

    if args.autopilot:
        corpus = AUTOPILOT[:args.turns] if args.turns else AUTOPILOT
        run_turns(eng, corpus, args.quiet, label="autopilot")

    if args.chat_file:
        turns = load_chat_file(args.chat_file)
        if not turns:
            say("[chat-file] no turns found in %s" % args.chat_file)
        else:
            run_turns(eng, turns, args.quiet, label="chat-file")

    if args.interactive or (not args.autopilot and not args.chat_file):
        say("")
        say("  Live chat — type your message. Commands: !profile !save !quit")
        try:
            while True:
                try:
                    u = input("  You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    say("")
                    break
                if not u:
                    continue
                low = u.lower()
                if low in ("!quit", "!exit", "bye", "goodbye"):
                    say("  RAVANA: bye! i'll remember what we talked about.")
                    break
                if low == "!profile":
                    say(render_profile_text(synthesize_personality(eng)))
                    continue
                if low == "!save":
                    say("  [%s]" % eng.save())
                    continue
                r = chat_turn(eng, u)
                say("  RAVANA: %s" % r)
        except Exception as e:
            say("[live loop error: %s]" % e)

    # Final synthesis + persist
    profile = synthesize_personality(eng)
    say(render_profile_text(profile))
    say("[%s]" % eng.save())
    if args.profile_out:
        _write_profile(args.profile_out, profile)
        say("[profile written to %s]" % args.profile_out)


def _write_profile(path, profile):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    md = ["# RAVANA Personality Profile", "",
          "_generated: %s_" % profile["generated_at"], ""]
    sc = profile["self_coherence"]
    wm = profile["world_model"]
    md.append("## Self-coherence")
    md.append("- identity strength: **%.2f** (%s)" % (sc["identity_strength"], sc["stage"]))
    md.append("- momentum %.2f · stability %.2f · trend %+.2f"
              % (sc["momentum"], sc["stability"], sc["trend"]))
    md.append("")
    md.append("## World-model")
    if wm["concepts"] is not None:
        md.append("- concepts: %d · typed edges: %d" % (wm["concepts"], wm["typed_edges"]))
    else:
        md.append("- concepts: n/a on reload (graph rebuilt fresh each boot)")
    md.append("- turns: %d · sleeps: %d · web-learnings: %d"
              % (wm["turns"], wm["sleep_cycles"], wm["web_learnings"]))
    md.append("- curious: %s" % wm["curious"])
    md.append("")
    md.append("## Attitudes")
    if profile["attitudes"]:
        for a in profile["attitudes"]:
            md.append("- **%s %s** (polarity %+.2f, confidence %.2f)"
                      % (_polarity_word(a["polarity"]), a["topic"], a["polarity"], a["confidence"]))
    else:
        md.append("- (none firmed up yet)")
    md.append("")
    md.append("## Attributes about you")
    if profile["attributes_about_you"]:
        for f in profile["attributes_about_you"]:
            md.append("- %s = %s (conf %.2f)" % (f["attribute"], f["value"], f["confidence"]))
    else:
        md.append("- (none extracted yet)")
    md.append("")
    md.append("## Your stated positions")
    for b in profile["your_stated_positions"]:
        md.append("- %s" % b)
    md.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    # Also drop a machine-readable JSON next to it
    jpath = os.path.splitext(path)[0] + ".json"
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)


if __name__ == "__main__":
    main()
