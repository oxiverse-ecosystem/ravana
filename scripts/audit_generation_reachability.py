"""Stage 1 reachability audit: which early returns actually fire?

No behaviour change. This wraps `process_turn` in a `sys.settrace` local-trace
and records, for each query, the LAST line in `engine.py` that executed before
the call returned. That line is the return site that produced the reply.

Run:
    uv run python scripts/audit_generation_reachability.py [--n 200] [--out FILE]

Output: a ranked table of return sites by hit count, plus per-query detail.
"""
import argparse
import collections
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

os.environ.pop("RAVANA_OFFLINE", None)
os.environ.pop("RAVANA_TEST_MODE", None)

for _p in (os.path.join(_REPO, "ravana_ml", "src"),
           os.path.join(_REPO, "ravana", "src"),
           os.path.join(_REPO, "ravana-v2", "src")):
    sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

# The ~200-query probe set. Deliberately ordinary: the question is what the
# engine does with the kinds of things a user actually types.
QUERIES = [
    # greetings / small talk
    "hi", "hello", "hey", "good morning", "how are you", "what's up",
    "who are you", "what are you", "tell me about yourself", "are you alive",
    "are you conscious", "do you feel", "what do you want", "are you happy",
    # wh- questions about ordinary things
    "what is love", "what is trust", "what is respect", "what is time",
    "what is water", "what is light", "what is sound", "what is memory",
    "what is a rumour", "what is a shadow", "what is gravity", "what is heat",
    "what is music", "what is colour", "what is language", "what is thought",
    "what is a dream", "what is fear", "what is hope", "what is freedom",
    "what is justice", "what is beauty", "what is truth", "what is knowledge",
    "what is a number", "what is a word", "what is a story", "what is a game",
    "what is a friend", "what is a family", "what is a home", "what is work",
    "what is money", "what is power", "what is art", "what is science",
    "what is a computer", "what is the internet", "what is a city",
    "what is the ocean", "what is a mountain", "what is the sky",
    "what is a star", "what is the moon", "what is the sun", "what is fire",
    "what is air", "what is soil", "what is a tree", "what is a bird",
    "what is a dog", "what is a cat", "what is a fish", "what is a plant",
    "why is the sky blue", "why do we sleep", "why do we dream",
    "why do people lie", "why do we forget", "why do we remember",
    "why is water wet", "why is fire hot", "why is the grass green",
    "why do birds sing", "why do we age", "why do we laugh", "why do we cry",
    "why is time short", "why is love hard", "why is trust important",
    "why do we need friends", "why do we work", "why do we play",
    "why do we learn", "why do we teach", "why do we build", "why do we ask",
    "how does the brain work", "how do we learn", "how do we remember",
    "how does memory work", "how does sleep help", "how do muscles grow",
    "how does a car move", "how does a phone work", "how does the web work",
    "how do plants grow", "how do birds fly", "how does rain form",
    "how does fire burn", "how does sound travel", "how does light move",
    "how do we see colour", "how do we hear", "how do we taste",
    "how do we smell", "how do we feel pain", "how do we balance",
    "how do we walk", "how do we speak", "how do we read", "how do we write",
    "how do we think", "how do we decide", "how do we choose",
    "how do we plan", "how do we focus", "how do we calm down",
    "how do i sleep better", "how do i learn faster", "how do i remember more",
    "how do i make friends", "how do i earn trust", "how do i be happy",
    "how do i be calm", "how do i be kind", "how do i be brave",
    "how heavy is a rumour", "how long is a shadow", "how far is the moon",
    "how big is the sun", "how old is the earth", "how deep is the ocean",
    "how fast is light", "how fast is sound", "how hot is the sun",
    "how cold is ice", "how strong is steel", "how soft is water",
    "tell me about love", "tell me about trust", "tell me about time",
    "tell me about water", "tell me about light", "tell me about music",
    "tell me about memory", "tell me about dreams", "tell me about fear",
    "tell me about the ocean", "tell me about the sky", "tell me about fire",
    "tell me about the brain", "tell me about language", "tell me about art",
    "tell me about science", "tell me about the city", "tell me about home",
    "tell me more", "go on", "and then", "what else", "why", "how",
    "describe the ocean", "describe the forest", "describe the desert",
    "describe a storm", "describe the night", "describe the morning",
    "describe a river", "describe a mountain", "describe a city",
    "describe a friend", "describe a teacher", "describe a leader",
    "describe happiness", "describe sadness", "describe fear",
    "describe trust", "describe respect", "describe love",
    "explain gravity", "explain memory", "explain sleep", "explain dreams",
    "explain language", "explain music", "explain colour", "explain time",
    "explain trust", "explain respect", "explain love", "explain fear",
    "explain the brain", "explain the heart", "explain the mind",
    "explain a computer", "explain the internet", "explain a city",
    "do you know me", "do you remember me", "do you like me",
    "do you trust me", "do you understand", "do you understand me",
    "can you help me", "can you learn", "can you remember",
    "can you feel", "can you think", "can you dream", "can you forget",
    "can you teach me", "can you show me", "can you tell me",
    "will you remember this", "will you learn this", "will you forget",
    "what do you know about me", "what have i told you", "what did i say",
    "remind me what i said", "what did you say", "what did we talk about",
    "i feel sad", "i feel happy", "i feel tired", "i feel lost",
    "i feel anxious", "i feel calm", "i feel angry", "i feel grateful",
    "i love you", "i trust you", "i hate this", "i like this",
    "i don't know", "i'm not sure", "maybe", "perhaps", "i think so",
    "thank you", "thanks", "sorry", "excuse me", "goodbye", "bye",
    "that's interesting", "i see", "okay", "alright", "fair enough",
    "what do you think", "what's your opinion", "what do you believe",
    "is this true", "is that right", "are you sure", "really",
    "why not", "how come", "what if", "what then", "so what",
    "compare trust and respect", "compare love and fear", "compare water and fire",
    "what is the difference between trust and respect",
    "what is the difference between memory and thought",
    "what is the difference between a star and a planet",
    "give me an example of trust", "give me an example of courage",
    "give me an example of kindness", "give me an example of fear",
    "what happens when it rains", "what happens when we sleep",
    "what happens when we die", "what happens when we learn",
    "what happens when we forget", "what happens when we love",
    "where does rain come from", "where does sound come from",
    "where does light come from", "where does memory live",
    "where does thought happen", "where does language live",
    "when did we start speaking", "when did we start writing",
    "when do we dream", "when do we forget", "when do we remember",
    "who made language", "who made music", "who made art",
    "who made the city", "who made the web", "who made the game",
]


def trace_last_return(engine, query):
    """Run one query under a local trace; return the last engine.py line hit.

    Traces `_process_turn_impl` (the real body) rather than `process_turn`
    (the thin wrapper), so the recorded line is an actual decision point
    rather than the wrapper's own return.
    """
    last = {"line": None}

    impl_code = engine._process_turn_impl.__code__

    def local_trace(frame, event, arg):
        # Match the CODE OBJECT, not the file: process_turn and
        # _process_turn_impl live in the same file, so a filename filter
        # traces both and the wrapper's own return (line 7741) wins.
        if event == "line" and frame.f_code is impl_code:
            last["line"] = frame.f_lineno
        return local_trace

    def global_trace(frame, event, arg):
        if event == "call" and frame.f_code is impl_code:
            return local_trace
        return None

    old = sys.gettrace()
    sys.settrace(global_trace)
    try:
        reply = engine.process_turn(query)
    finally:
        sys.settrace(old)
    return last["line"], reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dim", type=int, default=64)
    args = ap.parse_args()

    import tempfile
    d = tempfile.mkdtemp(prefix="ravana_reach_")
    print("building engine (dim=%d)..." % args.dim, flush=True)
    eng = CognitiveChatEngine(dim=args.dim, seed=42, baby_mode=True,
                              data_dir=d, user_suffix="_reach")

    queries = QUERIES[:args.n]
    print("running %d queries under trace...\n" % len(queries), flush=True)

    hits = collections.Counter()
    per_query = []
    errors = 0

    for i, q in enumerate(queries, 1):
        try:
            line, reply = trace_last_return(eng, q)
        except Exception as ex:
            errors += 1
            per_query.append((q, "ERROR", "%s: %s" % (type(ex).__name__, ex), ""))
            continue
        hits[line] += 1
        per_query.append((q, line, reply, ""))

    print("=" * 78)
    print("RETURN SITES RANKED BY HIT COUNT")
    print("=" * 78)
    print("%-8s %-6s %s" % ("line", "hits", "pct"))
    print("-" * 78)
    for line, n in hits.most_common():
        print("%-8s %-6d %.1f%%" % (line, n, 100.0 * n / len(queries)))
    print("-" * 78)
    print("distinct return sites: %d" % len(hits))
    print("queries: %d   errors: %d" % (len(queries), errors))

    print("\n" + "=" * 78)
    print("PER-QUERY DETAIL")
    print("=" * 78)
    for q, line, reply, err in per_query:
        r = (reply or "").replace("\n", " ")[:70]
        print("%-42s -> %-6s %s" % (q[:42], line, r))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("line\thits\tpct\n")
            for line, n in hits.most_common():
                fh.write("%s\t%d\t%.1f\n" % (line, n, 100.0 * n / len(queries)))
            fh.write("\nquery\tline\treply\n")
            for q, line, reply, err in per_query:
                fh.write("%s\t%s\t%s\n" % (q, line, (reply or "").replace("\t", " ")))
        print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
