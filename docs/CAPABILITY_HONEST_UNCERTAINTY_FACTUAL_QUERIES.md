# Capability: honest uncertainty for factual/world-knowledge queries

## What it does

When RAVANA runs offline and is asked a factual world-knowledge question it has
no stored fact for (e.g. "what is the capital of france?", "who invented the
telephone?"), it now emits honest uncertainty instead of confabulating an
associative reply. Personal/autobiographical queries (containing i/me/my/we/
our/you) are left untouched and continue through the normal recall/reasoning
path, as are non-wh inputs.

## How it grew from the conversation

The round report (ravana-2026-11-03T1533Z) identified the confabulation:

> When no stored fact exists for "capital of france", the engine fell through to
> analogical reasoning, which returned a GloVe-neighbour confabulation
> ("france is like second"). This is confabulation dressed as reasoning: the
> answer's content did not come from RAVANA's state.

The fix intercepts the query in `CognitiveChatEngine.process_turn` (in
`ravana/src/ravana/chat/engine.py`) before the episodic/recall path, returning
an honest "i don't know" for offline factual world queries.

## Design

**Location:** `ravana/src/ravana/chat/engine.py:8361-8385`

Inside `process_turn`, after the internal-knowledge short-circuit and only when
`RAVANA_OFFLINE=1`:

1. Read `user_input`, lowercased and stripped into `_t`.
2. If `_t` matches `^\s*(?:why|what|how|who|where|when|which)\b` (a wh-question)
   AND does not contain a first/second-person pronoun
   (`\b(i|me|my|we|our|you)\b`), treat it as a factual world-knowledge query.
3. Set `self._last_strategy = "honest_uncertainty"`, build the reply
   `"i don't really know that yet — i'd have to look it up."`, record it in
   `_last_responses` (capped at 10) and `_recent_user_turns` (capped at 12),
   call `_record_episode`, `notify_user_idle`, `_identity_end_of_turn`, and
   return the reply.
4. Otherwise fall through to the normal episodic/recall/reasoning path.

## Why this is not hardcoding

- The only authored string is the existing honest-uncertainty sentence (not new
  personality prose).
- No per-topic tables and no question-to-answer mappings.
- The gate is structural (wh-question shape + absence of personal pronouns),
  not a keyword list tuned to probe questions.

## Verified behaviour

- Live run on "what is the capital of france?" returns strategy
  `honest_uncertainty` (verified per parent task t_91a5483e).
- Personal/autobiographical wh-questions (containing i/me/my/we/our/you) bypass
  the gate and reach the normal recall path.
- Non-wh inputs bypass the gate.

## Related

- `ravana/src/ravana/chat/engine.py` — `process_turn` (the gate above)
- Strategy string: `honest_uncertainty` (`self._last_strategy`)

