# Overnight session report — 2026-11-11/12

**Merged:** PR #315 (`d3e250c`) — retrieval-vs-generation switch. CI green 10/0.
**Open:** PR #316 — delegation audit (docs only, 105 lines). Committed and
pushed; blocked on CI runners, not on content.
**Tests:** `tests/unit` 2999 passed, 22 skipped, 0 failed.

---

## 1. The brain's retrieve-vs-generate switch — implemented

Delegated the design question to two parallel subagents: one mapped the
codebase, one researched the neuroscience.

**The finding inverts the intuitive design.** The brain has no binary
retrieve-vs-generate switch. It runs a **thresholded race** between a fast
stored-representation pathway and a slower compositional one:

```
retrieve  iff  R(s) >= T(s)

R(s) = w1*definition + w2*web + w3*confidence + w4*stability
     + w5*association + w6*familiarity

T(s) = T_base * (1 + k_pe*PE) * (1 + k_d*D) * (1 - k_f*familiarity)
T(s) <- T(s) * (1 - k_p*tanh(N_compose / N_practice))     # practice term
```

Two properties matter:

- **High prediction error RAISES the bar.** It cannot invite generation, since
  `T` rises monotonically with `PE`. The brain's conservative response to
  uncertainty is to fall back on the most practiced stored form.
- **Practice LOWERS the bar**, so the decoder *earns* reachability (Logan's
  instance theory). A hand-set `use_linggen` flag cannot express this.

Delivered: `ravana/src/ravana/learn/retrieval_generation.py`, 26 tests, and a
635-line cited review (`docs/RETRIEVAL_VS_GENERATION.md`).

## 2. The switch is wired — and OFF, because the decoder loses

Wired into `_generate_response` behind `_rg_switch_enabled` (default `False`).

**Placement was the whole problem.** The first attempt put the gate above the
definition lookup, but an unconditional `return self._reflective_response(ctx)`
sits above that point — the gate was unreachable for exactly the turns it
exists to affect. Verified: `_rg_decide` fired 0 times. Moved above the
decomposition block; now it fires on precisely the turns that reach the funnel.

Then the measurement that decides it:

| query | switch OFF | switch ON |
|---|---|---|
| how does memory work | *"honestly, Memory is a bit outside what i know"* | **"It paves the way for brain. It is opposite to data..."** |
| why do we dream | *"honestly, Dream is a bit outside what i know"* | *"hmm, i'm not totally sure about Dream yet..."* |

**The decoder produces fluent nonsense where the template path gave an honest
admission of ignorance.** CE 1.570 rewards fluency, not groundedness.

So the switch stays off. Two independent methods agree: the CE promotion gate
says `promote_use_linggen: false`, and this says the same.

**Next work is decoder groundedness, not reachability.**

## 3. The 200-query audit corrected a wrong belief

An earlier 46-query audit reported **0/46 reach generation**. That number drove
several wrong recommendations across the session.

Measured properly:

```
11671  138  69.0%   generation tail
8711    27  13.5%   KB hit (correct routing)
10845   11   5.5%   greetings
8934     6   3.0%   abstract-meaning -> reflection (correct routing)
... 9 more sites, 1-4 hits each
```

**69% of turns reach the generation tail.** The early returns are not the wall.
Both top "bypasses" are the hierarchy working as intended.

Instrumenting every generation entry point over 16 queries:

| path | calls |
|---|---|
| `_consult_internal_knowledge` | 16 (100%) |
| `_web_direct_answer` | 11 |
| `_definition_response` | 7 |
| `_decomposition_generation_path` | 4 |
| `_reflective_response` | 1 |
| `_generate_with_situation_model` | **0** |
| `_generate_with_decoder_and_syntax` | **0** |

## 4. `use_linggen` is ON in production — corrected

An earlier note claimed the flag is never `True`. That was measured on a fresh
`baby_mode` engine with an empty `data_dir`, which never loads the persisted
weights.

```
weights/ravana_weights.pkl   use_linggen: True   turn_count: 18
```

And the decoder still fires 0 times. Two blockers: `_generate_response` is not
reached on most turns, and inside it the definition and reflective gates return
before the `use_linggen` gate. The flag was necessary but not sufficient.

## 5. Delegation audit — one claim I could not reproduce

The codebase dispatch map is accurate (spot-checked) and adopted as reference.

The agent's empirical claim — *"the decoder IS reachable when
`_reflective_response` returns `None`"*, with `decoder:1` on three queries —
**could not be reproduced**. Its own trace attributes those runs to
`strat='dorsal_reasoned'`, a different route than the one named.

Attempted both readings against the real persisted state:

- forcing `_reflective_response` to `None`: `refl_called=0`, `decoder=0`
- instrumenting `_dorsal_path`/`_ventral_path`: 0 calls

Recorded as **unverified** in `docs/DELEGATION_AUDIT_DISPATCH_MAP.md` rather
than adopted. If a dorsal route does reach the decoder in production that is
worth knowing, but it needs reproducing first.

**Lesson:** a subagent's self-report is not evidence. Its map was verifiable by
reading source and checked out; its measurement was not, and did not.

## 6. Harness lessons (now in the `ravana-harness` skill)

- **Gateway restarts SIGTERM background jobs** (`exit -15`,
  `termination_source: gateway_shutdown`). It looks like OOM; it is not. Two
  audit runs were lost this way. Long jobs need `persist_on_release=true` AND
  must be resumable.
- **`scripts/audit_generation_reachability.py` is now resumable** — reads back
  its `--out` file, rebuilds counters, skips completed queries. `--fresh`
  resets.
- **`sys.settrace` must match the code object, not the filename.**
  `process_turn` and `_process_turn_impl` share a file, so a filename filter
  traces both and the wrapper's return always wins — producing the fake result
  "100% of queries exit at line 7741".
- **LoopRunner's `auto_fix` is a no-op** ("Reserved for future auto-fix
  rounds"). It fingerprints failures and instructs an agent to fix via
  `loop_commit`; it never edits code.
- **`goal_cmd` is refused unattended** unless the user approved it
  interactively. A command the agent runs itself never registers as approved.
  Use `verify_cmd` as the acceptance test.

## Files

**New code:** `ravana/src/ravana/learn/retrieval_generation.py`,
`scripts/audit_generation_reachability.py`

**New tests:** `tests/unit/test_retrieval_generation.py` (26)

**New docs:** `RETRIEVAL_VS_GENERATION.md` (635), `GENERATION_REACHABILITY_200.md`,
`WHY_DECODER_NEVER_RUNS.md`, `DECODER_BLOCKED_BY_TWO_GATES.md`,
`RETRIEVAL_GENERATION_VALIDATION.md`, `SWITCH_IS_OFF.md`,
`DELEGATION_AUDIT_DISPATCH_MAP.md`

**New skill:** `ravana-harness`
