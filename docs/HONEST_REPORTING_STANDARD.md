# Honest Reporting Standard

Four enforceable rules for every RAVANA claim, measurement, and status report.
Created: 2026-09-14.

## Rule 1 — Every claim must have a number with N= and measurement method

A claim without a number is a guess. A number without N= or a stated measurement
method is unauditable.

| BAD | GOOD |
|-----|------|
| "Identity stability improved" | "Identity stability 0.42 → 0.67 (N=5 turns, mean over turns 6–10)" |
| "Tests pass" | "187/187 unit tests pass (pytest tests/unit/ -q, 3 shards)" |
| "Memory footprint is high" | "Peak RSS 307 MB (measured via `psutil.Process().memory_info()` at engine boot)" |
| "Reproducible" | "SHA-256 fingerprint `a1b2c3...` identical across 2 back-to-back runs (seed=42, RAVANA_OFFLINE=1, PYTHONHASHSEED=0)" |

The number must be a REAL measurement from this codebase or run — never fabricated,
never extrapolated from a single anecdote.

## Rule 2 — Module-level claims cite the real code path

Every claim about RAVANA's behavior must name the **file and function** that
produces it. "The engine does X" is too vague — `engine.py` is 9000+ lines.

| BAD | GOOD |
|-----|------|
| "Recall is robust" | "`_route_agent_own_recall` in `chat/engine.py:5841` relaxes `_min_overlap` to 1 when `len(_cands) <= 1`" |
| "Stances are stored" | "`UserStanceStore.put()` in `chat/user_model.py` writes `(topic, polarity, confidence)` tuples" |
| "Sleep consolidates" | "`sleep_cycle()` in `engine_generation.py:1289` graduates hippocampal buffer facts to graph" |

If you cannot name the function, the claim is not grounded.

## Rule 3 — Failures and limitations are reported honestly

A test that fails is not a "known limitation" until the failure is reproducible
AND the root cause is understood. A limitation is pre-existing only if you can
prove it existed BEFORE the change that supposedly introduced it.

| BAD | GOOD |
|-----|------|
| "Known limitation: routing test fails (pre-existing)" | "FAIL `tests/unit/test_foo.py::test_bar` — `_route_intent` returns `None` for input `'...'` (reproduced locally with cold GloVe; CI routing differs — see skill pitfall: 'local pytest unreliable for routing tests')" |
| "Works on my machine" | "Passes at CI; local cold-GloVe PFC classifies differently — not evidence of a fix or a regression" |
| "4 of 5 defects are known limitations" (without investigation) | Each defect cold-verified against the running engine OR reproduced in-process with `RAVANA_OFFLINE=1` |

The honest-failure hierarchy (use the first that applies):
1. **Reproduced** — same failure in-process AND in CI
2. **CI-only** — fails at CI, not locally (routing/GloVe-warmth gap — see skill)
3. **Pre-existing** — SAME failure on the baseline SHA (prove with `git checkout <baseline> && pytest`)
4. **Genuinely new** — introduced by this round's diff (fix it or file a card)

A claim of "pre-existing" without step 4 evidence is a lie of convenience.

## Rule 4 — No self-certification

A commit message that says "verified" or "grounded, not a script" is not evidence.
The diff is the evidence. Audit the diff, never the commit message.

```bash
# The hardcoding audit grep (run during EVERY round):
git diff <baseline> -- ravana/src/ | grep "^+" | grep -oE '"[a-z][^"]{45,}"'
```

Zero long authored-sentence hits = clean. Any hit → read in context, classify as
vocabulary (fine) or authored reply (violation).

A round that passes this grep AND has all claims satisfying Rules 1–3 is honest.
A round that fails any rule gets a FIX card, not a note in the changelog.

---

## Verification

Run before every PR:

```bash
# Rule 1+2: check that every README/BENCHMARKS/LEDGER number cites a file:line
grep -nE "^\| .+ \| .* \| .* \| \*\*GREEN\*\*" docs/ACCEPTANCE_LEDGER.md \
  | grep -v "Evidence" | head -1

# Rule 3: confirm no "known limitation" without a cited root cause
grep -ni "known limitation" docs/*.md | grep -v "root cause\|reproduced\|CI-only\|pre-existing"

# Rule 4: the diff audit
git diff HEAD~1 -- ravana/src/ | grep "^+" | grep -oE '"[a-z][^"]{45,}"' | head -5
```

If any of these return results that don't satisfy the rules, fix before opening the PR.

---

*This standard is derived from the failures the QA loop itself produced: self-certification,
overstated "known limitations", and claims without numbers. It exists because the loop
kept rubber-stamping its own work until an external audit forced honest measurement.*
