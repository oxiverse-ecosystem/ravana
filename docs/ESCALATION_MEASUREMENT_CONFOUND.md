# Escalation: first measurement was a confound

**Status:** the escalation measurement in PR #329's first run compared the
wrong arms. This records the error and the correct comparison.

## The mistake

I compared:

```
--cpg --cpg-relation --cpg-escalate     (480 subjects)
        vs
(no cpg flags)                          (446 subjects)
```

The second run had **CPG off entirely** — there is no `cpg :` line in its
output. So the comparison was CPG-off vs CPG-on+escalate, not
escalate-off vs escalate-on.

The result looked dramatic:

```
                    decoder   narrative   syntax
CPG off               226        217         1
CPG + escalate          0        179       303
```

Zero decoder outputs. I read that as "escalation kills the decoder". It does
not follow — CPG itself may account for all of it.

## Why I believed it

The 2-query probe in `docs/ESCALATION_VERIFIED.md` showed escalation changing
output on an absent subject. That was real but tiny. Scaling it to 480
subjects and comparing against the wrong baseline turned a small effect into
an apparent catastrophe.

## The correct comparison

```
--cpg --cpg-relation                  # escalate OFF (default)
--cpg --cpg-relation --cpg-escalate   # escalate ON
```

Both with CPG on, both with relation gating, both at theta 0.15. Only the
escalation flag differs.

## What the direct probe shows

On `gravity`, with `_generate_with_situation_model` and the gate disabled:

```
escalate=True   -> situation_model_narrative
escalate=False  -> situation_model_narrative
```

Identical. And `decide()` on gravity/ocean/water/light commits every time
(12/12, 10/10, 3/3, 3/3 admissible). So escalation is **not firing** for
graph-known subjects, which is the intended behaviour.

## Lesson

A flag that defaults to off is invisible in the output unless you print it.
The instrument prints `cpg : on (...)` only when `--cpg` is passed — so a
no-flag run looks like a CPG run in the summary. **Always check the config
line, not just the numbers.**

## Related

- PR #329 — adds `--cpg-escalate`
- `docs/ESCALATION_VERIFIED.md` — the 2-query probe
- `docs/THETA_LOWER_IS_BETTER.md` — the theta sweep
