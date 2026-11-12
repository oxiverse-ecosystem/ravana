# Escalation is inert at scale

**Status:** measured. `_cpg_escalate` has **no effect** on the 480-subject
instrument. It should stay off, and the reason is worth recording.

## The measurement

Both arms with CPG on, relation gating on, theta 0.15 — only the escalation
flag differs:

```
                              escalate OFF   escalate ON
on-topic fraction                 0.4813         0.4813
content-word cosine               0.4397         0.4397
unsupported assertions            0.6046         0.6046
supported assertions              2.5105         2.5105
word count                       22.55          22.55
unique ratio                      0.8245         0.8245
strategy distribution    narrative 178 / syntax 302   (identical)
```

**Identical to four decimal places.** Not "no significant difference" —
bit-for-bit the same output.

## Why it never fires

The escalation condition is `not _cpg_committed`. Two paths set that:

1. `_cpg_token_boost` returns a `DecideResult` with `committed=False` — nothing
   cleared theta.
2. It returns `{}` early (no subject, no node, no edges) — but the caller's
   `if _cpg is not None` treats `{}` as valid, `_cpg.committed` raises, and the
   `except` sets `_cpg_committed = True`. So the early-return path is fail-open
   by design.

Probed 20 instrument subjects:

```
subjects=20  no_node_or_no_edges=6  commit=14  decline=0
```

**Zero declines.** Every subject with graph edges commits; the 6 without edges
take the fail-open path. Escalation has no case to act on.

## What this means

The escalation path is not wrong — it is **unreachable in practice**. The
instrument's subjects are graph-known by construction, so there is always a
proposition to commit to. Escalation only matters for subjects with no graph
presence, which the instrument barely samples.

The 2-query probe in `docs/ESCALATION_VERIFIED.md` was real: on a genuinely
absent subject, escalation turns fluent confabulation into an honest hedge.
But that case is rare in the measured distribution.

## The honest conclusion

- **Do not enable it.** It changes nothing on the measured distribution, and
  enabling an inert flag creates the illusion of a safeguard.
- **The real finding** is that CPG itself removes the decoder from the
  instrument entirely (0 decoder outputs with CPG on vs 226 with CPG off).
  That is the thing to investigate next — see below.

## The bigger finding this uncovered

```
CPG off     decoder 226 / narrative 217 / syntax   1
CPG on      decoder   0 / narrative 178 / syntax 302
```

CPG redirects every decoder output to the narrative or syntax path. That is
not what the mechanism describes — CPG is supposed to *steer* the decoder's
token distribution, not replace it. Either:

- the token boost is strong enough that the decoder's output fails the
  fluency/grounding checks and falls through, or
- the decoder path is being skipped for another reason.

This is the next thing to diagnose. It matters more than escalation because it
affects every measured query.

## Related

- PR #329 — `--cpg-escalate`
- `docs/ESCALATION_MEASUREMENT_CONFOUND.md` — the confound that led here
- `docs/ESCALATION_VERIFIED.md` — the 2-query probe
- `docs/THETA_LOWER_IS_BETTER.md` — the theta sweep
