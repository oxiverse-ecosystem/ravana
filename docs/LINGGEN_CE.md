# LingGen P6 — decoder-CE vs template-CE promotion gate

Measured by `scripts/measure_linggen_ce.py` on a fixed held-out probe set
(N=5, method: per-path cross-entropy on the same probe strings).

Result: **PROMOTION BLOCKED.**

- decoder-CE: undefined (no trained decoder checkpoint present in the repo)
- template-CE: undefined (no trained template distribution present)
- `use_linggen` stays `False`.

Reason: no measurement harness could compute both CE values because the
required trained artifacts (decoder weights + template distribution) do not
exist in this checkout. Per the promotion contract, the flag flips to True
only when decoder-CE <= template-CE on a real measurement; absent that
evidence it remains False. No LLM, no retraining, no hand-switch was applied.
