# Production output: final state

Date: 2026-11-14 (end of session)
Instrument: `scripts/measure_production_output.py --no-web` — runs the real
`process_turn` entry point.

## Headline

**99 of 100 corpus subjects get a correct, grounded answer.**

The instrument reports 96/100 with 4 "salad" flags, but 3 of those are false
positives in the classifier (`natural`, `original`, `areas` are correct
definitions whose head word differs by inflection from the query). The one
remaining case is `being`, which is held as an incomplete question.

## The failure list, and what fixed each

| subject | before | after | cause / fix |
|---|---|---|---|
| `like` | "according to a web source, Love means to like something a lot." | "In English, the word like has a very wide range of uses..." | definitional queries routed to chitchat; function words definable |
| `being` | decoder salad from the chitchat path | "mm-hmm, what were you going to say?" | same routing bug; now held as an incomplete question |
| `zzzunknownqqq` | "Knowledge is an awareness of facts..." | "i don't really have a solid grasp on zzzunknownqqq..." | substring subject match |
| `natural`, `original`, `areas` | flagged "salad" | flagged "salad" | classifier false positives, not failures |

## The chain of fixes this session

| PR | fix | effect |
|---|---|---|
| #347/#348 | corpus → concept graph | 512/512 subjects graph-known |
| **#355** | corpus → `_definitions` | dispatch coverage **74% → 99%** |
| #354 | definition stutter guard | correct answers no longer withheld on a surface artifact |
| #351 | reference-check content filter | the gate no longer passes salad |
| #358 | substring subject match | nonsense queries no longer re-pointed at real concepts |
| #359 | definitional-query routing + function-word guard | `like`/`being` no longer produce salad |

## The reframe that mattered

`eval_decoder_groundedness.py` calls `_generate_with_situation_model` directly,
bypassing dispatch. Production consults stored knowledge FIRST:

```
consult_internal:  _definitions -> hippocampus -> graph edges -> belief -> MISS
                                                         |
                                            decoder / live web (fallback)
```

So the decoder's salad is the **fallback**, not the primary path. The "57% gate
withholding" I chased for most of the session measured the fallback path's
quality, not the engine's answers.

## Honest limits

- **99/100 is on corpus subjects** — concepts RAVANA was taught. Novel subjects
  still depend on the decoder and live web.
- **The decoder itself is unchanged and still weak.** It produces salad when
  consulted. It just isn't the primary path any more.
- **The salad classifier over-flags inflections.** Treat its counts as
  approximate; the samples are the evidence.
- **`being` is held, not answered.** "what is being" is genuinely incomplete, so
  this is defensible, but it is not a grounded answer.
