# WFA remaining hotspots — scratch A/B (2026-11-27)

**Scope:** scratch-only. Branch `perf/wfa-runtime-hotspots` from `main`
`51eee3e` (PR #549/#550/#552 included). No production file, gate, threshold,
item, prompt, scoring rule or store was changed. No full-bank verdict is
claimed; the previously recorded WFA **FAIL** remains authoritative.

## Diagnosis

The prior 180 s watchdog profile (`docs/investigations/
WFA_BOUNDED_TIMEOUT_HARNESS_ROOT_CAUSE.stats`) showed `run_pd_arms` 179.626 s,
`re._compile` 83.681 s / 390,670 lookups but only 79,892 real compiles,
`user_model.mine_personal_facts` 65.437 s / 488 calls, and `numpy.clip`
28.345 s / 3,073,760 calls. A focused 18-turn probe on unmodified production
code (`scratch/profile_regex_patterns.py`) found repeated *static* miner
patterns (e.g. the quantity and possession miners) compiled on nearly every
turn. CPython 3.14's pattern cache is only 512 entries (`_MAXCACHE`) / 256
(`_MAXCACHE2`), so the WFA prompt stream evicts patterns that are used again on
the next turn. This, not the regex matching itself, is the `re._compile`
hotspot and a large part of `mine_personal_facts`.

## Candidate A — enlarge the stdlib regex cache at harness startup

Set `re._MAXCACHE = re._MAXCACHE2 = 8192` in the scratch driver BEFORE the
unmodified harness/engine imports or runs. Compiled pattern objects and their
semantics are untouched; only eviction frequency changes.

### Focused 18-turn A/B, one engine per arm, separate processes
(`scratch/ab_re_cache.py`, `PYTHONHASHSEED=0`, identical prompt list)

| arm | real compiles | turn time | 18 byte-exact answers | state digest |
|---|---:|---:|---|---|
| A default (512/256) | 7,505 | 11.73 s | reference | `fcb77af…` |
| B 8192/8192 | 734 | 9.40 s | identical | `fcb77af…` |

−90.2 % compiles, −19.8 % turn time. **Order control:** two arms in ONE
process are NOT comparable — A/A runs (`ab_re_cache_aa.log`) and B/A runs
(`ab_re_cache_ba.log`) both show 8 answer differences tied to arm *position*
(the first engine boot leaves module-level warm state), not to the cache. The
valid control is one arm per fresh process / first position, which is what the
table above uses.

### Bounded WFA harness (`WFA_BOUNDED_DIAGNOSTIC=1`, max=2, 84 items)
(`scratch/run_wfa_bounded.py`; worktree had NO gitignored GloVe cache at this
point; 24 `process_turn` calls, 8 engine boots)

| arm | wall | real compiles | `learn5_wfa_probes.jsonl` SHA-256 |
|---|---:|---:|---|
| A first run | 23.3 s | 10,111 | `010e0f1611617c16…` |
| A repeat (A/A control) | 16.2 s | 10,094 | `010e0f1611617c16…` |
| B | 8.9 s | 747 | `010e0f1611617c16…` |

−92.6 % compiles; 1.8–2.6× wall speedup (wall itself varies between runs).
`bounded_A_summary.json` and `bounded_B_summary.json` are byte-identical
except the harness's own `elapsed_s` field; stdout logs are identical after
removing that line.

With the gitignored 144 MB `data/ravana_glove_cache.npz` copied from the parent
checkout into the worktree (so the engine boots the same 400,001 GloVe vectors
as the parent's store): A 18.4 s / 9,801 compiles, B 16.2 s / 751 compiles.
All three arms (A, B, B2 below) again produced the byte-identical probe file
SHA `486544c223aa8af44055…`. The GloVe-equipped run is boot-bound: 8 loads of
the projected 400k×64 matrix dominate, so the re-cache win is partly masked.

### Reprofile after the fix
`scratch/profile_bounded_after_re_cache.txt`: `re._compile` 42,421 lookups /
0.683 s (was 83.7 s in the 180 s baseline), 747 real compiles;
`mine_personal_facts` 1.903 s / 55 calls; `numpy.clip` 1.260 s / 160,003
calls. New dominant cost is the one-time NLTK WordNet corpus load (~9.8 s) and
repeated engine boots. `numpy.clip` is no longer a binding hotspot in the
bounded run and was deliberately NOT touched (changing scalar clip semantics
risks type/value drift for no measured benefit here).

## Candidate B — memoize the immutable GloVe boot payload

`scratch/run_wfa_bounded_glove.py` wraps `GraphMixin._init_glove` from the
scratch driver only. Cache key = (absolute npz path, size, mtime_ns, dim).
The first boot loads normally; later boots reuse `(words, word_index, proj,
vecs_proj)` and set fresh per-engine mutable dicts. `_glove_vector` calls
`astype(np.float32)` before normalising, so sharing the base matrix cannot
mutate it.

| arm (GloVe cache present) | wall | compiles | GloVe reuse hits | probes SHA |
|---|---:|---:|---:|---|
| A default | 18.4 s | 9,801 | 0 | `486544c2…` |
| B re-cache only | 16.2 s | 751 | 0 | `486544c2…` |
| B2 re-cache + GloVe memo | 9.3 s | 751 | 7 | `486544c2…` |

B2 is 1.98× faster than A on identical items/store and byte-identical output.
Vector-level control (`scratch/ab_glove_vector_equivalence.py`): 300-word
vector digest + projected-matrix digest equal across 6 sequential boots
(`3e53f1d6c8b34b98f7f905c5…`, `GLOVE_MEMO_EQUIVALENCE=PASS`).

## Full-bank checks (no gate/item/threshold change)

- With the re-cache candidate and NO GloVe cache in the worktree, the full
  272-item harness **completed** in 475.9 s with exit 1 (unchanged WFA FAIL,
  not a new verdict): `full_B_reonly.*`.
- With the GloVe cache present AND both candidates, the full run did **not**
  finish inside the 600 s foreground cap: it stopped inside the PD arm after
  308 `[ground_query] input=` lines (~100 turns) and only 1 GloVe load. No
  endpoint was produced; this is a timeout, not a FAIL or PASS. Full-bank
  runtime on the GloVe store therefore remains unresolved. Do not claim Phase
  0 unblocked on that store.

## Exact recommendation

1. Land Candidate A as a harness-startup (default-on for measurement runs)
   two-line cache enlargement, `re._MAXCACHE = re._MAXCACHE2 = 8192`, with an
   equivalence test that compares per-item probe rows against the frozen
   baseline. It is the only change that directly removes the reported
   `re._compile` hotspot and it is byte-identical here.
2. Treat Candidate B as a SEPARATE, default-OFF harness flag keyed by npz
   identity + dim, with the 300-word vector digest test as its regression
   guard. It is safe in this measured single-process sequential-boot pattern,
   but sharing a 100 MB-class array across engines deserves an explicit opt-in
   and a memory review before any production use.
3. Do not touch `numpy.clip` or WordNet/gates/scoring on this evidence. The
   next profiling target for a GloVe-equipped full bank is why turns slow to
   ~seconds each as the graph grows; that is a separate investigation.

All artefacts are scratch-only: scripts, logs, JSON/JSONL outputs, and this
finding. `data/ravana_glove_cache.npz` is gitignored and was NOT committed.
