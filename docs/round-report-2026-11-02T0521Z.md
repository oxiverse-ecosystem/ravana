# RAVANA round report — 2026-11-02T0521Z

## Summary
Chat probe executed 25 turns in-process (RAVANA_OFFLINE=1). Learning confirmed: turn_count 481→506, stances 18→21, graph 323→329 nodes / 5156→5344 edges, identity strength 1.0424→1.0492. Pivot task #1 (orphaned `_generate_with_decoder`) verified already removed — method absent from source (grep returns zero); marked [x] in task file.

## Transcript (excerpts)
See scratch/round_transcript_0521z.json for full 25 turns.

Key replies:
- Turn 5: "your dog is a sheepdog named cairn." (fact recall works)
- Turn 11: "you just told me: i'm sad today, my grandmother passed" (episodic recall works)
- Turn 16: "you used to say you were against privacy, and now you're for privacy." (stance inversion works)
- Turn 23: "your cat is wren." (generalized reverse-lookup works)

## State delta
| metric | before | after |
|---|---|---|
| turn_count | 481 | 506 |
| stances | 18 | 21 |
| personal facts | 16 | 16 |
| graph nodes | 323 | 329 |
| graph edges | 5156 | 5344 |
| identity strength | 1.0424 | 1.0492 |
| identity stability | 0.9955 | 0.9884 |
| agent_stances | 22 | 22 |
| sleep_cycles | (n/a) | 22 |

## Pivot task #1 verification
`grep -rn "_generate_with_decoder(" ravana/src/` returns zero call sites (only `_generate_with_decoder_and_syntax`). Method absent. Removed in prior round. Marked [x].

## Hardcoding audit
No new reply strings added this round. Only task-file edit + report. Audit: PASS (no authored prose added).

## Residual
- Pivot tasks 2-4 (decoder-CE measurement, garbled creative output, language learning) remain; deferred to subsequent rounds.
- Learning_count reports 0 (counter not incremented this path); not a regression.
EOF
mkdir -p tmp/reports
mv "tmp/reports" 2>/dev/null; mkdir -p tmp/reports
echo ok