# Capability: normalize garbled "mind - X" attributes and stance topics from "changed my mind - X is Y" disclosures

## What it does

When a user revises an opinion with a "changed my mind - X is Y" disclosure,
RAVANA's miner can mis-parse the disclosure and store a garbled meta-attribute
`"mind - X"` (and a garbled stance topic `"mind - X"` / `"changed"`) instead of
the real topic `X`. This capability normalizes those garbled values at the
storage boundary so recalls and stance replies render the real subject:

- "what do you think about privacy?" now returns "i'm against privacy." instead
  of the garbled "your mind - privacy is actually negotiable sometimes."

The normalization is structural: any attribute/topic/opinion phrase containing
`" - "` keeps only the part after the first `" - "`. No per-topic table.

**Locations:**
- `ravana/src/ravana/chat/user_model.py:2265` (`_put_fact`) — strips `" - "` lead-in from fact attributes.
- `ravana/src/ravana/chat/user_model.py:7142` (`_opinion_topic`) — strips `" - "` lead-in from opinion phrases.
- `ravana/src/ravana/chat/personal_fact_store.py:429` (`UserStanceStore.express_stance`) — strips `" - "` lead-in from stance topics.
- `ravana/src/ravana/chat/engine.py:3471` — strips `" - "` lead-in from the rendered fact attribute.

## How it grew from the conversation

The round probe (auto/round-2026-11-04T0634Z) surfaced a real defect: a
"changed my mind - X is Y" disclosure was mined into a garbled meta-attribute
`"mind - X"` and a garbled stance topic, so RAVANA's reply echoed the
meta-disclosure marker instead of the real subject ("your mind - privacy is
actually negotiable sometimes."). Root cause was that the miner treated the
"changed my mind - X" prefix as part of the attribute/topic rather than as a
meta-disclosure marker.

## Design

Normalization happens at the storage boundary (fact put, opinion topic
extraction, stance expression) plus the render site (engine fact recall), so
both persisted state and live replies are clean. The rule is purely structural
(split on `" - "`, keep the tail) — no per-topic lookup, no retraining, and
RAVANA can revise the value later through normal conversation.

## Verification

- Source change: commit `c6a13aa9` (23 insertions across 3 files).
- Verified live: "what do you think about privacy?" returns "i'm against
  privacy." instead of the garbled reply.
- Existing unit suite remains green.
