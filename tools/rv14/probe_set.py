"""FRESH probe set for t_98e02c67 (event polarity).

Deliberately NOT the utterances quoted in the card or in FIX-RV-14 — this is a
held-out set written to break any signal tuned on the development set.
Two families, because the card's premise has two shapes:
  ADVERSE  — something bad happened to the user, usually with NO emotion word
  BENIGN   — an event with the same grammar but nothing bad about it

Each item carries the label plus a note on what a false positive would cost.
"""

# (utterance, label, note)
PROBES = [
    # ── ADVERSE (18) ────────────────────────────────────────────────────────
    ("my landlord terminated my lease early", "adverse",
     "event, no emotion word"),
    ("the surgery got rescheduled for next month", "adverse",
     "passive event, no emotion word"),
    ("my laptop screen cracked again", "adverse",
     "short possessive-attr-ish"),
    ("they rejected my transfer request", "adverse",
     "passive event"),
    ("my mother was diagnosed with diabetes", "adverse",
     "kin + diagnosis, no emotion word"),
    ("the hospital cancelled my appointment", "adverse",
     "eventive, institutional"),
    ("my visa application was denied", "adverse",
     "passive event, parallels passport-issued"),
    ("i failed my driving test again", "adverse",
     "first-person event"),
    ("my scholarship was revoked last month", "adverse",
     "named in the card as the hard case"),
    ("my landlord raised the rent by 200", "adverse",
     "quantified event, no emotion word"),
    ("the workshop got cancelled due to funding", "adverse",
     "passive event"),
    ("my phone was stolen from the library", "adverse",
     "passive event"),
    ("my father lost his job in march", "adverse",
     "kin + job loss"),
    ("i was laid off from the internship", "adverse",
     "passive first-person event"),
    ("my car failed its emissions test", "adverse",
     "eventive, no emotion word"),
    ("the landlord evicted the tenants upstairs", "adverse",
     "impersonal event, no 'my'"),
    ("my prescription dosage got halved", "adverse",
     "passive event, health"),
    ("i was turned down for the fellowship", "adverse",
     "passive first-person event"),

    # ── BENIGN (20) ────────────────────────────────────────────────────────
    ("my passport was issued in 2019", "benign",
     "named in the card as the twin of revoked"),
    ("my lease was renewed for another year", "benign",
     "same grammar as terminated"),
    ("the workshop was rescheduled for friday", "benign",
     "twin of surgery-rescheduled"),
    ("my application was accepted", "benign",
     "twin of denied"),
    ("they approved my transfer request", "benign",
     "twin of rejected-transfer"),
    ("i passed my driving test on the first try", "benign",
     "twin of failed"),
    ("my scholarship was renewed last month", "benign",
     "the exact twin of the hard adverse case"),
    ("my mother was diagnosed with an early stage of something treatable",
     "benign", "diagonal: diagnosis word, benign content"),
    ("the hospital confirmed my appointment for monday", "benign",
     "twin of cancelled"),
    ("my laptop screen got replaced under warranty", "benign",
     "twin of cracked"),
    ("my car passed its emissions test", "benign",
     "exact twin"),
    ("my phone was upgraded at the store", "benign",
     "twin of stolen"),
    ("my father started a new job in march", "benign",
     "twin of lost-his-job"),
    ("the tenants upstairs moved out in april", "benign",
     "twin of evicted"),
    ("my prescription dosage got doubled", "benign",
     "twin of halved"),
    ("i got the fellowship", "benign", "twin of turned-down"),
    ("my rent went down by 50", "benign", "quantified twin of raised"),
    ("the flight was moved to an earlier time", "benign",
     "twin of cancelled"),
    ("my visa was approved in march", "benign", "twin of denied"),
    ("the store was renovated last week", "benign",
     "passive event, no stake"),
]


def labelled(label=None):
    return [(u, l, n) for (u, l, n) in PROBES if label is None or l == label]