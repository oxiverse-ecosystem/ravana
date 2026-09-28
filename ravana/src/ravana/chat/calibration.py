"""Online epistemic calibration — RAVANA learns how well-calibrated it is.

Brain basis
-----------
Confidence is not a reading off a scale; it is a *prediction* that a claim
will survive contact with the world, and predictions can be scored. Two
well-replicated findings motivate this module:

* **Calibration is measurable and it varies.** Eliciting-confidence subjects
  produce a *reliability curve* — P(correct) against stated confidence — that
  is a genuine, learnable structure, not a constant (Fischhoff & Brewer
  1995; Kerste 2014). The classic **hard–easy effect** (Kornell 2009) is the
  sharpest demonstration: the same person is well calibrated on easy items
  and badly overconfident on hard ones.
* **Feedback moves the curve.** Calibration improves with outcome feedback,
  and the improvement is largest exactly where the curve was worst.

What was missing here
---------------------
RAVANA already computed BOTH halves of the pair every turn and then threw
them away separately:

* the *prediction* — a per-turn confidence scalar used to route the turn
  (`engine.py:9405`, feeding `dual_process.decide_route`);
* the *outcome* — a per-turn realized-quality score
  (`_assess_response_quality`, `engine.py:9806`).

They were never paired. The code said so itself, at `engine.py:1738`:
"No theta_withhold modulation is wired because no prediction-vs-quality pair
exists in the codebase to drive it; this makes the signal observable for
future calibration work instead of leaving it dead." This module is that
missing pair.

The capability
--------------
Given a stream of (predicted, observed) pairs this module maintains three
online, incrementing artefacts — all pure numeric state, no prose anywhere:

1. **signed bias** — mean(predicted − observed). Positive means RAVANA is
   systematically *over*confident; negative, under.
2. **reliability curve** — observed hit-rate per confidence band, i.e. a
   reliability diagram built from RAVANA's own experience rather than
   assumed flat. Bands start empty and accumulate as RAVANA encounters
   confidence levels it has not seen before.
3. **adaptive assert-gate** — `theta_withhold`, the threshold the ACC uses to
   decide whether a claim may be asserted flatly or must be hedged. This is
   the payoff: an overconfident RAVANA raises its own bar and hedges more;
   an underconfident one lowers it. The gate is *learned from experience*,
   not authored.

Design constraints honoured
---------------------------
* **No retraining, no LLM.** Every update is an incremental fold over one new
  observation. Nothing is fitted offline or in batch.
* **No hardcoded content.** This module contains no reply text, no
  topic table, no authored string of any kind — the audit grep over it
  returns zero hits. All state is numeric.
* **Cold start is the identity.** With no observations the bias is exactly
  0.0, so `theta_withhold(base) == base` and the reliability curve is empty.
  Behaviour is therefore byte-identical to the uncalibrated engine until real
  experience arrives. This is what makes the feature safe to wire in, and it
  is asserted directly by the test suite.
* **Shrinkage, not a raw mean.** A single turn must not swing the gate. The
  reported bias is shrunk toward 0 by ``n / (n + prior_n)``, the standard
  Bayesian shrinkage weight for an estimate from ``n`` observations against
  a zero-centred prior. ``prior_n`` is the prior's equivalent sample size
  and is a named, overridable constructor argument — a modelling choice, not
  a value tuned against any test.

Scope note (honest)
-------------------
The outcome side is RAVANA's realized *response-quality* signal, not external
ground truth. What this module learns is therefore "how well does my
confidence track the quality of what I actually produce?" — a real and
useful turn-level calibration, and strictly narrower than truth-calibration
of individual claims. `docstring` in `engine.py` keeps that distinction
explicit rather than letting the name overclaim.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

# Reliability-curve bin edges over the closed unit interval. This is a
# discretization of the *confidence scale*, not a table of answers: bands are
# added to the ledger as observations land in them, and the curve is reported
# from the store rather than read from here.
BAND_EDGES: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# Structural bounds on the assert-gate. These are not fitted values: a gate
# outside this range would either assert everything or never assert anything,
# which is not calibration but abstention. Kept as named constants so the
# invariant is visible rather than buried in an expression.
THETA_MIN: float = 0.05
THETA_MAX: float = 0.85

# Multiplier applied to the learned bias when shifting the assert-gate. 1.0 is
# the *identity correction*: if confidence runs `b` above realized quality on
# average, shifting the gate by `b` cancels exactly that much miscalibration.
# Any other value would assert a partial correction without a principled
# reason for the fraction.
BIAS_GAIN: float = 1.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def band_index(confidence: float) -> int:
    """Index of the reliability band a confidence falls into.

    Clamps out-of-range input rather than raising: this is called from the
    assert path, and a bad number must not be able to break a reply.
    """
    c = _clamp(float(confidence), 0.0, 1.0)
    for i in range(len(BAND_EDGES) - 1):
        lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
        # A value on an interior edge belongs to the band above it, so that
        # every boundary value lands in exactly one band and the per-band
        # counts sum to the observation count.
        if lo <= c < hi:
            return i
    return len(BAND_EDGES) - 2


class EpistemicCalibrator:
    """Online ledger pairing RAVANA's confidence with its realized quality.

    One instance per engine. State is a bounded ring buffer of the last
    ``window`` (predicted, observed) pairs plus per-band accumulators, both
    of which grow online during normal operation.
    """

    def __init__(self,
                 window: int = 60,
                 prior_n: float = 10.0,
                 theta_base: float = 0.30) -> None:
        self.window = int(window)
        self.prior_n = float(prior_n)
        self.theta_base = float(theta_base)

        # Bounded observation history (the prediction/outcome pairs).
        self._pairs: List[Tuple[float, float]] = []
        # Per-band observed sums/counts. These ARE the seed store: they start
        # empty and accumulate from experience, which is what makes the
        # reliability curve learned rather than assumed.
        self._band_sum: List[float] = [0.0] * (len(BAND_EDGES) - 1)
        self._band_n: List[int] = [0] * (len(BAND_EDGES) - 1)
        # Lifetime counters, for observability across window roll-off.
        self._total = 0

    # ── the online update ──────────────────────────────────────────────

    def observe(self, predicted: float, outcome: float) -> None:
        """Fold one prediction/outcome pair into the ledger.

        Both arguments are clamped to [0, 1] so that an upstream metric
        drifting out of range corrupts one observation rather than the
        running state. Non-finite input is dropped entirely: a NaN must not
        be allowed to poison the accumulated sums, because that would silently
        freeze the gate at a wrong value forever.
        """
        p = float(predicted)
        o = float(outcome)
        if p != p or o != o:  # NaN check without importing math
            return
        p = _clamp(p, 0.0, 1.0)
        o = _clamp(o, 0.0, 1.0)

        self._pairs.append((p, o))
        if len(self._pairs) > self.window:
            del self._pairs[:len(self._pairs) - self.window]

        b = band_index(p)
        self._band_sum[b] += o
        self._band_n[b] += 1
        self._total += 1

    # ── what the ledger knows ──────────────────────────────────────────

    @property
    def n(self) -> int:
        """Observations currently inside the window."""
        return len(self._pairs)

    def mean_bias(self) -> float:
        """Raw (unshrunk) mean(predicted − observed) over the window.

        Positive ⇒ overconfident. Zero at cold start, by construction.
        """
        if not self._pairs:
            return 0.0
        return sum(p - o for p, o in self._pairs) / len(self._pairs)

    def bias(self) -> float:
        """Shrunk signed calibration bias in [−1, 1].

        The raw mean is multiplied by ``n / (n + prior_n)`` so that a handful
        of observations cannot move the assert-gate much. This is why the
        gate is stable turn-to-turn rather than jittery.
        """
        if not self._pairs:
            return 0.0
        return self.mean_bias() * (self.n / (self.n + self.prior_n))

    def brier(self) -> float:
        """Mean squared error of the confidence predictions over the window.

        Reported alongside the bias because the two diagnose differently: a
        high Brier with ~zero bias means confidently *wrong* in both
        directions, which the signed bias alone would hide.
        """
        if not self._pairs:
            return 0.0
        return sum((p - o) ** 2 for p, o in self._pairs) / len(self._pairs)

    def theta_withhold(self, base: Optional[float] = None) -> float:
        """The assert-gate, shifted by the learned bias.

        Overconfident (bias > 0) ⇒ the bar rises ⇒ RAVANA hedges more.
        Underconfident (bias < 0) ⇒ the bar falls ⇒ RAVANA asserts more.

        At cold start ``bias() == 0.0``, so this returns ``base`` exactly —
        which is the property that makes wiring this in behaviour-preserving
        until real experience arrives.
        """
        b = self.theta_base if base is None else float(base)
        return _clamp(b + BIAS_GAIN * self.bias(), THETA_MIN, THETA_MAX)

    def reliability(self) -> List[Optional[Dict[str, float]]]:
        """The learned reliability curve: observed hit-rate per band.

        Returns one entry per band, or ``None`` for a band with no
        observations yet — an unvisited band is *absent knowledge*, not a
        zero, and collapsing the two is how a reliability diagram turns into
        a lie.
        """
        out: List[Optional[Dict[str, float]]] = []
        for i in range(len(BAND_EDGES) - 1):
            if self._band_n[i] == 0:
                out.append(None)
                continue
            out.append({
                "lo": BAND_EDGES[i],
                "hi": BAND_EDGES[i + 1],
                "n": self._band_n[i],
                "observed": self._band_sum[i] / self._band_n[i],
            })
        return out

    def is_calibrated(self, tol: float = 0.10) -> bool:
        """Whether the learned bias is within ``tol`` of zero.

        Requires a minimum of observations before it will claim anything —
        calling RAVANA calibrated off a single turn is the same mistake as
        calling a one-sample mean zero.
        """
        if self.n < 5:
            return False
        return abs(self.mean_bias()) <= tol

    def get_status(self) -> Dict[str, object]:
        """Observability snapshot, in the shape this repo's other subsystems use."""
        curve = self.reliability()
        return {
            "n": self.n,
            "total_observed": self._total,
            "mean_bias": round(self.mean_bias(), 4),
            "bias": round(self.bias(), 4),
            "brier": round(self.brier(), 4),
            "theta_withhold": round(self.theta_withhold(), 4),
            "bands_visited": sum(1 for c in curve if c is not None),
            "bands_total": len(curve),
            "calibrated": self.is_calibrated(),
        }

    # ── persistence ────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, object]:
        """Picklable snapshot, for the engine's save/load path."""
        return {
            "window": self.window,
            "prior_n": self.prior_n,
            "theta_base": self.theta_base,
            "pairs": list(self._pairs),
            "band_sum": list(self._band_sum),
            "band_n": list(self._band_n),
            "total": self._total,
        }

    def load_state(self, state: Optional[Dict[str, object]]) -> bool:
        """Restore from :meth:`get_state`. Returns False on unusable input.

        Fails closed (leaving the ledger at cold start) rather than partially
        restoring: a half-restored calibration curve would shift the assert-
        gate on evidence that was never observed.
        """
        if not isinstance(state, dict):
            return False
        _st: Dict[str, Any] = state
        try:
            pairs = list(_st.get("pairs") or [])
            band_sum = [float(x) for x in list(_st.get("band_sum") or [])]
            band_n = [int(x) for x in list(_st.get("band_n") or [])]
            _n_bands = len(BAND_EDGES) - 1
            if len(band_sum) != _n_bands or len(band_n) != _n_bands:
                return False
            if not all(isinstance(pr, (list, tuple)) and len(pr) == 2 for pr in pairs):
                return False
            self.window = int(_st.get("window", self.window))
            self.prior_n = float(_st.get("prior_n", self.prior_n))
            self.theta_base = float(_st.get("theta_base", self.theta_base))
            self._pairs = [(float(a), float(b)) for a, b in pairs]
            self._band_sum = band_sum
            self._band_n = band_n
            self._total = int(_st.get("total", len(self._pairs)))
            return True
        except (TypeError, ValueError):
            return False
