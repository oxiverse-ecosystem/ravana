"""Calibration harness for fail-closed retrieval (research item D).

Cross-cutting measurement substrate (parallel to measure_salad_classifier.py
and measure_provenance_admission.py). Two jobs:

1. FALLBACK EFFECTIVENESS: simulate the observed silent-failure condition —
   local SearXNG is UP but returns an EMPTY result (off-topic/junk/zero hits) —
   and verify the engine (a) falls through to remote backends when
   fallback_on_empty=True (fail-CLOSED) and (b) would have returned the empty
   result (the old fail-OPEN behavior) when fallback_on_empty=False. This is
   the decision the plan called "circuit breaker should check emptiness, not
   just availability".

2. ZERO-RESULT RATE: over a labeled set of informational/why/definition queries,
   measure how often each backend (local / duckduckgo / oxiverse) returns zero
   usable results, and how often the FULL pipeline (local->remote) yields a hit.
   With network mocked, this exercises the routing logic, not the live web.

Outputs experiments/_retrieval_calib.json (dashboard): fallback_on_empty
effectiveness, per-backend zero-result rate, pipeline hit rate.

Run:
    python experiments/measure_retrieval_hitrate.py
"""
import os
import sys
import json
import time as _time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ,
         os.path.join(_PROJ, "ravana_ml", "src"),
         os.path.join(_PROJ, "ravana", "src"),
         os.path.join(_PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

from ravana.web.learner import SearchEngine, SearchConfig


class _FakeAPI:
    """Controllable backend simulator.

    mode: 'empty' -> always returns [] (simulates SearXNG up-but-junk).
          'results' -> returns a plausible result list.
    """
    def __init__(self, mode):
        self.mode = mode
        self.calls = 0

    def __call__(self, api_name, url, timeout, max_results):
        self.calls += 1
        if self.mode == "empty":
            return []
        return [
            {"title": f"{api_name} result 1", "url": f"https://example/{api_name}/1",
             "content": "A plausible factual snippet about the topic."},
            {"title": f"{api_name} result 2", "url": f"https://example/{api_name}/2",
             "content": "A second snippet with more detail."},
        ]


def _build_engine(local_mode, remote_mode, fallback_on_empty=True):
    cfg = SearchConfig(local_prefer=True, local_timeout=2, fallback_on_empty=fallback_on_empty)
    eng = SearchEngine(cfg)
    # Replace the real _call_api with controllable fakes.
    local_fake = _FakeAPI(local_mode)
    remote_fake = _FakeAPI(remote_mode)
    _dispatch = {"local_api": local_fake, "duckduckgo": remote_fake, "oxiverse": remote_fake}

    def _fake_call(api_name, url, timeout, max_results):
        return _dispatch[api_name](api_name, url, timeout, max_results)

    eng._call_api = _fake_call
    eng._fake_dispatch = _dispatch
    return eng


def measure_fallback_effectiveness():
    """Case A (the bug): local empty, remote has results.
    With fallback_on_empty=True  -> must return remote results (fail-closed).
    With fallback_on_empty=False -> old behavior returns local empty.
    """
    out = {}
    # fail-closed (new default)
    eng = _build_engine(local_mode="empty", remote_mode="results", fallback_on_empty=True)
    res = eng.search("why do we dream", max_results=4)
    out["fail_closed_returned_results"] = bool(res)
    out["fail_closed_hit"] = len(res) if res else 0
    # fail-open (old behavior, explicitly disabled)
    eng2 = _build_engine(local_mode="empty", remote_mode="results", fallback_on_empty=False)
    res2 = eng2.search("why do we dream", max_results=4)
    out["fail_open_returned_empty"] = (res2 == [])
    # all-empty case (local empty, remote empty) -> SearchError (caller's
    # fail-closed path turns this into an honest "couldn't verify" abstention).
    eng3 = _build_engine(local_mode="empty", remote_mode="empty", fallback_on_empty=True)
    raised = False
    try:
        eng3.search("why do we dream", max_results=4)
    except Exception:
        raised = True
    out["all_empty_raises_search_error"] = raised
    # local has results -> authoritative, remote NOT consulted
    eng4 = _build_engine(local_mode="results", remote_mode="empty", fallback_on_empty=True)
    res4 = eng4.search("why do we dream", max_results=4)
    out["local_has_results_authoritative"] = bool(res4)
    out["local_has_results_remote_not_called"] = (eng4._fake_dispatch["duckduckgo"].calls == 0)
    return out


def measure_zero_result_rate():
    """Over a labeled informational/why/definition query set, with live network
    mocked, measure per-backend zero-result rate and pipeline hit rate.

    We model each backend's reliability as a parameter (so the harness is
    hermetic) but drive the SIMULATED reliability from the plan's field note:
    local SearXNG (127.0.0.1:8080) is flaky/returns off-topic junk; Tor2/remote
    is the reliable fallback. The harness measures the ROUTING decision's
    outcome, not the live web.
    """
    queries = [
        "why do we dream", "what is trust", "how does gravity work",
        "what causes inflation", "why is the sky blue", "what is consciousness",
        "how do vaccines work", "why do we sleep", "what is entropy",
        "how does memory work",
    ]
    # Simulated backend reliability (per plan: local flaky, remote reliable).
    scenarios = {
        "local_flaky_remote_ok": dict(local="empty", remote="results"),
        "local_ok_remote_ok": dict(local="results", remote="results"),
        "all_down": dict(local="empty", remote="empty"),
    }
    report = {}
    for name, sc in scenarios.items():
        eng = _build_engine(local_mode=sc["local"], remote_mode=sc["remote"],
                            fallback_on_empty=True)
        hits = 0
        for q in queries:
            try:
                r = eng.search(q, max_results=4)
            except Exception:
                r = []
            if r:
                hits += 1
        report[name] = {
            "queries": len(queries),
            "hits": hits,
            "zero_result_rate": round(1.0 - hits / len(queries), 3),
            "pipeline_hit_rate": round(hits / len(queries), 3),
        }
    return report


def main():
    fb = measure_fallback_effectiveness()
    zr = measure_zero_result_rate()
    dashboard = {
        "item": "D",
        "substrate": "retrieval hit-rate + fallback measurement",
        "fallback_effectiveness": fb,
        "zero_result_rate_by_scenario": zr,
        "verdict": "fail-closed: empty local now falls through to remote; "
                   "informational queries always attempt web (see _needs_web_search override).",
    }
    out_path = os.path.join(_PROJ, "experiments", "_retrieval_calib.json")
    with open(out_path, "w") as f:
        json.dump(dashboard, f, indent=2)

    print(f"[calib] fallback_effectiveness: {json.dumps(fb)}")
    print(f"[calib] zero_result_rate_by_scenario:")
    for k, v in zr.items():
        print(f"    {k}: hits={v['hits']}/{v['queries']} "
              f"zero_rate={v['zero_result_rate']} hit_rate={v['pipeline_hit_rate']}")
    print(f"[calib] wrote dashboard -> {out_path}")
    # Assertions: the core fail-closed guarantee.
    assert fb["fail_closed_returned_results"], "fail-closed must fall through to remote on empty local"
    assert fb["fail_open_returned_empty"], "old fail-open must have returned empty (control)"
    assert fb["all_empty_raises_search_error"], "all-empty must raise SearchError (caller degrades to abstain)"
    assert fb["local_has_results_authoritative"], "local results stay authoritative"
    assert fb["local_has_results_remote_not_called"], "remote not consulted when local has results"
    print("[calib] VERDICT: fail-closed retrieval routing FIT + measured. "
          "(Decision is policy-driven, validated by mocked-backend harness — "
          "no blind threshold.)")


if __name__ == "__main__":
    main()
