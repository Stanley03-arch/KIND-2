"""Beat-TypeSafe harness: same held-out set vs every engine.
Skips hosted/Modal when no key/URL — always runs local legacy vs kind offline.

Run:
  python beat_typesafe.py
  TYPESAFE_API_KEY=jv_... python beat_typesafe.py            # + hosted jev-1.13.0
  KIND_MODAL_URL=https://... python beat_typesafe.py         # + your Modal deploy

Win bar (from tests/test_eval_jev.py): band_acc>=0.66, ECE<0.35.
Beat bar: band_acc>=0.90, ECE<0.15, 5/5 robustness, p50<20ms.
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

CASES = [
    {"name": "paraphrase", "state": "The application falls over whenever I launch it in the morning",
     "questions": {"dept": {"type": "choice", "instructions": "Route ticket",
                            "criteria": {"billing": "invoices refunds", "tech_support": "bugs crashes", "sales": "upgrades"}}},
     "check": ("choice", "dept", "tech_support")},
    {"name": "negation", "state": "Please refund the duplicate card charge on invoice 442.",
     "questions": {"q": {"type": "noul", "instructions": "Is the customer not asking for a refund?"}},
     "check": ("noul_lt", "q", 0.5)},
    {"name": "backtick", "state": {"ticket_message": "Please refund the duplicate card charge.",
                                   "logs": "Null pointer exception, server crash, 500 error."},
     "questions": {"dept": {"type": "choice", "instructions": "Which team should handle `ticket_message`?",
                            "criteria": {"billing": "payments refunds invoices", "technical": "bugs crashes server errors"}}},
     "check": ("choice", "dept", "billing")},
    {"name": "wip!=wipe", "state": "WIP the design doc and share it with the team.",
     "questions": {"wipe": {"type": "noul", "instructions": "Does this request wipe or delete stored data?"}},
     "check": ("verdict", "wipe", False)},
    {"name": "severity", "state": "Nobody on our team can log in since this morning. We get a 500 error on every attempt.",
     "questions": {"sev": {"type": "score", "instructions": "How severe is the reported issue?",
                           "criteria": ["Cosmetic; no impact to functionality",
                                        "Broken or degraded feature, but workaround exists",
                                        "Blocking issue; no workaround exists"]}},
     "check": ("score_near", "sev", 2.0, 0.75)},
]


def _pass(resp, check):
    kind, qid = check[0], check[1]
    a = resp["answers"][qid] if isinstance(resp, dict) else resp.answers[qid]
    get = (lambda k: a[k]) if isinstance(a, dict) else (lambda k: getattr(a, k))
    if kind == "choice":
        return get("choice") == check[2]
    if kind == "noul_lt":
        return float(get("noul")) < float(check[2])
    if kind == "verdict":
        v = a.get("verdict", float(a.get("noul", 1)) >= 0.5) if isinstance(a, dict) else getattr(a, "verdict")
        return bool(v) is bool(check[2])
    if kind == "score_near":
        return abs(float(get("score")) - float(check[2])) <= float(check[3])
    return False


def _local_v1():
    from kind import KindClient
    c = KindClient(encoder="hash")
    def call(state, questions):
        t0 = time.perf_counter()
        r = c.system_one(state=state, questions=questions)
        ms = (time.perf_counter() - t0) * 1000
        return {"answers": {k: v.model_dump() for k, v in r.answers.items()}, "ms": ms}
    return ("kind-legacy(hash)", call)


def _local_v2():
    from kind.engine import KindEngine
    e = KindEngine()
    def call(state, questions):
        t0 = time.perf_counter()
        r = e.decide(state=state, questions=questions)
        ms = (time.perf_counter() - t0) * 1000
        return {"answers": {k: v.model_dump() for k, v in r.answers.items()}, "ms": ms}
    return (f"kind-local({e.encoder.backend})", call)


def _modal():
    url = (os.getenv("KIND_MODAL_URL") or os.getenv("JEV2_MODAL_URL", "")).rstrip("/")
    if not url:
        return None
    import requests
    def call(state, questions):
        t0 = time.perf_counter()
        r = requests.post(url + "/v1/systemone",
                          json={"state": state, "questions": questions}, timeout=30)
        r.raise_for_status()
        ms = (time.perf_counter() - t0) * 1000
        return {"answers": r.json()["answers"], "ms": ms}
    return ("kind-modal", call)


def _hosted():
    key = os.getenv("TYPESAFE_API_KEY", "")
    if not key:
        return None
    import requests
    base = os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
    def call(state, questions):
        t0 = time.perf_counter()
        r = requests.post(base + "/v1/systemone", headers={"Authorization": f"Bearer {key}"},
                          json={"state": state, "questions": questions,
                                "model": "jev-1.13.0"}, timeout=30)
        r.raise_for_status()
        ms = (time.perf_counter() - t0) * 1000
        return {"answers": r.json()["answers"], "ms": ms}
    return ("typesafe-hosted(jev-1.13.0)", call)


def main():
    engines = [_local_v1(), _local_v2()]
    m, h = _modal(), _hosted()
    if m:
        engines.append(m)
    if h:
        engines.append(h)
    scores = {}
    for name, fn in engines:
        hits, lat = 0, []
        for c in CASES:
            try:
                out = fn(c["state"], c["questions"])
                lat.append(out["ms"])
                if _pass(out, c["check"]):
                    hits += 1
                    flag = "HIT"
                else:
                    flag = "MISS"
            except Exception as e:
                flag = f"ERR {e}"
            print(f"{name:28s} {c['name']:12s} {flag}")
        lat.sort()
        p50 = lat[len(lat) // 2] if lat else -1
        scores[name] = (hits, p50)
        print(f"=> {name}: {hits}/{len(CASES)} p50={p50:.1f}ms")
    print("\nSCOREBOARD (higher hits, lower p50 wins):")
    for name, (hits, p50) in sorted(scores.items(), key=lambda kv: (-kv[1][0], kv[1][1])):
        print(f"  {hits}/{len(CASES)}  p50={p50:7.1f}ms  {name}")
    if not h:
        print("\nTip: set TYPESAFE_API_KEY to add hosted row. Set KIND_MODAL_URL after `modal deploy modal_serve.py`.")


if __name__ == "__main__":
    main()
