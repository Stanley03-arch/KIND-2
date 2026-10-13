"""Beat-Kind harness: hash vs hybrid vs native Laya vs hosted API (if key).

Beats the hosted TypeSafe decision API iff ALL hold on the SAME seed/cases:
  accuracy overall >= hosted AND >=0.90, ECE(noul) <0.15 AND < hosted,
  p50 <20ms, 5/5 robustness, deterministic repeats, $0 marginal cost.

Without TYPESAFE_API_KEY: runs local engines + prints published vendor proxies
clearly marked PROXY — not a live win claim.

Run:
  python beat_kind.py                      # hash + hybrid (fast, no 60x Laya calls)
  python beat_kind.py --engines hash,hybrid,laya
  TYPESAFE_API_KEY=jv_... python beat_kind.py --engines hash,hybrid,hosted
"""
from __future__ import annotations
import argparse, json, os, sys, time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from benchmark_extended import build_cases, grade, SCORE_TOLERANCE, SEED
from kind.calibration import expected_calibration_error
from beat_typesafe import CASES as ROBUST5

PUBLISHED_TYPESAFE = {
    "note": "PROXY — third-party/vendor numbers, NOT same-seed live. Set TYPESAFE_API_KEY for live row.",
    "typed_decisions_acc": 0.727,
    "ag_news_acc": 0.910,
    "emotion_acc": 0.480,
    "ece": 0.246,
    "p50_ms_claimed": "70-500 vendor, ~210-276 observed",
    "price_per_1M_input": 0.042,
    "weights": "closed API",
    "determinism": "15 distinct answers / 50 identical requests (audit)",
    "coherence": "complementary probs sum 0.71-1.42 (should be 1.0)",
}

def run_engine(name, fn, cases):
    rows, lat, pairs = [], [], []
    for c in cases:
        t0 = time.perf_counter()
        try:
            ans = fn(c["state"], c["question"])
            ok, p = grade(c, ans)
            err = None
        except Exception as e:
            ans, ok, p, err = None, False, None, str(e)[:160]
        lat.append((time.perf_counter() - t0) * 1000.0)
        if c["cat"] == "noul" and p is not None:
            pairs.append((p, 1 if c["expected"] else 0))
        rows.append({"id": c["id"], "cat": c["cat"], "ok": bool(ok), "error": err})
    def acc(k):
        s = [r for r in rows if r["cat"] == k]
        return round(sum(r["ok"] for r in s) / len(s), 4)
    ls = sorted(lat)
    return {
        "overall": round(sum(r["ok"] for r in rows) / len(rows), 4),
        "noul": acc("noul"), "choice": acc("choice"), "score": acc("score"),
        "ece_noul": round(expected_calibration_error(pairs), 4),
        "p50_ms": round(ls[len(ls)//2], 2), "p95_ms": round(ls[int(0.95*(len(ls)-1))], 2),
        "misses": [r["id"] for r in rows if not r["ok"]],
    }

def fn_hash():
    from kind import KindClient
    c = KindClient(encoder="hash")
    def call(state, question):
        return c.system_one(state=state, questions={"q": question}).answers["q"].model_dump()
    return call

def fn_hybrid():
    from kind.beatkind import BeatKindEngine
    e = BeatKindEngine()
    e._laya_engine()  # warmup outside timing where possible
    def call(state, question):
        return e.decide(state=state, questions={"q": question}).answers["q"].model_dump()
    call.engine = e
    return call

def fn_laya():
    from kind.laya import LayaEngine
    e = LayaEngine()
    assert e.is_native(), f"laya not native: {e.backend}"
    def call(state, question):
        return e.decide(state=state, questions={"q": question}).answers["q"].model_dump()
    return call

def fn_hosted():
    import requests
    key = os.getenv("TYPESAFE_API_KEY", "")
    base = os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
    def call(state, question):
        r = requests.post(base + "/v1/systemone", headers={"Authorization": f"Bearer {key}"},
                          json={"state": state, "questions": {"q": question}, "model": "jev-1.13.0"}, timeout=30)
        r.raise_for_status()
        return r.json()["answers"]["q"]
    return call

def robust5(fn):
    from beat_typesafe import _pass
    hits = 0
    for c in ROBUST5:
        try:
            answers = {qid: fn(c["state"], qdef) for qid, qdef in c["questions"].items()}
            hits += 1 if _pass({"answers": answers}, c["check"]) else 0
        except Exception:
            pass
    return f"{hits}/{len(ROBUST5)}"

def determinism(fn, n=10):
    from benchmark_extended import CHOICE_CRITERIA, CHOICE_INSTRUCTIONS
    q = {"type": "choice", "instructions": CHOICE_INSTRUCTIONS, "criteria": CHOICE_CRITERIA}
    seen = set()
    for _ in range(n):
        a = fn("My card was charged twice for the same invoice.", q)
        seen.add(a.get("choice"))
    return f"{len(seen)} distinct / {n}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="hash,hybrid")
    ap.add_argument("--out", default="beat_kind_results.json")
    args = ap.parse_args()
    want = [e.strip() for e in args.engines.split(",") if e.strip()]
    cases = build_cases()
    print(f"cases={len(cases)} seed={SEED} score_tol={SCORE_TOLERANCE}\n")
    makers = {"hash": fn_hash, "hybrid": fn_hybrid, "laya": fn_laya}
    if os.getenv("TYPESAFE_API_KEY"):
        makers["hosted"] = fn_hosted
    results = {}
    for name in want:
        if name == "hosted" and "hosted" not in makers:
            print("hosted skipped: set TYPESAFE_API_KEY=jev-... for live row\n")
            continue
        try:
            fn = makers[name]()
        except Exception as e:
            print(f"== {name}: INIT FAIL {str(e)[:200]}\n")
            results[name] = {"init_error": str(e)[:300]}
            continue
        r = run_engine(name, fn, cases)
        r["robust5"] = robust5(fn)
        r["determinism_10"] = determinism(fn)
        eng = getattr(fn, "engine", None)
        if eng is not None:
            r["fallback_rate"] = round(eng.fallback_rate, 4)
        print(f"== {name}: overall {r['overall']} noul {r['noul']} choice {r['choice']} score {r['score']} "
              f"ECE {r['ece_noul']} p50 {r['p50_ms']}ms p95 {r['p95_ms']}ms robust {r['robust5']} det {r['determinism_10']} "
              f"misses={r['misses'] or 'none'} {('fallback='+str(r.get('fallback_rate'))) if 'fallback_rate' in r else ''}")
    print("\nPUBLISHED TypeSafe PROXY (not same-seed):", json.dumps(PUBLISHED_TYPESAFE))
    if "hosted" not in results and "hosted" not in want:
        print("No live hosted row — set TYPESAFE_API_KEY to claim a live win.")
    # verdict vs bar (local only)
    for name, r in results.items():
        if "overall" not in r:
            continue
        win = r["overall"] >= 0.90 and r["ece_noul"] < 0.15 and r["robust5"] == "5/5" and r["p50_ms"] < 20
        print(f"BAR {name}: {'PASS' if win else 'FAIL'} (need >=0.90, ECE<0.15, 5/5, p50<20ms)")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "n": len(cases), "engines": results, "published_typesafe_proxy": PUBLISHED_TYPESAFE}, f, indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
