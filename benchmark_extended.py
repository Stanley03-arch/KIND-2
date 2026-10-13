"""Extended LOCAL benchmark for Kind. No hosted API is ever called.

60 hand-labelled cases: 20 Noul, 20 Choice, 20 Score.
Case order is shuffled with a fixed seed (SEED = 42), so every run is identical.

Run:
  python benchmark_extended.py                    # both local engines
  python benchmark_extended.py --engines legacy   # hash engine only
  python benchmark_extended.py --out results.json
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SEED = 42
SCORE_TOLERANCE = 0.75

# ---------------------------------------------------------------- Noul
# (instructions, state, expected verdict)
NOUL_CASES = [
    ("Is the customer asking for a refund?", "I want my money back for order 88, the item broke.", True),
    ("Is the customer asking for a refund?", "Thanks for the help, everything works now.", False),
    ("Is the customer reporting a security incident?", "Someone logged into my account from another country and changed my password.", True),
    ("Is the customer reporting a security incident?", "How do I change my profile photo?", False),
    ("Is the customer asking for a human agent?", "Can I please talk to a real person?", True),
    ("Is the customer asking for a human agent?", "What are your opening hours?", False),
    ("Is this request asking to delete stored data?", "Please delete all records for user 42 permanently.", True),
    ("Is this request asking to delete stored data?", "Please summarize the quarterly report for me.", False),
    ("Is this request asking to delete stored data?", "WIP the design doc and share it with the team.", False),
    ("Is the customer NOT asking for a refund?", "Please refund the duplicate charge on invoice 442.", False),
    ("Is the customer NOT asking for a refund?", "Just checking my order status, no complaints.", True),
    ("Is the system currently down?", "Every request returns 503 since 9am.", True),
    ("Is the system currently down?", "The dashboard loads a little slowly sometimes.", False),
    ("Does the message mention a deadline?", "The report is due by Friday at noon.", True),
    ("Does the message mention a deadline?", "Great weather today, hope you are well.", False),
    ("Is the user expressing anger?", "This is the third time I've been ignored, I am furious.", True),
    ("Is the user expressing anger?", "Could you kindly update my address when convenient?", False),
    ("Is a backup mentioned?", "Last night's backup failed and we have no copy.", True),
    ("Is a backup mentioned?", "Let's schedule lunch next week.", False),
    ("Is the customer asking to cancel their subscription?", "Please cancel my plan effective today.", True),
]

# ---------------------------------------------------------------- Choice
CHOICE_CRITERIA = {
    "billing": "Invoices, payments, refunds, charges",
    "tech": "Bugs, crashes, errors, outages",
    "sales": "New purchases, upgrades, pricing, demos",
}
CHOICE_INSTRUCTIONS = "Which team should handle this request?"
# (state, expected key)
CHOICE_CASES = [
    ("My card was charged twice for the same invoice.", "billing"),
    ("The app crashes with a null pointer exception on startup.", "tech"),
    ("I'd like to upgrade to the enterprise plan and get a quote.", "sales"),
    ("Refund my last payment, it was a mistake.", "billing"),
    ("Server returns 500 errors for every API call.", "tech"),
    ("How much does the pro subscription cost per seat?", "sales"),
    ("The invoice total does not match what I agreed to pay.", "billing"),
    ("The login page shows a blank screen after I enter my password.", "tech"),
    ("We want to buy fifty licences for our team.", "sales"),
    ("The receipt amount is wrong and I need a credit.", "billing"),
    ("Build fails with a compiler error in module three.", "tech"),
    ("Can your sales team call us about a demo?", "sales"),
    ("The application falls over whenever I launch it in the morning", "tech"),
    ("I was overcharged last month, please send money back.", "billing"),
    ("Our dashboard keeps freezing during data sync.", "tech"),
    ("Interested in purchasing a yearly plan for my startup.", "sales"),
    ("The card on file was debited for a cancelled order.", "billing"),
    ("Exception thrown in the payment worker queue consumer.", "tech"),
    ("Which plan includes SSO? We're evaluating vendors.", "sales"),
    ("Please refund the shipping fee I was charged.", "billing"),
]

# ---------------------------------------------------------------- Score
SEVERITY_INSTRUCTIONS = "How severe is the reported issue?"
SEVERITY_LEVELS = [
    "Cosmetic; no impact to functionality",
    "Broken or degraded feature, but workaround exists",
    "Blocking issue; no workaround exists",
]
# (state, expected level index 0..2)
SCORE_CASES = [
    ("The button colour is slightly off on the settings page.", 0),
    ("Logo is misaligned by two pixels in the footer.", 0),
    ("Typo in the help text on the pricing page.", 0),
    ("Tooltip text is cut off on small screens.", 0),
    ("Sidebar icon is a little blurry on retina screens.", 0),
    ("Export to PDF does nothing but CSV export works fine.", 1),
    ("Search results sometimes load slowly; I refresh and it works.", 1),
    ("Dark mode toggle resets after reload, I reapply it each time.", 1),
    ("Notifications arrive late but do arrive eventually.", 1),
    ("Upload works but the progress bar stays at 0 until finished, workaround is to wait.", 1),
    ("Admins cannot delete users; they must ask support to do it.", 1),
    ("Nobody can log in since this morning, every attempt gets a 500.", 2),
    ("Payments are failing for all customers and checkout is down.", 2),
    ("The whole app crashes on launch for every user, no way around it.", 2),
    ("Production database is unreachable and we cannot serve any orders.", 2),
    ("Every invoice PDF is blank and no one can send invoices to clients.", 2),
    ("Entire team is locked out of the repository and deadlines are today.", 2),
    ("Data loss: the nightly job wiped customer records and there is no backup.", 2),
    ("Mobile app sometimes needs two taps to open a menu, tapping again works.", 1),
    ("Date format shows day-month in one widget, which is only a display quirk.", 0),
]


def build_cases():
    cases = []
    for i, (instr, state, exp) in enumerate(NOUL_CASES):
        cases.append({"id": f"noul-{i:02d}", "cat": "noul", "state": state,
                      "question": {"type": "noul", "instructions": instr}, "expected": exp})
    for i, (state, exp) in enumerate(CHOICE_CASES):
        cases.append({"id": f"choice-{i:02d}", "cat": "choice", "state": state,
                      "question": {"type": "choice", "instructions": CHOICE_INSTRUCTIONS,
                                   "criteria": CHOICE_CRITERIA}, "expected": exp})
    for i, (state, exp) in enumerate(SCORE_CASES):
        cases.append({"id": f"score-{i:02d}", "cat": "score", "state": state,
                      "question": {"type": "score", "instructions": SEVERITY_INSTRUCTIONS,
                                   "criteria": SEVERITY_LEVELS}, "expected": exp})
    random.Random(SEED).shuffle(cases)
    return cases


# ---------------------------------------------------------------- engines
def make_engines(names):
    engines = {}
    if "legacy" in names:
        from kind import KindClient
        client = KindClient(encoder="hash")

        def call_legacy(state, question):
            r = client.system_one(state=state, questions={"q": question})
            return r.answers["q"].model_dump()
        engines["kind-legacy(hash)"] = call_legacy
    if "local" in names:
        from kind.engine import KindEngine
        eng = KindEngine()

        def call_local(state, question):
            r = eng.decide(state=state, questions={"q": question})
            return r.answers["q"].model_dump()
        engines[f"kind-local({eng.encoder.backend})"] = call_local
    return engines


def grade(case, ans):
    cat, exp = case["cat"], case["expected"]
    if cat == "noul":
        p = float(ans["noul"])
        return (p >= 0.5) == exp, p
    if cat == "choice":
        return ans["choice"] == exp, None
    return abs(float(ans["score"]) - exp) <= SCORE_TOLERANCE, None


def run_engine(fn, cases):
    from kind.calibration import expected_calibration_error
    rows, lat, noul_pairs = [], [], []
    for case in cases:
        t0 = time.perf_counter()
        ans, err, ok, p = None, None, False, None
        try:
            ans = fn(case["state"], case["question"])
            ok, p = grade(case, ans)
        except Exception as exc:  # a crash counts as a miss, never skipped
            err = str(exc)
        lat.append((time.perf_counter() - t0) * 1000.0)
        if case["cat"] == "noul" and p is not None:
            noul_pairs.append((p, 1 if case["expected"] else 0))
        rows.append({"id": case["id"], "cat": case["cat"], "ok": ok,
                     "expected": case["expected"], "answer": ans, "error": err})

    def acc(cat):
        sel = [r for r in rows if r["cat"] == cat]
        return round(sum(r["ok"] for r in sel) / len(sel), 4) if sel else 0.0

    lat_s = sorted(lat)
    return {
        "accuracy": {c: acc(c) for c in ("noul", "choice", "score")},
        "overall_accuracy": round(sum(r["ok"] for r in rows) / len(rows), 4),
        "ece_noul": round(expected_calibration_error(noul_pairs, n_bins=10), 4),
        "p50_ms": round(lat_s[len(lat_s) // 2], 2),
        "p95_ms": round(lat_s[int(0.95 * (len(lat_s) - 1))], 2),
        "cases": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="legacy,local", help="comma list: legacy,local")
    ap.add_argument("--out", default="benchmark_extended_results.json")
    args = ap.parse_args()

    cases = build_cases()
    engines = make_engines([e.strip() for e in args.engines.split(",") if e.strip()])
    results = {}
    for name, fn in engines.items():
        results[name] = run_engine(fn, cases)

    print(f"Cases: {len(cases)} (seed {SEED}, local only, no hosted API)\n")
    for name, r in results.items():
        misses = [x["id"] for x in r["cases"] if not x["ok"]]
        print(f"== {name}")
        print(f"   overall {r['overall_accuracy']:.3f}  noul {r['accuracy']['noul']:.3f}  "
              f"choice {r['accuracy']['choice']:.3f}  score {r['accuracy']['score']:.3f}")
        print(f"   ECE(noul) {r['ece_noul']:.3f}  p50 {r['p50_ms']:.1f}ms  p95 {r['p95_ms']:.1f}ms")
        print(f"   misses: {', '.join(misses) if misses else 'none'}\n")

    payload = {
        "seed": SEED,
        "n_cases": len(cases),
        "score_tolerance": SCORE_TOLERANCE,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "engines": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
