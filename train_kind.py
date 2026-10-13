"""Train Kind System One decision model from scratch.

Fine-tunes SentenceTransformer on System-One decision triples:
- Routing (lexical & semantic paraphrasing)
- Binary guardrails & negation (WIP vs wipe, not asking vs asking)
- Rubric scoring (incident severity, continuous expectation)
- Platt / RLCD temperature calibration for probability outputs

Saves trained weights to models/kind-ft/
"""
from __future__ import annotations
import os
import json
import math
import time
import numpy as np

# Hugging Face authentication
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
# Force offline HF loading to use cached weights cleanly
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "kind-ft"))
MODELS_DIR = os.path.dirname(OUTPUT_DIR)


def build_kind_training_data():
    """Comprehensive synthetic System-One pairs for Kind."""
    pos = []
    
    # 1. Paraphrases and Semantic Invariance (No lexical overlap -> same meaning)
    pairs = [
        ("Route ticket || The application falls over whenever I launch it in the morning",
         "tech_support bugs crashes server errors stacktrace"),
        ("Route ticket || My card was charged twice, please reverse transaction ASAP",
         "billing invoices refunds payments duplicate charge credit"),
        ("Route ticket || Prove Fermat's last theorem with a novel mathematical approach",
         "frontier_reasoner complex proof math advanced reasoning logic"),
        ("Route ticket || Hi! Translate 'hello' to Spanish please.",
         "slm_fast greeting translation simple factual fast"),
        ("Route ticket || We need an enterprise quote for 500 team seats and SOC2 audit report",
         "sales enterprise quote contract commercial pricing license"),
        ("Route ticket || Can somebody reset my password? I forgot it over the weekend.",
         "auth credentials password login account access"),
         
        # 2. Guardrails & Negative Boundary Hardening (WIP vs wipe, negation)
        ("Does this request wipe or delete stored data? || WIP the design doc and share it with the team.",
         "Safe benign read-only draft work in progress documentation"),
        ("Does this request wipe or delete stored data? || Delete all records from production customer table",
         "Destructive catastrophic data wiping drop delete permanent removal"),
        ("Does this request wipe or delete stored data? || Wipe database clean and reset root passwords",
         "Destructive catastrophic data wiping drop delete permanent removal"),
        ("Is the customer not asking for a refund? || Please refund the duplicate card charge on invoice 442.",
         "not asking no refund needed satisfied customer keep payment"),
        ("Is the customer asking for a refund? || Please refund the duplicate card charge on invoice 442.",
         "refund invoices payments duplicate charge credit dispute money back"),
        ("Which team should handle `ticket_message`? || Please refund the duplicate card charge.",
         "billing payments refunds invoices charges"),
         
        # 3. Rubric & Severity Graders
        ("Rate severity || The export button is misaligned by a few pixels on safari",
         "Cosmetic minor styling no impact to functionality low priority"),
        ("Rate severity || Users cannot download CSV reports, but can view table on screen",
         "Degraded degraded feature workaround exists medium priority"),
        ("Rate severity || Nobody on our team can log in. 500 error on every single attempt.",
         "Blocking issue service outage no workaround exists critical p0"),
        ("Rate urgency || I have a quick question whenever you get a chance next week",
         "low casual relaxed non-urgent low priority"),
        ("Rate urgency || Immediate production database lockup affecting all checkout flows NOW",
         "critical emergency immediate blocker catastrophic outage"),
    ]
    for anchor, target in pairs:
        pos.append((anchor, target))

    # Explicit hard negative pairs to sharpen boundaries
    neg = [
        ("Does this request wipe or delete stored data? || WIP the design doc and share it with the team.",
         "Destructive catastrophic data wiping drop delete permanent removal"),
        ("Is the customer asking for a refund? || Prove Fermat's last theorem with a novel approach",
         "refund invoices payments duplicate charge credit dispute money back"),
        ("Is the customer not asking for a refund? || I demand a full refund right now",
         "not asking no refund needed satisfied customer keep payment"),
        ("Route ticket || The application falls over whenever I launch it in the morning",
         "billing invoices refunds payments duplicate charge credit"),
        ("Rate severity || The export button is misaligned by a few pixels on safari",
         "Blocking issue service outage no workaround exists critical p0"),
    ]

    return pos, neg


def train_kind(epochs: int = 3, batch_size: int = 8):
    print("=" * 60)
    print(" KIND System One -- Local Model Trainer")
    print("=" * 60)
    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training Device: {device.upper()}")
    print(f"[*] Base Architecture: {BASE_MODEL}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pos, neg = build_kind_training_data()
    print(f"[*] Prepared {len(pos)} contrastive pairs and {len(neg)} explicit negative boundaries.")

    t_start = time.time()
    print("[*] Loading base encoder...")
    model = SentenceTransformer(BASE_MODEL, device=device, local_files_only=True)

    # 1. Multiple Negatives Ranking Loss (Contrastive Pass)
    print(f"[*] Phase 1: Contrastive fine-tuning ({epochs} epochs)...")
    train_ex = [InputExample(texts=[a, b]) for a, b in pos]
    loader = DataLoader(train_ex, batch_size=batch_size, shuffle=True)
    mnr_loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(loader, mnr_loss)],
        epochs=epochs,
        warmup_steps=min(10, len(loader)),
        show_progress_bar=False,
    )

    # 2. Triplet Loss Pass on Hard Boundaries (WIP vs wipe, negation)
    print("[*] Phase 2: Triplet boundary sharpening...")
    trip_ex = []
    for i, (a_neg, n_text) in enumerate(neg):
        # find matching anchor in pos
        matching_pos = [p for a, p in pos if a == a_neg]
        if matching_pos:
            trip_ex.append(InputExample(texts=[a_neg, matching_pos[0], n_text]))
        elif i < len(pos):
            trip_ex.append(InputExample(texts=[pos[i][0], pos[i][1], n_text]))

    if trip_ex:
        trip_loader = DataLoader(trip_ex, batch_size=min(4, len(trip_ex)), shuffle=True)
        trip_loss = losses.TripletLoss(model=model)
        model.fit(
            train_objectives=[(trip_loader, trip_loss)],
            epochs=1,
            warmup_steps=2,
            show_progress_bar=False,
        )

    # 3. Save fine-tuned Kind model
    print(f"[*] Saving fine-tuned Kind model to: {OUTPUT_DIR}")
    model.save(OUTPUT_DIR)

    # 4. Fit calibration temperatures (RLCD / Platt scaling)
    print("[*] Phase 3: Fitting Platt / RLCD probability temperatures...")
    model.eval()
    with torch.inference_mode():
        def enc(text):
            return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

        margins = []
        for a, p in pos:
            va, vp = enc(a), enc(p)
            margins.append((float(np.dot(va, vp)) * 8.0, 1))
        for a, n in neg:
            va, vn = enc(a), enc(n)
            margins.append((float(np.dot(va, vn)) * 8.0, 0))

    def sig(x):
        return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))

    best_t, best_nll = 1.0, float("inf")
    for t_cand in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        nll = sum(
            -(y * math.log(min(0.999, max(0.001, sig(m / t_cand))))
              + (1 - y) * math.log(1.0 - min(0.999, max(0.001, sig(m / t_cand)))))
            for m, y in margins
        )
        if nll < best_nll:
            best_nll, best_t = nll, t_cand

    temps = {
        "model_name": "kind-ft",
        "noul_temp": round(best_t, 3),
        "choice_temp": 1.0,
        "score_temp": 1.0,
        "fitted_at": time.time(),
    }
    temps_file = os.path.join(MODELS_DIR, "temps.json")
    with open(temps_file, "w") as f:
        json.dump(temps, f, indent=2)

    elapsed = round(time.time() - t_start, 2)
    metrics = {
        "model": "kind-ft",
        "base_model": BASE_MODEL,
        "epochs": epochs,
        "train_samples": len(pos),
        "negative_samples": len(neg),
        "fitted_noul_temp": best_t,
        "nll_score": round(best_nll, 4),
        "train_duration_sec": elapsed,
        "status": "ready",
    }
    metrics_file = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 60)
    print(f" [SUCCESS] KIND Model training completed successfully in {elapsed}s!")
    print(f"    - Output: {OUTPUT_DIR}")
    print(f"    - Calibrated Temperatures: {temps}")
    print(f"    - Metrics: {metrics}")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    train_kind()
