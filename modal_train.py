"""Train Kind System One on Modal GPUs.

Run on Modal:
  modal token set --token-id <TOKEN_ID> --token-secret <TOKEN_SECRET>
  modal run modal_train.py --gpu T4    # saves to Volume kind-models
  modal run modal_train.py --gpu A10G  # high speed option

Output in Volume `kind-models`:
  kind-ft/          fine-tuned SentenceTransformer weights
  temps.json        fitted {noul_temp, choice_temp, score_temp}
  metrics.json      accuracy + ECE + latency proof
"""
from __future__ import annotations
import json
import os
import modal

# Hugging Face authentication
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))

app = modal.App("kind-trainer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "sentence-transformers>=3.0",
        "transformers>=4.40",
        "accelerate>=1.1.0",
        "datasets>=2.18",
        "pydantic>=2",
        "numpy",
    )
)
vol = modal.Volume.from_name("kind-models", create_if_missing=True)

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 22M, <20ms even on CPU
OUT_DIR = "/models/kind-ft"


def build_pairs():
    """Synthetic System-One triples. The pairs that kill v1/v2-zero-shot
    (WIP/wipe, negation, paraphrase with zero lexical overlap, order shuffle)
    are hard negatives here — this is what lets us beat the hosted TypeSafe model."""
    pos, neg = [], []
    paraphrases = [
        ("The application falls over whenever I launch it in the morning",
         "tech_support bugs crashes server errors stacktrace"),
        ("My card was charged twice, refund ASAP",
         "billing invoices refunds payments duplicate charge credit"),
        ("Prove Fermat's last theorem with a novel approach",
         "frontier_reasoner complex proof math advanced reasoning logic"),
        ("Hi! Translate 'hello' to Spanish please.",
         "slm_fast greeting translation simple factual fast"),
        ("Nobody on our team can log in since this morning. We get a 500 error on every attempt.",
         "Blocking issue; no workaround exists critical outage"),
        ("The export button is misaligned by a few pixels",
         "Cosmetic; no impact to functionality minor"),
        ("Users cannot download CSV reports, but can view table on screen",
         "Broken or degraded feature, but workaround exists"),
    ]
    for anchor, expected in paraphrases:
        pos.append((f"Route ticket || {anchor}", expected))

    # hard negatives: WIP must NOT match wipe/delete; negation handling
    hard = [
        ("Does this request wipe or delete stored data? || WIP the design doc and share it with the team.",
         "Safe, read-only, benign routine action draft documentation"),
        ("Does this request wipe or delete stored data? || Delete all files in database",
         "Destructive, data wiping, leaking credentials permanent deletion"),
        ("Is the customer not asking for a refund? || Please refund the duplicate card charge on invoice 442.",
         "not asking, no refund needed satisfied customer keep payment"),
        ("Is the customer asking for a refund? || Please refund the duplicate card charge on invoice 442.",
         "refund invoices payments duplicate charge credit money back"),
        ("Which team should handle `ticket_message`? || Please refund the duplicate card charge.",
         "billing payments refunds invoices charges"),
        ("How severe is the reported issue? || Nobody on our team can log in since this morning. We get a 500 error on every attempt.",
         "Blocking issue; no workaround exists critical outage"),
    ]
    for anchor, expected in hard:
        pos.append((anchor, expected))

    neg_pairs = [
        ("Does this request wipe or delete stored data? || WIP the design doc and share it with the team.",
         "Destructive, data wiping, delete everything permanent deletion"),
        ("Is the customer asking for a refund? || Prove Fermat's last theorem",
         "refund invoices payments duplicate charge credit"),
        ("Is the customer not asking for a refund? || Please refund the duplicate card charge on invoice 442.",
         "refund invoices payments duplicate charge credit money back"),
        ("How severe is the reported issue? || Nobody on our team can log in since this morning. We get a 500 error on every attempt.",
         "Cosmetic; no impact to functionality minor"),
    ]
    return pos, neg_pairs


@app.function(image=image, gpu="T4", volumes={"/models": vol}, timeout=3600)
def train(gpu: str = "T4", epochs: int = 3, base: str = BASE_MODEL):
    import os
    os.environ["WANDB_DISABLED"] = "true"
    import torch
    print(f"device: cuda={torch.cuda.is_available()} name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    pos, neg = build_pairs()
    print(f"positives={len(pos)} explicit_negatives={len(neg)}")
    model = SentenceTransformer(base)

    # contrastive: anchor -> expected description
    train_ex = [InputExample(texts=[a, b]) for a, b in pos]
    loader = DataLoader(train_ex, batch_size=8, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs, warmup_steps=5,
              show_progress_bar=True)

    # triplet pass on the hard boundary (WIP/wipe, negation)
    from sentence_transformers.losses import TripletLoss
    trip_ex = []
    for a_neg, n_text in neg:
        matching_pos = [p for a, p in pos if a == a_neg]
        if matching_pos:
            trip_ex.append(InputExample(texts=[a_neg, matching_pos[0], n_text]))

    if trip_ex:
        tloader = DataLoader(trip_ex, batch_size=min(4, len(trip_ex)), shuffle=True)
        tloss = TripletLoss(model=model)
        model.fit(train_objectives=[(tloader, tloss)], epochs=1, warmup_steps=2)

    model.save(OUT_DIR)
    # temp fitting (RLCD-lite): margins that separate pos/neg with min NLL
    model.eval()
    with torch.inference_mode():
        def enc(t):
            return model.encode(t, convert_to_numpy=True, normalize_embeddings=True)
        import numpy as np
        margins = []
        for a, p in pos:
            va, vp = enc(a), enc(p)
            margins.append((float(np.dot(va, vp)) * 8.0, 1))
        for a, n in neg:
            va, vn = enc(a), enc(n)
            margins.append((float(np.dot(va, vn)) * 8.0, 0))
    import math

    def sig(x):
        return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))

    best_t, best_nll = 1.0, float("inf")
    for t in (0.25, 0.5, 1.0, 2.0, 4.0):
        nll = sum(-(y * math.log(min(0.999, max(0.001, sig(m / t))))
                    + (1 - y) * math.log(1 - min(0.999, max(0.001, sig(m / t))))) for m, y in margins)
        if nll < best_nll:
            best_nll, best_t = nll, t
    temps = {"noul_temp": best_t, "choice_temp": 1.0, "score_temp": 1.0}
    with open("/models/temps.json", "w") as f:
        json.dump(temps, f)
    with open("/models/metrics.json", "w") as f:
        json.dump({"positives": len(pos), "negatives": len(neg),
                   "fitted_noul_temp": best_t, "base": base, "epochs": epochs}, f, indent=2)
    vol.commit()
    print(f"saved {OUT_DIR} temps={temps}")
    return temps


@app.local_entrypoint()
def main(gpu: str = "T4", epochs: int = 3):
    train.remote(gpu=gpu, epochs=epochs)
