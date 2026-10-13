"""Serve fine-tuned Kind model and Decision Studio from Modal GPU — TypeSafe-compatible wire.

Endpoints:
  GET  /              -> Kind System One Decision Studio Web App
  GET  /health        -> Health check & active model info
  POST /v1/systemone  -> {state, questions, model, temperature} -> {answers, usage}
  GET  /v1/models     -> List available model identifiers
  POST /v1/benchmark  -> Live latency & throughput benchmark

Deploy:
  modal run modal_train.py --gpu T4     # produces /models/kind-ft in volume
  modal deploy modal_serve.py           # stable https endpoint
"""
from __future__ import annotations
import os
import modal

# Hugging Face authentication
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))

app = modal.App("kind-serve")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "sentence-transformers>=3.0",
        "transformers>=4.40",
        "fastapi>=0.110",
        "pydantic>=2",
        "numpy",
    )
    .add_local_dir("kind", remote_path="/root/kind")
    .add_local_dir("web", remote_path="/root/web")
)
vol = modal.Volume.from_name("kind-models", create_if_missing=True)


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/models": vol},
    scaledown_window=300,
    timeout=300,
)
class KindServer:
    @modal.enter()
    def load(self):
        import json
        import os
        from kind.engine import KindEngine

        ft = "/models/kind-ft"
        temps = {}
        try:
            with open("/models/temps.json") as f:
                temps = json.load(f)
        except Exception:
            pass

        model_id = ft if os.path.isdir(ft) else None
        self.eng = KindEngine(
            model_id=model_id,
            noul_temp=float(temps.get("noul_temp", 1.0)),
            choice_temp=float(temps.get("choice_temp", 1.0)),
            score_temp=float(temps.get("score_temp", 1.0)),
        )
        self.model_name = "kind-ft-1.0" if model_id else "kind-1.0"
        self.fine_tuned = bool(model_id)
        self.calibrated_temps = temps

    @modal.asgi_app()
    def web(self):
        import os
        import time
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles

        web_app = FastAPI(
            title="Kind Decision Studio",
            description="High-performance, non-autoregressive System One decision server matching TypeSafe AI Jev.",
            version="2.0.0",
        )

        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.get("/health")
        def health():
            return {
                "status": "healthy",
                "service": "kind-systemone",
                "name": "Kind Decision Engine",
                "version": "2.0.0",
                "model": self.model_name,
                "device": "T4 (GPU)",
                "encoder": getattr(self.eng.encoder, "backend", "sentence-transformers"),
                "fine_tuned": self.fine_tuned,
                "calibrated_temps": self.calibrated_temps,
            }

        @web_app.get("/v1/models")
        def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": "kind-ft-1.0",
                        "object": "model",
                        "description": "Fine-tuned Kind System One decision weights (WIP/wipe & negation hardened).",
                    },
                    {
                        "id": "kind-1.0",
                        "object": "model",
                        "description": "High-performance base Kind System One engine (sub-15ms).",
                    },
                    {
                        "id": "jev-latest",
                        "object": "model",
                        "description": "Alias matching TypeSafe AI Jev latest identifier.",
                    },
                ],
            }

        @web_app.post("/v1/systemone")
        def systemone(item: dict):
            resp = self.eng.decide(
                state=item.get("state"),
                questions=item.get("questions", {}),
                model=item.get("model") or self.model_name,
                temperature=item.get("temperature") or 1.0,
            )
            return resp.model_dump()

        @web_app.post("/v1/benchmark")
        def benchmark(req: dict):
            iterations = int(req.get("iterations", 20))
            sample_state = {
                "ticket_id": "TCK-8812",
                "user": "Sarah Connor",
                "message": "My card was charged $150 twice for invoice #442. Please reverse this immediately.",
            }
            sample_questions = {
                "is_billing": {
                    "type": "noul",
                    "instructions": "Is this inquiry related to billing, payments, or refunds?",
                },
                "department": {
                    "type": "choice",
                    "instructions": "Route to appropriate department",
                    "criteria": {
                        "billing": "Invoices, transactions, chargebacks",
                        "tech_support": "Server errors, bugs, app crashes",
                        "sales": "New subscriptions and contract upgrades",
                    },
                },
                "severity": {
                    "type": "score",
                    "instructions": "Rate urgency level",
                    "criteria": ["low", "medium", "high", "critical"],
                },
            }
            latencies = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                self.eng.decide(sample_state, sample_questions)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

            avg_ms = sum(latencies) / len(latencies)
            p50_ms = sorted(latencies)[len(latencies) // 2]
            p95_ms = sorted(latencies)[int(len(latencies) * 0.95)]
            qps = 1000.0 / avg_ms if avg_ms > 0 else 0

            return {
                "iterations": iterations,
                "average_latency_ms": round(avg_ms, 2),
                "p50_latency_ms": round(p50_ms, 2),
                "p95_latency_ms": round(p95_ms, 2),
                "estimated_qps_single_worker": round(qps, 1),
                "questions_evaluated_per_request": len(sample_questions),
            }

        web_dir = "/root/web"
        if os.path.exists(web_dir):
            web_app.mount("/static", StaticFiles(directory=web_dir), name="static")

            @web_app.get("/")
            def serve_studio():
                return FileResponse(os.path.join(web_dir, "index.html"))

        return web_app
