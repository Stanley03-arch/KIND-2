# Benchmark Quickstart

## Local checks

```bash
pip install -r requirements.txt
python benchmark_extended.py
python run_server.py
```

## What to look for
- Latency targets remain in the low millisecond range.
- Classification outputs remain stable across repeated runs.
- Guardrail logic stays deterministic on known hard cases.
