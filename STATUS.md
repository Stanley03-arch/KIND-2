# Status — KIND-2

## Current

- Training / benchmark scripts and Modal helpers are in-repo
- Docs cover setup, security, troubleshooting, and release checklist
- Local secrets stay out of git (`.env.example` only)

## Verify

```bash
git clone https://github.com/Stanley03-arch/KIND-2.git
cd KIND-2
cp .env.example .env   # fill in keys if needed
pip install -r requirements.txt
python run_server.py --help || true
```

## Focus

- Keep benchmarks reproducible
- Prefer safe defaults for local runs
- Document any breaking CLI or env changes in CHANGELOG
