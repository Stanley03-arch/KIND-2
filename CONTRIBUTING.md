# Contributor guide

## Setup

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env   # fill secrets locally only
```

## Workflow

- Create a feature branch for each change
- Keep commits small and descriptive
- Never commit `.env` or API keys
- Update `CHANGELOG.md` / `STATUS.md` when behavior changes
- Push regularly and review diffs before merging

## Useful docs

- `LOCAL_SETUP.md` — run locally
- `SECURITY.md` — secret handling
- `TROUBLESHOOTING.md` — common issues
