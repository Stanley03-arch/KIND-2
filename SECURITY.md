# Security notes

## Secret handling
- Never commit real `.env` or token files.
- Keep secrets in local environment variables only.
- Rotate any leaked credentials immediately.

## Recommended flow
1. Copy `.env.example` to `.env`.
2. Fill in required environment values.
3. Confirm `.env` is ignored by Git.
