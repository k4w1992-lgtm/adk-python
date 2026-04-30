# Security Policy — Google ADK

## Reporting a Vulnerability

If you discover a security vulnerability in this repository, please report it via:

- Google OSS VRP: https://bughunters.google.com
- Or open a GitHub Security Advisory

## Credential Handling

**NEVER** commit any of the following to this repository:

- OAuth access tokens (`ya29.*`)
- API keys (`AIzaSy*`)
- Service account JSON keys
- Private keys (`.pem`, `.key`)
- `.env` files with real credentials
- Any form of password or secret

Use environment variables or a secrets manager instead.

## Pre-commit Checks

Run before committing:
```bash
pip install detect-secrets
detect-secrets scan --all-files
```

Reported by: @k4w1992-lgtm | Google Issue: #504158909
