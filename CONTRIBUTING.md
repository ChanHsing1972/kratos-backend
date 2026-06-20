# Contributing

1. Create a focused branch from `dev`.
2. Keep credentials and generated data out of the repository. Use `.env.example` only for empty placeholders.
3. Run the checks before opening a pull request:

```bash
python -m compileall -q app tests
PYTHONPATH=. python -m pytest -q
```

4. Describe the change, its user impact, and the validation performed in the pull request.

Security vulnerabilities should be reported privately as described in `SECURITY.md`.
