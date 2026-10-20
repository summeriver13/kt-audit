# Contributing

`kt-audit` is intentionally small. The v0.1 scope is frozen to three checks:

- `temporal_split`
- `multi_kc`
- `alignment`

A new check needs an issue with a concrete failure mode and a minimal example first. Please do not open a large refactor or a new framework PR.

## Development

```bash
uv sync --extra dev
uv run pytest --cov=kt_audit --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
```

## Rules

- Keep runtime dependencies to `numpy`, `pandas`, and `scikit-learn`.
- Add a test for every new branch. Synthetic data only; do not commit restricted datasets.
- Every finding must have a severity, a count, and bounded examples.
- Document limitations next to the check, not only in the README.
- Disclose AI assistance in the PR body. The maintainer reviews every check definition and test.

## Reporting a false positive

Open an issue with:

1. the command,
2. the input schema (not the full private dataset),
3. the actual finding,
4. why you think it is wrong.

A reproducible counterexample is more useful than a general complaint.
