# Release

Publishing uses GitHub Actions trusted publishing. No long-lived PyPI token is stored in the repository.

## One-time setup

Create a pending publisher on both indexes before the first upload.

TestPyPI:
- URL: https://test.pypi.org/manage/account/publishing/
- PyPI project name: `kt-audit`
- Owner: `summeriver13`
- Repository: `kt-audit`
- Workflow name: `publish.yml`
- Environment: `testpypi`

PyPI:
- URL: https://pypi.org/manage/account/publishing/
- Same owner, repository, and workflow
- Environment: `pypi`

## TestPyPI

Run the workflow manually:

```bash
gh workflow run publish.yml
gh run watch
```

Verify from TestPyPI:

```bash
uv run --isolated --no-project \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  --with kt-audit==0.1.0 kt-audit --version
```

## PyPI

1. Bump `__version__` in `src/kt_audit/__init__.py`.
2. Commit and push; wait for CI.
3. Tag the same version and push the tag:

```bash
git tag -a v0.1.0 -m v0.1.0
git push origin v0.1.0
```

The tag triggers `publish.yml`. The build job fails if the tag does not match `__version__`.

TestPyPI and PyPI are independent indexes. A successful TestPyPI upload does not publish to PyPI.
