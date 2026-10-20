#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

uv build --out-dir "$tmp/dist" >/dev/null
wheel="$(ls "$tmp/dist"/kt_audit-*.whl | head -n 1)"

uv venv "$tmp/venv" >/dev/null
if [[ -x "$tmp/venv/bin/python" ]]; then
  python="$tmp/venv/bin/python"
else
  python="$tmp/venv/Scripts/python.exe"
fi
uv pip install --python "$python" "$wheel" >/dev/null

"$python" -c 'import kt_audit; print(f"version: {kt_audit.__version__}")'
"$python" -m kt_audit --version

set +e
"$python" -m kt_audit run examples/leaky.csv --out "$tmp/report" >/dev/null
code=$?
set -e
if [[ "$code" -ne 1 ]]; then
  echo "expected exit 1 for leaky.csv, got $code" >&2
  exit 1
fi

"$python" - "$tmp/report/report.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload["schema_version"] == "0.1"
assert payload["verdict"] == "fail"
print("report: ok")
PY
