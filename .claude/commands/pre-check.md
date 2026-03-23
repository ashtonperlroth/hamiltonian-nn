Run a pre-commit quality check:
```bash
echo "=== Syntax Check ==="
python -m py_compile pswm/**/*.py 2>&1 || true

echo "=== Import Check ==="
python -c "import pswm" 2>&1

echo "=== Ruff ==="
ruff check pswm/ tests/ 2>&1 | tail -20

echo "=== Tests ==="
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Report a summary: how many files checked, any syntax errors, any import errors, lint issues, test failures.