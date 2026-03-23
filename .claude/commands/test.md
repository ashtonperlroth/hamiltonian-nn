Run the test suite and report results:
```bash
python -m pytest tests/ -v --tb=short 2>&1 | head -50
```

If any tests fail, fix them. If all pass, say so.