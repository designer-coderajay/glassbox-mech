# /test — Run Test Suite with Coverage

Run the full Glassbox test suite and report coverage.

## Usage

```
/test [--unit] [--integration] [--fast] [--file <path>]
```

Flags:
- `--unit`: unit tests only (no model loading, fast)
- `--integration`: integration + E2E tests only
- `--fast`: skip `@pytest.mark.slow` tests
- `--file`: run a specific test file

## Commands

```bash
# Full suite
pytest --cov=glassbox --cov-report=term-missing -v

# Fast (unit only)
pytest tests/unit/ -v

# Skip slow (model loading) tests
pytest -m "not slow" -v

# With HTML coverage report
pytest --cov=glassbox --cov-report=html -v
# Open: htmlcov/index.html
```

## Pass Criteria

| Check | Required |
|-------|---------|
| All tests pass | Yes |
| Overall coverage | ≥ 80% |
| `faithfulness.py` coverage | 100% |
| `patching.py` coverage | 100% |
| No new test warnings | Yes |

## On Failure

1. Print failing test name, file, line, and error
2. Print the nearest assertion that failed
3. Suggest which agent to invoke (`pytorch-build-resolver` for tensor errors, `python-reviewer` for logic errors)
4. Never mark tests as "expected failures" unless discussed with researcher

## Invariants to Assert

These must hold in every test run:

```python
# From tests/test_faithfulness.py — these values are paper results
assert abs(sufficiency - 1.00) < 0.01
assert abs(comprehensiveness - 0.22) < 0.01
assert abs(f1 - 0.64) < 0.01
assert abs(r_correlation - 0.009) < 0.005
```

If any of these fail, something changed in the core computation — investigate before merging.
