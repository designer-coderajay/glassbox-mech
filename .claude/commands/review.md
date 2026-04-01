# /review — Python Code Review

Run a comprehensive Python code review on all changed files.

## Usage

```
/review [--files <glob>] [--strict]
```

## Steps

1. Run `git diff --staged && git diff` to identify changed files
2. Run automated tools:
   ```bash
   ruff check glassbox/ tests/
   mypy glassbox/
   black --check glassbox/ tests/
   bandit -r glassbox/ -ll
   pytest --cov=glassbox --cov-report=term-missing -q
   ```
3. Manual review: correctness, type hints, docstrings, research number preservation
4. Report all findings by severity: CRITICAL → HIGH → MEDIUM → LOW
5. Print pass/fail verdict

## Approval Criteria

- Zero CRITICAL issues
- `ruff`, `mypy`, `black --check` all pass
- Coverage ≥ 80% overall, 100% on faithfulness.py and patching.py
- All public functions have type hints and docstrings
- Research numbers (r=0.009, sufficiency=1.00, F1=0.64, 76 tests, 37×) unchanged

## Output Format

```
[CRITICAL] Hook not reset after patching
File: glassbox/core/patching.py:87
Issue: model.reset_hooks() not called after run_with_cache
Fix: Add model.reset_hooks() in finally block

[HIGH] Missing type hint on public function
File: glassbox/faithfulness.py:42
Issue: compute_comprehensiveness has no return type annotation
Fix: Add -> float

VERDICT: FAIL (1 CRITICAL, 1 HIGH)
```
