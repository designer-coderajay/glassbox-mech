---
name: python-testing
description: pytest patterns and TDD workflow for Glassbox. Activates when writing new tests, reviewing test coverage, or debugging failing tests. Enforces 80%+ overall coverage and 100% on critical paths.
origin: Glassbox (adapted from ECC python-testing)
---

# Python Testing Patterns for Glassbox

## When to Activate

- Writing tests for new features (write test FIRST)
- Reviewing test coverage reports
- Debugging failing tests
- Adding parametrized tests for new models or prompts

---

## Coverage Requirements

| Scope | Required Coverage |
|-------|------------------|
| Overall (`glassbox/`) | ≥ 80% |
| `glassbox/faithfulness.py` | 100% |
| `glassbox/core/patching.py` | 100% |
| `glassbox/core/attention.py` | ≥ 90% |
| `glassbox/core/compliance.py` | ≥ 90% |

Run:
```bash
pytest --cov=glassbox --cov-report=term-missing -v
```

---

## TDD Cycle for Glassbox

### Step 1 — RED: Write Failing Test

```python
# tests/test_faithfulness.py
def test_sufficiency_perfect_attribution():
    """Sufficiency should be ~1.00 when cited heads explain full logit diff."""
    cited_heads = [(9, 9), (9, 6), (10, 0)]
    head_effects = {(9, 9): 0.584, (9, 6): 0.211, (10, 0): 0.208}
    clean_logit_diff = 1.003

    result = compute_sufficiency(cited_heads, head_effects, clean_logit_diff)

    assert abs(result - 1.00) < 0.01, f"Expected ~1.00, got {result:.4f}"
```

Run — confirm it fails (function not written yet). Commit: `test: add sufficiency unit test (RED)`

### Step 2 — GREEN: Make It Pass

Write minimal implementation in `glassbox/faithfulness.py`. Run the test. Confirm it passes.

Commit: `feat(faithfulness): implement compute_sufficiency (GREEN)`

### Step 3 — REFACTOR

Clean up implementation if needed. All tests must still pass.

Commit: `refactor(faithfulness): extract contribution sum helper`

---

## Test Structure for Glassbox

```
tests/
├── unit/
│   ├── test_faithfulness.py     # compute_sufficiency, compute_comprehensiveness, F1
│   ├── test_attention.py        # AttentionAnalyzer
│   ├── test_patching.py         # Attribution patching engine
│   ├── test_logit_lens.py       # LogitLens
│   └── test_compliance.py       # Compliance report generation
├── integration/
│   ├── test_full_pipeline.py    # End-to-end: prompt → circuit → report
│   └── test_cli.py              # CLI commands
└── conftest.py                  # Shared fixtures
```

---

## Core Fixtures (conftest.py)

```python
# tests/conftest.py
import pytest
import torch
from transformer_lens import HookedTransformer


@pytest.fixture(scope="session")
def model():
    """Load GPT-2 once for the test session."""
    m = HookedTransformer.from_pretrained(
        "gpt2",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
    )
    m.eval()
    return m


@pytest.fixture
def ioi_prompts():
    """Standard IOI test prompts from the paper."""
    return {
        "clean":     "When Mary and John went to the store, John gave a drink to",
        "corrupted": "When John and Mary went to the store, Mary gave a drink to",
        "target":    " Mary",
        "distractor": " John",
    }


@pytest.fixture
def known_circuit():
    """Known circuit from Glassbox paper (arXiv 2603.09988)."""
    return {
        "cited_heads": [(9, 9), (9, 6), (10, 0)],
        "head_effects": {(9, 9): 0.584, (9, 6): 0.211, (10, 0): 0.208},
        "clean_logit_diff": 3.36,
        "sufficiency": 1.00,
        "comprehensiveness": 0.22,
        "f1": 0.64,
    }
```

---

## Test Patterns for MI Code

### Testing Faithfulness Metrics

```python
@pytest.mark.parametrize("sufficiency,comprehensiveness,expected_f1", [
    (1.00, 0.22, 0.36),   # Glassbox paper result
    (0.80, 0.80, 0.80),   # Perfect balance
    (1.00, 0.00, 0.00),   # Zero comprehensiveness
    (0.00, 1.00, 0.00),   # Zero sufficiency
])
def test_f1_computation(sufficiency, comprehensiveness, expected_f1):
    f1 = compute_f1(sufficiency, comprehensiveness)
    assert abs(f1 - expected_f1) < 0.01
```

### Testing Tensor Outputs

```python
def test_attribution_scores_shape(model, ioi_prompts):
    """Attribution scores must be (n_layers, n_heads)."""
    from glassbox import GlassboxAnalyzer
    analyzer = GlassboxAnalyzer(model=model)
    results = analyzer.analyze(**ioi_prompts)

    expected_shape = (model.cfg.n_layers, model.cfg.n_heads)
    assert results.attribution_scores.shape == expected_shape, \
        f"Expected {expected_shape}, got {results.attribution_scores.shape}"

def test_attribution_scores_not_all_zero(model, ioi_prompts):
    """At least some heads must have non-zero attribution."""
    from glassbox import GlassboxAnalyzer
    analyzer = GlassboxAnalyzer(model=model)
    results = analyzer.analyze(**ioi_prompts)

    assert results.attribution_scores.abs().max() > 1e-6
```

### Testing Model Cleanup

```python
def test_hooks_cleaned_after_analysis(model, ioi_prompts):
    """No hooks should remain after analysis completes."""
    from glassbox import GlassboxAnalyzer
    analyzer = GlassboxAnalyzer(model=model)
    analyzer.analyze(**ioi_prompts)

    for name, hook in model.hook_dict.items():
        assert len(hook.fwd_hooks) == 0, f"Hook {name} not cleaned up"
```

### Testing Compliance Output

```python
def test_compliance_report_has_all_sections(model, ioi_prompts):
    """Compliance report must contain all 9 Annex IV sections."""
    from glassbox import GlassboxAnalyzer
    analyzer = GlassboxAnalyzer(model=model)
    results = analyzer.analyze(**ioi_prompts)
    report = analyzer.generate_report(results)
    report_dict = report.to_dict()

    annex = report_dict["annex_iv"]
    for i in range(1, 10):
        section_key = f"section_{i}_"
        matching = [k for k in annex if k.startswith(section_key)]
        assert len(matching) == 1, f"Section {i} missing from Annex IV"

def test_compliance_grade_from_f1_not_confidence(known_circuit):
    """Grade must be computed from F1, not model confidence."""
    from glassbox.core.compliance import compute_compliance_grade
    grade = compute_compliance_grade(f1_score=known_circuit["f1"])  # 0.64
    assert grade == "B"
```

---

## Running Tests

```bash
# Full suite with coverage
pytest --cov=glassbox --cov-report=term-missing -v

# Fast: unit tests only (no model loading)
pytest tests/unit/ -v

# Single file
pytest tests/unit/test_faithfulness.py -v

# With markers
pytest -m "not slow" -v      # Skip slow integration tests
pytest -m "slow" -v          # Only slow tests

# Stop on first failure
pytest -x -v
```

## Marking Slow Tests

```python
import pytest

@pytest.mark.slow
def test_full_pipeline_with_model_loading(ioi_prompts):
    """Full end-to-end test — loads GPT-2, runs analysis."""
    ...
```

Add to `pytest.ini`:
```ini
[pytest]
markers =
    slow: marks tests as slow (model loading, full pipeline)
```

---

## Property-Based Testing with Hypothesis

For mathematical properties of faithfulness metrics (which must hold for any valid inputs), use Hypothesis:

```python
# Install: pip install hypothesis

from hypothesis import given, strategies as st, settings
from hypothesis import assume
import pytest

@given(
    sufficiency=st.floats(min_value=0.0, max_value=1.0),
    comprehensiveness=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=500)
def test_f1_always_between_0_and_1(sufficiency, comprehensiveness):
    """F1 must always be in [0, 1] for any valid inputs."""
    assume(sufficiency + comprehensiveness > 0)  # Avoid division by zero
    f1 = compute_f1(sufficiency, comprehensiveness)
    assert 0.0 <= f1 <= 1.0

@given(
    sufficiency=st.floats(min_value=0.0, max_value=1.0),
    comprehensiveness=st.floats(min_value=0.0, max_value=1.0)
)
def test_f1_never_exceeds_max_component(sufficiency, comprehensiveness):
    """F1 harmonic mean is never greater than either component."""
    assume(sufficiency + comprehensiveness > 0)
    f1 = compute_f1(sufficiency, comprehensiveness)
    assert f1 <= max(sufficiency, comprehensiveness) + 1e-10

@given(
    score=st.floats(min_value=0.0, max_value=1.0)
)
def test_perfect_symmetry_gives_same_f1(score):
    """When sufficiency == comprehensiveness, F1 should equal them."""
    f1 = compute_f1(score, score)
    assert abs(f1 - score) < 1e-6

@given(
    n_heads=st.integers(min_value=1, max_value=100),
    effects=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=100)
)
def test_sufficiency_clips_to_range(n_heads, effects):
    """Sufficiency must always be clipped to [0, 1]."""
    cited_heads = [(0, i) for i in range(min(n_heads, len(effects)))]
    head_effects = {(0, i): e for i, e in enumerate(effects[:n_heads])}
    clean_ld = sum(effects[:n_heads]) + 0.1  # Ensure non-zero denominator

    result = compute_sufficiency(cited_heads, head_effects, clean_ld)
    assert 0.0 <= result <= 1.0
```

---

## Snapshot Testing for Reports

Compliance reports must be deterministic. Use snapshot tests to catch unexpected changes:

```python
# pip install syrupy

from syrupy.assertion import SnapshotAssertion

def test_compliance_report_snapshot(snapshot: SnapshotAssertion, known_circuit):
    """Compliance report structure must not change unexpectedly."""
    report = generate_compliance_report(
        model_id="gpt2",
        metrics=known_circuit,
        seed=42
    )
    # First run: creates snapshot. Subsequent runs: compares.
    assert report.to_dict() == snapshot
```

---

## Test Fixtures for CI (No GPU Required)

When running in CI without GPU, use these lightweight fixtures:

```python
@pytest.fixture(scope="session")
def mock_attribution_scores():
    """Pre-computed attribution scores — no model loading required."""
    # Realistic scores from the paper (GPT-2 Small, IOI task)
    scores = torch.zeros(12, 12)
    scores[9, 9] = 0.584
    scores[9, 6] = 0.211
    scores[10, 0] = 0.208
    return scores

@pytest.fixture(scope="session")
def mock_cache(mock_attribution_scores):
    """Minimal mock of TransformerLens ActivationCache for unit tests."""
    class MockCache:
        def __getitem__(self, key):
            return torch.randn(1, 15, 768)  # (batch, seq, d_model)
    return MockCache()
```

Add to `pytest.ini`:
```ini
[pytest]
markers =
    slow: requires model loading (~30s)
    gpu: requires CUDA
    unit: fast unit tests (<1s each)
    integration: integration tests (model + full pipeline)
```

CI command (fast, no GPU):
```bash
pytest -m "unit and not gpu" --cov=glassbox -q
```
