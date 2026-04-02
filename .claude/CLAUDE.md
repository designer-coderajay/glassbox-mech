# Glassbox — Mechanistic Interpretability Library
## Claude Code Project Configuration

**Stack**: Python 3.10+ · PyTorch · TransformerLens · Gradio · arXiv 2603.09988
**Package**: `glassbox-mech-interp` on PyPI
**Live demo**: [HuggingFace Space](https://huggingface.co/spaces/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool)
**Website**: [glassbox-ai.vercel.app](https://project-gu05p.vercel.app)

---

## What Glassbox Does

Causal circuit discovery for transformer models. Given a prompt, it:
1. Runs attribution patching to rank every attention head by causal contribution
2. Identifies the minimal circuit of heads that explains the prediction
3. Computes faithfulness metrics (sufficiency, comprehensiveness, F1)
4. Optionally generates EU AI Act Annex IV compliance documentation (9 sections)

**Key numbers to preserve across all code and docs:**
- 3 forward passes per analysis
- 1.2s on CPU
- 37× faster than ACDC (Conmy et al. 2023)
- r = 0.009 (confidence–faithfulness correlation)
- Sufficiency: 1.00, Comprehensiveness: 0.22, F1: 0.64
- 76 automated tests
- 9 EU AI Act Annex IV sections

---

## Directory Structure

```
glassbox/
├── glassbox/              # Core library
│   ├── core/
│   │   ├── attention.py   # AttentionAnalyzer
│   │   ├── patching.py    # Attribution patching engine
│   │   └── logit_lens.py  # Logit lens visualization
│   ├── faithfulness.py    # Sufficiency / comprehensiveness / F1
│   ├── models/            # Model loader (TransformerLens)
│   ├── viz/               # Circuit + attention visualizations
│   └── utils/
├── tests/                 # 76 pytest tests
├── dashboard/             # Gradio HF Space app
├── docs/                  # Vercel website (index.html)
├── requirements/
│   ├── base.txt
│   ├── api.txt
│   └── dev.txt
└── pyproject.toml
```

---

## Agents Available

| Agent | When to Use |
|-------|-------------|
| `python-reviewer` | After any `.py` file changes |
| `pytorch-build-resolver` | PyTorch / TransformerLens / CUDA errors |
| `interpretability-researcher` | Designing new MI experiments |
| `compliance-generator` | EU AI Act Annex IV work |
| `code-reviewer` | General quality + security pass |
| `doc-updater` | After feature changes — sync README, CHANGELOG, docstrings |

---

## Skills Available

| Skill | Activates When |
|-------|----------------|
| `mechanistic-interpretability` | Writing or reviewing MI logic |
| `circuit-discovery` | Attribution patching or circuit work |
| `eu-ai-act-compliance` | Annex IV generation or compliance output |
| `pytorch-transformerlens` | PyTorch + TransformerLens code |
| `python-testing` | Writing or reviewing tests |
| `security-review` | API keys, model loading, user input |

---

## Commands

| Command | Action |
|---------|--------|
| `/audit` | Run full model audit (circuit + faithfulness + report) |
| `/compliance` | Generate EU AI Act Annex IV JSON |
| `/review` | Python code review (ruff + mypy + manual) |
| `/test` | Run pytest with coverage report |
| `/circuit` | Run attribution patching on a prompt |

---

## Coding Standards

### Python
- **Type hints** on all public functions — no `Any` unless unavoidable
- **Docstrings** on all public classes and functions (Google style)
- **No bare except** — catch specific exceptions
- **No print()** in library code — use `logging`
- **Context managers** for file and resource handling
- Max function length: 50 lines. Max params: 5. Use dataclass if more.

### Testing
- **TDD**: write the test first, make it pass, refactor
- **80%+ coverage** across the whole package
- **100% coverage** on `faithfulness.py` and `patching.py` (critical paths)
- Run: `pytest --cov=glassbox --cov-report=term-missing -v`

### Commits
- Format: `type(scope): message`
- Types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`
- Example: `feat(patching): add activation patching for MLP layers`

### Key numbers are facts — never change them
The metrics (r=0.009, sufficiency=1.00, F1=0.64, 76 tests, 37× speedup) are verified experimental results from the paper. Never round, estimate, or alter them.

---

## Environment Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest --cov=glassbox -v

# Lint
ruff check glassbox/
mypy glassbox/

# Format
black glassbox/ tests/
```

---

## Research Context

- **Paper**: arXiv 2603.09988 (Mahale, 2026)
- **Method**: Attribution patching via activation difference × gradient
- **Key finding**: Confidence ≠ faithfulness (r=0.009). Compliance auditors cannot rely on confidence scores.
- **Regulatory relevance**: EU AI Act Article 13 (transparency), Article 17 (quality management), Annex IV (technical documentation)
- **Baseline**: ACDC (Conmy et al., NeurIPS 2023)
