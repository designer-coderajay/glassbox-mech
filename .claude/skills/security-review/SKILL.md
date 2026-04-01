---
name: security-review
description: Security checklist for Glassbox. Activates when handling model loading from user input, API keys, HuggingFace tokens, file paths, or adding new CLI/API endpoints.
origin: Glassbox (adapted from ECC security-review)
---

# Security Review for Glassbox

## When to Activate

- Adding new model loading paths (user-supplied model names)
- Handling API keys or HuggingFace tokens
- Adding new CLI flags that accept file paths
- Building new Gradio UI inputs
- Adding new API endpoints

---

## Glassbox-Specific Threat Surface

| Risk | Where | Severity |
|------|-------|---------|
| Arbitrary model loading | CLI `--model` arg, Gradio input | HIGH |
| Pickle deserialization | Model file loading | HIGH |
| Path traversal | File export, JSON save | HIGH |
| HF token exposure | `HfApi`, environment vars | HIGH |
| Prompt injection in reports | Compliance JSON generation | MEDIUM |
| Unconstrained memory use | Large model + long prompt | MEDIUM |

---

## Checklist

### 1. Secrets Management

```python
# WRONG — hardcoded token
HF_TOKEN = "hf_xxxxxxxxxxxxx"

# CORRECT — environment variable
import os
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable not set")
```

Verify:
- [ ] No tokens/API keys in source code
- [ ] `.env` file in `.gitignore`
- [ ] `HF_TOKEN`, `OPENAI_API_KEY` loaded from environment only
- [ ] No secrets in git history (`git log -p | grep -i "hf_"`)

### 2. Model Name Validation

Never load arbitrary model names from user input without validation:

```python
# WRONG — loads any model the user requests
model = HookedTransformer.from_pretrained(user_input_model_name)

# CORRECT — validate against allowlist
ALLOWED_MODELS = {
    "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
}

def load_model_safe(model_name: str) -> HookedTransformer:
    """Load a model from the allowlist only."""
    if model_name not in ALLOWED_MODELS:
        raise ValueError(
            f"Model '{model_name}' not in allowlist. "
            f"Allowed: {sorted(ALLOWED_MODELS)}"
        )
    return HookedTransformer.from_pretrained(model_name)
```

### 3. File Path Validation

When saving reports or accepting file paths from users:

```python
import os

# WRONG — path traversal risk
def save_report(report: dict, path: str):
    with open(path, "w") as f:
        json.dump(report, f)

# CORRECT — validate and normalize
def save_report(report: dict, path: str, allowed_dir: str = "./outputs"):
    # Normalize and verify the path stays within allowed_dir
    abs_allowed = os.path.abspath(allowed_dir)
    abs_path = os.path.abspath(path)

    if not abs_path.startswith(abs_allowed):
        raise PermissionError(
            f"Path '{path}' is outside allowed directory '{allowed_dir}'"
        )

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w") as f:
        json.dump(report, f, indent=2)
```

### 4. Pickle Safety

TransformerLens uses pickle internally for model weights (via `torch.load`). Only load from trusted sources:

```python
# WRONG — loading from user-supplied path
model = torch.load(user_provided_path)

# CORRECT — use HuggingFace Hub only, never raw user paths
model = HookedTransformer.from_pretrained(validated_model_name)

# If you must load a local checkpoint, restrict to a known directory
CHECKPOINTS_DIR = os.path.abspath("./checkpoints")
def load_local_checkpoint(filename: str):
    path = os.path.join(CHECKPOINTS_DIR, filename)
    if not path.startswith(CHECKPOINTS_DIR):
        raise PermissionError("Checkpoint path traversal detected")
    return torch.load(path, weights_only=True)  # weights_only=True is safer
```

### 5. Gradio Input Validation

In `dashboard/app.py`:

```python
# WRONG — accepts any string as model name
model_input = gr.Textbox(label="Model name")

# CORRECT — restrict to dropdown
model_input = gr.Dropdown(
    choices=list(ALLOWED_MODELS),
    value="gpt2",
    label="Select model"
)

# WRONG — prompt with no length limit
prompt_input = gr.Textbox(label="Prompt")

# CORRECT — limit prompt length
MAX_PROMPT_TOKENS = 512

def validate_prompt(prompt: str) -> str:
    if len(prompt.strip()) == 0:
        raise gr.Error("Prompt cannot be empty")
    tokens = model.to_tokens(prompt)
    if tokens.shape[1] > MAX_PROMPT_TOKENS:
        raise gr.Error(f"Prompt too long ({tokens.shape[1]} tokens, max {MAX_PROMPT_TOKENS})")
    return prompt.strip()
```

### 6. Memory Limits

Prevent OOM from large models or prompts:

```python
MAX_PROMPT_TOKENS = 512
MAX_MODEL_PARAMS = 1_500_000_000  # 1.5B parameters

def check_memory_requirements(model_name: str, prompt_len: int):
    model_params = MODEL_PARAM_COUNTS.get(model_name, float('inf'))
    if model_params > MAX_MODEL_PARAMS:
        raise MemoryError(f"Model {model_name} exceeds parameter limit")
    if prompt_len > MAX_PROMPT_TOKENS:
        raise MemoryError(f"Prompt length {prompt_len} exceeds token limit")
```

---

## Security Scan Commands

```bash
# Scan for hardcoded secrets
grep -r "hf_[a-zA-Z0-9]" glassbox/ --include="*.py"
grep -r "sk-" glassbox/ --include="*.py"

# Bandit security scan
pip install bandit
bandit -r glassbox/ -ll

# Safety check for vulnerable dependencies
pip install safety
safety check -r requirements/base.txt

# Check for eval/exec usage
grep -rn "eval\|exec\|pickle\.loads" glassbox/ --include="*.py"
```

---

## Security Approval Criteria

- [ ] No hardcoded secrets anywhere in `glassbox/`
- [ ] All user-supplied model names validated against allowlist
- [ ] All user-supplied file paths validated against allowed directory
- [ ] No `eval()`, `exec()`, or `pickle.loads()` on untrusted data
- [ ] `bandit -r glassbox/ -ll` shows no HIGH severity issues
- [ ] `safety check` shows no known vulnerabilities in dependencies

---

## Supply Chain Security

```bash
# Audit all dependencies for known CVEs
pip install pip-audit
pip-audit -r requirements/base.txt -r requirements/dev.txt

# Check for typosquatting / malicious packages
# TransformerLens correct name: transformer-lens (not transformerlens, transformer_lens)
# PyTorch correct name: torch (not pytorch or torchvision standalone)

# Lock file for reproducibility
pip install pip-tools
pip-compile requirements/base.in --output-file requirements/base.txt
pip-compile requirements/dev.in --output-file requirements/dev.txt
```

## Model Card Validation

Hugging Face models should have model cards. Validate before loading:

```python
from huggingface_hub import hf_hub_download, HfApi

def validate_model_has_card(model_name: str) -> bool:
    """Check model has a model card before loading."""
    api = HfApi()
    try:
        info = api.model_info(model_name)
        return info.card_data is not None
    except Exception:
        return False

# Use in loading flow
if not validate_model_has_card(model_name):
    import logging
    logging.warning(f"Model {model_name} has no model card — loading anyway but verify source")
```

## Rate Limiting for the Gradio Demo

The HuggingFace Space should rate-limit analysis requests to prevent abuse:

```python
import time
from collections import defaultdict

# Simple in-memory rate limiter
_request_times = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = 5

def check_rate_limit(client_id: str = "default") -> bool:
    """Returns True if request is allowed, False if rate limited."""
    now = time.time()
    window = 60  # 1 minute
    
    # Clean old entries
    _request_times[client_id] = [
        t for t in _request_times[client_id] if now - t < window
    ]
    
    if len(_request_times[client_id]) >= MAX_REQUESTS_PER_MINUTE:
        return False
    
    _request_times[client_id].append(now)
    return True
```
