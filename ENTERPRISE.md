# Deployment and Licensing Guide — Glassbox AI v4.6.1

This guide is for engineering and compliance teams who want to run Glassbox on their own
infrastructure. It covers installation, a first documentation draft, CI use, container use
and licensing. It states what has been verified and what has not.

> **Not legal advice.** Glassbox drafts technical documentation from measured model
> behaviour. It does not decide whether a system is high-risk, does not produce a
> declaration of conformity (Article 47) or a notified-body certificate (Article 43), and
> does not by itself make anyone compliant. Confirm classification and sufficiency with
> qualified counsel.

---

## What is available

| Capability | Status |
|---|---|
| Annex IV draft: 9-section structure; measured fields filled from the analysis, provider-only fields (including the §8 declaration of conformity) marked `[PROVIDER TO COMPLETE]` | Released |
| Circuit discovery and faithfulness metrics (suff / comp / F1, grade A–D) | Released; validated 82M → 12B across 9 architecture families (see `docs/VALIDATION_LOG.md`) |
| Larger models (> ~13B) via gradient checkpointing and offload | Implemented, **not yet validated live** |
| CircuitDiff (version-to-version circuit comparison) | Released |
| SHA-256 hash-chained audit log (and hashed evidence packages for investigations) | Released. Hashing supports integrity checks; it is **not** a digital signature. |
| Container image for self-hosted scans (`scan.Dockerfile`) | Available; build it yourself |
| AI-system investigation layer (`glassbox/v7/`) | In the PyPI package from 4.6.0; research stage (see `ASSURANCE.md`) |

There is currently **no hosted service, no SLA and no paid support plan**.

---

## 1. Install

```bash
pip install "glassbox-mech-interp[compliance]"
```

## 2. Draft your first Annex IV document

```python
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2
from glassbox.compliance import AnnexIVReport, DeploymentContext

model = HookedTransformer.from_pretrained("gpt2")   # your open-weight model
gb    = GlassboxV2(model)

result = gb.analyze(
    prompt    = "Your decision prompt here",
    correct   = " expected_output_token",
    incorrect = " alternative_output_token",
)

report = AnnexIVReport(
    model_name         = "YourModelV1",
    system_purpose     = "What your AI system does",
    provider_name      = "Your legal entity",
    provider_address   = "Your registered address",
    deployment_context = DeploymentContext.FINANCIAL_SERVICES,
)
report.add_analysis(result)
report.to_pdf("annex_iv_draft.pdf")    # review before adding to your technical file
with open("annex_iv_draft.json", "w", encoding="utf-8") as f:
    f.write(report.to_json())
```

Fields marked `[PROVIDER TO COMPLETE]` must be filled in by you. A low grade (C or D) means
Glassbox could not find a faithful explanation; it does not certify the system either way.

## 3. CI use

`action.yml` provides a GitHub Action that fails a build when the explainability grade
falls below a threshold.

> **Data warning.** By default the action calls the **public demo** on Hugging Face Spaces
> (`api_url`). Do not send confidential or personal data to it. For real data, run your
> own instance and set `api_url` to it.

## 4. Container use

```bash
git clone https://github.com/designer-coderajay/glassbox-mech
cd glassbox-mech
docker build -f scan.Dockerfile -t glassbox-scan .

docker run --rm -v "$(pwd)/output:/output" glassbox-scan \
    --model gpt2 \
    --prompt "Your decision prompt" \
    --correct " correct_token" \
    --incorrect " wrong_token" \
    --purpose "Your system purpose" \
    --provider "Your organisation" \
    --output /output/annex_iv_draft.pdf
```

**Offline / air-gapped use:** the container downloads model weights from Hugging Face
unless they are already available. For an offline environment, pre-load the weights into
the image or a mounted cache and set `HF_HUB_OFFLINE=1`. This setup has not been verified
end-to-end by the maintainer; test it in your environment.

---

## Licensing

| Component | License | Your use |
|---|---|---|
| Interpretability core and everything not listed below | MIT | Free for any use |
| Compliance engine (`compliance.py`, `circuit_diff.py`, `risk_register.py`, `bias.py`, `audit_log.py`, `cross_model.py`) and the investigation layer (`glassbox/v7/`, `dashboard/v7_investigation.py`, `dashboard/v7_agent_demo.py`) | BSL 1.1 | Free for non-production use and for production use inside your own organisation. A separate commercial license is needed to offer them to third parties as a compliance, governance, investigation, debugging or observability service or product. Each version converts to Apache-2.0 four years after first publication. |

`NOTICE` is the authoritative file-by-file map; `LICENSE` and `LICENSE-COMMERCIAL` contain
the license texts. Commercial license and design-partner enquiries:
**mahale.ajay01@gmail.com**.

---

