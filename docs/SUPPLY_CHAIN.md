# Supply chain checks

*Last run 2026-10-31, for release 4.6.1 (V8 phase B2). Re-run before each release.
None of this is legal advice.*

## 1. Secret scan (tree and full history)

**Method.** A pattern scan was run over every tracked file, and over every added line
in every commit of every branch (`git log --all -p`, 550 commits at the time). The
patterns were:
- AWS, GitHub, Hugging Face, PyPI, OpenAI, Anthropic, Slack, Google and Stripe key
  formats;
- private-key headers;
- generic `key/secret/token/password = "…"` assignments of 24+ characters.

The patterns were checked against planted test tokens (all detected). File names ever
committed were also checked for `.env`, `*.pem`, `*.key`, `id_rsa*`,
`credentials*.json`, `.pypirc` and `secrets.*`.

**Result.**
- 0 matches in the tree, 0 in history.
- The only sensitive-looking file name ever committed is `.env.example`. Its value is
  a placeholder.

**Limits.**
- A pattern scan only finds the listed formats.
- Run an independent scanner too. For example, gitleaks (MIT), on your own machine:

  ```
  brew install gitleaks
  gitleaks git --redact -v .
  ```

  Older gitleaks versions use `gitleaks detect --redact -v` instead.

## 2. Dependency licences

**Method.** `scripts/license_audit.py` reads the metadata of the installed packages
and follows runtime requirements (extras excluded) from the base dependencies:
`torch`, `transformer-lens`, `einops`, `numpy` and `scipy`. Verdicts:
- **ALLOW:** permissive.
- **REVIEW:** weak copyleft, or no licence information.
- **DENY:** GPL, AGPL or SSPL. The project rules exclude these. They can never be
  accepted.

CI runs the audit on the pinned stack on every full-lane run.

**Result** (local Python 3.11 environment with torch 2.10.0, transformer-lens 2.17.0
and transformers 4.57.6; some transitive versions differ from the CI pins):
- 68 packages: 64 ALLOW, 4 REVIEW, **0 DENY**.

The 4 REVIEW packages were read by hand and are passed to the audit with `--accept`:

| Package | Metadata says | Finding |
|---|---|---|
| certifi | MPL-2.0 | Weak copyleft at file level. Used unmodified as an installed dependency; Glassbox does not copy or ship its files. |
| tqdm | MPL-2.0 AND MIT | Same as certifi. |
| python-dateutil | "Dual License" | Its classifiers list BSD and Apache Software License. |
| sentencepiece | (no licence in metadata) | Upstream `google/sentencepiece` LICENSE is Apache-2.0 (fetched 2026-10-31). |

Any new REVIEW or DENY package fails CI until it has been looked at.

Optional extras (`sae`, `api`, `mlflow`, ...) are not audited by this run. Audit them
with `--roots <extra packages>` before relying on them commercially.

## 3. Publishing

From 4.6.1, `.github/workflows/publish.yml` uploads to PyPI with **Trusted Publishing**
(OIDC):
- no API token is stored in GitHub;
- **attestations** are switched on.

The PyPI side is configured by the owner, as described at the top of the workflow file.
