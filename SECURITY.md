# Security & Privacy

## Supported Versions

Security fixes are provided for the current minor release line. Always run the
latest `4.3.x` from PyPI (`pip install -U glassbox-mech-interp`), which carries
the verified, version-pinned dependency stack (`numpy<2`, `torch<2.11`,
`transformer_lens<3`).

| Version | Supported          |
| ------- | ------------------ |
| 4.3.x   | :white_check_mark: |
| < 4.3   | :x:                |

---

## API Key Handling

Glassbox never stores, logs, or retains your model provider API keys.

**Architecture (v4.2.6+):**
- API keys are passed via the `X-Provider-Api-Key` HTTP request header — not in the request body
- All log filters explicitly scrub any string matching known key patterns before writing to disk
- Keys are used in-memory only for the duration of the HTTP request
- Keys are never written to `_REPORT_STORE`, PDF reports, or any persistent storage
- Compliance reports (JSON/PDF) contain zero credential data — only model behaviour metrics

**Verification:** You can confirm this yourself by reading [`api/main.py`](api/main.py):
- `_StripKeyFilter` class: active on all log handlers
- `_REPORT_STORE` write: contains only `json`, `mode`, `created_at`
- `BlackBoxRequest` model: contains no `api_key` field

---

## Recommended: Self-Hosting

For production compliance audits, run Glassbox locally. Your keys never leave your infrastructure:

```bash
git clone https://github.com/designer-coderajay/glassbox-mech
docker build -t glassbox .
docker run -p 8000:8000 glassbox
```

The hosted instance at `https://designer-coderajay-glassbox-ai-2-0-mechanistic-interpretability-tool.hf.space` is provided for evaluation and testing only. Do not use the hosted instance for production compliance workflows involving sensitive data.

---

## Reporting Vulnerabilities

Please report security issues **privately** to: [mahale.ajay01@gmail.com](mailto:mahale.ajay01@gmail.com)

Do **not** open a public GitHub issue for security vulnerabilities.

---

## Data Processing and GDPR

**No personal data is intentionally collected.** Glassbox processes:
- Model prompts and outputs (text provided by the user)
- API keys (in-memory only, never persisted — see above)
- Compliance metrics and scores (logged to the local report store with a random report ID)

**GDPR applicability (Regulation (EU) 2016/679):**

If you use Glassbox to audit an AI system that processes personal data (e.g., credit scoring prompts containing applicant details), you are the data controller for that processing. Glassbox acts as a data processor only for the duration of a live API call to the hosted instance.

For self-hosted deployments, all processing stays within your own infrastructure and Glassbox's author is not a data processor.

API keys are not personal data under GDPR Article 4(1) — they do not identify a natural person.

**Legitimate interest (Article 6(1)(f)):** Where the hosted API processes text snippets incidentally, the processing is for the legitimate interest of providing the requested compliance documentation service, and no data is retained beyond the individual request.

---

## Threat Model and Mitigations (reviewed 2026-09-28)

Attack classes checked against the website, the waitlist endpoint (`api/waitlist.js`),
the REST API (`api/main.py`) and the library. "N/A" means the vulnerable component does
not exist in Glassbox; it is listed so the review is complete.

| Attack class | Where it could apply | Status and mitigation |
|---|---|---|
| Stored XSS / HTML injection | Annex IV HTML vault renders model, provider and use-case strings supplied by users | **Fixed 2026-09-28.** All interpolated values pass through `evidence_vault._esc` (`html.escape`). Regression test: `tests/test_security_hardening.py`. |
| Markup injection in PDF | ReportLab `Paragraph` interprets tags in provider/purpose/address strings | **Fixed 2026-09-28.** Values escaped with `xml.sax.saxutils.escape`. |
| SSTI (server-side template injection) | Would need a template engine rendering user input | **N/A.** No Jinja or other template engine renders user input; reports use escaped f-strings. |
| Unsafe deserialisation (pickle RCE) | `torch.load` of SAE checkpoints and steering vectors from user paths | **Fixed 2026-09-28.** `torch.load(..., weights_only=True)`; a test fails if any `torch.load` omits it. Only load checkpoints you trust. |
| ReDoS | Waitlist email regex; API head-label regex `L(\d+)H(\d+)` | Email length capped at 254 before the regex (fixed 2026-09-28). The API regex is linear. |
| Resource-exhaustion DoS (oversized prompts / fields) | API request bodies drive model compute | **Fixed 2026-09-28.** `max_length` on every string field (prompts 4,000 chars), `top_k` 1–64, at most 64 heads; model allowlist; 20 req/min per IP. Waitlist fields truncated server-side. |
| SQL / NoSQL injection | Waitlist storage | **N/A for SQL** (no database). Vercel KV (Upstash) is called with a JSON command array, so user input is a value, never parsed as a command. |
| Secret key leaks | Repository history, logs | History scanned 2026-09-28 for OpenAI, Hugging Face, PyPI, Resend, AWS, GitHub tokens and private keys: none found; only `.env.example` is tracked. Provider API keys travel in a header, are never stored, and are scrubbed from logs by `_StripKeyFilter`. PyPI publishing uses OIDC (no stored token). |
| Replay attack | Outgoing signed webhooks | **Fixed 2026-09-28.** Each delivery carries `X-Glassbox-Delivery` (UUID) and `X-Glassbox-Timestamp`, both covered by `X-Glassbox-Signature-V2 = HMAC-SHA256(secret, ts.id.body)`. Receivers should reject timestamps older than 300 s and repeated delivery ids. The legacy body-only signature is still sent for compatibility. |
| Zip Slip (archive path traversal) | Extracting user archives | **N/A.** No archive extraction in the library or API. |
| CORS abuse | Waitlist endpoint | The function enforces an origin allowlist. Note: `vercel.json` also sets `Access-Control-Allow-Origin: *` on `/api/*`; worst case is spam signups from other sites. Tightening it is tracked, pending a deploy test. |
| Privacy leaks via third parties | Website fonts, embeds, trackers | Fonts self-hosted since 2026-09-28; no iframes, analytics or cookies. See `docs/privacy.html`. |

## Legal Jurisdiction and Governing Law

This project is developed under the laws of the Federal Republic of Germany. EU law (including Regulation (EU) 2024/1689 and Regulation (EU) 2016/679) applies directly.

---

## Disclaimer on Regulatory Adequacy

**Nothing in this security notice, or in the Glassbox software, constitutes a guarantee that use of Glassbox satisfies any obligation under Regulation (EU) 2024/1689 (EU AI Act), GDPR, or any other applicable law.** Whether your deployment of Glassbox in a compliance workflow satisfies your specific regulatory obligations is a matter for qualified legal counsel.

See also: [`README.md — Legal Notices & Regulatory Disclaimer`](README.md#legal-notices--regulatory-disclaimer)

---

## References

- [EU AI Act Regulation (EU) 2024/1689](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689) — Article 12 (logging), Article 99(4) (penalties)
- [GDPR Regulation (EU) 2016/679](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679) — Articles 4, 6, 28, 35
