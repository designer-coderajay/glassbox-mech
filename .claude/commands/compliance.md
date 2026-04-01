# /compliance — Generate EU AI Act Annex IV Report

Generate a complete EU AI Act Annex IV technical documentation report from a Glassbox analysis.

## Usage

```
/compliance [model_name] [--prompt <text>] [--output <path>] [--format json|pdf]
```

## Steps

1. **Validate inputs**: model name against allowlist, prompt length ≤ 512 tokens
2. **Run Glassbox analysis**: attribution patching → circuit → faithfulness metrics
3. **Populate all 9 Annex IV sections**: use analysis results + model metadata
4. **Compute compliance grade**: from F1 score (NOT from confidence)
5. **Serialize to JSON**: GRC-importable schema with schema_version
6. **Optionally generate PDF**: if `--format pdf` specified
7. **Print summary + file path**

## Key Rules

- Grade is computed from F1, never from model confidence (r=0.009 makes confidence unreliable)
- All 9 sections must be non-empty in the output
- JSON must pass `python -m json.tool <output>` validation
- Timestamps in ISO 8601 format

## Output Schema

```json
{
  "schema_version": "1.0",
  "generated_by": "glassbox-mech-interp",
  "paper_reference": "arXiv:2603.09988",
  "generated_at": "<ISO 8601>",
  "model_id": "<model_name>",
  "glassbox_metrics": {
    "sufficiency": 1.00,
    "comprehensiveness": 0.22,
    "f1_score": 0.64,
    "confidence_faithfulness_correlation": 0.009,
    "cited_heads": [[9, 9], [9, 6], [10, 0]]
  },
  "compliance_grade": "B",
  "annex_iv": {
    "section_1_general_description": {},
    "section_2_development_process": {},
    "section_3_monitoring": {},
    "section_4_testing_validation": {},
    "section_5_risk_assessment": {},
    "section_6_transparency": {},
    "section_7_data_governance": {},
    "section_8_human_oversight": {},
    "section_9_cybersecurity": {}
  }
}
```

## Verify Output

After generating:
```bash
python -m json.tool outputs/compliance_report.json > /dev/null && echo "Valid JSON"
python -c "
import json
with open('outputs/compliance_report.json') as f:
    r = json.load(f)
assert len(r['annex_iv']) == 9, 'Missing sections'
assert r['compliance_grade'] in 'ABCD', 'Invalid grade'
print('Compliance report: OK')
"
```
