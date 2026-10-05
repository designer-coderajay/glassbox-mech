# Evidence package: v7-exp1-rag-fault

**Claim.** In a controlled RAG pipeline with one injected cause (a stale index missing four documents), the Glassbox trace diff placed the first divergence at the retrieval step for 4/4 affected questions, reported no divergence for 4/4 unaffected questions, confirmed identical behaviour after the fault was undone (8/8) and was deterministic on re-run (8/8).

**Status:** `EXPERIMENTALLY_SUPPORTED` (ASSURANCE.md taxonomy)

**Scope and limits.** Pre-registered (protocol commit 0e70405, before the run). GPT-2 small, teacher-forced two-option scoring, 8 fictional questions, bag-of-words retriever (top_k=1), no sampling, single machine (CPU). The cause was known by construction; this does not show Glassbox localises unknown causes in real systems (V7 Hard Gate 3, open). Criteria recomputed from the traces by the author; not independently reproduced.

## Provenance

- assurance_row: `V2.3`
- model: `gpt2 (TransformerLens)`
- protocol_commit: `0e70405`
- results_commit: `aedfebf`

## Contents

- `README.md`
- `artifacts/code/run.py`
- `artifacts/measurements/RESULTS.md`
- `artifacts/measurements/results.json`
- `artifacts/protocol/PROTOCOL.md`
- `artifacts/protocol/corpus.json`
- `artifacts/traces_baseline/q1.trace.jsonl`
- `artifacts/traces_baseline/q2.trace.jsonl`
- `artifacts/traces_baseline/q3.trace.jsonl`
- `artifacts/traces_baseline/q4.trace.jsonl`
- `artifacts/traces_baseline/q5.trace.jsonl`
- `artifacts/traces_baseline/q6.trace.jsonl`
- `artifacts/traces_baseline/q7.trace.jsonl`
- `artifacts/traces_baseline/q8.trace.jsonl`
- `artifacts/traces_baseline_repeat/q1.trace.jsonl`
- `artifacts/traces_baseline_repeat/q2.trace.jsonl`
- `artifacts/traces_baseline_repeat/q3.trace.jsonl`
- `artifacts/traces_baseline_repeat/q4.trace.jsonl`
- `artifacts/traces_baseline_repeat/q5.trace.jsonl`
- `artifacts/traces_baseline_repeat/q6.trace.jsonl`
- `artifacts/traces_baseline_repeat/q7.trace.jsonl`
- `artifacts/traces_baseline_repeat/q8.trace.jsonl`
- `artifacts/traces_fault/q1.trace.jsonl`
- `artifacts/traces_fault/q2.trace.jsonl`
- `artifacts/traces_fault/q3.trace.jsonl`
- `artifacts/traces_fault/q4.trace.jsonl`
- `artifacts/traces_fault/q5.trace.jsonl`
- `artifacts/traces_fault/q6.trace.jsonl`
- `artifacts/traces_fault/q7.trace.jsonl`
- `artifacts/traces_fault/q8.trace.jsonl`
- `artifacts/traces_restored/q1.trace.jsonl`
- `artifacts/traces_restored/q2.trace.jsonl`
- `artifacts/traces_restored/q3.trace.jsonl`
- `artifacts/traces_restored/q4.trace.jsonl`
- `artifacts/traces_restored/q5.trace.jsonl`
- `artifacts/traces_restored/q6.trace.jsonl`
- `artifacts/traces_restored/q7.trace.jsonl`
- `artifacts/traces_restored/q8.trace.jsonl`
- `claim.json`
- `provenance.json`

## Verify

`python -m glassbox.v7.evidence verify <path-to>/v7-exp1-rag-fault`

**Integrity is not truth.** a verifying manifest shows these files are unchanged since packaging; it does not show the experiment or the claim is correct.
