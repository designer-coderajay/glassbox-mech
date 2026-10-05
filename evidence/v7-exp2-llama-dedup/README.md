# Evidence package: v7-exp2-llama-dedup

**Claim.** On a real pre-existing bug in llama-index-core (issue #21033), Glassbox traces and diff placed the sync-vs-async behavioural difference at the retrieval step under 0.14.17, showed no difference under 0.14.19, showed the version change altered the sync path's retrieval but not the async path's, and were deterministic on re-run.

**Status:** `EXPERIMENTALLY_SUPPORTED` (ASSURANCE.md taxonomy)

**Scope and limits.** Pre-registered (protocol commit d866eea) with a disclosed sandbox dry run (Addendum 1). Not blind: the cause was known from the issue. Minimal repro from the issue, no LLM, two pinned versions. Retrieved-document hashes agreed across macOS and Linux runs by the same author; not independently reproduced.

## Provenance

- assurance_row: `V2.5`
- issue: `https://github.com/run-llama/llama_index/issues/21033`
- protocol_commit: `d866eea`
- versions: `llama-index-core 0.14.17 (bug), 0.14.19 (fixed)`

## Contents

- `README.md`
- `artifacts/code/evaluate.py`
- `artifacts/code/subject.py`
- `artifacts/measurements/RESULTS.md`
- `artifacts/measurements/results.json`
- `artifacts/protocol/PROTOCOL.md`
- `artifacts/protocol/RUN.md`
- `artifacts/traces_0_14_17_async/r1.trace.jsonl`
- `artifacts/traces_0_14_17_async/r2.trace.jsonl`
- `artifacts/traces_0_14_17_sync/r1.trace.jsonl`
- `artifacts/traces_0_14_17_sync/r2.trace.jsonl`
- `artifacts/traces_0_14_19_async/r1.trace.jsonl`
- `artifacts/traces_0_14_19_async/r2.trace.jsonl`
- `artifacts/traces_0_14_19_sync/r1.trace.jsonl`
- `artifacts/traces_0_14_19_sync/r2.trace.jsonl`
- `claim.json`
- `provenance.json`

## Verify

`python -m glassbox.v7.evidence verify <path-to>/v7-exp2-llama-dedup`

**Integrity is not truth.** a verifying manifest shows these files are unchanged since packaging; it does not show the experiment or the claim is correct.
