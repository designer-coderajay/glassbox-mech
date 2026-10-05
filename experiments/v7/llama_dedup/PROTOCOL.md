# V7 Experiment 2 — A real third-party retrieval bug: llama_index #21033

*Protocol v1.0, written 2026-10-05, BEFORE any run of the repro. It is committed before
execution and not edited afterwards; deviations go in dated addenda below the line.*

## 1. Question

Experiment 1 used a fault we injected ourselves. This experiment uses a **real,
pre-existing bug in a widely used framework**:
- llama_index issue **#21033**, *Sync recursive retrieval misses `ref_doc_id` in dedup
  key* — https://github.com/run-llama/llama_index/issues/21033

The question: when the same retriever is called through the sync and async paths, do
Glassbox traces and their diff:
- (a) localise the behavioural difference to retrieval under the buggy version;
- (b) show it disappears under a version where the sync path is fixed;
- (c) leave the unaffected path untouched?

**Honesty about blindness.** We have read the issue, so the cause is known to us. This
is a test on a **real, third-party failure we did not create**. It is **not a blind
test**: it does not show Glassbox finds unknown causes (ASSURANCE V2.4 stays
`UNRESOLVED`).

## 2. Facts established before the run (static inspection only, 2026-10-05)

Source of `llama_index/core/base/base_retriever.py` in the PyPI wheels:

| Version | Sync path dedup key | Async path dedup key |
|---|---|---|
| 0.14.17 | `node.hash` | `(node.hash, node.ref_doc_id)` |
| 0.14.18 | `node.hash` (still buggy, although fix PR #21034 was merged 2026-03-18) | `(node.hash, node.ref_doc_id)` |
| 0.14.19 | `node.node_id` | `node.node_id` |

The repro was **not executed** before this protocol was written.

## 3. System under test (fixed)

- **Repro:** the issue's own minimal example, as given in the issue body. A
  `BaseRetriever` subclass returns two nodes:
  - both have text `"shared content"` and empty metadata;
  - their sources are `doc-1` (score 0.9) and `doc-2` (score 0.8);
  - the query is `"test"`.
- **No LLM** is involved.
- **Versions:** `llama-index-core==0.14.17` (bug) and `llama-index-core==0.14.19`
  (fixed sync path). Each runs in its own virtual environment.
- **Paths:** `sync` = `retriever.retrieve("test")`; `async` =
  `await retriever.aretrieve("test")`.
- **Trace:** one Glassbox trace per (version, path, repetition), policy `hash_only`.
  - The retrieval span records `gen_ai.retrieval.documents` (hashed) as a list of
    `{ref_doc_id, score, text_sha256}`.
  - Node IDs are not recorded, because llama_index generates them randomly per run.
  - The root records the installed `llama-index-core` version in
    `glassbox.env.packages`.
  - The subject loads `glassbox/v7/trace.py` by file path, which is stdlib-only, so
    the venv needs no Glassbox install or torch.

## 4. Runs

For each version ∈ {0.14.17, 0.14.19} and path ∈ {sync, async}: a run `r1` and a
repeat `r2`. That is 8 traces.

## 5. Pre-declared criteria

All are evaluated with `glassbox.v7.diff.diff_traces`.

| ID | Criterion | Pass rule |
|---|---|---|
| C1 Localisation (bug) | 0.14.17: `async` vs `sync` → first divergence at the retrieval span | yes |
| C2 Fix confirmed | 0.14.19: `async` vs `sync` → no behavioural divergence | yes |
| C3a Intervention on affected path | `sync` 0.14.17 vs 0.14.19 → first divergence at the retrieval span | yes |
| C3b Specificity on unaffected path | `async` 0.14.17 vs 0.14.19 → no behavioural divergence | yes |
| C4 Determinism | every (version, path): `r2` vs `r1` → no behavioural divergence | 4/4 |
| C5 Descriptive | number of documents returned per (version, path) | reported, not gated |

**Overall:** PASS iff C1, C2, C3a, C3b and C4 all pass. A FAIL is reported as FAIL;
the diff, the repro and the criteria are not changed afterwards.

Environment differences (package versions) are expected between versions. They are
reported separately by the diff and are not divergences.

## 6. What a PASS would and would not show

- **Would show:** Glassbox traces + diff localise a real third-party retrieval bug to
  the retrieval step, and confirm the fix and its specificity, on the issue's own
  repro.
- **Would not show:**
  - blind localisation;
  - behaviour on full RAG pipelines with an LLM;
  - that the first divergence is a cause in general.

## 7. Execution (owner's Mac)

The exact commands are in `RUN.md`:
1. Two venvs are created with the pinned versions.
2. `subject.py` writes the traces.
3. `evaluate.py` (main env) writes `results/results.json`. It never overwrites.

---
*(Addenda, if any, go below this line, with dates.)*
