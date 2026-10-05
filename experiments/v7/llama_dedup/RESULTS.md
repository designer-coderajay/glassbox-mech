# V7 Experiment 2 — llama_index #21033: RESULTS

*Protocol `PROTOCOL.md` v1.0, registered in `d866eea` before the run. The disclosed
sandbox dry run is Addendum 1. The official run was on the owner's Mac (macOS arm64,
Python 3.11.9) on 2026-10-05. Traces and `results.json` in `results/` are committed
unchanged.*

## Outcome: **PASS** (all pre-declared gated criteria met)

| Criterion | Rule | Result |
|---|---|---|
| C1 Localisation (bug) | 0.14.17 async vs sync → first divergence at retrieval | **PASS**: a `content` divergence on `gen_ai.retrieval.documents` at `retrieval repro-21033` |
| C2 Fix confirmed | 0.14.19 async vs sync → no divergence | **PASS** |
| C3a Intervention, affected path | sync 0.14.17 vs 0.14.19 → first divergence at retrieval | **PASS** |
| C3b Specificity, unaffected path | async 0.14.17 vs 0.14.19 → no divergence | **PASS** |
| C4 Determinism | r2 vs r1 for all 4 (version, path) | **4/4** |
| C5 Documents returned (descriptive) | | 0.14.17 sync **1**, async 2; 0.14.19 sync 2, async 2 |

The diff reported the package version as an **environment** difference
(`glassbox.env.packages`), separately from the behavioural divergences, as designed.

## Verification

1. All 8 traces pass the integrity check.
2. Recomputing C1–C5 from the traces after the run reproduces the stored
   `results.json` exactly.
3. **Cross-environment agreement.** The retrieved-document hashes of the official run
   (macOS arm64, Python 3.11.9) are identical, 4/4, to the pre-official sandbox dry run
   (Linux, Python 3.10). This is a reproduction across environments **by the same
   author**, not an independent reproduction.
4. No document text appears in any trace or in `results.json`; content is hashed only.

## What this establishes, and what it does not

**Established.** On a real, pre-existing bug in a widely used third-party framework,
which we did not create, Glassbox traces + diff:
- localised the behavioural difference to the retrieval step;
- confirmed that a later version removes it;
- showed that the version change did not alter the unaffected (async) path;

all deterministically, on the issue's own repro.

**Not established:**
- **Blind localisation.** We had read the issue and knew the cause; the protocol states
  this. ASSURANCE V2.4 remains `UNRESOLVED`.
- **Behaviour on a full application** (an LLM in the loop, real data, many steps). The
  repro is a minimal retriever.
- **That the first divergence is a cause in general.** Here the version change is the
  only difference between conditions, by design.
- **That the fix in 0.14.19 is "correct" in general.** It changes deduplication on both
  paths to `node_id` (protocol §2). This experiment only shows it removes the sync/async
  difference for this input.

**A real-world detail the experiment surfaced.** The merged fix (PR #21034) was not in
0.14.18. The shipped fix in 0.14.19 used a different deduplication key. A version-diff
investigation has to deal with exactly this kind of mismatch between "merged" and
"released".
