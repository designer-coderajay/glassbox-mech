# V6 is frozen (V7 Step 1, 2026-10-05)

V6 is a completed, preregistered study. Its results are reported in
`experiments/v6/RESULTS_confirmatory_v1.md`, and the protocol is locked at tag
`v6-amendment3-lock` (commit 116035c). From V7 onward, the V6 artefacts are
**read-only**.

## What is frozen

Every git-tracked file under these two roots, as of commit `e81dccc`:

- `experiments/v6/`: preregistration, amendments, audits, gate results, the
  runner, the locked prompts, run records and the confirmatory cache.
- `glassbox/v6/`: the library code that produced them.

That is 119 files, recorded with sha256 in `experiments/V6_FROZEN_MANIFEST.json`.

## The rule

- V7 code may **import or read** V6 artefacts.
- V7 must **not modify, delete or add** files under the frozen roots.
- New work goes in new modules, e.g. `glassbox/v7/`.
- If a genuine error is ever found in a V6 artefact, it is reported in a new,
  dated erratum document outside the frozen roots. The original file stays as it
  is.

## How to check

```
python3 scripts/check_v6_frozen.py
pytest tests/test_v6_frozen.py
```

- Exit code 0 means intact.
- Exit code 1 lists each file as `added`, `modified` or `removed`.
- `--write` creates the manifest once and refuses to overwrite an existing
  manifest.

## Note on prompt hashes

The prompt-set hashes quoted in the RESULTS report (`aa3e32c8…`, `0849499b…`)
hash the canonical JSON of the prompt *items*. This is how the runner checks them;
see `items_hash` in `prompts/build_prompt_sets.py`. The manifest instead hashes
the *file bytes*, so its values for the two prompt files differ. Both are
correct, and they measure different things.

The record (`18d11300…`) and runner (`477953a4…`) hashes in the manifest equal
the values reported in RESULTS.
