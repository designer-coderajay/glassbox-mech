# V8 phase B: GPU sanity check. RESULTS

**Overall: PASS** (G1 and G2), attempt 2. Pre-registered in `PROTOCOL.md` v1.0.

*Provenance, in commit order:*
- Protocol v1.0: `c075cfb`, rebased to `ac9b3fa` before push.
- Runner, tests and Colab runbook: `c5bd4c2`.
- Addendum 1 (attempt 1 infrastructure failure, setup fix): `6467d56`.
- Addendum 2 (attempt 2 stopped at the setup check, isolated venv): `1545453`.
- Attempt 2 ran on 2026-10-30 at code `15454531d1dabf4be9daa8b75d260c802e5770a8`. That
  is the Addendum 2 commit, as recorded in `results.json` → `software.git_commit`.

*Files, committed unchanged:*

| File | sha256 |
|---|---|
| `attempt1/run.log` | `abb136eef9e14985134918a8af5ccd4c3602996f9eb3ac98649780254348e682` |
| `attempt2/results.json` | `d605558872a02f7ce41e654d70f6617d63cc551f3341b81ace30310efded7a88` |
| `attempt2/run.log` | `4e251b6e246be23864380e45718f484840533e0bd6299ab34cf1b9645148c2b1` |

## 1. Environment (attempt 2, from `results.json`)

All three configurations ran in one Colab session:
- Tesla T4, CUDA 12.8 (torch build `2.10.0+cu128`).
- Python 3.11.18 in an isolated `uv` venv (uv 0.13.6), Linux 6.6 x86-64.
- glassbox-mech-interp 4.6.0 (editable, from the checkout), torch 2.10.0,
  transformer-lens 2.17.0, transformers 4.57.6, numpy 1.26.4.
- Settings as pre-registered: 30 runs per cell, 999 permutations, alpha 0.05, seed
  `gpu-check`.

## 2. Gated results

| ID | Criterion | Observed | Result |
|---|---|---|---|
| G1 | Decisions A (cpu/float32) vs B (cuda/float32) | `factor_summary` identical; all 8 cases have identical `attributed` and `reverting_matters` | **PASS** |
| G2 | Environment recorded | B: Tesla T4, CUDA 12.8. A: device cpu | **PASS** |

The decisions shared by A and B:
- `index` was attributed, and reverting it alone mattered, on q1–q4.
- Reverting `template` alone mattered on q7.
- `log_level` (the inert change) was never attributed.

Beyond the gate (descriptive): the raw outcome counts of A and B are identical in
every cell of every case. Moving from CPU to GPU in float32 did not change a single
sampled draw.

## 3. Descriptive results (not gated)

**C (cuda/bfloat16): 7 of 8 cases match A.** The one difference is q3:
- In C, `template` is also attributed.
- The difference is one sampled draw: in cell `only_template`, A had 23 correct /
  7 wrong and C had 22 / 8. Every other cell of q3 is identical.
- On load, TransformerLens warned that with reduced precision
  `from_pretrained_no_processing` is advised. The runner used `from_pretrained` as
  written.

**Runtime of the investigation step** (model loading excluded; one run each, so no
comparison claim):

| Config | Seconds |
|---|---|
| A cpu | 14.1 |
| B T4 float32 | 3.9 |
| C T4 bfloat16 | 3.9 |

## 4. What this shows and what it does not

- **Shows:** on this incident and model (GPT-2 small, built-in RAG demo, 8 cases),
  the investigation reaches the same decisions on a Colab T4 GPU in float32 as on CPU
  in the same session. The sampled counts were identical too.
- **Does not show** (protocol §5):
  - anything about other or larger models, or other incidents;
  - anything about other GPUs;
  - anything about bfloat16 beyond the count above;
  - independence: the author ran it.

**Observation, not a test.** C's q3 difference sits exactly where earlier work said
the method is weak. Experiment 5 (ASSURANCE V3.3) recorded the q3 template effect as
small and not detected at 30 runs per cell. One draw moved this decision. Decisions
near the detection limit can flip with tiny numeric changes on any device.
Consequences for later GPU work:
- Gated runs use float32.
- A reduced-precision run is compared only with runs in the same precision.

## 5. Deviations from the runbook (disclosed; no effect on the gated result expected)

1. **Commit placeholder.**
   - `git checkout NEWHASH` was pasted literally and failed with "pathspec 'NEWHASH'
     did not match".
   - The checkout therefore stayed on `main`, which at that moment was `1545453`, the
     intended Addendum 2 commit.
   - `results.json` records the full hash `15454531…770a8`, so the code that ran is the
     protocol's.
2. **Runtime not deleted.**
   - The clone path was `/content/glassbox-mech/glassbox-mech/glassbox-mech`, so
     earlier clones were still present. The runtime was reused, not deleted as the
     runbook asked.
   - Everything ran in a newly created venv that cannot see Colab's or the earlier
     attempts' packages.
   - The isolation was not checked independently.

## 6. Attempts

| Attempt | What happened | Scored |
|---|---|---|
| 1 | Crashed while loading GPT-2 for A (Colab's torchvision built for a newer torch). Log kept. | No (infrastructure failure, Addendum 1) |
| 2 (setup) | Stopped at the new model-load check (Colab's TensorFlow imported by `transformers`). `run.py` not started. | No (Addendum 2) |
| 2 (run) | Isolated Python 3.11 venv. Completed. | **Yes: PASS** |
