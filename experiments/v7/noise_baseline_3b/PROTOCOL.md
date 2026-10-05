# V7 Experiment 3b: group diff with redundant-field merging, and a power curve

*Protocol v1.0. Written 2026-10-05, after Experiment 3 FAILED
(`experiments/v7/noise_baseline/RESULTS.md`, commit b160e4f) and BEFORE any 3b code or
run. It is committed first and not changed afterwards. Deviations go in dated addenda
below the line.*

## 0. Relation to Experiment 3 (read first)

- **Experiment 3 stays FAIL.** Nothing in 3b changes or re-scores it. ASSURANCE V2.7
  stays `REFUTED` whatever 3b finds.
- **3b tests a modified method.** The modification comes from a post-hoc diagnosis of
  Experiment 3: three of the tool step's fields were exact copies of one partition of
  the runs, which inflated the Holm correction. Because the idea came from looking at
  Experiment 3's data, 3b uses:
  1. a **new master seed**;
  2. **only fresh replications**;
  3. the **same thresholds** as Experiment 3, not easier ones.
- **Known weakness, stated in advance.** The synthetic worlds are the same as in
  Experiment 3, so 3b checks the fix on the setting where the problem was seen. A pass
  shows the fix works *there*. It says nothing about other pipelines.

## 1. Question

1. With redundant fields merged, does the group diff reach ≥ 0.80 localisation at
   N = 30 while keeping false alarms ≤ 0.08?
2. How does detection power change with group size, N ∈ {10, 20, 30, 50}?
3. Descriptive: how much of the change is due to merging? The un-merged Experiment 3
   method is run on the same 3b data, but not gated.

## 2. Method change (fixed before implementation)

`glassbox.v7.groupdiff.group_diff` gains a keyword `merge_redundant: bool`. For
library users the default becomes `True`; `False` reproduces Experiment 3 exactly. Steps
1–4 of the Experiment 3 method are unchanged (step keys, fields, classification,
TVD + permutation). Then:

5. **Merge.** Among the fields classified *tested* in one comparison, two fields are
   **redundant** if their value vectors over the pooled runs A ∪ B induce the **same
   partition of the runs**. That is, for every pair of runs, the two runs are equal on
   one field iff they are equal on the other. Example: tool `present` and tool
   `status`.
   - Each group of redundant fields is tested **once**.
   - The **representative** is the group member earliest in run order. Ties are broken
     by the same order as the field listing (`present` first, then field name).
   - The other members are listed under `merged_with` on the representative and are
     not tested separately.
   - Merging uses only the pooled values, never the group labels, so it cannot look at
     the outcome. This keeps the permutation test valid.
6. **Holm** across representatives only, at α = 0.05.
7. **Output.** As in Experiment 3, plus `merged_with` per tested field.
   - **Run-order rule, unchanged:** `first_beyond_noise` is the earliest significant
     *representative*.
   - When a representative is significant, its merged fields are reported as
     significant too, each at its own step position.

## 3. Design (Part A only; synthetic)

The pipeline, baseline parameters and worlds W0–W3 are **identical** to Experiment 3
§3. The script reuses `experiments/v7/noise_baseline/part_a.py`'s `simulate_run`,
`BASELINE` and `CANDIDATE` by import, unchanged.

| Setting | Value |
|---|---|
| Master seed | `20261006` (Experiment 3 used `20261005`) |
| Group sizes | N = M ∈ {10, 20, 30, 50} |
| Replications | R = 200 per (world, N) |
| Permutations | `n_perm` = 999 |
| α | 0.05 (Holm) |
| Seed string per group | `"<master>:<world>:<N>:<rep>:A"` / `":B"` |

No Part B is run. Experiment 3's GPT-2 null had only one testable field per question,
so merging could not change it.

## 4. Pre-declared criteria

| ID | Criterion | Pass rule |
|---|---|---|
| A1′ Type I | W0: fraction of replications with ≥1 significant field, at **every** N | ≤ 0.08 at each N |
| A2′ Chat localisation | W1, N = 30: `first_beyond_noise` at `chat#0` | ≥ 0.80 |
| A2b′ No false upstream flag | W1: any `retrieval#0` field significant, at every N | ≤ 0.08 at each N |
| A3′ Retrieval localisation | W2, N = 30: `first_beyond_noise` at `retrieval#0` | ≥ 0.80 |
| A4′ Free text untestable | Output hash classified untestable | 100%, all worlds and N |
| A7 Merge validity | W1, N = 30: `chat#0` label, tool `present`, tool `status` and tool `gen_ai.tool.name` end up in one merged group | ≥ 0.95 of replications. *Rationale:* by construction they are the same partition whenever both labels occur, so failures can only come from degenerate replications or a bug. |
| D1 Power curve (**descriptive**) | Localisation rate (W1 → `chat#0`, W2 → `retrieval#0`) and any-detection rate per N | Reported |
| D2 Merge effect (**descriptive**) | Same 3b data, `merge_redundant=False`: A2′ and A1′ rates | Reported next to the merged rates |
| D3 One vs two causes (**descriptive**) | `chat#0` flagged rate in W2 vs W3 per N | Reported. Expected: still cannot separate the two |

**Overall result.**
- **PASS** iff A1′, A2′, A2b′, A3′, A4′ and A7 all pass.
- D1–D3 are descriptive.
- A FAIL is reported as a FAIL. If 3b fails, the group diff's localisation claim stays
  unsupported, and the next step is a larger N or a different test. It is not another
  re-tuning on these worlds.

## 5. What a PASS would and would not show

**Would show:** on this synthetic pipeline, merging redundant fields restores ≥ 0.80
localisation at N = 30 without raising false alarms, plus a measured power curve to
guide group sizes.

**Would not show:**
- Validity on other pipelines or real systems. The worlds are the ones where the
  problem was found.
- Any change to Experiment 3's FAIL.
- Separation of one cause from two (D3, V2.2).
- Anything about unknown causes (V2.4).

## 6. Execution

1. Commit this protocol.
2. Implement `merge_redundant` test-first, with unit tests on hand-built traces only.
   Implement `experiments/v7/noise_baseline_3b/run.py`. Experiment 3's tests must still
   pass with `merge_redundant=False`.
3. Commit the code.
4. Run on the owner's Mac:

   `python3 experiments/v7/noise_baseline_3b/run.py --out experiments/v7/noise_baseline_3b/results`

   The traces go to a temporary directory and are not committed (regenerable from the
   seeds).
5. Commit `results.json` unchanged.

---
*(Addenda, if any, go below this line, with dates.)*
