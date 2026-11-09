# E2 on a rented GPU: step by step

You (the owner) rent and pay for the machine. Nothing here needs a Hugging Face,
PyPI or cloud token: both models are public and not gated. If a step's output is
not what is described, stop and keep the output.

There are two separate runs, in this order:
1. **Pilot** (dev split only).
2. **Test run**, after Addendum 2 is committed.

Between them, you send me the pilot results and I write Addendum 2.

## 0. The machine

- **GPU:** one card with 80 GB of memory (an A100 80GB or H100 80GB class), so that
  both the 7B and the 14B model fit in bf16 (PROTOCOL §10). A smaller card is only
  acceptable if the gate picks the 7B model and it fits. In that case, tell me the
  card, and it is recorded.
- **System:** Ubuntu 22.04 or 24.04 with a recent NVIDIA driver, and at least 150 GB
  of disk (models about 45 GB, plus the venv and traces).
- **Price, provider and runtime** are your choice. I have not estimated the runtime;
  the pilot measures it (step 6).
- **Copy results off the machine** (step 9) before you stop or delete it. A deleted
  instance loses its disk.

## 1. Connect and open a session that survives disconnects

```
ssh USER@HOST
nvidia-smi
tmux new -s e2
```

- `nvidia-smi` must list the GPU.
- If the connection drops, reconnect with `ssh USER@HOST` and then `tmux attach -t e2`.

## 2. Code at the run commit

Replace `RUNHASH` with the hash of the E2 code commit, the one that adds Addendum 1.
Paste the real hash, not the word.

```
git clone https://github.com/designer-coderajay/glassbox-mech.git ~/glassbox-mech
cd ~/glassbox-mech
git checkout RUNHASH
git log -1 --oneline
sha256sum experiments/v8/e2_golden_agent/questions_test.json
```

The last line must start with
`da640ddc3d080520d14e4f177f52040d4a592ba4b09fcf15603fe03168b6f897`.

## 3. Environment

```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv ~/gbenv --python 3.11 --clear
uv pip install --python ~/gbenv/bin/python "numpy==1.26.4" "scipy==1.17.1" "einops==0.8.2" "torch==2.10.0" "transformer-lens==2.17.0" "transformers==4.57.6" "pyarrow==23.0.0" "datasets==4.5.0" "pandas==3.0.0" accelerate huggingface_hub pytest
uv pip install --python ~/gbenv/bin/python -e . --no-deps
~/gbenv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
~/gbenv/bin/python -m pytest -q tests/test_v8_e2.py tests/test_v7_taxonomy.py tests/test_v7_checkpoint.py tests/test_v7_intervene_direction.py tests/test_v7_intervene.py tests/test_v7_investigate.py
```

- The `torch` line must print `2.10.0`, `True` and the GPU name.
- The tests should end with **108 passed**.

## 4. Model check on one dev question (not part of the run)

```
~/gbenv/bin/python -c "import sys; sys.path[:0]=['.','experiments/v8/e2_golden_agent']; import system as s; m=s.HFAgentModel('Qwen/Qwen2.5-7B-Instruct','a09a35458c702b33eeacc393d103063234e8bc28','cuda','bfloat16'); x=s.System(m,'host-check',2); q=s.load_questions('dev')[0]; e=x.episode(s.BASE,q,0); print(e['label'], e['calls'], e['final'][:200])"
```

- It should print `correct` or `wrong`, then a list of tool names such as
  `['lookup_order']`, and a short answer about an order status.
- This uses only a **dev** question.
- The first time, it downloads the 7B model (about 15 GB).

## 5. Pilot (dev split only)

```
~/gbenv/bin/python experiments/v8/e2_golden_agent/run.py --stage pilot --out ~/runs/e2/pilot1 2>&1 | tee -a ~/runs/e2_pilot1.log
```

- It runs the competence gate (7B, then 14B if the 7B fails), then the pilot.
- It ends with `overall: PILOT (not scored)`. If no model passes the gate, it ends
  with `overall: INCONCLUSIVE`; send me that too.
- If it stops for any reason, run the **same command** again. It resumes from
  `~/runs/e2/pilot1/checkpoint`.

## 6. Send me the pilot

```
cd ~/runs/e2 && zip -r ~/e2_pilot1.zip pilot1 ../e2_pilot1.log
```

Copy `~/e2_pilot1.zip` to your laptop (step 9) and upload it here. From it I write
Addendum 2. It contains:
- the gate result;
- each factor's effect on dev;
- **the measured episodes per hour, and the projected hours for the test run.**

Use that projection to decide whether to keep renting for the test run. The sample
sizes do not change.

If a designed factor shows no effect, the protocol allows **one** wording revision
and **one** re-pilot. That would be a new commit, and you would repeat steps 2 and 5
with a new `--out` (`pilot2`).

## 7. Test run (only after Addendum 2 is committed)

```
cd ~/glassbox-mech && git fetch && git checkout ADDENDUM2HASH
~/gbenv/bin/python experiments/v8/e2_golden_agent/run.py --stage test --out ~/runs/e2/attempt1 2>&1 | tee -a ~/runs/e2_attempt1.log
```

- It ends with `overall: PASS`, `FAIL` or `INCONCLUSIVE`. All three are reported as
  they are.
- If it stops, run the same command again. It resumes per question.
- Do not change `--out` between resumes.

## 8. Do not

- Edit any file in the checkout.
- Read or open `questions_test.json` yourself before the run ends. Checking its
  hash is fine.
- Change `--gate-min`. It only works with `--backend toy`.

## 9. Copy results to your laptop, then stop the machine

From your laptop:

```
scp USER@HOST:~/e2_pilot1.zip ~/Downloads/
```

After the test run, on the machine:

```
cd ~/runs/e2 && zip -r ~/e2_attempt1.zip attempt1 ../e2_attempt1.log
```

Then, on your laptop:

```
scp USER@HOST:~/e2_attempt1.zip ~/Downloads/
```

Check that the zip opens on your laptop **before** deleting the instance.
