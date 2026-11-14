# E2b on Colab: step by step

Each block is one Colab cell. If a cell's output is not what is described, stop and
keep the output. Nothing here needs a token: both models are public.

There are two runs, in this order: the **pilot** (dev split only), then, after
Addendum 2 is committed, the **test run**. Replace `RUNHASH` with the commit named in
the chat for that run.

**Before cell 1:** Runtime → Disconnect and delete runtime, then Runtime → Change
runtime type → **A100**. Do not use an L4 or T4 for the test run.

## 1. GPU

```
!nvidia-smi
```

It must list an A100.

## 2. Google Drive

```
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/glassbox_runs/e2b
```

## 3. Code at the run commit

```
!git clone https://github.com/designer-coderajay/glassbox-mech.git /content/glassbox-mech
%cd /content/glassbox-mech
!git checkout RUNHASH
!git log -1 --oneline
!sha256sum experiments/v8/e2b_agent_retry/questions_test.json
```

The last line must start with
`a4c3eee088716d72b13f6e71bdc1b44a08011364b59c83cdf322221d23e9988e`.

## 4. Environment

```
!pip install -q uv
!uv venv /content/gbenv --python 3.11 --clear
!uv pip install --python /content/gbenv/bin/python "numpy==1.26.4" "scipy==1.17.1" "einops==0.8.2" "torch==2.10.0" "transformer-lens==2.17.0" "transformers==4.57.6" "pyarrow==23.0.0" "datasets==4.5.0" "pandas==3.0.0" accelerate huggingface_hub pytest
!uv pip install --python /content/gbenv/bin/python -e . --no-deps
!/content/gbenv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
!/content/gbenv/bin/python -m pytest -q tests/test_v8_e2b_agent.py tests/test_v7_taxonomy.py tests/test_v7_checkpoint.py tests/test_v7_intervene_direction.py tests/test_v7_intervene.py tests/test_v7_investigate.py
```

- The torch line must show `2.10.0+cu128 True NVIDIA A100…`.
- The tests should end with **115 passed**.

## 5. Model check on one dev question (not part of the run)

```
!/content/gbenv/bin/python -c "import sys; sys.path[:0]=['.','experiments/v8/e2b_agent_retry']; import system as s; m=s.HFAgentModel('Qwen/Qwen2.5-7B-Instruct','a09a35458c702b33eeacc393d103063234e8bc28','cuda','bfloat16'); x=s.System(m,'host-check',2); q=s.load_questions('dev')[0]; e=x.episode(s.BASE,q,0); print(e['label'], e['calls'], e['final'][:200])"
```

It should print `correct` or `wrong`, a tool list such as `['lookup_order']`, and a
short answer about an order status. Keep this output.

## 6a. Pilot (dev split only)

```
!/content/gbenv/bin/python experiments/v8/e2b_agent_retry/run.py --stage pilot --out /content/drive/MyDrive/glassbox_runs/e2b/pilot1 2>&1 | tee -a /content/drive/MyDrive/glassbox_runs/e2b/e2b_pilot1.log
```

- It ends with `overall: PILOT (not scored)`, or `overall: INCONCLUSIVE` if no model
  passes the gate.
- If Colab disconnects, rerun cells 2, 3, 4 and this cell unchanged. The pilot saves
  its result only at the end, so it starts the pilot again.

```
!cd /content/drive/MyDrive/glassbox_runs/e2b && zip -r e2b_pilot1.zip pilot1 e2b_pilot1.log
from google.colab import files
files.download('/content/drive/MyDrive/glassbox_runs/e2b/e2b_pilot1.zip')
```

Upload `e2b_pilot1.zip` in the chat. A second pilot, if the protocol's one revision
is used, writes to `pilot2`.

## 6b. Test run (only after Addendum 2 is committed)

```
!/content/gbenv/bin/python experiments/v8/e2b_agent_retry/run.py --stage test --out /content/drive/MyDrive/glassbox_runs/e2b/attempt1 2>&1 | tee -a /content/drive/MyDrive/glassbox_runs/e2b/e2b_attempt1.log
```

- It logs one line per question and ends with `overall: PASS`, `FAIL` or
  `INCONCLUSIVE`. All three are reported as they are.
- Keep the tab open for the whole run.
- If it disconnects, rerun cells 2, 3, 4 and this cell unchanged. It resumes from the
  last finished step. Never change `--out`.
- If you see "CUDA out of memory" or another CUDA error, stop and paste the output.
  Do not change any setting.

```
!cd /content/drive/MyDrive/glassbox_runs/e2b && zip -r e2b_attempt1.zip attempt1 e2b_attempt1.log
from google.colab import files
files.download('/content/drive/MyDrive/glassbox_runs/e2b/e2b_attempt1.zip')
```

## Do not

- Edit any file in the checkout.
- Open `questions_test.json` before the test run ends. Checking its hash is fine.
- Change `--gate-min`. It only works with `--backend toy`.
