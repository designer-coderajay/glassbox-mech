# E1b on Colab: step by step

This uses the same environment recipe as E1: an isolated Python 3.11 venv with the CI
pins. Each block below is one Colab cell. If a cell's output is not what is described,
stop and keep the output.

**Start with a fresh runtime.**
1. Runtime → Disconnect and delete runtime.
2. Runtime → Change runtime type → T4 GPU.

E1 showed that a reused runtime makes the steps harder to follow.

**Expected length:** probably 4–5 hours of GPU time for the 1.5B model. That is an
estimate. Keep the tab open, because free Colab disconnects idle sessions. If it does
disconnect, step 6 resumes where it stopped.

## 1. GPU

```
!nvidia-smi
```

## 2. Code at the run commit

Replace `RUNHASH` with the hash of the E1b code commit, the one that adds Addendum 1.
Paste the real hash, not the word.

```
!git clone https://github.com/designer-coderajay/glassbox-mech.git /content/glassbox-mech
%cd /content/glassbox-mech
!git checkout RUNHASH
!git log -1 --oneline
!sha256sum experiments/v8/e1b_directed_rag/questions_test.json
```

- `git log` must show the E1b code commit.
- The last line must be
  `a55b085c5ed9e7a74e081583107549cbd0f5ccda0aaa3c54855097e402a670fe`.

## 3. Environment

```
!pip install -q uv
!uv venv /content/gbenv --python 3.11 --clear
!uv pip install --python /content/gbenv/bin/python "numpy==1.26.4" "scipy==1.17.1" "einops==0.8.2" "torch==2.10.0" "transformer-lens==2.17.0" "transformers==4.57.6" "pyarrow==23.0.0" "datasets==4.5.0" "pandas==3.0.0" sentence-transformers bitsandbytes accelerate pytest
!uv pip install --python /content/gbenv/bin/python -e . --no-deps
!/content/gbenv/bin/python -m pytest -q tests/test_v8_e1b.py tests/test_v7_checkpoint.py tests/test_v7_intervene_direction.py tests/test_v7_intervene.py tests/test_v7_investigate.py
```

The last line should show **81 passed**.

## 4. Model check on a dev question (not part of the run)

```
!/content/gbenv/bin/python -c "import sys; sys.path.insert(0,'experiments/v8/e1b_directed_rag'); import system as s; e=s.STEmbedder('sentence-transformers/all-MiniLM-L6-v2','1110a243fdf4706b3f48f1d95db1a4f5529b4d41','cuda'); g=s.HFGenerator('Qwen/Qwen2.5-1.5B-Instruct','989aa7980e4cf806f80c7fef2b1adb7bc71aa306','cuda','float32'); x=s.System(e,g,'colab-check'); q=s.load_questions('dev')[0]; print(x.answer(s.BASE,q,0))"
```

It should print `('correct', '...', [...])` or `('wrong', ...)`, with a short answer
about the exit fee. This uses only a **dev** question.

## 5. Google Drive (for checkpoints)

```
from google.colab import drive
drive.mount("/content/drive")
```

Approve the pop-up yourself.

## 6. The run (repeat this exact cell after a disconnect)

```
!mkdir -p /content/drive/MyDrive/glassbox_runs/e1b && cd /content/glassbox-mech && /content/gbenv/bin/python experiments/v8/e1b_directed_rag/run.py --backend hf --out /content/drive/MyDrive/glassbox_runs/e1b/attempt1 2>&1 | tee -a /content/drive/MyDrive/glassbox_runs/e1b/attempt1_log.txt
```

- Progress shows one line per question: `question n/24 …`.
- It ends with `overall: PASS`, `FAIL` or `INCONCLUSIVE`. All three are results. Do
  not re-run because of the answer.
- The installed package list is written to `attempt1/pip_freeze.txt` automatically.
- **After a disconnect:**
  1. Run steps 1–5 again in the new session (step 2 with the same hash).
  2. Run this same cell. Finished questions are loaded from Drive, not re-run.
  3. Tell me it happened, so the resume is disclosed.

## 7. Download

```
!cd /content/drive/MyDrive/glassbox_runs/e1b && zip -r /content/e1b_attempt1.zip attempt1 attempt1_log.txt
from google.colab import files
files.download("/content/e1b_attempt1.zip")
```

Send me the zip. It is committed unchanged, and then RESULTS.md is written.
