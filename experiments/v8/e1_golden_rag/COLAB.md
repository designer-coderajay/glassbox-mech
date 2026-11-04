# E1 on Colab: step by step

This follows `docs/v7/COLAB_GPU.md`: an isolated Python 3.11 venv with the CI pins.
Each block below is one Colab cell. If a cell's output is not what is described, stop
and keep the output.

**Start with a fresh runtime.** Runtime → Disconnect and delete runtime. Then Runtime →
Change runtime type → T4 GPU.

**Expected length:** probably 1.5–3 hours of GPU time for the 1.5B model. That is an
estimate. Keep the tab open: free Colab disconnects idle sessions. If it disconnects,
step 6 resumes where it stopped.

## 1. GPU

```
!nvidia-smi
```

## 2. Code at the run commit

Replace `RUNHASH` with the commit that adds the E1 code and Addendum 1. Paste the real
hash, not the placeholder.

```
!git clone https://github.com/designer-coderajay/glassbox-mech.git /content/glassbox-mech
%cd /content/glassbox-mech
!git checkout RUNHASH
!git log -1 --oneline
!sha256sum experiments/v8/e1_golden_rag/questions_test.json
```

The last line must be
`a6dbd869e45567a02f51dab4dc9b402bf3fa3e253275fdb60c84267657acbb09`.

## 3. Environment

```
!pip install -q uv
!uv venv /content/gbenv --python 3.11
!uv pip install --python /content/gbenv/bin/python "numpy==1.26.4" "scipy==1.17.1" "einops==0.8.2" "torch==2.10.0" "transformer-lens==2.17.0" "transformers==4.57.6" "pyarrow==23.0.0" "datasets==4.5.0" "pandas==3.0.0" sentence-transformers bitsandbytes accelerate pytest
!uv pip install --python /content/gbenv/bin/python -e . --no-deps
!/content/gbenv/bin/python -m pytest -q tests/test_v8_e1.py tests/test_v7_checkpoint.py
```

All tests should pass. If uv cannot resolve `sentence-transformers` with
`transformers==4.57.6`, stop and send the output.

## 4. Model check on a dev question (not part of the run)

```
!/content/gbenv/bin/python -c "import sys; sys.path.insert(0,'experiments/v8/e1_golden_rag'); import system as s; e=s.STEmbedder('sentence-transformers/all-MiniLM-L6-v2','1110a243fdf4706b3f48f1d95db1a4f5529b4d41','cuda'); g=s.HFGenerator('Qwen/Qwen2.5-1.5B-Instruct','989aa7980e4cf806f80c7fef2b1adb7bc71aa306','cuda','float32'); x=s.System(e,g,'colab-check'); q=s.load_questions('dev')[0]; print(x.answer(s.BASE,q,0))"
```

It should print `('correct', '...', [...])` or `('wrong', ...)` with a short answer
about returns. It only uses a **dev** question, which the protocol allows for checking
the pipeline.

## 5. Google Drive (for checkpoints)

```
from google.colab import drive
drive.mount("/content/drive")
```

Approve the pop-up yourself.

## 6. The run (repeat this exact cell after a disconnect)

```
!cd /content/glassbox-mech && /content/gbenv/bin/python experiments/v8/e1_golden_rag/run.py --backend hf --out /content/drive/MyDrive/glassbox_runs/e1/attempt1 2>&1 | tee -a /content/drive/MyDrive/glassbox_runs/e1/attempt1_log.txt
```

It ends with `overall: PASS`, `FAIL` or `INCONCLUSIVE`. All three are results. Do not
re-run because of the answer.

After a disconnect, run steps 1–5 again in the new session, then this same cell. Finished
questions are loaded from Drive, not re-run. Tell me it happened, so the resume is
disclosed.

## 7. Download

```
!cd /content/drive/MyDrive/glassbox_runs/e1 && zip -r /content/e1_attempt1.zip attempt1 attempt1_log.txt
!/content/gbenv/bin/pip freeze > /content/pip_freeze.txt
from google.colab import files
files.download("/content/e1_attempt1.zip")
files.download("/content/pip_freeze.txt")
```

Send me both files. They are committed unchanged, and then RESULTS.md is written.
