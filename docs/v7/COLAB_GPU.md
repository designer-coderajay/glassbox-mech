# Running a V7 experiment on a Colab GPU

*Runbook for the V8 phase B GPU sanity check (`experiments/v7/gpu_check/`). Later GPU
experiments reuse steps 1–4.*

Each step is one Colab cell. Paste it, run it, and check the expected output before
going on. If a step does not show what is expected, **stop** and keep the output. A
setup failure is an infrastructure failure under the protocol (re-run in a new
session); it is never patched mid-run.

Nothing here needs a token or a login beyond your Google account. The repository is
public and GPT-2 small downloads without an HF token.

## 1. GPU runtime

In the Colab menu: **Runtime → Change runtime type → a GPU (e.g. T4) → Save**. Then:

```
!nvidia-smi
```

Expected: a table naming a GPU. If it says no device was found, the runtime has no GPU;
change it and start again.

## 2. Get the code at the exact commit

Replace `RUN_COMMIT` with the commit the protocol names for this attempt (for
attempt 2, the commit that adds Addendum 2).

```
!git clone https://github.com/designer-coderajay/glassbox-mech.git
%cd glassbox-mech
!git checkout RUN_COMMIT
!git log -1 --oneline
```

Expected: the last line shows that commit.

## 3. Install the pinned stack in its own environment

Colab's preinstalled packages (TensorFlow, torchvision, JAX and others) are built for
newer torch and NumPy. When they are present, `transformers` imports them and breaks
(gpu_check protocol, Addenda 1 and 2). So everything runs in a separate virtual
environment with **Python 3.11, the CI's version**, containing only the CI pins
(`.github/workflows/ci.yml`). `uv` downloads that Python; no system change is needed.

```
!pip install -q uv
!uv --version
!uv venv /content/gbenv --python 3.11
!uv pip install --python /content/gbenv/bin/python "numpy==1.26.4" "scipy==1.17.1" "einops==0.8.2" "torch==2.10.0" "transformer-lens==2.17.0" "transformers==4.57.6" "pyarrow==23.0.0" "datasets==4.5.0" "pandas==3.0.0" pytest
!uv pip install --python /content/gbenv/bin/python -e . --no-deps
!/content/gbenv/bin/python --version
```

Expected: no error, and the last line says `Python 3.11.x`. (uv only builds the
environment; its version is printed for the record. The package pins are what matter.) From here on, every command
uses `/content/gbenv/bin/python`, never `python3`.

If an install fails, stop and send the output. The setup is then changed in a dated
addendum before anything runs.

## 4. Check the GPU from torch, run the tests, load the model

```
!/content/gbenv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
!/content/gbenv/bin/python -m pytest -q tests/test_v7_gpu_check.py tests/test_v7_compute.py
!/content/gbenv/bin/python -c "from transformer_lens import HookedTransformer; m = HookedTransformer.from_pretrained('gpt2', device='cuda'); print('gpt2 loaded on', m.cfg.device)"
```

Expected:
- torch `2.10.0`, a CUDA version, `True` and a GPU name;
- all tests passing;
- `gpt2 loaded on cuda`.

The tests use fakes, so the last line is the real check that the model stack works.

## 5. Run the check

```
!/content/gbenv/bin/python experiments/v7/gpu_check/run.py --out experiments/v7/gpu_check/attempt2
```

Expected: it ends with `overall: PASS`, `overall: FAIL` or
`overall: INFRASTRUCTURE_FAILURE`, plus the path of `results.json`. **Any of the three
is a result to keep.** Do not re-run because of a FAIL.

If the session disconnects before this line appears, nothing was produced. Start again
from step 1 in a new session and mention the disconnect when sending the results.

## 6. Download the results

```
!cd experiments/v7/gpu_check && zip -r gpu_check_results.zip attempt2
from google.colab import files
files.download("experiments/v7/gpu_check/gpu_check_results.zip")
```

On your Mac, unzip it into `experiments/v7/gpu_check/` so that
`experiments/v7/gpu_check/attempt2/results.json` and `run.log` exist. (A later
attempt uses the next free name, `attempt3`, and so on, everywhere above.) Do not edit them.
They are committed unchanged, and `RESULTS.md` is written afterwards.

## 7. Long runs: survive a disconnect

Colab can disconnect, and a free session ends when the runtime is recycled. For a run
longer than a few minutes, two things protect the work:

1. **Write to Google Drive, not to `/content`.** Mount Drive once per session. Colab
   asks you to approve access in a pop-up, and you do that yourself. Then pass an
   output directory on Drive:

   ```
   from google.colab import drive
   drive.mount("/content/drive")
   ```
   ```
   !/content/gbenv/bin/python experiments/v7/<experiment>/run.py --out /content/drive/MyDrive/glassbox_runs/<experiment>/attemptN
   ```

2. **Experiment runners save each finished unit as it completes.** They use
   `glassbox.v7.checkpoint.Checkpoint`:
   - each finished case, model or configuration is one JSON file, written atomically
     and never overwritten;
   - a fingerprint of the settings is stored alongside, and reopening the folder with
     different settings is refused.

   After a disconnect, run steps 1–4 again in a new session and repeat the **same**
   command with the **same** `--out`. Finished units are loaded, not re-run.

   - Resuming is only allowed if the protocol of that experiment says so.
   - The resume is disclosed in its RESULTS.md, with the units completed before and
     after.

Experiments whose code is already committed (all of `experiments/v7/` up to and
including `gpu_check`) are not changed to add checkpoints. Their code is hashed into
evidence packages. New runners (phase E onwards) use `Checkpoint` from the start, and
load models through `glassbox.v7.demo_rag.GPT2Scorer(model_name, compute=...)`, which
accepts any TransformerLens model name together with `--device` / `--dtype`.
