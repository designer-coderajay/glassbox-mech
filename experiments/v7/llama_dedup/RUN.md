# Running V7 Experiment 2

Run these only **after** `PROTOCOL.md` is committed. Run them from the repo root. The
virtual environments live in `~/.glassbox-venvs`, outside the repo.

## 1. Create the two pinned environments

```
python3 -m venv ~/.glassbox-venvs/li-0.14.17
~/.glassbox-venvs/li-0.14.17/bin/pip install -q "llama-index-core==0.14.17"
python3 -m venv ~/.glassbox-venvs/li-0.14.19
~/.glassbox-venvs/li-0.14.19/bin/pip install -q "llama-index-core==0.14.19"
```

## 2. Record the 8 traces

```
for v in 0.14.17 0.14.19; do for p in sync async; do for r in r1 r2; do ~/.glassbox-venvs/li-$v/bin/python experiments/v7/llama_dedup/subject.py --path $p --run $r --out experiments/v7/llama_dedup/results; done; done; done
```

## 3. Evaluate (main environment, the one that runs the Glassbox tests)

```
python3 experiments/v7/llama_dedup/evaluate.py --results experiments/v7/llama_dedup/results
```

Commit `results/` unchanged, whether the overall result is PASS or FAIL.
