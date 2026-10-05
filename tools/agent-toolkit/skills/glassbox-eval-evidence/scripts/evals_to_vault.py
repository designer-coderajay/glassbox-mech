#!/usr/bin/env python3
"""Turn evaluation results into Glassbox Annex IV vault entries.

Reads any mix of:
  * lm-evaluation-harness result files (results_*.json written by `lm_eval --output_path`)
  * Inspect AI logs (.eval or .json, written by `inspect eval`)

and writes a JSON list of VaultEntry dicts for Annex IV accuracy/robustness evidence.

    python evals_to_vault.py out/lm-eval/ logs/ --min "hellaswag:acc_norm=0.5" -o eval-evidence.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ARTICLES = ["Article 15(1)", "Article 11"]


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _entry(source, task, metric, value, stderr, model, thresholds, raw):
    key = f"{task}:{metric}"
    threshold = thresholds.get(key)
    passed = None if threshold is None else bool(value >= threshold)
    err = f" ± {stderr:.4f} (stderr)" if stderr is not None and not math.isnan(stderr) else ""
    return {
        "section": "§2",
        "article_refs": ARTICLES,
        "title": f"{task} — {metric}",
        "description": f"{source} result for model '{model}' on task '{task}': {metric} = {value:.4f}{err}.",
        "evidence_type": "general",
        "metric_name": key,
        "metric_value": float(value),
        "threshold": threshold,
        "passed": passed,
        "raw": raw,
        "timestamp_utc": _now(),
    }


def from_lm_eval(path: Path, thresholds):
    data = json.loads(path.read_text())
    if "results" not in data:
        return []
    cfg = data.get("config", {})
    model = cfg.get("model_args") or cfg.get("model") or data.get("model_name", "unknown")
    out = []
    for task, metrics in data["results"].items():
        for name, value in metrics.items():
            if name == "alias" or "_stderr" in name or not isinstance(value, (int, float)):
                continue
            metric, _, filt = name.partition(",")
            stderr = metrics.get(f"{metric}_stderr,{filt}") if filt else metrics.get(f"{metric}_stderr")
            stderr = stderr if isinstance(stderr, (int, float)) else None
            raw = {"source_file": str(path), "filter": filt or None,
                   "n_samples": data.get("n-samples", {}).get(task),
                   "lm_eval_version": data.get("lm_eval_version"), "git_hash": data.get("git_hash")}
            out.append(_entry("lm-evaluation-harness", task, metric, value, stderr, model, thresholds, raw))
    return out


def from_inspect(path: Path, thresholds):
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(path), header_only=True)
    if log.status != "success" or log.results is None:
        print(f"skip {path}: status={log.status}", file=sys.stderr)
        return []
    out = []
    for score in log.results.scores:
        stderr = score.metrics.get("stderr")
        for mname, m in score.metrics.items():
            if mname in ("stderr", "std"):
                continue
            metric = f"{score.name}/{mname}" if score.name != log.eval.task else mname
            raw = {"source_file": str(path), "scorer": score.scorer, "dataset": log.eval.dataset.name,
                   "samples": log.results.completed_samples, "inspect_version": log.eval.packages.get("inspect_ai")}
            out.append(_entry("Inspect AI", log.eval.task, metric, m.value,
                              stderr.value if stderr else None, log.eval.model, thresholds, raw))
    return out


def collect(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(x for x in p.rglob("*") if x.suffix in {".json", ".eval"})
        else:
            yield p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="result files or directories")
    ap.add_argument("--min", action="append", default=[], metavar="TASK:METRIC=VALUE",
                    help="pass threshold (metric >= value); repeatable")
    ap.add_argument("-o", "--output", default="eval-evidence.json")
    args = ap.parse_args(argv)

    thresholds = {}
    for spec in args.min:
        key, _, val = spec.rpartition("=")
        thresholds[key] = float(val)

    entries = []
    for path in collect(args.paths):
        try:
            if path.suffix == ".eval":
                entries += from_inspect(path, thresholds)
                continue
            head = json.loads(path.read_text())
            if "results" in head and "config" in head:
                entries += from_lm_eval(path, thresholds)
            elif "eval" in head and "status" in head:
                entries += from_inspect(path, thresholds)
        except (ValueError, KeyError) as exc:
            print(f"skip {path}: {exc}", file=sys.stderr)

    if not entries:
        print("no evaluation results found", file=sys.stderr)
        return 1
    Path(args.output).write_text(json.dumps({"entries": entries}, indent=2, default=str))
    for e in entries:
        flag = {True: "PASS", False: "FLAG", None: "----"}[e["passed"]]
        print(f"{flag}  {e['title']}: {e['metric_value']:.4f}")
    print(f"wrote {len(entries)} vault entries -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
