#!/usr/bin/env python3
"""Group-fairness metrics for a model's predictions, written as Glassbox Annex IV
vault entries (a JSON list of VaultEntry dicts).

Input: a CSV with one row per decision: true label, predicted label, and one or
more sensitive-attribute columns. Labels must be binary (0/1, or pass --positive).

    python fairness_evidence.py preds.csv --y-true label --y-pred pred \
        --sensitive gender --sensitive age_band -o fairness-evidence.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score

ARTICLES = ["Article 10(2)(f)", "Article 10(2)(g)", "Article 15(1)"]


def _binarize(series: pd.Series, positive) -> pd.Series:
    if positive is None:
        values = set(series.dropna().unique())
        if not values <= {0, 1, True, False}:
            raise SystemExit(f"labels are not binary 0/1 ({sorted(map(str, values))[:5]}); pass --positive")
        return series.astype(int)
    return (series.astype(str) == str(positive)).astype(int)


def entry(title, description, metric_name, value, threshold, passed, raw):
    return {
        "section": "§2",
        "article_refs": ARTICLES,
        "title": title,
        "description": description,
        "evidence_type": "bias",
        "metric_name": metric_name,
        "metric_value": None if value is None else float(value),
        "threshold": threshold,
        "passed": passed,
        "raw": raw,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--y-true", required=True)
    p.add_argument("--y-pred", required=True)
    p.add_argument("--sensitive", action="append", required=True, help="repeat for several attributes")
    p.add_argument("--positive", default=None, help="label value meaning the positive outcome")
    p.add_argument("--min-dp-ratio", type=float, default=0.8,
                   help="demographic-parity ratio floor (default 0.8, the US 'four-fifths' heuristic; not an EU legal threshold)")
    p.add_argument("--max-eo-diff", type=float, default=0.1,
                   help="equalized-odds difference ceiling (default 0.1, a common convention; not a legal threshold)")
    p.add_argument("--min-group-size", type=int, default=30, help="warn when a group has fewer rows")
    p.add_argument("-o", "--output", default="fairness-evidence.json")
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    missing = [c for c in [args.y_true, args.y_pred, *args.sensitive] if c not in df.columns]
    if missing:
        raise SystemExit(f"columns not found: {missing}; available: {list(df.columns)}")
    df = df.dropna(subset=[args.y_true, args.y_pred, *args.sensitive])
    y_true = _binarize(df[args.y_true], args.positive)
    y_pred = _binarize(df[args.y_pred], args.positive)

    entries = []
    for attr in args.sensitive:
        sf = df[attr].astype(str)
        mf = MetricFrame(
            metrics={
                "n": count,
                "selection_rate": selection_rate,
                "accuracy": accuracy_score,
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate,
            },
            y_true=y_true, y_pred=y_pred, sensitive_features=sf,
        )
        by_group = mf.by_group.reset_index().to_dict(orient="records")
        small = [str(r[attr]) for r in by_group if r["n"] < args.min_group_size]
        dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sf)
        dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
        eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sf)
        raw = {
            "sensitive_attribute": attr,
            "n_rows": int(len(df)),
            "by_group": by_group,
            "demographic_parity_difference": float(dp_diff),
            "small_groups": small,
            "thresholds_are_conventions_not_law": True,
        }
        caveat = f" Groups with fewer than {args.min_group_size} rows: {', '.join(small)}; treat their rates as unreliable." if small else ""
        entries.append(entry(
            f"Demographic parity ratio by {attr}",
            f"Ratio of the lowest to highest positive-prediction rate across '{attr}' groups "
            f"on {len(df)} decisions. Floor {args.min_dp_ratio} is a screening convention.{caveat}",
            "demographic_parity_ratio", dp_ratio, args.min_dp_ratio, bool(dp_ratio >= args.min_dp_ratio), raw,
        ))
        entries.append(entry(
            f"Equalized odds difference by {attr}",
            f"Largest gap in true- or false-positive rate across '{attr}' groups. "
            f"Ceiling {args.max_eo_diff} is a screening convention.{caveat}",
            "equalized_odds_difference", eo_diff, args.max_eo_diff, bool(eo_diff <= args.max_eo_diff),
            {"sensitive_attribute": attr},
        ))

    with open(args.output, "w") as fh:
        json.dump({"entries": entries}, fh, indent=2, default=str)
    for e in entries:
        flag = "PASS" if e["passed"] else "FLAG"
        print(f"{flag}  {e['title']}: {e['metric_value']:.3f} (threshold {e['threshold']})")
    print(f"wrote {len(entries)} vault entries -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
