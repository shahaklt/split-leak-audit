"""Real-data check: does split-leak-audit catch leakage that inflates held-out scores?

Data: MIT-BIH Arrhythmia Database (PhysioNet, ODC Attribution License v1.0).
Each record is one patient. Beat annotations are turned into 5-minute segments with
RR-interval features (no ECG signal needed). Label: segment contains >= 1 ventricular
ectopic beat (V). The V count itself is never used as a feature.

Scenarios:
  leaky   - segments split at random, so the same patient is on both sides.
  patient - records split by patient: test patients are never seen in training.

For each scenario we run the audit and fit a random forest on the training segments.

Usage:
    PYTHONPATH=src python3 validation/validate_mitbih_split.py --data-dir DIR --records ... --out results.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import wfdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from split_leak_audit.core import audit, summarize

BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
SUPRAVENTRICULAR = set("AaSJ")
SEGMENT_S = 300.0
MIN_BEATS = 5


def segments_for_record(data_dir, record):
    ann = wfdb.rdann(os.path.join(data_dir, record), "atr")
    fs = float(ann.fs)
    idx = [i for i, s in enumerate(ann.symbol) if s in BEAT_SYMBOLS]
    samples = ann.sample[idx].astype(float)
    symbols = np.array([ann.symbol[i] for i in idx])
    t = samples / fs
    seg = np.floor(t / SEGMENT_S).astype(int)
    rows = []
    for s in np.unique(seg):
        m = seg == s
        if m.sum() < MIN_BEATS + 1:
            continue
        rr = np.diff(t[m])
        rows.append({
            "patient_id": record,
            "seg_index": int(s),
            "n_beats": int(m.sum()),
            "mean_rr": float(rr.mean()),
            "sdrr": float(rr.std(ddof=1)),
            "rr_range": float(rr.max() - rr.min()),
            "n_supraventricular": int(np.isin(symbols[m], list(SUPRAVENTRICULAR)).sum()),
            "has_pvc": int((symbols[m] == "V").any()),
        })
    return rows


def build_table(data_dir, records):
    rows = []
    for r in records:
        rows += segments_for_record(data_dir, r)
    return pd.DataFrame(rows)


FEATURES = ["n_beats", "mean_rr", "sdrr", "rr_range", "n_supraventricular"]


def fit_and_score(train, test, seed):
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced")
    clf.fit(train[FEATURES], train["has_pvc"])
    prob = clf.predict_proba(test[FEATURES])[:, 1]
    pred = (prob >= 0.5).astype(int)
    out = {"balanced_accuracy": float(balanced_accuracy_score(test["has_pvc"], pred))}
    if test["has_pvc"].nunique() == 2:
        out["auc"] = float(roc_auc_score(test["has_pvc"], prob))
    return out


def audit_frame(df):
    """Hand the audit only patient_id, features, and label; keep seg_index out of the feature set."""
    return df[["patient_id", "has_pvc"] + FEATURES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    table = build_table(args.data_dir, sorted(args.records, key=int))
    records = sorted(table["patient_id"].unique(), key=int)
    patient_test = set(records[::3])

    leaky_scores, patient_scores = [], []
    leaky_report = patient_report = None
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        mask = rng.random(len(table)) < 0.7
        train_l, test_l = table[mask], table[~mask]
        if seed == 0:
            leaky_report = summarize(audit(audit_frame(train_l), audit_frame(test_l), "patient_id", "has_pvc"))
        leaky_scores.append(fit_and_score(train_l, test_l, seed))

        is_test = table["patient_id"].isin(patient_test)
        train_p, test_p = table[~is_test], table[is_test]
        if seed == 0:
            patient_report = summarize(audit(audit_frame(train_p), audit_frame(test_p), "patient_id", "has_pvc"))
        patient_scores.append(fit_and_score(train_p, test_p, seed))

    def mean_of(scores, key):
        vals = [s[key] for s in scores if key in s]
        return round(float(np.mean(vals)), 3) if vals else None

    result = {
        "n_segments": int(len(table)),
        "n_patients": int(table["patient_id"].nunique()),
        "pvc_segment_fraction": round(float(table["has_pvc"].mean()), 3),
        "patient_test_records": sorted(patient_test, key=int),
        "leaky_split": {
            "audit_errors": leaky_report["n_errors"],
            "audit_warnings": leaky_report["n_warnings"],
            "audit_findings": [f["check"] for f in leaky_report["findings"]],
            "mean_auc": mean_of(leaky_scores, "auc"),
            "mean_balanced_accuracy": mean_of(leaky_scores, "balanced_accuracy"),
        },
        "patient_split": {
            "audit_errors": patient_report["n_errors"],
            "audit_warnings": patient_report["n_warnings"],
            "audit_findings": [f["check"] for f in patient_report["findings"]],
            "mean_auc": mean_of(patient_scores, "auc"),
            "mean_balanced_accuracy": mean_of(patient_scores, "balanced_accuracy"),
        },
        "seeds": args.seeds,
    }
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
