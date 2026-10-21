"""Real-data check: does a random epoch split inflate sleep-staging scores compared with a
subject-level split, and does split-leak-audit catch it?

Data: Sleep-EDF Database Expanded, sleep-cassette subset (PhysioNet, ODC Attribution License v1.0).
Subjects: SC4001, SC4011, SC4021, SC4031, SC4041, SC4051, one night each (E0 PSG file).
Hypnograms: expert stage annotations, 30-second epochs.

Task: wake vs. sleep per 30-second epoch (Wake = 0; N1, N2, N3, REM = 1). Movement and unscored
epochs are dropped. Features per epoch and channel (Fpz-Cz, Pz-Oz): log band power in delta, theta,
alpha, sigma and beta bands, relative band power, and log variance.

Evaluations:
  random_epoch_split  - 75/25 split of epochs, shuffled across subjects (leaky)
  leave_one_subject_out - one subject held out per fold (patient-level)
Both are also run through split_leak_audit.audit, using the subject as the ID column.

Usage:
    PYTHONPATH=src python3 validation/validate_sleep_split.py --data-dir DIR --out validation/sleep_results.json
"""

import argparse
import json
import os

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from split_leak_audit.core import audit, summarize

trapz = getattr(np, "trapezoid", None) or np.trapz

EPOCH_SEC = 30.0
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30)}
STAGE_TO_Y = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 1,
              "Sleep stage 3": 1, "Sleep stage 4": 1, "Sleep stage R": 1}
SUBJECTS = {
    "SC4001": ("SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf"),
    "SC4011": ("SC4011E0-PSG.edf", "SC4011EH-Hypnogram.edf"),
    "SC4021": ("SC4021E0-PSG.edf", "SC4021EH-Hypnogram.edf"),
    "SC4031": ("SC4031E0-PSG.edf", "SC4031EC-Hypnogram.edf"),
    "SC4041": ("SC4041E0-PSG.edf", "SC4041EC-Hypnogram.edf"),
    "SC4051": ("SC4051E0-PSG.edf", "SC4051EC-Hypnogram.edf"),
}


def epoch_labels(hyp_path, n_epochs):
    ann = mne.read_annotations(hyp_path)
    y = np.full(n_epochs, -1)
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        if desc not in STAGE_TO_Y:
            continue
        first = int(np.floor(onset / EPOCH_SEC))
        last = int(np.floor((onset + dur) / EPOCH_SEC))
        for e in range(first, last):
            if 0 <= e < n_epochs:
                y[e] = STAGE_TO_Y[desc]
    return y


def epoch_features(psg_path):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
    raw.pick(CHANNELS)
    raw.filter(0.5, 40, verbose="ERROR")
    sf = raw.info["sfreq"]
    data = raw.get_data()
    n = int(data.shape[1] // (EPOCH_SEC * sf))
    rows = []
    for e in range(n):
        seg = data[:, int(e * EPOCH_SEC * sf):int((e + 1) * EPOCH_SEC * sf)]
        row = {}
        for ci, ch in enumerate(CHANNELS):
            f, p = welch(seg[ci], fs=sf, nperseg=int(sf * 4))
            total = trapz(p, f)
            for band, (lo, hi) in BANDS.items():
                m = (f >= lo) & (f < hi)
                bp = trapz(p[m], f[m])
                row[f"{ch}_{band}_log"] = np.log10(bp + 1e-12)
                row[f"{ch}_{band}_rel"] = bp / total if total > 0 else 0.0
            row[f"{ch}_var_log"] = np.log10(np.var(seg[ci]) + 1e-12)
        rows.append(row)
    return pd.DataFrame(rows), n


def build_table(data_dir):
    frames = []
    for subject, (psg, hyp) in SUBJECTS.items():
        if not os.path.exists(os.path.join(data_dir, psg)):
            print(f"skipping {subject}: {psg} not found", flush=True)
            continue
        feats, n = epoch_features(os.path.join(data_dir, psg))
        y = epoch_labels(os.path.join(data_dir, hyp), n)
        feats["y"] = y
        feats["subject"] = subject
        frames.append(feats[feats["y"] >= 0])
    return pd.concat(frames, ignore_index=True)


def fit_and_score(train, test, feats, seed=0):
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=seed)
    clf.fit(train[feats], train["y"])
    prob = clf.predict_proba(test[feats])[:, 1]
    pred = (prob >= 0.5).astype(int)
    out = {"balanced_accuracy": float(balanced_accuracy_score(test["y"], pred))}
    if test["y"].nunique() == 2:
        out["auc_roc"] = float(roc_auc_score(test["y"], prob))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = build_table(args.data_dir)
    feats = [c for c in df.columns if c not in ("y", "subject")]

    rng = np.random.default_rng(0)
    mask = rng.random(len(df)) < 0.75
    train_r, test_r = df[mask], df[~mask]
    random_metrics = fit_and_score(train_r, test_r, feats)
    random_audit = summarize(audit(train_r[["subject", "y"] + feats], test_r[["subject", "y"] + feats],
                                   "subject", "y"))

    loso = []
    for held in sorted(df["subject"].unique()):
        train, test = df[df["subject"] != held], df[df["subject"] == held]
        m = fit_and_score(train, test, feats)
        m["held_out_subject"] = held
        m["n_test_epochs"] = int(len(test))
        loso.append(m)

    held = sorted(df["subject"].unique())[0]
    loso_audit = summarize(audit(df[df["subject"] != held][["subject", "y"] + feats],
                                 df[df["subject"] == held][["subject", "y"] + feats], "subject", "y"))

    result = {
        "dataset": "Sleep-EDF Expanded, sleep-cassette, one night per subject",
        "subjects": sorted(df["subject"].unique().tolist()),
        "n_epochs": int(len(df)),
        "wake_fraction": round(float((df["y"] == 0).mean()), 3),
        "random_epoch_split": {
            **{k: round(v, 3) for k, v in random_metrics.items()},
            "audit_errors": random_audit["n_errors"],
            "audit_findings": [f["check"] for f in random_audit["findings"]],
        },
        "leave_one_subject_out": {
            "folds": loso,
            "mean_balanced_accuracy": round(float(np.mean([r["balanced_accuracy"] for r in loso])), 3),
            "mean_auc_roc": round(float(np.mean([r["auc_roc"] for r in loso if "auc_roc" in r])), 3),
            "audit_first_fold_errors": loso_audit["n_errors"],
            "audit_first_fold_findings": [f["check"] for f in loso_audit["findings"]],
        },
    }
    text = json.dumps(result, indent=2, default=float)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
