"""Leakage checks for train/test splits of patient-level data."""

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Finding:
    check: str
    severity: str  # "error", "warn", or "info"
    message: str
    details: dict = field(default_factory=dict)


def feature_columns(df, id_col, label_col=None):
    skip = {id_col, label_col}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def check_patient_overlap(train, test, id_col):
    overlap = sorted(set(train[id_col]) & set(test[id_col]), key=str)
    if not overlap:
        return []
    return [Finding(
        "patient_overlap", "error",
        f"{len(overlap)} patients appear in both train and test",
        {"patient_ids": [str(p) for p in overlap[:50]]},
    )]


def _row_keys(df, feats):
    return df[feats].round(8).astype(str).agg("|".join, axis=1)


def check_exact_duplicates(train, test, feats):
    if not feats:
        return [Finding("exact_duplicate_rows", "info", "no numeric feature columns; check skipped")]
    shared = set(_row_keys(train, feats)) & set(_row_keys(test, feats))
    if not shared:
        return []
    return [Finding(
        "exact_duplicate_rows", "error",
        f"{len(shared)} distinct feature rows occur in both train and test",
        {"count": len(shared)},
    )]


def check_near_duplicates(train, test, feats, threshold=0.05, chunk=1000):
    """Test rows whose nearest train row is within `threshold` RMS difference in standardized units."""
    if not feats:
        return []
    mu = train[feats].mean()
    sd = train[feats].std(ddof=0).replace(0, 1.0)
    a = ((train[feats] - mu) / sd).to_numpy(dtype=float)
    b = ((test[feats] - mu) / sd).to_numpy(dtype=float)
    a_sq = (a**2).sum(axis=1)
    near = 0
    for start in range(0, len(b), chunk):
        block = b[start:start + chunk]
        d2 = (block**2).sum(axis=1)[:, None] + a_sq[None, :] - 2 * block @ a.T
        rms = np.sqrt(np.maximum(d2.min(axis=1), 0) / len(feats))
        near += int((rms < threshold).sum())
    if near == 0:
        return []
    return [Finding(
        "near_duplicate_rows", "warn",
        f"{near} of {len(test)} test rows sit within RMS {threshold} of a train row",
        {"n_test_rows": int(near), "fraction": round(near / len(test), 4)},
    )]


def check_preprocessing_fit(manifest):
    """Scalers, imputers, and feature selectors must be fit on train only."""
    if manifest is None:
        return [Finding("preprocessing_fit", "info", "no preprocessing manifest; check skipped")]
    fit_on = manifest.get("fit_on")
    if fit_on == "train":
        return []
    return [Finding(
        "preprocessing_fit", "error",
        f"preprocessing was fit on {fit_on!r}, which includes test data",
        {"fit_on": fit_on},
    )]


def audit(train, test, id_col, label_col=None, manifest=None, near_threshold=0.05):
    feats = feature_columns(train, id_col, label_col)
    findings = []
    findings += check_patient_overlap(train, test, id_col)
    findings += check_exact_duplicates(train, test, feats)
    findings += check_near_duplicates(train, test, feats, threshold=near_threshold)
    findings += check_preprocessing_fit(manifest)
    return findings


def summarize(findings):
    return {
        "n_errors": sum(f.severity == "error" for f in findings),
        "n_warnings": sum(f.severity == "warn" for f in findings),
        "findings": [asdict(f) for f in findings],
    }
