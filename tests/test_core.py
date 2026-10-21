import numpy as np
import pandas as pd

from split_leak_audit.core import (
    audit,
    check_exact_duplicates,
    check_near_duplicates,
    check_patient_overlap,
    check_preprocessing_fit,
    feature_columns,
)


def make_split(seed=0, n_train=60, n_test=30):
    rng = np.random.default_rng(seed)
    train = pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n_train)],
        "f1": rng.normal(size=n_train),
        "f2": rng.normal(size=n_train),
        "label": rng.integers(0, 2, n_train),
    })
    test = pd.DataFrame({
        "patient_id": [f"q{i}" for i in range(n_test)],
        "f1": rng.normal(size=n_test),
        "f2": rng.normal(size=n_test),
        "label": rng.integers(0, 2, n_test),
    })
    return train, test


def test_clean_split_has_no_errors():
    train, test = make_split()
    findings = audit(train, test, "patient_id", label_col="label")
    assert not [f for f in findings if f.severity == "error"]


def test_patient_overlap_is_error():
    train, test = make_split()
    test.loc[0, "patient_id"] = "p3"
    findings = check_patient_overlap(train, test, "patient_id")
    assert findings[0].severity == "error"
    assert findings[0].details["patient_ids"] == ["p3"]


def test_exact_duplicate_rows_are_error():
    train, test = make_split()
    test.loc[5, ["f1", "f2"]] = train.loc[7, ["f1", "f2"]].values
    findings = check_exact_duplicates(train, test, ["f1", "f2"])
    assert findings[0].severity == "error"


def test_near_duplicate_rows_are_warn():
    train, test = make_split()
    test.loc[2, ["f1", "f2"]] = train.loc[10, ["f1", "f2"]].values + 1e-4
    findings = check_near_duplicates(train, test, ["f1", "f2"], threshold=0.05)
    assert findings[0].severity == "warn"
    assert findings[0].details["n_test_rows"] >= 1


def test_unrelated_rows_are_not_near_duplicates():
    train, test = make_split()
    assert check_near_duplicates(train, test, ["f1", "f2"], threshold=0.001) == []


def test_preprocessing_fit_on_test_is_error():
    assert check_preprocessing_fit({"fit_on": "train+test"})[0].severity == "error"
    assert check_preprocessing_fit({"fit_on": "train"}) == []
    assert check_preprocessing_fit(None)[0].severity == "info"


def test_feature_columns_skip_id_and_label():
    train, _ = make_split()
    assert feature_columns(train, "patient_id", "label") == ["f1", "f2"]


def test_leak_free_split_with_all_checks_clean_except_info():
    train, test = make_split()
    findings = audit(train, test, "patient_id", label_col="label", manifest={"fit_on": "train"})
    assert all(f.severity != "error" for f in findings)
