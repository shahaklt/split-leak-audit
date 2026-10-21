# split-leak-audit

Audits a train/test split for leakage before you trust a held-out score. It targets the failure modes that inflate results in patient-level biomedical ML.

## Checks

| Check | Severity | Triggers when |
|---|---|---|
| `patient_overlap` | error | the same `patient_id` is in train and test |
| `exact_duplicate_rows` | error | identical numeric feature rows appear in both splits |
| `near_duplicate_rows` | warn | a test row is within RMS distance `--near-threshold` (standardized) of a train row |
| `preprocessing_fit` | error | a manifest says scalers/imputers were fit on data that includes test |

## Usage

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m split_leak_audit.cli train.csv test.csv --id-col patient_id --label-col label --manifest prep.json --out report.json
```

`prep.json` example: `{"fit_on": "train"}`. The tool cannot see your code, so this manifest is how you declare what preprocessing was fit on.

Exit code is 1 on any error.

## Limits

- Numeric features only. Categorical columns are ignored.
- Near-duplicate search is brute force. It is fine to a few tens of thousands of rows.
- A clean report does not prove there is no leakage. It rules out the common, checkable cases.
- Unit tests use synthetic splits. The real-data check is in `validation/` (MIT-BIH, 29 patients): the audit flags the leaky split and not the patient-level split. Small sample, see `validation/README.md`.
