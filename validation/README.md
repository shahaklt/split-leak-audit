# Real-data validation: MIT-BIH Arrhythmia

Data: **MIT-BIH Arrhythmia Database** (PhysioNet; Moody & Mark 2001; Goldberger et al. 2000), ODC Attribution License v1.0. 29 records (one patient each), beat annotations only, downloaded from `https://physionet.org/files/mitdb/1.0.0/`.

## Setup

Each patient is split into 5-minute segments with RR-interval features (mean RR, SD of RR, RR range, beat count, supraventricular beat count). Label: segment contains at least one ventricular ectopic beat. The ventricular beat count is never a feature.

- 199 segments from 29 patients (53% positive)
- Random forest, 200 trees, balanced class weights, 5 seeds

Two splits:

| Split | How | Audit result (seed 0) | Mean AUC | Mean balanced accuracy |
|---|---|---|---|---|
| leaky | segments assigned to train/test at random (~70/30), so the same patient appears on both sides | **1 error: `patient_overlap`**; warnings: `near_duplicate_rows` | **0.936** | 0.874 |
| patient | 10 of 29 records held out; their segments never appear in training | no errors; warning: `near_duplicate_rows` | **0.766** | 0.773 |

Reproduce:

```bash
PYTHONPATH=src python3 validation/validate_mitbih_split.py --data-dir <dir with .hea/.atr> --records 100 101 103 105 106 107 108 109 111 112 113 115 116 117 118 119 121 122 123 124 200 201 202 203 205 207 208 209 210 --out validation/results.json
```

Full output: `results.json`.

## What this shows

- The audit catches the leakage that matters here. The leaky split overlaps patients, and the audit reports it as an error. The patient-level split is clean on that check.
- The leaky split overstates performance: AUC 0.94 vs. 0.77 on the same features and model. Much of the leaky score comes from a patient being in both train and test, not from the features generalizing.
- The `near_duplicate_rows` warning also fires on the clean patient split. With standardized features, segments from different patients can still be close. Treat that warning as "look at this," not "this is leakage." Its threshold is a default.

## Limits

- 29 patients and 199 segments. Each patient-level test set has only 10 records, so the AUC estimate is noisy.
- Features are simple RR statistics computed from annotations, not from the ECG signal.
- One label definition, one model. This shows the audit's behavior on real data, not a claim about arrhythmia detection performance.

## Sleep-EDF validation (wake vs. sleep, 3 subjects)

Data: **Sleep-EDF Database Expanded**, sleep-cassette subset (PhysioNet; Kemp et al. 2000; ODC Attribution License v1.0). Subjects SC4001, SC4031, SC4041, one night each, EEG channels Fpz-Cz and Pz-Oz, expert hypnograms in 30-second epochs. Wake = 0; N1, N2, N3, and REM = 1. Movement and unscored epochs dropped. 8,039 epochs, 69% wake.

Features per epoch: log band power (delta, theta, alpha, sigma, beta), relative band power, and log variance per channel. Random forest, 200 trees, balanced class weights.

| Split | Balanced accuracy | AUC | `split-leak-audit` result |
|---|---|---|---|
| random epoch split (75/25, leaky) | 0.989 | 0.999 | 1 error: `patient_overlap` |
| leave-one-subject-out | 0.936 (folds 0.948, 0.930, 0.929) | 0.993 | no errors |

Reproduce:

```bash
PYTHONPATH=src python3 validation/validate_sleep_split.py --data-dir <dir with the SC40x1E0-PSG.edf and hypnogram files> --out validation/sleep_results.json
```

Full output: `sleep_results.json`.

**What this shows:** the random split inflates balanced accuracy by about 5 points, and the audit catches it. The AUC gap is small because wake vs. sleep is an easy task from EEG band power.

**Limits:** 3 of the 6 planned subjects; one night each; healthy, older cassette-study subjects, not clinical sleep patients; one binary task (wake vs. sleep), not full staging.
