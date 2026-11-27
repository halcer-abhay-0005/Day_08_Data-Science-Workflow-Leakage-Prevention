# Day 08: Ticket-category prediction, leakage-safe workflow

## Setup
```
python3 -m venv .venv && .venv/bin/pip install scikit-learn pandas numpy pytest
.venv/bin/python src/make_data.py                  # data/tickets.csv (synthetic, seed 42)
.venv/bin/python src/audit.py                      # Task B
PYTHONPATH=src .venv/bin/python src/run_baseline.py   # Task C -> results/results.json
.venv/bin/pytest -q                                # 6 tests
```

## Task A: Problem specification
- **Target:** `category` (billing / technical / account / shipping), one label per ticket.
- **Unit of prediction:** one support ticket.
- **Prediction time:** at ticket creation, before any human triage, to route it automatically.
- **Available features:** `text`, `channel`, `priority`.
- **Excluded:** `assigned_team` and `resolution_hours` (post-outcome); `ticket_id`, `customer_id` (identifiers); `created_at` (split only).
- **Metric:** macro-F1 (primary), accuracy secondary. Classes are imbalanced (shipping ~10%) and every category's routing matters equally.

## Task B: Audit findings (actual output, `src/audit.py`)
| Check | Result |
|---|---|
| Rows | 4120 |
| Missing labels | 61 (dropped) |
| Class counts | technical 1579, billing 1448, account 622, shipping 410 |
| Duplicate texts | 120 (resubmissions; earliest kept) |
| Missing features | `priority` 218 (imputed inside the pipeline) |
| Leakage screen | `assigned_team` normalised MI with label = **1.0** (a perfect proxy for the label); `customer_id` 0.28 (memorisation risk) |
| Resolution time | differs by class (6h to 24h), but it is only known after closing |

## Task C: Split & baseline
- **Split:** chronological 70/15/15 (train Jan to Sep, val, test Nov to Dec). Tickets are time-ordered and the model will predict future tickets, so a random split would be optimistic. The sort is stable and the data seed is fixed, so it is fully reproducible.
- **Preprocessing:** TF-IDF, imputer and one-hot encoder all live inside one sklearn `Pipeline`/`ColumnTransformer`, so they are fitted on train only. A test checks the vocabulary is unchanged after `predict`.
- **Duplicates** are removed before splitting, so no text appears in both train and test.
- **Test set** is scored once, after the model was chosen on validation.

| Model | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|
| Majority class | 0.141 | 0.382 | 0.138 |
| TF-IDF + LogReg (balanced) | 0.972 | 0.978 | 0.976 |

Leakage demonstrations (test set):
- Adding `assigned_team` gives accuracy 1.000, which is too good to be true.
- A random split on the raw data (duplicates kept) gives macro-F1 0.987 vs 0.976 for the time split. The gap is small here, and the next point explains why.

## Task D: Assumptions & limitations
- **The data is synthetic.** Keywords map almost directly to categories, so 0.98 is much higher than real tickets would give. Only the pipeline, the leakage mechanics and the relative comparisons carry over.
- The synthetic data has no concept drift, so the time split costs little. Real data would likely show a larger gap.
- Duplicates are matched on exact text only. Near-duplicates would need fuzzy matching.
- No hyperparameter tuning was done. If it were, I'd use `TimeSeriesSplit` on train and keep val/test untouched.
- Ticket IDs are assumed unique and `created_at` trustworthy.

## Viva answers
1. **Target leakage:** a feature carries information about the label that wouldn't be available at prediction time (e.g. `assigned_team`), so validation looks great and production fails.
2. **Time-based split:** when the model predicts the future (or data has drift or ordering), so training on the past and testing on the future mimics deployment.
3. **Untouched test set:** every time you look at it and adjust something, it leaks into your choices. Untouched, it gives an unbiased estimate of generalisation.
4. **Accuracy on imbalanced classes:** predicting the majority class gets high accuracy with zero minority recall. Here the majority baseline gets 38% accuracy and 0.14 macro-F1. In a 99/1 problem it would be 99%.

## Daily report
- **Topics learned:** problem framing, time-based splits, fit/transform boundaries, target and duplicate leakage.
- **Implementation completed:** data generator, audit, leakage-safe pipeline, two baselines, leakage demos, 6 tests.
- **Tests/experiment results:** 6 passed; see `results/results.json`.
- **Challenge (placeholder, replace with your own):** which copy of a duplicate to keep. Keeping the earliest stops a later resubmission landing in test while the original is in train.
- **Design decision:** a time split plus `Pipeline` so leakage is prevented structurally, not by discipline.
- **Blocker:** none.
- **Tomorrow:** TimeSeriesSplit CV, and test on messier text.
