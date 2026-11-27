# Decisions, assumptions, limitations (Task D)
**Baseline:** majority class: val macro-F1 0.141, test 0.138. TF-IDF + balanced logistic regression: val 0.972, test 0.976 (accuracy 0.978). Reproduce with the commands in README; results in `results/`.

**Decisions**
- Time-based split, because the model predicts future tickets.
- Single sklearn Pipeline so leakage is prevented structurally.
- Macro-F1 over accuracy because of imbalance.
- `class_weight="balanced"` rather than resampling (resampling before the split would leak).

**Assumptions:** `created_at` is reliable; ticket text is the only strong signal; exact-text matching is enough for dedup.

**Limitations:** the data is synthetic and far easier than real tickets; no drift is simulated; near-duplicates are not detected; no hyperparameter tuning (would use `TimeSeriesSplit` on train only).
