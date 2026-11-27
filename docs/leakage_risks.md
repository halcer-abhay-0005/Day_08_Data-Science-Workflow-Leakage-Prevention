# Leakage risks (documented)
| Risk | Type | Evidence | Mitigation |
|---|---|---|---|
| `assigned_team` | Target / post-outcome | Normalised MI with label = 1.0; adding it gives 100% test accuracy | Excluded (`EXCLUDED` in `src/pipeline.py`) |
| `resolution_hours` | Post-outcome | Differs by class; known only after closing | Excluded |
| Duplicate texts (120) | Duplicate leakage | Same text in train and test inflates score | Dedup before splitting; keep earliest; test asserts no shared text |
| `customer_id` | Identifier memorisation | MI 0.28 | Excluded |
| Random split | Temporal leakage | Future tickets in train | Chronological split; test asserts ordering |
| TF-IDF / imputer / encoder | Preprocessing leakage | Fitting on all data lets test stats leak | Fitted inside a Pipeline on train only; test checks vocabulary unchanged after predict |
| Test set reuse | Selection leakage | Tuning on test | Test scored once after choices frozen on validation |
