# Daily report – Day 08
- **Topics learned:** problem framing, time-based splits, fit/transform boundaries, target and duplicate leakage.
- **Implementation completed:** data generator, audit, leakage-safe pipeline, two baselines, leakage demos, 6 tests.
- **Tests/experiment results:** 6 passed; see `results/results.json`.
- **Challenge (placeholder, replace with your own):** which copy of a duplicate to keep. Keeping the earliest stops a later resubmission landing in test while the original is in train.
- **Design decision:** a time split plus `Pipeline` so leakage is prevented structurally, not by discipline.
- **Blocker:** none.
- **Tomorrow:** TimeSeriesSplit CV, and test on messier text.
