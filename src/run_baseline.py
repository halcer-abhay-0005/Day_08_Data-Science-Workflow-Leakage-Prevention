"""Task C: split + baselines + leakage demonstration. Writes results/results.json."""
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from pipeline import FEATURES, TARGET, build_majority_baseline, build_model, clean, time_split


def score(model, X, y):
    p = model.predict(X)
    return {"accuracy": round(accuracy_score(y, p), 4), "macro_f1": round(f1_score(y, p, average="macro"), 4)}


def main():
    raw = pd.read_csv("data/tickets.csv", parse_dates=["created_at"])
    df = clean(raw)
    tr, va, te = time_split(df)
    print(f"raw={len(raw)} clean={len(df)} train={len(tr)} val={len(va)} test={len(te)}")
    print("train range:", tr.created_at.min().date(), "->", tr.created_at.max().date())
    print("test  range:", te.created_at.min().date(), "->", te.created_at.max().date())
    assert tr.created_at.max() <= va.created_at.min() and va.created_at.max() <= te.created_at.min()

    res = {}
    maj = build_majority_baseline().fit(tr[FEATURES], tr[TARGET])
    res["majority_val"] = score(maj, va[FEATURES], va[TARGET])
    lr = build_model().fit(tr[FEATURES], tr[TARGET])
    res["logreg_val"] = score(lr, va[FEATURES], va[TARGET])

    # Test set touched ONCE, after all choices are frozen
    res["majority_test"] = score(maj, te[FEATURES], te[TARGET])
    res["logreg_test"] = score(lr, te[FEATURES], te[TARGET])
    print(classification_report(te[TARGET], lr.predict(te[FEATURES]), zero_division=0))

    # --- Leakage demo: add assigned_team (post-outcome) and watch the score become 'too good' ---
    leak_feats = FEATURES + ["assigned_team"]
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), ["assigned_team"])])
    leaky = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))]).fit(tr[leak_feats], tr[TARGET])
    res["LEAKY_assigned_team_only_test"] = score(leaky, te[leak_feats], te[TARGET])

    # --- Leakage demo 2: duplicates + random split inflates the score vs the time split ---
    raw_l = raw.dropna(subset=[TARGET])
    a, b = train_test_split(raw_l, test_size=0.15, random_state=42, stratify=raw_l[TARGET])
    m = build_model().fit(a[FEATURES], a[TARGET])
    res["random_split_with_duplicates_test"] = score(m, b[FEATURES], b[TARGET])

    # --- Leakage demo 3: fitting TF-IDF on ALL data before splitting (wrong) vs inside pipeline ---
    print(json.dumps(res, indent=2))
    with open("results/results.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
