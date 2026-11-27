"""Task B: data audit. Prints class counts, missing labels, duplicates, leakage screening."""
import pandas as pd
from sklearn.metrics import mutual_info_score


def run(df):
    out = {}
    out["rows"] = len(df)
    out["missing_labels"] = int(df["category"].isna().sum())
    out["class_counts"] = df["category"].value_counts(dropna=True).to_dict()
    out["duplicate_text_rows"] = int(df.duplicated("text", keep="first").sum())
    out["missing_by_col"] = df.isna().sum()[lambda s: s > 0].to_dict()
    lab = df.dropna(subset=["category"])
    # leakage screen: how well does a single column "explain" the label?
    # normalised mutual information for categoricals; ANOVA-style spread for numeric
    leak = {}
    for col in ["channel", "priority", "assigned_team", "customer_id"]:
        x = lab[col].fillna("missing").astype(str)
        mi = mutual_info_score(x, lab["category"])
        h = mutual_info_score(lab["category"], lab["category"])
        leak[col] = round(mi / h, 3)
    out["normalised_MI_with_label"] = leak
    out["mean_resolution_hours_by_class"] = lab.groupby("category")["resolution_hours"].mean().round(1).to_dict()
    return out


if __name__ == "__main__":
    import json
    df = pd.read_csv("data/tickets.csv", parse_dates=["created_at"])
    print(json.dumps(run(df), indent=2, default=str))
