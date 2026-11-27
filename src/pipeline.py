"""Split + preprocessing pipeline + baselines. Leak-free by construction."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "category"
# Available at ticket creation time (prediction time):
FEATURES = ["text", "channel", "priority"]
# Excluded: post-outcome (assigned_team, resolution_hours) and identifiers (ticket_id, customer_id).
EXCLUDED = {
    "assigned_team": "post-outcome: derived from the category after triage (target leakage)",
    "resolution_hours": "post-outcome: only known after ticket is closed",
    "ticket_id": "identifier, no signal",
    "customer_id": "identifier; would let model memorise customers across splits",
    "created_at": "used for splitting only",
}


def clean(df):
    """Row-level cleaning that uses no statistics (safe to do before splitting)."""
    df = df.dropna(subset=[TARGET]).copy()                 # unlabeled rows can't train/evaluate
    df = df.sort_values("created_at", kind="stable")
    # keep the EARLIEST copy of each duplicated text so a later resubmission can't leak into test
    df = df.drop_duplicates(subset=["text"], keep="first")
    return df.reset_index(drop=True)


def time_split(df, train=0.70, val=0.15):
    """Chronological split: train < val < test. Requires df sorted by created_at."""
    assert df["created_at"].is_monotonic_increasing
    n = len(df)
    i, j = int(n * train), int(n * (train + val))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def build_model(clf=None):
    pre = ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2), "text"),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), ["channel", "priority"]),
    ])
    return Pipeline([("pre", pre), ("clf", clf or LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])


def build_majority_baseline():
    return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
