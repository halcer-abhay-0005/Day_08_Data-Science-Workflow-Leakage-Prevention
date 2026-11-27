import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import numpy as np
import pandas as pd
from make_data import make_tickets
from pipeline import EXCLUDED, FEATURES, TARGET, build_model, clean, time_split


def df():
    return clean(make_tickets())


def test_make_data_is_reproducible():
    pd.testing.assert_frame_equal(make_tickets(seed=1), make_tickets(seed=1))


def test_clean_removes_missing_labels_and_duplicates():
    c = df()
    assert c[TARGET].notna().all()
    assert not c.duplicated("text").any()


def test_time_split_is_chronological_and_disjoint():
    tr, va, te = time_split(df())
    assert tr.created_at.max() <= va.created_at.min()
    assert va.created_at.max() <= te.created_at.min()
    assert not (set(tr.ticket_id) & set(te.ticket_id))
    assert not (set(tr.text) & set(te.text))          # no duplicate text across train/test


def test_no_leaky_feature_used():
    assert not set(FEATURES) & set(EXCLUDED)


def test_preprocessing_fit_on_train_only():
    tr, _, te = time_split(df())
    m = build_model().fit(tr[FEATURES], tr[TARGET])
    vocab = set(m.named_steps["pre"].named_transformers_["text"].vocabulary_)
    # a token that exists only in test must not be in the fitted vocabulary
    te_only = "zzzonlyintest"
    te2 = te.copy(); te2.iloc[0, te2.columns.get_loc("text")] = te_only + " " + te_only
    m.predict(te2[FEATURES])                           # predict must not refit
    vocab_after = set(m.named_steps["pre"].named_transformers_["text"].vocabulary_)
    assert te_only not in vocab_after and vocab == vocab_after


def test_baseline_beats_majority():
    from pipeline import build_majority_baseline
    from sklearn.metrics import f1_score
    tr, _, te = time_split(df())
    maj = build_majority_baseline().fit(tr[FEATURES], tr[TARGET])
    lr = build_model().fit(tr[FEATURES], tr[TARGET])
    f = lambda m: f1_score(te[TARGET], m.predict(te[FEATURES]), average="macro")
    assert f(lr) > f(maj) + 0.2
