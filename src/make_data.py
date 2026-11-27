"""Generate a synthetic historical support-ticket dataset with deliberate leakage traps."""
import numpy as np
import pandas as pd

CATS = {
    "billing":   ["invoice", "refund", "charge", "payment", "subscription", "card"],
    "technical": ["error", "crash", "bug", "timeout", "login", "server"],
    "account":   ["password", "profile", "email", "username", "access", "verify"],
    "shipping":  ["delivery", "package", "tracking", "late", "address", "courier"],
}
PRIORS = {"billing": 0.35, "technical": 0.40, "account": 0.15, "shipping": 0.10}  # imbalanced
FILLER = ["please", "help", "issue", "my", "the", "urgent", "need", "problem", "with", "again"]
TEAM = {"billing": "finance", "technical": "engineering", "account": "support", "shipping": "logistics"}


def make_tickets(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    cats = list(PRIORS)
    y = rng.choice(cats, size=n, p=list(PRIORS.values()))
    start = pd.Timestamp("2025-01-01")
    created = start + pd.to_timedelta(np.sort(rng.uniform(0, 365 * 24 * 3600, n)), unit="s")
    texts = []
    for c in y:
        # 60% of words are category-specific, rest filler/noise -> task is learnable but not trivial
        words = [rng.choice(CATS[c]) if rng.random() < 0.35 else rng.choice(FILLER)
                 for _ in range(rng.integers(6, 14))]
        if rng.random() < 0.15:
            words.append(rng.choice(CATS[rng.choice(cats)]))
        texts.append(" ".join(words))
    df = pd.DataFrame({
        "ticket_id": np.arange(n),
        "created_at": created,
        "customer_id": rng.integers(0, 800, n),
        "channel": rng.choice(["email", "chat", "phone", "web"], n),
        "priority": rng.choice(["low", "medium", "high", None], n, p=[.3, .4, .25, .05]),
        "text": texts,
        "category": y,
    })
    # --- LEAKAGE TRAPS ---
    df["assigned_team"] = df["category"].map(TEAM)                         # set AFTER triage = post-outcome
    df["resolution_hours"] = rng.gamma(2, 6, n) * df["category"].map(      # known only after closing
        {"billing": 1.0, "technical": 2.0, "account": 0.5, "shipping": 1.5})
    # --- DATA QUALITY PROBLEMS ---
    df.loc[rng.choice(n, 60, replace=False), "category"] = None            # missing labels
    dups = df.sample(120, random_state=seed).copy()                        # exact-text duplicates, new ids
    dups["ticket_id"] = np.arange(n, n + len(dups))
    dups["created_at"] = dups["created_at"] + pd.Timedelta(days=30)        # resubmitted later
    df = pd.concat([df, dups], ignore_index=True).sort_values("created_at").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = make_tickets()
    df.to_csv("data/tickets.csv", index=False)
    print(f"wrote data/tickets.csv {df.shape}")
