# Problem specification (Task A)
Predict the support-ticket `category` (billing / technical / account / shipping) for a single ticket at creation time, before human triage, so it can be routed automatically.

- **Unit of prediction:** one ticket
- **Target:** `category`
- **Features available at prediction time:** `text`, `channel`, `priority`
- **Excluded:** `assigned_team`, `resolution_hours` (post-outcome); `ticket_id`, `customer_id` (identifiers); `created_at` (split only)
- **Metric:** macro-F1 (primary); accuracy (secondary). Classes are imbalanced and every category matters equally.
- **Success:** beat the majority baseline (macro-F1 0.138) clearly on a chronologically later test set.
