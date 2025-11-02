
# ARA (Pure-ML Demo)

This version replaces heuristics with **scikit-learn** models:

- **SafetyModel:** TF-IDF (char+word ngrams) → Logistic Regression (harmful vs benign)
- **ConsistencyDetector:** IsolationForest trained on *benign* TF-IDF vectors for anomaly/OOD

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .[dev]
python scripts/seed_data.py
python -m arms_race_alignment.cli --config configs/default.yaml
```

Metrics printed at the end: Time-to-Mitigation, Robustness-Regret, Rolling-robustness.
