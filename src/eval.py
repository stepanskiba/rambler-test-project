import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

MODEL_PATH = Path("models/model.joblib")
TEST_PATH = Path("data/test.jsonl")
METRICS_PATH = Path("data/metrics.json")

ALLOW_BELOW, BLOCK_ABOVE = 0.3, 0.8

model = joblib.load(MODEL_PATH)

test = pd.read_json(TEST_PATH, lines=True)
X_test, y_test = test["text"].fillna(""), (test["label"] == 0).astype(int)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

allow = int((proba < ALLOW_BELOW).sum())
block = int((proba >= BLOCK_ABOVE).sum())
review = len(proba) - allow - block

metrics = {
    "n_samples": len(y_test),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "roc_auc": round(roc_auc_score(y_test, proba), 4),
    "zones": {
        "ALLOW": round(allow / len(proba), 4),
        "REVIEW": round(review / len(proba), 4),
        "BLOCK": round(block / len(proba), 4),
    },
}

METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(metrics, indent=2, ensure_ascii=False))