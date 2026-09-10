import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

load_dotenv()

ALLOW_BELOW = float(os.getenv("ALLOW_BELOW", "0.3"))
BLOCK_ABOVE = float(os.getenv("BLOCK_ABOVE", "0.8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/model.joblib"))
    parser.add_argument("--test-path", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--metrics-path", type=Path, default=Path("data/metrics.json"))
    parser.add_argument("--prior", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pi = args.prior

    model = joblib.load(args.model_path)
    test = pd.read_json(args.test_path, lines=True)

    proba = model.predict_proba(test["text"].fillna(""))[:, 1]
    p_toxic = proba[test["toxic"] == 1]
    p_normal = proba[test["toxic"] == 0]

    fnr = float((p_toxic < ALLOW_BELOW).mean())
    tnr = float((p_normal < ALLOW_BELOW).mean())
    tpr = float((p_toxic >= BLOCK_ABOVE).mean())
    fpr = float((p_normal >= BLOCK_ABOVE).mean())

    allow_share = fnr * pi + tnr * (1 - pi)
    block_share = tpr * pi + fpr * (1 - pi)

    metrics = {
        "n_samples": len(test),
        "positive_rate": round(float(test["toxic"].mean()), 4),
        "assumed_prior": pi,
        "roc_auc": round(roc_auc_score(test["toxic"], proba), 4),
        "thresholds": {"allow_below": ALLOW_BELOW, "block_above": BLOCK_ABOVE},
        "allow": {
            "share": round(allow_share, 4),
            "contamination": round(fnr * pi / allow_share, 4) if allow_share else 0.0,
        },
        "review": {"share": round(1 - allow_share - block_share, 4)},
        "block": {
            "share": round(block_share, 4),
            "precision": round(tpr * pi / block_share, 4) if block_share else 0.0,
            "recall": round(tpr, 4),
            "fpr": round(fpr, 4),
        },
    }

    args.metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()