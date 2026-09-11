import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

load_dotenv()

ALLOW_BELOW = float(os.getenv("ALLOW_BELOW", "0.33"))
BLOCK_ABOVE = float(os.getenv("BLOCK_ABOVE", "0.88"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/model.joblib"))
    parser.add_argument("--test-path", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--metrics-path", type=Path, default=Path("data/metrics.json"))
    parser.add_argument("--toxic-share", type=float, default=0.05, help="ожидаемая доля токсичных в потоке")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t = args.toxic_share

    model = joblib.load(args.model_path)
    test = pd.read_json(args.test_path, lines=True)

    proba = model.predict_proba(test["text"].fillna(""))[:, 1]
    p_toxic = proba[test["toxic"] == 1]
    p_normal = proba[test["toxic"] == 0]

    toxic_allowed = float((p_toxic < ALLOW_BELOW).mean())
    normal_allowed = float((p_normal < ALLOW_BELOW).mean())
    toxic_blocked = float((p_toxic >= BLOCK_ABOVE).mean())
    normal_blocked = float((p_normal >= BLOCK_ABOVE).mean())

    allow = toxic_allowed * t + normal_allowed * (1 - t)
    block = toxic_blocked * t + normal_blocked * (1 - t)

    metrics = {
        "n_samples": len(test),
        "toxic_share_in_test": round(float(test["toxic"].mean()), 4),
        "toxic_share": t,
        "roc_auc": round(roc_auc_score(test["toxic"], proba), 4),
        "thresholds": {"allow_below": ALLOW_BELOW, "block_above": BLOCK_ABOVE},
        "allow": {
            "share": round(allow, 4),
            "toxic_in_allow": round(toxic_allowed * t / allow, 4) if allow else 0.0,
        },
        "review": {"share": round(1 - allow - block, 4)},
        "block": {
            "share": round(block, 4),
            "precision": round(toxic_blocked * t / block, 4) if block else 0.0,
            "recall": round(toxic_blocked, 4),
            "fpr": round(normal_blocked, 4),
        },
    }

    args.metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()