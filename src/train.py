from pathlib import Path

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

MODEL_PATH = Path("models/model.joblib")
TEST_PATH = Path("data/test.jsonl")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
TEST_PATH.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset("Mnwa/russian-toxic")
train, test = ds["train"].to_pandas(), ds["test"].to_pandas()

# label=0 -> токсичный, label=1 -> нормальный; хотим 1 = токсичный
X_train, y_train = train["text"].fillna(""), (train["label"] == 0).astype(int)
X_test, y_test = test["text"].fillna(""), (test["label"] == 0).astype(int)

model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", verbose=1)),
])
model.fit(X_train, y_train)

joblib.dump(model, MODEL_PATH)

test[["text", "label"]].to_json(TEST_PATH, orient="records", lines=True, force_ascii=False)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print(f"precision: {precision_score(y_test, pred):.4f}")
print(f"recall: {recall_score(y_test, pred):.4f}")
print(f"f1: {f1_score(y_test, pred):.4f}")
print(f"roc_auc: {roc_auc_score(y_test, proba):.4f}")
