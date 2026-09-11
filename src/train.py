from pathlib import Path

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_PATH = Path("models/model.joblib")
VALID_PATH = Path("data/valid.jsonl")
TEST_PATH = Path("data/test.jsonl")
RANDOM_STATE = 42

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
TEST_PATH.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset("Mnwa/russian-toxic")
train_full, test = ds["train"].to_pandas(), ds["test"].to_pandas()

# сделаем предсказания toxic = 1 -> токсичный
for df in (train_full, test):
    df["toxic"] = (df["label"] == 0).astype(int)
    df["text"] = df["text"].fillna("")

train, valid = train_test_split(
    train_full, test_size=0.2, random_state=RANDOM_STATE, stratify=train_full["toxic"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", verbose=1)),
])
model.fit(train["text"], train["toxic"])

joblib.dump(model, MODEL_PATH)

valid[["text", "toxic"]].to_json(VALID_PATH, orient="records", lines=True, force_ascii=False)
test[["text", "toxic"]].to_json(TEST_PATH, orient="records", lines=True, force_ascii=False)

pred = model.predict(test["text"])
proba = model.predict_proba(test["text"])[:, 1]

print(f"precision: {precision_score(test['toxic'], pred):.4f}")
print(f"recall: {recall_score(test['toxic'], pred):.4f}")
print(f"f1: {f1_score(test['toxic'], pred):.4f}")
print(f"roc_auc: {roc_auc_score(test['toxic'], proba):.4f}")