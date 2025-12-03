import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
import numpy as np


CSV_PATH = "data/df_ready.csv"   # <- your dataset


def load_beauty_data(csv_path: str = CSV_PATH):
    df = pd.read_csv(csv_path)
    if "rating" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain 'rating' and 'text' columns")

    df = df[["rating", "text"]].dropna()
    df = df[df["text"].astype(str).str.strip() != ""]

    # MULTICLASS LABELS: 1–5 → 0–4
    df["label"] = df["rating"].astype(int) - 1

    X = df["text"].astype(str).values
    y = df["label"].values  # 0–4

    # 80 / 10 / 10 split with stratification
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.10,
        random_state=42,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=1/9,  # 10% of original (since 0.1 / 0.9)
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_logreg_models():
    models = {}

    # 1. TF-IDF (unigrams) + L2 (binary)
    models["tfidf_uni_l2"] = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 1),
                min_df=5,
                max_df=0.95,
            )),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="liblinear",
                max_iter=1000,
                class_weight="balanced",
            )),
        ]
    )

    # 2. TF-IDF (unigrams) + L1 (binary)
    models["tfidf_uni_l1"] = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 1),
                min_df=5,
                max_df=0.95,
            )),
            ("clf", LogisticRegression(
                penalty="l1",
                solver="liblinear",
                max_iter=1000,
                class_weight="balanced",
            )),
        ]
    )

    # 3. TF-IDF (uni+bi) + L2 (binary)
    models["tfidf_uni_bi_l2"] = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.95,
            )),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="liblinear",
                max_iter=1000,
                class_weight="balanced",
            )),
        ]
    )

    # 4. TF-IDF (uni+bi) + L2 (MULTICLASS RIDGE)  ← YOU REQUESTED THIS CHANGE
    models["tfidf_uni_bi_l2_multiclass"] = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.95,
            )),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="lbfgs",        # supports multiclass softmax
                max_iter=2000,
                class_weight="balanced",
                multi_class="multinomial"  # ← MULTICLASS MODE
            )),
        ]
    )

    return models


def evaluate_split(split_name, y_true, y_pred, y_scores):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"  # macro for multiclass
    )
    try:
        roc = roc_auc_score(y_true, y_scores, multi_class="ovr")
    except:
        roc = float("nan")

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[{split_name}]")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 (macro)     : {f1:.4f}")
    print(f"ROC-AUC (OVR)  : {roc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)


def train_and_evaluate_all(csv_path: str = CSV_PATH):
    X_train, X_val, X_test, y_train, y_val, y_test = load_beauty_data(csv_path)
    models = build_logreg_models()

    for name, model in models.items():
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)

        # Validation
        y_val_pred = model.predict(X_val)
        y_val_scores = model.predict_proba(X_val)  # now shape (N, 5)
        evaluate_split("Validation", y_val, y_val_pred, y_val_scores)

        # Test
        y_test_pred = model.predict(X_test)
        y_test_scores = model.predict_proba(X_test)
        evaluate_split("Test", y_test, y_test_pred, y_test_scores)


if __name__ == "__main__":
    train_and_evaluate_all()
