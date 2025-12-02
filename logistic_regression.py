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


CSV_PATH = "./data/df_ready.csv"   # <- your dataset


def load_beauty_data(csv_path: str = CSV_PATH):
    df = pd.read_csv(csv_path)
    if "rating" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain 'rating' and 'text' columns")

    df = df[["rating", "text"]].dropna()
    df = df[df["text"].astype(str).str.strip() != ""]
    df["label"] = (df["rating"] >= 4).astype(int)

    X = df["text"].astype(str).values
    y = df["label"].values

    # 80 / 10 / 10 split with stratification
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=1 / 9,  # 10% of original (since 0.1 / 0.9)
        random_state=42,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_logreg_models():
    models = {}

    # 1. TF-IDF (unigrams) + L2
    models["tfidf_uni_l2"] = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 1),
                    min_df=5,
                    max_df=0.95,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # 2. TF-IDF (unigrams) + L1
    models["tfidf_uni_l1"] = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 1),
                    min_df=5,
                    max_df=0.95,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # 3. TF-IDF (uni+bi) + L2
    models["tfidf_uni_bi_l2"] = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=5,
                    max_df=0.95,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # 4. TF-IDF (uni+bi) + L1
    models["tfidf_uni_bi_l1"] = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=5,
                    max_df=0.95,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    return models


def evaluate_split(split_name, y_true, y_pred, y_scores):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    roc = roc_auc_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[{split_name}]")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1             : {f1:.4f}")
    print(f"ROC-AUC        : {roc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)


def train_and_evaluate_all(csv_path: str = CSV_PATH):
    X_train, X_val, X_test, y_train, y_val, y_test = load_beauty_data(csv_path)
    models = build_logreg_models()

    # ============================================================
    # Run your 4 existing models
    # ============================================================
    for name, model in models.items():
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)

        # Validation
        y_val_pred = model.predict(X_val)
        y_val_scores = model.predict_proba(X_val)[:, 1]
        evaluate_split("Validation", y_val, y_val_pred, y_val_scores)

        # Test
        y_test_pred = model.predict(X_test)
        y_test_scores = model.predict_proba(X_test)[:, 1]
        evaluate_split("Test", y_test, y_test_pred, y_test_scores)

    # ============================================================
    # 5. NEW MODEL: TF-IDF full → drop lowest IDF → truncate to 250 chars
    # ============================================================
    print("\n=== tfidf_250chars_uni_bi_l2 (feature-pruned) ===")

    # Step 1: Fit TF-IDF on full corpus
    full_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
    )
    X_train_full = full_vec.fit_transform(X_train)

    # Step 2: Drop lowest 20% IDF features
    idf = full_vec.idf_
    vocab = np.array(full_vec.get_feature_names_out())

    sorted_idx = np.argsort(idf)[::-1]      # high IDF → low
    keep_n = int(len(idf) * 0.80)           # keep top 80%
    keep_idx = sorted_idx[:keep_n]
    kept_vocab = vocab[keep_idx]

    reduced_vec = TfidfVectorizer(
        vocabulary=list(kept_vocab),
        ngram_range=(1, 2),
    )

    # Step 3: Truncate to first 250 characters
    X_train_250 = [t[:250] for t in X_train]
    X_val_250   = [t[:250] for t in X_val]
    X_test_250  = [t[:250] for t in X_test]

    # Step 4: TF-IDF on truncated text (reduced vocab)
    X_train_tfidf = reduced_vec.fit_transform(X_train_250)
    X_val_tfidf   = reduced_vec.transform(X_val_250)
    X_test_tfidf  = reduced_vec.transform(X_test_250)

    # Step 5: Logistic Regression (ridge / L2)
    clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
    )
    clf.fit(X_train_tfidf, y_train)

    # Validation
    y_val_pred = clf.predict(X_val_tfidf)
    y_val_scores = clf.predict_proba(X_val_tfidf)[:, 1]
    evaluate_split("Validation", y_val, y_val_pred, y_val_scores)

    # Test
    y_test_pred = clf.predict(X_test_tfidf)
    y_test_scores = clf.predict_proba(X_test_tfidf)[:, 1]
    evaluate_split("Test", y_test, y_test_pred, y_test_scores)


if __name__ == "__main__":
    train_and_evaluate_all()
