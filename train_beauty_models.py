# train_beauty_models.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

INPUT_FILE = "df_ready.csv"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
MODEL_FILE = "logreg_model.pkl"

def print_basic_stats(y_true, y_pred, y_proba=None, split_name=""):
    print(f"\n=== {split_name} performance ===")
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")

    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
            print(f"AUC: {auc:.4f}")
        except ValueError:
            print("AUC: could not be computed (only one class present).")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def main():
    print(f"Loading cleaned data from {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)

    # Basic check
    print(df.head())
    print(df["label"].value_counts())

    X = df["text"].astype(str).values
    y = df["label"].values

    # Split train/valid/test: 80 / 10 / 10
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.1111,  # ~0.1 of original
        random_state=42, stratify=y_train_full
    )

    print(f"Train size: {len(X_train)}")
    print(f"Valid size: {len(X_valid)}")
    print(f"Test size:  {len(X_test)}")

    # -------- Majority baseline on validation set --------
    majority_class = int(np.bincount(y_train).argmax())
    print(f"\nMajority baseline class: {majority_class}")
    y_valid_maj = np.full_like(y_valid, majority_class)
    print_basic_stats(y_valid, y_valid_maj, split_name="VALID (majority baseline)")

    # -------- TF-IDF vectorization (this is the slow part) --------
    print("\nFitting TF-IDF vectorizer (this may take a while)...")
    vectorizer = TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_tf = vectorizer.fit_transform(X_train)
    X_valid_tf = vectorizer.transform(X_valid)
    X_test_tf  = vectorizer.transform(X_test)

    print("TF-IDF shapes:")
    print("  Train:", X_train_tf.shape)
    print("  Valid:", X_valid_tf.shape)
    print("  Test: ", X_test_tf.shape)

    # -------- Logistic Regression training --------
    print("\nTraining Logistic Regression model (may also take some time)...")
    clf = LogisticRegression(
        max_iter=200,
        n_jobs=-1,
        verbose=1
    )
    clf.fit(X_train_tf, y_train)

    # -------- Evaluation --------
    y_valid_pred = clf.predict(X_valid_tf)
    y_valid_proba = clf.predict_proba(X_valid_tf)[:, 1]
    print_basic_stats(y_valid, y_valid_pred, y_valid_proba, split_name="VALID (LogReg)")

    y_test_pred = clf.predict(X_test_tf)
    y_test_proba = clf.predict_proba(X_test_tf)[:, 1]
    print_basic_stats(y_test, y_test_pred, y_test_proba, split_name="TEST (LogReg)")

    # -------- Save model + vectorizer for later --------
    print(f"\nSaving vectorizer to {VECTORIZER_FILE}")
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print(f"Saving model to {MODEL_FILE}")
    joblib.dump(clf, MODEL_FILE)

    print("\nDone. You now have:")
    print(f"- {VECTORIZER_FILE} (TF-IDF features)")
    print(f"- {MODEL_FILE} (trained Logistic Regression)")
    print("Use these later in your notebook + presentation.")

if __name__ == "__main__":
    main()
