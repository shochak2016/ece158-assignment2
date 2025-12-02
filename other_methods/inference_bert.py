import csv
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MODEL_NAME = "distilbert-base-uncased"

WEIGHTS = {
    "A": 0.8,
    "B": 0.1,
    "C": 0.1
}


# ============================================================
# Load CSV manually
# ============================================================
def load_csv(path):
    texts = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip()
            rating = float(row["rating"])
            label = 1 if rating >= 4 else 0
            texts.append(text)
            labels.append(label)
    return texts, labels


# ============================================================
# Slice splitting
# ============================================================
def split_text_slices(text):
    words = text.split()
    A = " ".join(words[:100])
    B = " ".join(words[100:200])
    C = " ".join(words[200:])
    return A, B, C


# ============================================================
# Batch inference helper
# ============================================================
def predict_batch(model, tokenizer, texts):
    all_probs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Batch inference"):
        batch = texts[i:i+BATCH_SIZE]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        all_probs.extend(probs)

    return np.array(all_probs)


# ============================================================
# Load trained BERT slice model
# ============================================================
def load_trained_slice(path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


# ============================================================
# MAIN
# ============================================================
def main():
    # Load validation CSV
    print("Loading CSV...")
    texts, labels = load_csv("../data/df_ready.csv")

    # Split into slices
    print("Splitting slices...")
    slicesA = []
    slicesB = []
    slicesC = []

    for t in texts:
        A, B, C = split_text_slices(t)
        slicesA.append(A)
        slicesB.append(B)
        slicesC.append(C)

    # Load trained slice models
    print("Loading fine-tuned models...")
    tokA, modelA = load_trained_slice("bert_A.pt")
    tokB, modelB = load_trained_slice("bert_B.pt")
    tokC, modelC = load_trained_slice("bert_C.pt")

    # Predict
    print("\nPredicting slice A...")
    pA = predict_batch(modelA, tokA, slicesA)

    print("\nPredicting slice B...")
    pB = predict_batch(modelB, tokB, slicesB)

    print("\nPredicting slice C...")
    pC = predict_batch(modelC, tokC, slicesC)

    # Weighted ensemble
    print("\nCombining predictions with weights:", WEIGHTS)
    p_final = (
        WEIGHTS["A"] * pA +
        WEIGHTS["B"] * pB +
        WEIGHTS["C"] * pC
    )

    preds = (p_final >= 0.5).astype(int)

    # Evaluation
    print("\n=== FINAL EVALUATION ===")
    print("Accuracy:", accuracy_score(labels, preds))
    print("ROC-AUC:", roc_auc_score(labels, p_final))
    print(classification_report(labels, preds))


if __name__ == "__main__":
    main()
