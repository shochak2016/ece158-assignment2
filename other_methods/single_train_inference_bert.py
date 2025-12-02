import csv
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler
)
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3
SAVE_PATH = "bert_full.pt"


# ============================================================
# Load CSV (no pandas)
# ============================================================
def load_csv(path):
    texts, labels = [], []
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
# Slice review to first 512 tokens
# ============================================================
def slice_512(text):
    words = text.split()
    return " ".join(words[:512])


# ============================================================
# Dataset class
# ============================================================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ============================================================
# Train a single BERT model
# ============================================================
def train_single_model(train_texts, train_labels, val_texts, val_labels):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_set = ReviewDataset(train_texts, train_labels, tokenizer, max_len=256)
    val_set   = ReviewDataset(val_texts, val_labels, tokenizer, max_len=256)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler("linear", optimizer, 0, total_steps)

    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        # ---------------------------------------
        # TRAIN
        # ---------------------------------------
        model.train()
        for batch in tqdm(train_loader, desc="Training"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # ---------------------------------------
        # VALIDATION
        # ---------------------------------------
        model.eval()
        preds, labs = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits = model(**batch).logits
                pred = logits.argmax(dim=1).cpu().numpy()
                label = batch["labels"].cpu().numpy()

                preds.extend(pred)
                labs.extend(label)

        acc = accuracy_score(labs, preds)
        print("Val accuracy:", acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
            print("Saved best model →", SAVE_PATH)

    return best_acc


# ============================================================
# Inference on full dataset
# ============================================================
def run_inference(texts, labels):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    sliced = [slice_512(t) for t in texts]

    all_probs = []

    for i in tqdm(range(0, len(sliced), BATCH_SIZE), desc="Inference"):
        batch = sliced[i:i+BATCH_SIZE]

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

    preds = (np.array(all_probs) >= 0.5).astype(int)

    print("\n=== FINAL EVALUATION ===")
    print("Accuracy:", accuracy_score(labels, preds))
    print("ROC-AUC:", roc_auc_score(labels, all_probs))
    print(classification_report(labels, preds))


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading CSV...")
    texts, labels = load_csv("../data/df_ready.csv")

    print("Slicing reviews to first 512 tokens...")
    sliced = [slice_512(t) for t in texts]

    # train/val split
    idxs = list(range(len(texts)))
    random.seed(42)
    random.shuffle(idxs)

    split = int(0.8 * len(idxs))
    train_idx = idxs[:split]
    val_idx   = idxs[split:]

    X_train = [sliced[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]

    X_val   = [sliced[i] for i in val_idx]
    y_val   = [labels[i] for i in val_idx]

    print("Training single BERT model...")
    acc = train_single_model(X_train, y_train, X_val, y_val)
    print("Best val accuracy:", acc)

    print("\nRunning inference on full dataset...")
    run_inference(texts, labels)


if __name__ == "__main__":
    main()
