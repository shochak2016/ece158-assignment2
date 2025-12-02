import csv
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 3
MODEL_NAME = "distilbert-base-uncased"


# ============================================================
# Load CSV manually (NO PANDAS)
# ============================================================
def load_csv_text_and_labels(path):
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
# Dataset class (from earlier)
# ============================================================
class ReviewSliceDataset(Dataset):
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
# Training function for each slice model (A/B/C)
# ============================================================
def train_slice_model(texts_train, labels_train, texts_val, labels_val, save_path):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_set = ReviewSliceDataset(texts_train, labels_train, tokenizer)
    val_set   = ReviewSliceDataset(texts_val, labels_val, tokenizer)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler("linear", optimizer, 0, total_steps)

    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
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

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits = model(**batch).logits
                preds = logits.argmax(dim=1)
                correct += (preds == batch["labels"]).sum().item()
                total += preds.size(0)

        acc = correct / total
        print(f"Val accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model → {save_path}")

    return best_acc


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading CSV...")
    texts, labels = load_csv_text_and_labels("../data/df_ready.csv")

    print("Splitting slices...")
    slicesA, slicesB, slicesC = [], [], []
    for t in texts:
        A, B, C = split_text_slices(t)
        slicesA.append(A)
        slicesB.append(B)
        slicesC.append(C)

    # Train/val split (manual)
    idxs = list(range(len(texts)))
    random.seed(42)
    random.shuffle(idxs)

    split = int(len(idxs) * 0.8)
    train_idx = idxs[:split]
    val_idx = idxs[split:]

    # Build training + validation sets for each slice
    A_train = [slicesA[i] for i in train_idx]
    A_val   = [slicesA[i] for i in val_idx]

    B_train = [slicesB[i] for i in train_idx]
    B_val   = [slicesB[i] for i in val_idx]

    C_train = [slicesC[i] for i in train_idx]
    C_val   = [slicesC[i] for i in val_idx]

    y_train = [labels[i] for i in train_idx]
    y_val   = [labels[i] for i in val_idx]

    # Train three models
    print("\nTraining Model A (0–100 words)...")
    accA = train_slice_model(A_train, y_train, A_val, y_val, "bert_A.pt")

    print("\nTraining Model B (100–200 words)...")
    accB = train_slice_model(B_train, y_train, B_val, y_val, "bert_B.pt")

    print("\nTraining Model C (200+ words)...")
    accC = train_slice_model(C_train, y_train, C_val, y_val, "bert_C.pt")

    print("\nDone.")
    print("A accuracy:", accA)
    print("B accuracy:", accB)
    print("C accuracy:", accC)


if __name__ == "__main__":
    main()
