# preprocess_beauty.py
import pandas as pd

INPUT_FILE = "bp.csv"  # your 1% sample
OUTPUT_FILE = "df_ready.csv"          # cleaned + labeled dataset

def main():
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)

    # Drop rows with missing text or rating
    df = df.dropna(subset=["text", "rating"])

    # Basic text cleaning
    df["text"] = df["text"].astype(str).str.strip().str.lower()
    df["title"] = df["title"].fillna("").astype(str).str.strip().str.lower()

    # Create binary sentiment label: 1 if rating >= 4, else 0
    df["label"] = (df["rating"] >= 4).astype(int)

    # Remove super-short reviews (optional but helpful)
    df = df[df["text"].str.len() > 10]

    print(f"Remaining rows after cleaning: {len(df)}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cleaned data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
