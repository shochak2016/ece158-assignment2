import json
import random
import pandas as pd

def main():
    rows = []
    sample_prob = 0.01

    with open("BP.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if random.random() < sample_prob:
                data = json.loads(line)
                rows.append({
                    "text": data.get("text", ""),
                    "title": data.get("title", ""),
                    "rating": data.get("rating"),
                    "user_id": data.get("user_id"),
                    "asin": data.get("asin"),
                })

    df = pd.DataFrame(rows)
    df.to_csv("bp.csv", index=False)

if __name__ == "__main__":
    main()