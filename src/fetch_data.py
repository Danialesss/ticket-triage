"""Download the Hugging Face customer support ticket dataset, keep English
rows, and reshape it into the text,category format the rest of the pipeline
expects.

Run:
    python -m src.fetch_data
"""
import pandas as pd

URL = "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/main/dataset-tickets-multi-lang-4-20k.csv"
OUT_PATH = "data/support_tickets_en.csv"


def main():
    df = pd.read_csv(URL)
    df = df[df["language"] == "en"].copy()

    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["text"] = (df["subject"] + " " + df["body"]).str.strip()
    df["category"] = df["queue"]

    out = df[["text", "category"]].dropna()
    out = out[out["text"].str.len() > 0]

    out.to_csv(OUT_PATH, index=False)
    print(f"saved {len(out)} rows to {OUT_PATH}")
    print(out["category"].value_counts())


if __name__ == "__main__":
    main()