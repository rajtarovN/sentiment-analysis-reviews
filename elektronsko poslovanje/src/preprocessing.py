import pandas as pd
import re
from datasets import load_dataset

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    print("Loading dataset")
    dataset = load_dataset("fancyzhx/amazon_polarity")

    train_data = dataset["train"]
    df = pd.DataFrame(train_data[:20000])
    df_pos = df[df["label"] == 1].sample(2500, random_state=42)
    df_neg = df[df["label"] == 0].sample(2500, random_state=42)
    df_balanced = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_balanced["text"] = df_balanced["title"] + " " + df_balanced["content"]
    df_balanced["clean_text"] = df_balanced["text"].apply(clean_text)
    df_balanced[["clean_text", "label"]].to_csv("../data/reviews_clean.csv", index=False)
    print("Finished")
if __name__ == "__main__":
    main()