from datasets import load_dataset
import os

os.makedirs("data/raw", exist_ok=True)

# Load Wikitext-103 raw text (English)
ds = load_dataset("wikitext", "wikitext-103-v1", split="train[:1000]")  # first 1000 lines

for i, item in enumerate(ds):
    text = item['text']
    if text.strip() == "":
        continue
    with open(f"data/raw/wiki_{i}.txt", "w", encoding="utf-8") as f:
        f.write(text)

print("Saved 1000 Wikipedia lines to data/raw/")
