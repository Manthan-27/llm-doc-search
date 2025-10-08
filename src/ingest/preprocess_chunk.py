import os
import re

# Install tiktoken if needed for token-based chunking
import tiktoken

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load tiktoken encoding for GPT-3.5/4
enc = tiktoken.get_encoding("cl100k_base")  # adjust if using another model
CHUNK_SIZE = 500  # tokens per chunk

def preprocess(text):
    # remove multiple newlines and whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE):
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i+chunk_size]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

file_count = 0
for filename in os.listdir(RAW_DIR):
    if not filename.endswith(".txt"):
        continue
    with open(os.path.join(RAW_DIR, filename), "r", encoding="utf-8") as f:
        text = f.read()
    text = preprocess(text)
    chunks = chunk_text(text)
    for j, chunk in enumerate(chunks):
        out_file = os.path.join(PROCESSED_DIR, f"{filename[:-4]}_chunk{j}.txt")
        with open(out_file, "w", encoding="utf-8") as f_out:
            f_out.write(chunk)
    file_count += 1

print(f"Processed and chunked {file_count} files into {PROCESSED_DIR}")
