import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = "data/processed"
INDEX_FILE = "data/processed/faiss.index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # lightweight and fast

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

# Read all chunks
chunk_texts = []
chunk_files = []
for filename in os.listdir(PROCESSED_DIR):
    if filename.endswith(".txt") and "chunk" in filename:
        path = os.path.join(PROCESSED_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            chunk_texts.append(text)
            chunk_files.append(path)

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, INDEX_FILE)

# Save mapping for retrieval
import json
with open("data/processed/chunk_mapping.json", "w", encoding="utf-8") as f:
    json.dump(chunk_files, f)

print(f"FAISS index built with {len(chunk_texts)} chunks and saved to {INDEX_FILE}")
