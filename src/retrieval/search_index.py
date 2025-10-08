import json
import faiss
from sentence_transformers import SentenceTransformer

INDEX_FILE = "data/processed/faiss.index"
MAPPING_FILE = "data/processed/chunk_mapping.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5  # number of results to retrieve

# Load FAISS index
index = faiss.read_index(INDEX_FILE)

# Load chunk mapping
with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    chunk_files = json.load(f)

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

def search(query, top_k=TOP_K):
    # Encode query
    query_vec = model.encode([query], convert_to_numpy=True)
    
    # Search index
    distances, indices = index.search(query_vec, top_k)
    
    # Retrieve text chunks
    results = []
    for idx in indices[0]:
        path = chunk_files[idx]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        results.append(text)
    return results

if __name__ == "__main__":
    q = input("Enter your query: ")
    top_chunks = search(q)
    print("\nTop results:\n")
    for i, chunk in enumerate(top_chunks):
        print(f"Result {i+1}:\n{chunk[:500]}...\n")  # show first 500 chars
