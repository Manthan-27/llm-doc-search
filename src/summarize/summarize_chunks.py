from transformers import pipeline

from src.retrieval.search_index import search

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

if __name__ == "__main__":
    query = input("Enter your query: ")
    top_chunks = search(query, top_k=5)

    # Combine top chunks into one text
    combined_text = " ".join(top_chunks)

    # Summarize (adjust max_length and min_length)
    summary = summarizer(combined_text, max_length=200, min_length=50, do_sample=False)
    print("\nSummary:\n")
    print(summary[0]['summary_text'])
