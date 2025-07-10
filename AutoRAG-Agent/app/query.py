
import numpy as np

def search_query(query, model, faiss_index, text_chunks, top_k=3):
    """
    Perform semantic search for a user query against the FAISS index.

    Returns top_k most relevant text chunks.
    """
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(text_chunks):
            results.append(text_chunks[idx])
    return results
