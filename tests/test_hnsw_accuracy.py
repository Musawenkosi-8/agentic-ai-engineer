import faiss
import numpy as np
from src.embeddings import model
import time

def test_accuracy_comparison():
    """Test accuracy comparison between Flat and HNSW indices."""
    
    # Generate small test dataset
    documents = [
        "Python is a programming language",
        "Machine learning uses data to train models",
        "Deep learning is a subset of machine learning",
        "Natural language processing works with text data",
        "Computer vision analyzes images and videos"
    ]
    
    embeddings = model.encode(documents)
    
    # Build indices
    flat_index = faiss.IndexFlatL2(embeddings.shape[1])
    flat_index.add(embeddings.astype('float32'))
    
    hnsw_index = faiss.IndexHNSWFlat(embeddings.shape[1], 16)
    hnsw_index.hnsw.efConstruction = 200
    hnsw_index.add(embeddings.astype('float32'))
    hnsw_index.hnsw.efSearch = 50
    
    # Test query
    query = "What is machine learning?"
    query_embedding = model.encode([query]).astype('float32')
    
    # Search with both indices
    _, flat_indices = flat_index.search(query_embedding, 3)
    _, hnsw_indices = hnsw_index.search(query_embedding, 3)
    
    print("Flat L2 results:", [documents[idx] for idx in flat_indices[0]])
    print("HNSW results:", [documents[idx] for idx in hnsw_indices[0]])
    
    # Check if HNSW found the same results
    match = np.array_equal(flat_indices[0], hnsw_indices[0])
    print(f"\nResults match: {match}")
    if not match:
        print("HNSW provides approximate results (faster but slightly different)")

if __name__ == "__main__":
    test_accuracy_comparison()