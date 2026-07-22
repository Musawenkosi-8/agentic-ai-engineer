import chromadb
from src.embeddings import model  # Reusing your MiniLM model from Day 1
from src.logger import logger

# 1. Initialize the Persistent Client (stores data in the 'data/' folder)
client = chromadb.PersistentClient(path="./chroma_db")

# 2. Create or Get a Collection
collection = client.get_or_create_collection(name="research_knowledge")

def add_to_memory(text: str, doc_id: str, metadata: dict, priority: str = "Medium"):
    """
    Encodes text and stores it in ChromaDB with priority metadata.
    
    Args:
        text: The text to store
        doc_id: Unique identifier for the document
        metadata: Additional metadata dictionary
        priority: Priority level ("High", "Medium", "Low")
    """
    # Add priority to metadata
    metadata["priority"] = priority
    
    embedding = model.encode(text).tolist()
    collection.add(
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata],
        ids=[doc_id]
    )
    logger.info(f"Document {doc_id} added to long-term memory with priority: {priority}")

def query_memory(query_text: str, n_results: int = 2, priority_filter: str = None):
    """
    Searches memory for the most relevant documents.
    
    Args:
        query_text: The search query
        n_results: Number of results to return
        priority_filter: Optional priority level to filter by ("High", "Medium", "Low")
    
    Returns:
        Query results from ChromaDB
    """
    query_embedding = model.encode(query_text).tolist()
    
    # Build the where clause if priority filter is specified
    where_filter = None
    if priority_filter:
        where_filter = {"priority": priority_filter}
        logger.info(f"Filtering by priority: {priority_filter}")
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter  # Apply the metadata filter
    )
    return results

def get_all_documents():
    """Helper function to view all documents in the collection."""
    return collection.get()

def delete_collection():
    """Helper function to delete the collection (use with caution)."""
    try:
        client.delete_collection("research_knowledge")
        logger.info("Collection deleted successfully")
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")

if __name__ == "__main__":
    # Test the persistent storage with different priorities
    
    # Add documents with different priority levels
    print("Adding documents to memory...")
    
    add_to_memory(
        "Agentic AI uses LLMs to perform autonomous tasks.", 
        "id1", 
        {"topic": "AI", "source": "textbook"},
        priority="High"
    )
    
    add_to_memory(
        "Python is a programming language used for data science.", 
        "id2", 
        {"topic": "Programming", "source": "tutorial"},
        priority="Medium"
    )
    
    add_to_memory(
        "ChromaDB is a vector database for AI applications.", 
        "id3", 
        {"topic": "Database", "source": "documentation"},
        priority="High"
    )
    
    add_to_memory(
        "VS Code is a popular code editor.", 
        "id4", 
        {"topic": "Tools", "source": "blog"},
        priority="Low"
    )
    
    print("\n" + "="*50)
    
    # Query without priority filter (returns all)
    print("Query: 'What can an agent do?' (No filter)")
    results = query_memory("What can an agent do?", n_results=3)
    print(f"Results: {results['documents']}")
    print(f"Priorities: {[meta['priority'] for meta in results['metadatas'][0]]}")
    
    print("\n" + "="*50)
    
    # Query with High priority filter only
    print("Query: 'What can an agent do?' (High priority only)")
    high_priority_results = query_memory(
        "What can an agent do?", 
        n_results=3, 
        priority_filter="High"
    )
    print(f"Results: {high_priority_results['documents']}")
    print(f"Priorities: {[meta['priority'] for meta in high_priority_results['metadatas'][0]]}")
    
    print("\n" + "="*50)
    
    # Check persistence - show all documents in collection
    print(" All documents in collection (persistence check):")
    all_docs = get_all_documents()
    for i, doc_id in enumerate(all_docs['ids']):
        print(f"  - {doc_id}: {all_docs['documents'][i][:50]}... (Priority: {all_docs['metadatas'][i]['priority']})")
    
    print("\n" + "="*50)
    print("To test persistence:")
    print("1. Run this script again (without the add_to_memory calls)")
    print("2. The documents should still be in the ./chroma_db folder")
    print("3. Try querying with different priority filters")