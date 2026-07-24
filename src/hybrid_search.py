from rank_bm25 import BM25Okapi
from src.faiss_retriever import build_faiss_index, search_faiss
from src.embeddings import model
from src.logger import logger
import numpy as np
from typing import List, Tuple, Dict
import math

class HybridRetriever:
    def __init__(self, texts: List[str]):
        """
        Initialize the Hybrid Retriever with BM25 (keyword) and FAISS (vector) search.
        
        Args:
            texts: List of documents to index
        """
        self.texts = texts
        self.doc_id_map = {doc: idx for idx, doc in enumerate(texts)}
        
        # 1. Initialize BM25 (Keyword Search)
        logger.info("Initializing BM25 index...")
        tokenized_corpus = [doc.split(" ") for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 initialized with {len(texts)} documents")
        
        # 2. Initialize FAISS (Vector Search)
        logger.info("Initializing FAISS index...")
        self.vector_index = build_faiss_index(texts)
        logger.info(f"FAISS initialized with {len(texts)} documents")

    def keyword_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform keyword-based search using BM25.
        
        Returns:
            List of (document, score) tuples
        """
        tokenized_query = query.split(" ")
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k documents with their scores
        top_indices = np.argsort(scores)[-k:][::-1]
        results = [(self.texts[idx], scores[idx]) for idx in top_indices if scores[idx] > 0]
        
        return results

    def vector_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform semantic/vector-based search using FAISS.
        
        Returns:
            List of (document, distance) tuples (lower distance = better match)
        """
        # search_faiss returns documents, modify to get distances too
        query_embedding = self._get_query_embedding(query)
        distances, indices = self.vector_index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], distances[0][i]))
        
        return results

    def reciprocal_rank_fusion(self, 
                              keyword_results: List[Tuple[str, float]], 
                              vector_results: List[Tuple[str, float]], 
                              k: int = 60) -> List[Tuple[str, float]]:
        """
        Apply Reciprocal Rank Fusion (RRF) to combine results from multiple retrievers.
        
        RRF Formula: score = sum(1 / (k + rank)) for each document across all result lists
        
        Why RRF helps:
        - When keyword search and vector search give different "top 1" results,
          RRF prevents either system from dominating the final ranking.
        - It normalizes scores across different retrieval methods (BM25 scores vs FAISS distances).
        - A document that appears at position 1 in one list gets high score even if absent in the other.
        - This is particularly important when keyword search finds an exact match (GHOST-99) 
          but vector search finds a semantically related document.
        
        Args:
            keyword_results: List of (document, score) from keyword search
            vector_results: List of (document, distance) from vector search
            k: Constant to prevent division by zero (typically 60)
        
        Returns:
            List of (document, fusion_score) sorted by score descending
        """
        # Create a dictionary to accumulate RRF scores
        fusion_scores = {}
        
        # Process keyword results (rank 1 = highest score)
        for rank, (doc, score) in enumerate(keyword_results, start=1):
            if doc not in fusion_scores:
                fusion_scores[doc] = 0
            fusion_scores[doc] += 1 / (k + rank)
        
        # Process vector results (rank 1 = smallest distance = best match)
        for rank, (doc, distance) in enumerate(vector_results, start=1):
            if doc not in fusion_scores:
                fusion_scores[doc] = 0
            fusion_scores[doc] += 1 / (k + rank)
        
        # Sort by fusion score descending
        sorted_results = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Helper method to get query embedding for FAISS."""
        from src.embeddings import model
        return model.encode([query]).astype('float32')

    def search(self, 
               query: str, 
               k: int = 3, 
               use_rrf: bool = True,
               keyword_weight: float = 0.5,
               vector_weight: float = 0.5) -> List[str]:
        """
        Perform hybrid search combining keyword and vector retrieval.
        
        Args:
            query: Search query string
            k: Number of results to return
            use_rrf: If True, use Reciprocal Rank Fusion for merging
                    If False, use simple union
            keyword_weight: Weight for keyword scores (only used with weighted fusion)
            vector_weight: Weight for vector scores (only used with weighted fusion)
        
        Returns:
            List of unique documents sorted by relevance
        """
        logger.info(f"Running Hybrid Search for: '{query}'")
        
        # Get results from both retrievers
        keyword_results = self.keyword_search(query, k=5)  # Get more results for fusion
        vector_results = self.vector_search(query, k=5)
        
        logger.info(f"Keyword search found {len(keyword_results)} results")
        logger.info(f"Vector search found {len(vector_results)} results")
        
        if use_rrf:
            # Use Reciprocal Rank Fusion for optimal combination
            logger.info("Applying Reciprocal Rank Fusion (RRF)")
            fused_results = self.reciprocal_rank_fusion(keyword_results, vector_results)
            
            # Return top k documents
            final_results = [doc for doc, score in fused_results[:k]]
            
        else:
            # Simple union (original method)
            keyword_docs = [doc for doc, _ in keyword_results]
            vector_docs = [doc for doc, _ in vector_results]
            combined = list(set(keyword_docs + vector_docs))
            final_results = combined[:k]
        
        logger.info(f"Hybrid search returned {len(final_results)} unique matches.")
        
        # Log detailed results for debugging
        for i, doc in enumerate(final_results, 1):
            logger.info(f"   {i}. {doc[:100]}...")
        
        return final_results

    def explain_search(self, query: str, k: int = 3) -> Dict:
        """
        Detailed explanation of search results showing contribution from each method.
        """
        keyword_results = self.keyword_search(query, k=5)
        vector_results = self.vector_search(query, k=5)
        
        # Get RRF scores
        fused_results = self.reciprocal_rank_fusion(keyword_results, vector_results)
        
        explanation = {
            'query': query,
            'keyword_results': keyword_results[:k],
            'vector_results': vector_results[:k],
            'fused_results': fused_results[:k],
            'keyword_count': len(keyword_results),
            'vector_count': len(vector_results)
        }
        
        return explanation


def run_vibe_check_challenge():
    """
    Run the "Vibe Check" Challenge tests.
    """
    logger.info("\n" + "="*60)
    logger.info("VIBE CHECK CHALLENGE")
    logger.info("="*60 + "\n")
    
    # Create knowledge base with unique code and semantically related documents
    kb = [
        "The part number for the motor is AX-500.",
        "How to repair an electric engine.",
        "Agentic AI is built on LangChain and FastAPI.",
        "FastAPI is a modern web framework for Python.",
        # Secret project document with unique code
        "Secret Project: GHOST-99 is a stealth initiative for autonomous surveillance.",
        "Stealth technology involves reducing detection across multiple domains.",
        "Autonomous systems require advanced sensor fusion and AI.",
        "GHOST-99 uses advanced AI for covert operations.",
        "The project focuses on undetectable intelligence gathering."
    ]
    
    retriever = HybridRetriever(kb)
    
    print("\n" + "="*60)
    print("KNOWLEDGE BASE")
    print("="*60)
    for i, doc in enumerate(kb, 1):
        print(f"{i}. {doc}")
    
    # Test 1: Precision Test - Find unique code
    print("\n" + "="*60)
    print("TEST 1: PRECISION TEST (GHOST-99)")
    print("="*60)
    
    precision_query = "GHOST-99"
    print(f"\nQuery: '{precision_query}'")
    
    # Show detailed search explanation
    explanation = retriever.explain_search(precision_query)
    print(f"\nSearch Explanation:")
    print(f"   Keyword Results ({len(explanation['keyword_results'])}):")
    for doc, score in explanation['keyword_results']:
        print(f"     - BM25 Score: {score:.3f} | {doc[:80]}...")
    print(f"\n   Vector Results ({len(explanation['vector_results'])}):")
    for doc, dist in explanation['vector_results']:
        print(f"     - Distance: {dist:.3f} | {doc[:80]}...")
    
    # Get final results
    results = retriever.search(precision_query, k=3, use_rrf=True)
    
    print(f"\nFinal Results:")
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc}")
    
    # Verify GHOST-99 is found
    ghost_found = any("GHOST-99" in doc for doc in results)
    print(f"\nGHOST-99 Found: {ghost_found}")
    assert ghost_found, "Precision Test Failed: GHOST-99 not found!"
    print("Precision Test PASSED!")
    
    # Test 2: Semantic Test - Find GHOST-99 via semantics
    print("\n" + "="*60)
    print("TEST 2: SEMANTIC TEST (stealth initiatives)")
    print("="*60)
    
    semantic_query = "stealth initiatives"
    print(f"\nQuery: '{semantic_query}'")
    
    # Show detailed search explanation
    explanation = retriever.explain_search(semantic_query)
    print(f"\nSearch Explanation:")
    print(f"   Keyword Results ({len(explanation['keyword_results'])}):")
    for doc, score in explanation['keyword_results']:
        print(f"     - BM25 Score: {score:.3f} | {doc[:80]}...")
    print(f"\n   Vector Results ({len(explanation['vector_results'])}):")
    for doc, dist in explanation['vector_results']:
        print(f"     - Distance: {dist:.3f} | {doc[:80]}...")
    
    # Get final results
    results = retriever.search(semantic_query, k=3, use_rrf=True)
    
    print(f"\nFinal Results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc}")
    
    # Verify GHOST-99 is found via semantic search
    ghost_found_semantic = any("GHOST-99" in doc for doc in results)
    print(f"\nGHOST-99 Found via Semantic Search: {ghost_found_semantic}")
    assert ghost_found_semantic, "Semantic Test Failed: GHOST-99 not found via 'stealth initiatives'!"
    print("Semantic Test PASSED!")
    
    # Test 3: Compare with/without RRF
    print("\n" + "="*60)
    print("TEST 3: RRF COMPARISON")
    print("="*60)
    
    test_query = "autonomous surveillance"
    print(f"\nQuery: '{test_query}'")
    
    print("\nWith RRF:")
    results_rrf = retriever.search(test_query, k=3, use_rrf=True)
    for i, doc in enumerate(results_rrf, 1):
        print(f"   {i}. {doc}")
    
    print("\nWithout RRF (Simple Union):")
    results_no_rrf = retriever.search(test_query, k=3, use_rrf=False)
    for i, doc in enumerate(results_no_rrf, 1):
        print(f"   {i}. {doc}")
    
    # Explain RRF benefits
    print("\n" + "="*60)
    print("RRF EXPLANATION")
    print("="*60)
    print("""
    Reciprocal Rank Fusion (RRF) helps when keyword and vector search give different top results:
    
    1. Example Scenario:
       - Keyword search: "GHOST-99" → exact match at rank 1
       - Vector search: "stealth initiatives" → semantically related docs at rank 1
       
    2. Without RRF:
       - Simple union might prioritize one system's results
       - Could miss documents from the other system if their scores aren't comparable
       
    3. With RRF:
       - Score = sum(1/(60 + rank)) for each result list
       - A document that's rank 1 in EITHER list gets high score
       - No need to normalize different scoring systems (BM25 scores vs FAISS distances)
       - Ensures both exact matches AND semantic matches are considered
       
    4. Why RRF works:
       - Democratizes the ranking process
       - Prevents any single retriever from dominating
       - Particularly important when queries contain both specific codes AND general concepts
    """)
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)

