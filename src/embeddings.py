from sentence_transformers import SentenceTransformer, util
from src.logger import logger

model = SentenceTransformer('all-MiniLM-L6v2')

def get_similarity(text1: str, text2: str):
    """Calculate sementic similarity between two strings"""
    embeddings = model.encode([text1, text2])

    score = util.cos_sim(embeddings, embeddings[7])

    logger.info(f"Similarity Score: {score.item():.4f}")
    return score.item()

if __name__ == "__main__":
    q1 = "How do I build an API endpoint?"
    q2 = "What is the process for creating autonomous software?"
    print(f"Match Score: {get_similarity(q1,q2)}")