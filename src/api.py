from fastapi import FastAPI, HTTPException, status
from src.async_researcher import run_concurrent_research
from src.logger import logger # Using the logger we defined above

app = FastAPI(title="Reliable Research Gateway")

@app.get("/research")
async def research_endpoint(topic: str):
    if not topic or len(topic) < 3:
        logger.warning(f"Validation failed: Invalid topic received: '{topic}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Topic must be at least 3 characters long."
        )

    logger.info(f" Starting research process for topic: {topic}")
    
    try:
        # Attempt the concurrent LLM calls
        results = await run_concurrent_research(topic)
        logger.info(f" Research completed successfully for: {topic}")
        return {"topic": topic, "research_data": results}

    except Exception as e:
        # Catch unexpected errors (API timeouts, rate limits, etc.)
        logger.error(f"Critical error during research for '{topic}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_AVAILABLE,
            detail="The research agent is currently unavailable. Please try again later."
        )