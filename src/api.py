from fastapi import FastAPI, HTTPException
from src.schemas import ResearchRequest # Import your new gatekeeper
from src.async_researcher import run_concurrent_research
from src.logger import logger

app = FastAPI(title="Schema-Validated Research Gateway")

@app.post("/research")
async def research_endpoint(request: ResearchRequest):

    logger.info(
        f"🚀 Topic: {request.topic} | Perspective: {request.perspective}"
    )

    try:
        results = await run_concurrent_research(
            topic=request.topic,
            perspective=request.perspective
        )

        return {
            "status": "success",
            "topic": request.topic,
            "perspective": request.perspective,
            "data": results
        }

    except Exception as e:
        logger.error(f"❌ Execution error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal agent error."
        )