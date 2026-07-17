from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from src.schemas import ResearchRequest # Import the gatekeeper/ real-time validation
from src.async_researcher import run_concurrent_research, client
from src.logger import logger
import asyncio

app = FastAPI(title="Production-Grade Research Gateway")

async def groq_stream_generator(topic: str):
    """Generator function with internal error recovery logic"""
    try:
        logger.info(f"Initializing stream for topic: {topic}")
        stream = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"Provide a bief expert analysis on: {topic}"}],
            stream=True
        )
        async for chunk in stream:

            content = chunk.choices[0].delta.content

            if content:
                yield content           
            

        logger.info(f"Stream successfully completed for: {topic}")

    except Exception as e:
        logger.error(f"MID-STREAM FAILURE for '{topic}': {str(e)}")
        yield f"\n\n[SYSTEM ERROR]: Reasoning loop interrupted. Details logged. {str(e)}"

@app.post("/stream-research")
async def stream_research(request: ResearchRequest):
    """Secure, validated POST endpoint with full traceability"""
    logger.info(f"Processing validated research for: {request.topic}")
    if not client.api_key:
        logger.critical(" Missing API Configuration: GROQ_API_KEY is not set!")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Agent core is misconfigured. Please check environment secrets.")
    return StreamingResponse(groq_stream_generator(request.topic),
                             media_type="text/plain")