from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse

from src.schemas import ResearchRequest
from src.async_researcher import run_concurrent_research, client
from src.logger import logger


app = FastAPI(
    title="Production-Grade Research Gateway"
)


@app.get("/")
async def root():
    return {
        "status": "Online",
        "agent_version": "1.0.0"
    }


@app.post("/research")
async def research(request: ResearchRequest):

    logger.info(
        f"Starting research for: {request.topic}"
    )

    try:
        results = await run_concurrent_research(
            request.topic
        )

        return {
            "topic": request.topic,
            "answer": results
        }

    except Exception as e:

        logger.error(
            f"Research failed: {str(e)}"
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research agent unavailable"
        )


async def groq_stream_generator(topic: str):

    stream = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"Research the impact of {topic}"
            }
        ],
        stream=True
    )


    async for chunk in stream:

        content = chunk.choices[0].delta.content

        if content:
            yield content



@app.post("/stream-research")
async def stream_research(request: ResearchRequest):

    return StreamingResponse(
        groq_stream_generator(request.topic),
        media_type="text/plain"
    )