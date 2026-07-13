import asyncio
from groq import Groq

from src.config import GROQ_API_KEY
from src.logger import logger

client = Groq(api_key=GROQ_API_KEY)


def ask_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


async def get_opinion(topic, perspective):

    prompt = f"""
    Give a detailed {perspective} perspective on:

    {topic}
    """

    max_retries = 3

    for attempt in range(max_retries):

        try:

            logger.info(
                f"Getting {perspective} opinion. Attempt {attempt + 1}"
            )

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content

        except Exception as e:

            logger.warning(
                f"Attempt {attempt + 1} failed for {perspective}: {e}"
            )

            if attempt == max_retries - 1:

                logger.error(
                    f"Final failure for {perspective}"
                )

                return (
                    f"Error retrieving {perspective} perspective."
                )

            await asyncio.sleep(1)
