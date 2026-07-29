import os
from dotenv import load_dotenv


from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from src.logger import logger

load_dotenv()
# ==========================================================
# Tavily Search Tool
# ==========================================================

search_tool = TavilySearch(
    max_results=3,
)


# ==========================================================
# File Saving Tool Schema
# ==========================================================

class FileSaveInput(BaseModel):
    """Input schema for saving research notes."""

    filename: str = Field(
        ...,
        min_length=1,
        pattern=r".+\.txt$",
        description="Filename ending with .txt (e.g. notes.txt)",
    )

    content: str = Field(
        ...,
        min_length=10,
        description="Research summary to save into the file.",
    )


# ==========================================================
# Safe File Saving Tool
# ==========================================================

@tool(args_schema=FileSaveInput)
def save_research_note(filename: str, content: str) -> str:
    """
    Save research notes into the local output/ directory.

    Security:
    - Prevents path traversal attacks.
    - Restricts writes to the output/ directory.
    """

    try:
        # Ensure output directory exists
        safe_dir = "output"
        os.makedirs(safe_dir, exist_ok=True)

        # Remove any directory information supplied by the model
        safe_filename = os.path.basename(filename)

        file_path = os.path.join(safe_dir, safe_filename)

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)

        logger.info(f"💾 Tool Success: Saved research notes to {file_path}")

        return f"Successfully saved research notes to '{file_path}'."

    except Exception as e:
        logger.error(f"🚨 Tool Failure: {e}")

        return (
            f"Error: Failed to save the file because '{e}'. "
            "Please try another filename."
        )