from pydantic import BaseModel, Field
from typing import Literal

class ResearchRequest(BaseModel):
    """Schema for a research task request."""

    topic: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="The subject to be researched by the agent."
    )

    max_analysts: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of expert agents to launch concurrently."
    )

    perspective: Literal[
        "economic",
        "historical",
        "technological",
        "political"
    ] = Field(
        default="economic",
        description="Research perspective to guide the analysis."
    )