from pydantic import BaseModel, Field
from typing import List

class MaintenanceAudit(BaseModel):
    """Structured output for the Domain Expert Agent."""
    reasoning: str = Field(..., description="CoT trace explaining the audit findings.")
    severity_level: str = Field(..., pattern="^(Low|Medium|High|Critical)$")
    detected_risks: List[str] = Field(..., min_items=1)
    estimated_cost_usd: float = Field(..., gt=0)