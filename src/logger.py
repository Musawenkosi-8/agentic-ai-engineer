import logging
import sys
import os

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout), # console output
        logging.FileHandler("logs/agent_execution.log")
    ]
)

logger = logging.getLogger("AgenticResearcher")