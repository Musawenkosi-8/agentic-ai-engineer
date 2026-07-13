# Async Multi-Agent Research System 🤖

An Agentic AI research system built with Python, AsyncIO, and Groq LLMs.  
The system uses multiple specialized AI agents running concurrently to analyze a topic from different perspectives and generate a consolidated research report.

## Overview

This project demonstrates a production-style Agentic AI architecture where multiple specialized agents collaborate to solve a research task.

Instead of using a single LLM call, the system creates multiple expert agents:

- Economist Agent
- Technologist Agent
- Ethicist Agent

Each agent analyzes the same topic from a different domain perspective. Async programming allows these agents to run concurrently, reducing execution time.

## Architecture
             Research Topic
                   |
                   v
          Async Agent Orchestrator
                   |
    --------------------------------
    |              |               |
    v              v               v
    Economist Technologist Ethicist
Agent Agent Agent
| | |
--------------------------------
|
v
Consolidated Report

## Features

✅ Multi-agent architecture  
✅ Asynchronous execution using Python AsyncIO  
✅ Concurrent LLM calls  
✅ Groq Llama model integration  
✅ Environment-based API key management  
✅ Production-style Python project structure  

## Project Structure
agentic-ai-project/

├── src/
│ ├── init.py
│ ├── async_researcher.py
│ ├── config.py
│ └── utils.py
│
├── notebooks/
│
├── tests/
│
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md


## Technologies Used

- Python 3.10+
- AsyncIO
- Groq API
- Llama 3.1 model
- python-dotenv

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd agentic-ai-project

2. Create virtual environment
python -m venv .venv

Activate:

Windows:

.venv\Scripts\activate

Linux/Mac:

source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Configure environment variables

Create a .env file:

GROQ_API_KEY=your_api_key_here
5. Run the agent
python -m src.async_researcher
Example Output
Launching 3 agents concurrently...

ECONOMIST:
Analysis of nuclear fusion from an economic perspective...

TECHNOLOGIST:
Analysis of nuclear fusion technology...

ETHICIST:
Analysis of ethical implications...

Total time with Asyncio: 3.42 seconds
Future Improvements
Add a synthesizer agent for final report generation
Add memory using vector databases
Implement LangGraph workflow orchestration
Add tool calling capabilities
Deploy with FastAPI and Docker
Add automated testing

Author
Musawenkosi Nyawo

Building Agentic AI systems, LLM applications, and intelligent automation solutions.