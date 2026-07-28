from typing import TypedDict


class GraphState(TypedDict):
    """
    Shared state passed between LangGraph nodes.

    Each node reads from and writes to this state.
    """

    # User input
    question: str

    # HyDE-generated retrieval query
    hyde_query: str

    # Retrieved document chunks
    documents: list[str]

    # Retrieved metadata
    metadata: list[dict]

    # Combined context passed to the LLM
    context: str

    # Result of context grading
    context_approved: bool

    # Final generated answer
    answer: str