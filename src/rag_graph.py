from langgraph.graph import StateGraph, START, END
from src.checkpointer import checkpointer

from src.graph_state import GraphState
from src.rag_engine import (
    generate_hyde_query,
    retrieve_documents,
    grade_context,
    generate_answer,
    corrective_retrieval,
)

from src.logger import logger


# ======================================================
# HyDE Node
# ======================================================

def hyde_node(state: GraphState):

    logger.info("🧠 HyDE Node")

    question = state["question"]

    hyde_query = generate_hyde_query(question)

    return {
        "hyde_query": hyde_query
    }


# ======================================================
# Retrieval Node
# ======================================================

def retrieve_node(state: GraphState):

    logger.info("📚 Retrieval Node")

    documents, metadata = retrieve_documents(
        state["hyde_query"]
    )

    if not documents:

        logger.warning(
            "No documents retrieved."
        )

        return {
            "documents": [],
            "metadata": [],
            "context": "",
            "context_approved": False
        }


    context = "\n\n".join(documents)

    return {
        "documents": documents,
        "metadata": metadata,
        "context": context
    }

# ======================================================
# Context Grading Node
# ======================================================

def grade_node(state: GraphState):

    logger.info("🧪 Context Grading Node")

    approved = grade_context(
        state["context"],
        state["question"]
    )

    return {
        "context_approved": approved
    }


# ======================================================
# Router
# ======================================================

def grade_router(state: GraphState):

    if state["context_approved"]:

        return "generate"

    return "corrective"


# ======================================================
# Corrective Retrieval
# ======================================================

def corrective_node(state: GraphState):

    logger.info("🔄 Corrective Retrieval Node")

    documents, metadata = corrective_retrieval(
        state["question"]
    )

    context = "\n\n".join(documents)

    approved = grade_context(
        context,
        state["question"]
    )

    return {

        "documents": documents,

        "metadata": metadata,

        "context": context,

        "context_approved": approved
    }


# ======================================================
# Router After CRAG
# ======================================================

def corrective_router(state: GraphState):

    if state["context_approved"]:

        return "generate"

    return "fallback"


# ======================================================
# Answer Generation
# ======================================================

def generate_node(state: GraphState):

    logger.info("✍️ Answer Generation Node")

    answer = generate_answer(

        state["context"],

        state["question"]

    )

    return {

        "answer": answer

    }


# ======================================================
# Safe Fallback
# ======================================================

def fallback_node(state: GraphState):

    logger.warning(
        "🚨 Fallback Node"
    )

    return {

        "answer":
        (
            "I cannot answer this confidently because "
            "my knowledge base does not contain "
            "reliable information."
        )

    }


# ======================================================
# Build Graph
# ======================================================

builder = StateGraph(GraphState)


builder.add_node(

    "hyde",

    hyde_node

)

builder.add_node(

    "retrieve",

    retrieve_node

)

builder.add_node(

    "grade",

    grade_node

)

builder.add_node(

    "corrective",

    corrective_node

)

builder.add_node(

    "generate",

    generate_node

)

builder.add_node(

    "fallback",

    fallback_node

)


builder.add_edge(

    START,

    "hyde"

)

builder.add_edge(

    "hyde",

    "retrieve"

)

builder.add_edge(

    "retrieve",

    "grade"

)


builder.add_conditional_edges(

    "grade",

    grade_router

)


builder.add_conditional_edges(

    "corrective",

    corrective_router

)


builder.add_edge(

    "generate",

    END

)

builder.add_edge(

    "fallback",

    END

)


graph = builder.compile(
    checkpointer=checkpointer
)

def get_graph():
    return graph