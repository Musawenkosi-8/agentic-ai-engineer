from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.vector_store import query_memory
from src.logger import logger


# ===========================
# 1. Setup LLM
# ===========================

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)


# ===========================
# 2. HyDE Prompt
# ===========================

hyde_prompt = ChatPromptTemplate.from_template(
    """
You are an expert assistant.

Generate ONE concise hypothetical answer that could help
retrieve information from a knowledge base.

Question:
{question}

Return only the hypothetical answer.
"""
)


# ===========================
# 3. Context Grader Prompt
# ===========================

grader_prompt = ChatPromptTemplate.from_template(
    """
You are a context relevance grader.

Your job is to decide whether the context contains useful
information to answer the question.

Context:
{context}

Question:
{question}

Reply with ONLY:

relevant

or

irrelevant
"""
)


# ===========================
# 4. Answer Generation Prompt
# ===========================

answer_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the question using ONLY the provided context.

If the context does not contain the answer,
say you do not know.

Context:
{context}


Question:
{question}
"""
)


# ===========================
# 5. Smart RAG Pipeline
# ===========================

def smart_rag(question: str):

    logger.info(
        f"🔎 Processing question: {question}"
    )


    # ---------------------------------
    # STEP 1: HyDE Query Generation
    # ---------------------------------

    logger.info(
        "🧠 Generating hypothetical answer..."
    )

    hypothetical_answer = llm.invoke(
        hyde_prompt.format(
            question=question
        )
    ).content


    logger.info(
        f"📝 HyDE Query: {hypothetical_answer}"
    )


    # ---------------------------------
    # STEP 2: ChromaDB Retrieval
    # ---------------------------------

    logger.info(
        "📚 Searching ChromaDB memory..."
    )


    results = query_memory(
        hypothetical_answer,
        n_results=3
    )


    documents = results.get(
        "documents",
        [[]]
    )[0]


    if not documents:

        logger.warning(
            "🚨 No documents retrieved."
        )

        return (
            "My internal knowledge base does not contain "
            "information related to this question."
        )


    context_text = "\n\n".join(
        documents
    )


    logger.info(
        f"Retrieved {len(documents)} documents."
    )


    # ---------------------------------
    # STEP 3: Context Grading
    # ---------------------------------

    logger.info(
        "🧪 Checking context relevance..."
    )


    grade = llm.invoke(
        grader_prompt.format(
            context=context_text,
            question=question
        )
    ).content.strip().lower()


    logger.info(
        f"Context Grade: {grade}"
    )


    # IMPORTANT:
    # Do not use:
    # if "relevant" in grade
    #
    # because:
    # "irrelevant" contains "relevant"


    if grade == "relevant":


        logger.info(
            "✅ Context approved. Generating answer..."
        )


        response = llm.invoke(
            answer_prompt.format(
                context=context_text,
                question=question
            )
        )


        return response.content



    # ---------------------------------
    # STEP 4: Safe Fallback
    # ---------------------------------

    logger.warning(
        "🚨 Context rejected. Preventing hallucination."
    )


    return (
        "I cannot answer this confidently because "
        "my knowledge base does not contain reliable "
        "information about this topic."
    )



# ===========================
# Manual Test
# ===========================

if __name__ == "__main__":

    while True:

        question = input(
            "\nAsk something (type exit): "
        )


        if question.lower() == "exit":
            break


        answer = smart_rag(question)


        print(
            "\nAnswer:"
        )

        print(answer)