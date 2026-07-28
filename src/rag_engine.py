from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.vector_store import query_memory
from src.logger import logger


# =====================================================
# 1. LLM Configuration
# =====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)


# =====================================================
# 2. HyDE Prompt
# =====================================================

hyde_prompt = ChatPromptTemplate.from_template(
    """
You are an expert knowledge retrieval assistant.

Generate a hypothetical answer that would likely
appear in a knowledge base containing the answer.

The answer will be used only for document retrieval.

Question:
{question}

Return only the hypothetical answer.
"""
)


# =====================================================
# 3. Context Grading Prompt
# =====================================================

grader_prompt = ChatPromptTemplate.from_template(
    """
You are a strict RAG context evaluator.

Determine whether the provided context contains enough
information to answer the user question.

Context:
{context}


Question:
{question}


Respond ONLY with:

YES

or

NO
"""
)


# =====================================================
# 4. Answer Generation Prompt
# =====================================================

answer_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- Do not hallucinate.
- If the answer is missing, say you do not know.


Context:
{context}


Question:
{question}
"""
)


# =====================================================
# 5. HyDE Query Generation
# =====================================================

def generate_hyde_query(question: str):

    logger.info(
        "🧠 Generating HyDE retrieval query..."
    )

    response = llm.invoke(
        hyde_prompt.format(
            question=question
        )
    )

    return response.content



# =====================================================
# 6. Retrieve Documents From Memory
# =====================================================

def retrieve_documents(query: str, n_results: int = 3):

    logger.info(
        "📚 Searching ChromaDB..."
    )

    results = query_memory(
        query,
        n_results=n_results
    )


    documents = results.get(
        "documents",
        [[]]
    )[0]


    metadata = results.get(
        "metadatas",
        [[]]
    )[0]


    return documents, metadata



# =====================================================
# 7. Context Grading
# =====================================================

def grade_context(
    context: str,
    question: str
):

    logger.info(
        "🧪 Evaluating retrieved context..."
    )


    response = llm.invoke(
        grader_prompt.format(
            context=context,
            question=question
        )
    )


    grade = (
        response.content
        .strip()
        .upper()
    )


    logger.info(
        f"Context grade: {grade}"
    )


    return grade == "YES"



# =====================================================
# 8. Generate Final Answer
# =====================================================

def generate_answer(
    context: str,
    question: str
):

    logger.info(
        "✍️ Generating final answer..."
    )


    response = llm.invoke(
        answer_prompt.format(
            context=context,
            question=question
        )
    )


    return response.content



# =====================================================
# 9. CRAG Corrective Retrieval
# =====================================================

def corrective_retrieval(question: str):

    logger.warning(
        "🔄 Starting Corrective RAG retrieval..."
    )


    documents, metadata = retrieve_documents(
        question,
        n_results=5
    )


    return documents, metadata



# =====================================================
# 10. Main Smart RAG Pipeline
# =====================================================

def smart_rag(question: str):

    logger.info(
        f"🔎 User question: {question}"
    )


    try:

        # ---------------------------------
        # Step 1: HyDE Retrieval
        # ---------------------------------

        hyde_query = generate_hyde_query(
            question
        )


        logger.info(
            f"HyDE query generated: {hyde_query[:100]}"
        )


        documents, metadata = retrieve_documents(
            hyde_query
        )


        if not documents:

            logger.warning(
                "No documents retrieved."
            )

            return (
                "I could not find relevant information "
                "in my knowledge base."
            )


        context = "\n\n".join(
            documents
        )


        # ---------------------------------
        # Step 2: Grade Context
        # ---------------------------------

        approved = grade_context(
            context,
            question
        )


        # ---------------------------------
        # Step 3: CRAG fallback
        # ---------------------------------

        if not approved:

            logger.warning(
                "Initial retrieval failed grading."
            )


            documents, metadata = corrective_retrieval(
                question
            )


            if documents:

                context = "\n\n".join(
                    documents
                )


                approved = grade_context(
                    context,
                    question
                )


        # ---------------------------------
        # Step 4: Final Decision
        # ---------------------------------

        if not approved:

            logger.warning(
                "Context rejected after correction."
            )


            return (
                "I cannot answer this confidently because "
                "the retrieved knowledge does not contain "
                "reliable information about this question."
            )


        answer = generate_answer(
            context,
            question
        )


        # ---------------------------------
        # Step 5: Attach Sources
        # ---------------------------------

        sources = []


        for item in metadata:

            if item:

                source = item.get(
                    "filename",
                    item.get(
                        "source",
                        "Unknown"
                    )
                )

                sources.append(source)


        if sources:

            answer += "\n\nSources:\n"

            for source in set(sources):

                answer += f"- {source}\n"


        return answer



    except Exception as e:

        logger.exception(
            f"RAG pipeline failure: {e}"
        )


        return (
            "An error occurred while processing "
            "your request."
        )



# =====================================================
# Manual Testing
# =====================================================

if __name__ == "__main__":


    while True:

        question = input(
            "\nAsk something (exit): "
        )


        if question.lower() == "exit":

            break


        answer = smart_rag(
            question
        )


        print(
            "\nAnswer:"
        )

        print(answer)