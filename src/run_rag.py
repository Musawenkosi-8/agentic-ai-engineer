from src.ingestor import ingest_source
from src.vector_store import add_to_memory
from src.rag_engine import smart_rag

import os


SOURCE = "data/agentic_ai.pdf"


def index_document(source: str):

    print("=" * 60)
    print("📄 Loading document...")
    print("=" * 60)


    chunks = ingest_source(source)


    if not chunks:

        print(
            "❌ No chunks created."
        )

        return False


    print(
        f"✅ Created {len(chunks)} chunks."
    )


    print(
        "\n🧠 Adding chunks to ChromaDB memory..."
    )


    for index, chunk in enumerate(chunks):

        metadata = chunk.metadata.copy()

        metadata["chunk"] = index


        add_to_memory(

            text=chunk.page_content,

            doc_id=(
                f"{os.path.basename(source)}_{index}"
            ),

            metadata=metadata,

            priority="Medium"
        )


    print(
        "✅ Document indexed successfully."
    )


    return True



def main():

    success = index_document(
        SOURCE
    )


    if not success:

        return


    print("\n" + "=" * 60)
    print("🤖 Agentic RAG Ready")
    print("=" * 60)


    while True:

        question = input(
            "\nAsk a question (type exit): "
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



if __name__ == "__main__":

    main()