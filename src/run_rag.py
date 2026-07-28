from src.ingestor import process_document
from src.vector_store import add_to_memory
from src.rag_engine import smart_rag

import os


PDF_PATH = "data/agentic_ai.pdf"


def index_document(pdf_path: str):

    print("=" * 60)
    print("📄 Loading document...")
    print("=" * 60)

    chunks = process_document(pdf_path)


    if not chunks:
        print("❌ No chunks created.")
        return False


    print(
        f"✅ Created {len(chunks)} chunks."
    )


    print("\n🧠 Adding chunks to ChromaDB memory...")


    for index, chunk in enumerate(chunks):

        add_to_memory(
            text=chunk.page_content,
            doc_id=f"{os.path.basename(pdf_path)}_{index}",
            metadata={
                "source": pdf_path,
                "chunk": index
            },
            priority="Medium"
        )


    print(
        "✅ Document indexed successfully."
    )

    return True



def main():

    success = index_document(
        PDF_PATH
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