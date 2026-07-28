from src.ingestor import process_document
from src.vector_store import add_to_memory


def index_document(path):

    chunks = process_document(path)


    for i, chunk in enumerate(chunks):

        add_to_memory(
            text=chunk.page_content,
            doc_id=f"doc_{i}",
            metadata={
                "source": path,
                "chunk": i
            },
            priority="Medium"
        )


if __name__ == "__main__":

    index_document(
        "data/python.pdf"
    )