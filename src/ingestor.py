from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.logger import logger
import os


def process_document(source: str, is_web: bool = False):
    """
    Ingests and splits documents with a Senior Mindset:
    - Supports multiple data sources
    - Preserves context integrity
    - Handles ingestion failures safely
    """

    try:

        # 1. Flexible Loading
        if is_web:
            logger.info(f"🌐 Loading web content from: {source}")
            loader = WebBaseLoader(source)

        else:
            extension = os.path.splitext(source)[1].lower()

            if extension == ".pdf":
                logger.info(f"📄 Loading PDF: {source}")
                loader = PyPDFLoader(source)

            elif extension == ".csv":
                logger.info(f"📊 Loading CSV: {source}")
                loader = CSVLoader(source)

            else:
                raise ValueError(
                    f"Unsupported file type: {extension}"
                )


        docs = loader.load()


        # 2. Strategic Splitting
        #
        # RecursiveCharacterTextSplitter preserves natural boundaries:
        # paragraph -> sentence -> word -> character
        #
        # For technical documentation, semantic chunking can sometimes
        # outperform character splitting because it groups text based
        # on meaning rather than arbitrary character length.
        #
        # Example:
        # SemanticChunker analyzes embeddings and attempts to keep
        # conceptually related sentences together.

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )


        chunks = text_splitter.split_documents(docs)


        logger.info(
            f"✅ Processed {len(docs)} documents into {len(chunks)} chunks."
        )

        return chunks


    except Exception as e:

        logger.error(
            f"🚨 Ingestion Failure for {source}: {e}"
        )

        return []


if __name__ == "__main__":

    url = "https://python.langchain.com/docs/introduction/"

    chunks = process_document(
        url,
        is_web=True
    )


    if chunks:
        print(
            "First Chunk Preview:"
        )

        print(
            chunks[0].page_content[:500]
        )