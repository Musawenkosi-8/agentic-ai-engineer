import os
from datetime import datetime, UTC
from urllib.parse import urlparse

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    WebBaseLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.logger import logger


load_dotenv()


USER_AGENT = os.getenv(
    "USER_AGENT",
    "AgenticAIResearchBot/1.0"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def _get_filename(source: str):
    """
    Generates a readable filename/source identifier.
    """

    if source.startswith(("http://", "https://")):
        parsed_url = urlparse(source)

        return parsed_url.netloc

    return os.path.basename(source)


def _get_loader(source: str):
    """
    Returns the correct LangChain loader based on source type.
    """

    if source.startswith(("http://", "https://")):

        logger.info(
            f"🌐 Loading website: {source}"
        )

        return (
            WebBaseLoader(
                source,
                header_template={
                    "User-Agent": USER_AGENT
                }
            ),
            "web"
        )


    extension = os.path.splitext(source)[1].lower()


    if extension == ".pdf":

        logger.info(
            f"📄 Loading PDF: {source}"
        )

        return (
            PyPDFLoader(source),
            "pdf"
        )


    if extension == ".csv":

        logger.info(
            f"📊 Loading CSV: {source}"
        )

        return (
            CSVLoader(source),
            "csv"
        )


    raise ValueError(
        f"Unsupported source type: {extension}"
    )


def ingest_source(source: str):
    """
    Loads, enriches metadata, and splits documents.

    Supported:
    - PDF
    - CSV
    - Website

    Returns:
        List of LangChain Document chunks
    """

    try:

        loader, source_type = _get_loader(source)


        docs = loader.load()


        if not docs:

            logger.warning(
                f"No documents loaded from {source}"
            )

            return []


        logger.info(
            f"Loaded {len(docs)} document(s) from {source_type.upper()} source."
        )


        # Metadata enrichment happens here
        # because docs exists at this point

        for doc in docs:

            doc.metadata["source_type"] = source_type

            doc.metadata["source"] = source

            doc.metadata["ingested_at"] = (
                datetime.now(UTC).isoformat()
            )

            if not doc.metadata.get("filename"):

                doc.metadata["filename"] = (
                    _get_filename(source)
                )


        splitter = RecursiveCharacterTextSplitter(

            chunk_size=CHUNK_SIZE,

            chunk_overlap=CHUNK_OVERLAP,

            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )


        chunks = splitter.split_documents(docs)


        logger.info(
            f"Created {len(chunks)} chunks from {len(docs)} document(s)."
        )


        return chunks


    except Exception as e:

        logger.exception(
            f"Ingestion failed for '{source}': {e}"
        )

        return []


if __name__ == "__main__":

    source = (
        "https://python.langchain.com/docs/introduction/"
    )


    chunks = ingest_source(source)


    if chunks:

        print("\nFirst Chunk\n")

        print(
            chunks[0].page_content[:500]
        )


        print("\nMetadata\n")

        print(
            chunks[0].metadata
        )