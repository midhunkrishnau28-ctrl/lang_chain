
"""Module for handling retrieval functionality."""

import os
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma  # Changed to Chroma

from src.indexing import create_embeddings


def create_vector_store(
    documents: List[Document],
    embeddings: Optional[Embeddings] = None,
    persist_directory: str = "./data/chroma_db",  # Added persistence directory parameter
) -> VectorStore:
    """
    Create and populate a vector store.

    Args:
        documents: Document chunks to add to the vector store.
        embeddings: The embedding model. If not specified, a default model is created.
        persist_directory: The directory where the Chroma database is persisted.

    Returns:
        A vector store populated with documents.
    """
    if embeddings is None:
        embeddings = create_embeddings()

    # Ensure the persistence directory exists
    os.makedirs(persist_directory, exist_ok=True)

    # Create a Chroma vector database and add documents
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # Persist the database to disk
    vector_store.persist()

    return vector_store


def load_vector_store(
    persist_directory: str = "./data/chroma_db",
    embeddings: Optional[Embeddings] = None,
) -> VectorStore:
    """
    Load an existing vector database.

    Args:
        persist_directory: The directory where the Chroma database is persisted.
        embeddings: The embedding model. If not specified, a default model is created.

    Returns:
        The loaded vector database.
    """
    if embeddings is None:
        embeddings = create_embeddings()

    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"Vector database directory does not exist: {persist_directory}"
        )

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


def retrieve_documents(
    vector_store: VectorStore,
    query: str,
    top_k: int = 4,
    filter_dict: Optional[Dict] = None,
) -> List[Document]:
    """
    Retrieve documents from the vector store based on a query.

    Args:
        vector_store: The vector store to search.
        query: The query text.
        top_k: The number of documents to retrieve.
        filter_dict: Optional filtering conditions.

    Returns:
        A list of retrieved documents.
    """
    if filter_dict:
        docs = vector_store.similarity_search(
            query, k=top_k, filter=filter_dict
        )
    else:
        docs = vector_store.similarity_search(query, k=top_k)

    return docs

