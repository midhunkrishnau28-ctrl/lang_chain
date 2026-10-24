
"""Module for handling document indexing."""

import os
from typing import Dict, List, Optional, Union

import bs4
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import get_env_var


def load_document(source: str) -> List[Document]:
    """
    Load document content from a web URL or a local file.

    Args:
        source: The document source, which can be a URL or a file path.

    Returns:
        A list of documents containing the loaded content.
    """
    if source.startswith(("http://", "https://")):
        # Load web page content
        loader = WebBaseLoader(
            web_paths=(source,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=(
                        "post-content",
                        "post-title",
                        "post-header",
                        "article",
                        "content",
                    )
                )
            ),
        )
        docs = loader.load()

    elif source.endswith(".pdf"):
        # Load a PDF file
        loader = PyPDFLoader(source)
        docs = loader.load()

    elif source.endswith((".txt", ".md")):
        # Load a text file
        loader = TextLoader(source)
        docs = loader.load()

    else:
        raise ValueError(f"Unsupported document type: {source}")

    return docs


def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into smaller chunks.

    Args:
        docs: The list of documents to split.
        chunk_size: The maximum size of each chunk in characters.
        chunk_overlap: The number of overlapping characters between chunks.

    Returns:
        A list of split document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    doc_chunks = text_splitter.split_documents(docs)

    return doc_chunks


def create_embeddings() -> Embeddings:
    """
    Create a text embedding model.

    Returns:
        The embedding model.
    """
    model_name = get_env_var(
        "EMBEDDING_MODEL",
        "text-embedding-3-small",
    )

    return OpenAIEmbeddings(model=model_name)


def load_and_process_data(
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Load and process a document, returning its split chunks.

    Args:
        source: The document source, which can be a URL or a file path.
        chunk_size: The maximum size of each chunk in characters.
        chunk_overlap: The number of overlapping characters between chunks.

    Returns:
        The processed document chunks.
    """
    # Load the document
    docs = load_document(source)

    # Split the document into chunks
    doc_chunks = split_documents(
        docs,
        chunk_size,
        chunk_overlap,
    )

    return doc_chunks
