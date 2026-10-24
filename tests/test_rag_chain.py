
"""Test RAG chain."""

import os
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.rag_chain import create_rag_chain


class TestRAGChain(unittest.TestCase):
    """Test class for the RAG chain."""

    def setUp(self):
        """Set up the test environment."""
        # Create test documents
        self.test_docs = [
            Document(
                page_content="Artificial intelligence is a branch of computer science that aims to create systems capable of simulating human intelligence.",
                metadata={"source": "test"}
            ),
            Document(
                page_content="Machine learning is a subfield of artificial intelligence that focuses on developing algorithms that can learn from data.",
                metadata={"source": "test"}
            ),
            Document(
                page_content="Deep learning is a method of machine learning that uses neural networks for learning.",
                metadata={"source": "test"}
            ),
        ]

    @patch("src.rag_chain.create_vector_store")
    @patch("src.rag_chain.create_embeddings")
    @patch("src.rag_chain.ChatOpenAI")
    def test_create_rag_chain(
        self,
        mock_chat_openai,
        mock_create_embeddings,
        mock_create_vector_store
    ):
        """Test the creation of the RAG chain."""
        # Configure mock objects
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        mock_vector_store = MagicMock()
        mock_create_vector_store.return_value = mock_vector_store

        # Create the RAG chain
        rag_chain = create_rag_chain(documents=self.test_docs)

        # Verify that the RAG chain was created successfully
        self.assertIsNotNone(rag_chain)
        mock_create_vector_store.assert_called_once()


if __name__ == "__main__":
    unittest.main()
