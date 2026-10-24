
"""Basic RAG application example."""

import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src.indexing import load_and_process_data
from src.rag_chain import create_rag_chain, create_simple_chain

# Load environment variables
load_dotenv()

# Check for the API key
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError(
        "The OPENAI_API_KEY environment variable is not set. "
        "Please set it in the .env file or export it directly."
    )


def main():
    """Run the RAG demonstration."""
    print("=== Simple RAG Application Demo ===")

    # Load sample data - use Lilian Weng's blog post as an example
    print("Loading and processing documents...")
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    docs = load_and_process_data(url)
    print(f"Documents have been split into {len(docs)} chunks")

    # Create the RAG chain
    print("Creating the RAG chain...")
    rag_chain = create_rag_chain(documents=docs)

    # Alternatively, use the simple chain
    # simple_chain = create_simple_chain(docs)

    # Interactive question-answering
    print("\n=== RAG Question-Answering System ===")
    print("Type 'exit' to quit")

    while True:
        question = input("\nEnter your question: ")

        if question.lower() == "exit":
            break

        # Get an answer using the RAG chain
        print("Processing your query...")
        result = rag_chain.invoke({"question": question})

        # Print the answer
        print("\nAnswer:")
        print(result["answer"])

        # If using the simple chain
        # answer = simple_chain.invoke({"question": question})
        # print("\nAnswer:")
        # print(answer)


if __name__ == "__main__":
    main()
