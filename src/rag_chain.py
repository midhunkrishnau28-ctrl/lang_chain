
"""Implementation of the RAG chain."""

from typing import Dict, List, Optional, TypedDict, Union

from langchain import hub
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph

from src.indexing import create_embeddings, load_and_process_data
from src.retrieval import create_vector_store, retrieve_documents
from src.utils import get_env_var

# Define the application state
class State(TypedDict):
    """State of the RAG application."""

    question: str
    context: List[Document]
    answer: str


def create_rag_chain(
    documents: Optional[List[Document]] = None,
    source: Optional[str] = None,
    vector_store: Optional[VectorStore] = None,
    llm: Optional[BaseChatModel] = None,
) -> StateGraph:
    """
    Create a RAG application chain.

    Args:
        documents: An optional list of preprocessed documents.
        source: The document source. Required if the documents parameter is not provided.
        vector_store: An optional pre-created vector store.
        llm: An optional language model. OpenAI is used by default.

    Returns:
        A compiled StateGraph instance.
    """
    # Process input parameters
    if documents is None and source is None and vector_store is None:
        raise ValueError(
            "At least one of the following parameters must be provided: "
            "documents, source, or vector_store."
        )

    # Load and process data if only the source parameter is provided
    if documents is None and source:
        documents = load_and_process_data(source)

    # Create the vector store
    if vector_store is None and documents:
        embeddings = create_embeddings()
        vector_store = create_vector_store(documents, embeddings)

    # Create the language model
    if llm is None:
        model_name = get_env_var("LLM_MODEL", "gpt-3.5-turbo")
        llm = ChatOpenAI(model=model_name)

    # Load the RAG prompt template
    prompt = hub.pull("rlm/rag-prompt")

    # Define the retrieval step
    def retrieve(state: State):
        """Retrieve relevant documents from the vector store."""
        if not vector_store:
            raise ValueError("The vector store has not been initialized.")

        retrieved_docs = retrieve_documents(
            vector_store,
            state["question"],
            top_k=4,
        )
        return {"context": retrieved_docs}

    # Define the answer generation step
    def generate(state: State):
        """Generate an answer based on the retrieved context."""
        docs_content = "\n\n".join(
            doc.page_content for doc in state["context"]
        )
        messages = prompt.invoke(
            {
                "question": state["question"],
                "context": docs_content,
            }
        )
        response = llm.invoke(messages)
        return {"answer": response.content}

    # Create and compile the graph
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph


def create_simple_chain(
    documents: List[Document],
    llm: Optional[BaseChatModel] = None,
) -> Runnable:
    """
    Create a simple RAG chain without using LangGraph.

    Args:
        documents: A list of documents.
        llm: An optional language model.

    Returns:
        A runnable chain.
    """
    # Create the vector store
    embeddings = create_embeddings()
    vector_store = create_vector_store(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Create the language model
    if llm is None:
        model_name = get_env_var("LLM_MODEL", "gpt-3.5-turbo")
        llm = ChatOpenAI(model=model_name)

    # Create the prompt template
    template = """You are a helpful assistant. Use the following retrieved context to answer the question.

    Context:
    {context}

    Question: {question}

    If you don't know the answer, simply say that you don't know.
    Do not try to make up an answer. Keep your response concise.
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Assemble the chain
    chain = (
        {
            "context": retriever,
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
