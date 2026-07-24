from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import Config
from src.vector_store import VectorStoreManager
from src.logger import logger


class RAGQAError(Exception):
    """Exception raised for execution issues during the RAG pipeline."""
    pass


def _format_docs(docs: List[Document]) -> str:
    """Joins retrieved document page_content into a single context string.

    Args:
        docs: List of LangChain Document objects returned by the retriever.

    Returns:
        str: Concatenated text from all documents separated by double newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)


class RAGService:
    """Retrieval-Augmented Generation QA engine using OpenAI Chat Models and LCEL."""

    def __init__(self, vector_store_manager: VectorStoreManager) -> None:
        """Initializes the RAG Service.

        Args:
            vector_store_manager: The active VectorStoreManager instance.

        Raises:
            RAGQAError: If the OpenAI API key is missing or invalid.
        """
        self.vector_store = vector_store_manager

        # Verify API key presence
        api_key = Config.OPENAI_API_KEY
        if not api_key:
            logger.error("RAGService initialization aborted: OpenAI API key is missing from config/.env.")
            raise RAGQAError(
                "OpenAI API Key is missing. Please add your key to the '.env' file "
                "or export it as OPENAI_API_KEY."
            )

        logger.info("Initializing OpenAI Chat Model 'gpt-4o-mini' for RAG QA...")
        try:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=api_key,
                temperature=0.0  # Zero temperature for deterministic factual extraction
            )

            # Modern LCEL prompt template
            self.prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a helpful, expert AI assistant. Answer the user's question "
                    "clearly and factually, using ONLY the provided context below. "
                    "If the context doesn't contain the answer, say "
                    "'I'm sorry, but I cannot find the answer within the indexed documents.' "
                    "Do not make up facts or extrapolate beyond the text.\n\n"
                    "CONTEXT:\n{context}"
                ),
                ("human", "{question}")
            ])

            # Build retriever from vector store
            self.retriever = self.vector_store.db.as_retriever(search_kwargs={"k": 5})

            # LCEL chain: retrieve → format → prompt → llm → parse
            self.rag_chain = (
                {
                    "context": self.retriever | _format_docs,
                    "question": RunnablePassthrough()
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

            logger.info("RAG QA LCEL pipeline configured successfully.")

        except Exception as e:
            logger.error(f"Failed to load RAG LLM pipeline: {e}")
            raise RAGQAError(f"LLM Setup Error: {e}")

    def query(self, user_question: str) -> Dict[str, Any]:
        """Queries the vector database for matching context and answers the question via LLM.

        Args:
            user_question: The semantic search query / question.

        Returns:
            Dict[str, Any]: A dictionary containing:
                "answer": String response from the model.
                "source_documents": List of source Document objects with page reference metadata.
        """
        logger.info(f"Invoking RAG pipeline for query: '{user_question}'")

        try:
            # Retrieve source documents separately so we can display them
            source_docs = self.retriever.invoke(user_question)
            logger.info(f"Retrieved {len(source_docs)} source documents for context.")

            # Run the full LCEL chain for the answer
            answer = self.rag_chain.invoke(user_question)

            logger.info("RAG query finished successfully.")

            return {
                "answer": answer.strip(),
                "source_documents": source_docs
            }

        except Exception as e:
            logger.error(f"Error during RAG pipeline query execution: {e}")
            raise RAGQAError(f"RAG Pipeline invocation failed: {e}")
