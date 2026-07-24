from pathlib import Path
from typing import Dict, Any, List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.config import Config
from src.embedding_model import EmbeddingModelProvider
from src.logger import logger

class VectorStoreManager:
    """Manager to wrap ChromaDB Vector Database interactions."""

    def __init__(self, persist_directory: str | Path = None) -> None:
        """Initializes the persistent ChromaDB Vector Store.
        
        Args:
            persist_directory: Path to ChromaDB persistence folder.
        """
        self.persist_directory = str(persist_directory or Config.CHROMA_DB_PATH)
        self.embeddings = EmbeddingModelProvider.get_embeddings()
        
        logger.info(f"Initializing ChromaDB connection at: {self.persist_directory}")
        
        try:
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="rag_documents"
            )
            logger.info("ChromaDB vector store loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Chroma DB: {e}")
            raise RuntimeError(f"Database Initialization Error: {e}")

    def add_documents(self, documents: List[Document]) -> int:
        """Inserts text documents into vector store while bypassing duplicate entries.
        
        Uses chunk_id metadata to verify existing chunks in the collection.
        
        Args:
            documents: A list of LangChain Document objects to index.
            
        Returns:
            int: Number of new documents added.
        """
        if not documents:
            logger.warning("No documents provided to index. Skipping database insert.")
            return 0
            
        # Collect chunk_ids to check for duplicates in the DB
        ids_to_add = [doc.metadata["chunk_id"] for doc in documents]
        
        logger.info(f"Checking {len(ids_to_add)} document chunks for duplicates in ChromaDB...")
        
        try:
            # Query existing document chunks by ID
            existing = self.db.get(ids=ids_to_add)
            existing_ids = set(existing.get("ids", []))
            
            logger.debug(f"Found {len(existing_ids)} pre-existing chunks in the database.")
            
            # Filter documents to only include new chunks
            new_documents = []
            new_ids = []
            for doc, chunk_id in zip(documents, ids_to_add):
                if chunk_id not in existing_ids:
                    new_documents.append(doc)
                    new_ids.append(chunk_id)
            
            num_new_docs = len(new_documents)
            
            if num_new_docs > 0:
                logger.info(f"Indexing {num_new_docs} new document chunks into ChromaDB...")
                self.db.add_documents(documents=new_documents, ids=new_ids)
                logger.info("New chunks successfully stored and persisted.")
            else:
                logger.info("All uploaded document chunks are already indexed. 0 new chunks written.")
                
            return num_new_docs
            
        except Exception as e:
            logger.error(f"Error during document indexing database transaction: {e}")
            raise RuntimeError(f"Database indexing failed: {e}")

    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Performs cosine distance similarity search against ChromaDB indices.
        
        Args:
            query: The question or search text.
            k: The number of highest matching documents to fetch from the collection.
            
        Returns:
            List[Tuple[Document, float]]: Pairs of matching Documents and distance scores.
        """
        logger.info(f"Executing similarity search for query: '{query}' (top K: {k})")
        
        try:
            results = self.db.similarity_search_with_score(query, k=k)
            logger.info(f"Retrieved {len(results)} search results.")
            return results
        except Exception as e:
            logger.error(f"Error executing similarity search: {e}")
            raise RuntimeError(f"Database query error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Gathers collection metrics, listing unique documents and chunks indexed.
        
        Returns:
            Dict[str, Any]: Stats containing:
                "total_chunks": Number of chunks in db.
                "indexed_files": List of unique document filenames.
                "num_files": Count of unique filenames.
        """
        try:
            # Query all metadata elements in Chroma DB
            data = self.db.get(include=["metadatas"])
            metadatas = data.get("metadatas", [])
            
            unique_docs = set()
            for meta in metadatas:
                if meta and "document_name" in meta:
                    unique_docs.add(meta["document_name"])
            
            return {
                "total_chunks": len(metadatas),
                "indexed_files": list(unique_docs),
                "num_files": len(unique_docs)
            }
        except Exception as e:
            logger.error(f"Error fetching database stats: {e}")
            return {
                "total_chunks": 0,
                "indexed_files": [],
                "num_files": 0,
                "error": str(e)
            }

    def clear_database(self) -> None:
        """Deletes/resets the database collection contents."""
        logger.warning(f"Resetting ChromaDB collection at: {self.persist_directory}")
        try:
            self.db.delete_collection()
            logger.info("ChromaDB collection deleted successfully.")
            # Recreate collection instance
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="rag_documents"
            )
            logger.info("Database collection reinitialized.")
        except Exception as e:
            logger.error(f"Error during collection reset: {e}")
            raise RuntimeError(f"Database clear failed: {e}")
