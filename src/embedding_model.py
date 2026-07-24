from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import Config
from src.logger import logger

class EmbeddingModelProvider:
    """Singleton provider for HuggingFace embeddings model (sentence-transformers)."""

    _instance: Optional[HuggingFaceEmbeddings] = None

    @classmethod
    def get_embeddings(cls, model_name: str = None) -> HuggingFaceEmbeddings:
        """Retrieves or instantiates the cached Hugging Face model.
        
        Uses sentence-transformers under the hood. Caches the object in-memory
        to prevent duplicate downloads/memory allocation during run lifecycle.
        
        Args:
            model_name: Model identifier name, defaults to Config.EMBEDDING_MODEL_NAME.
            
        Returns:
            HuggingFaceEmbeddings: Langchain compatible huggingface embeddings instance.
        """
        model_name = model_name or Config.EMBEDDING_MODEL_NAME

        if cls._instance is None:
            logger.info(f"Loading embedding model: [bold]{model_name}[/bold]. This might take a moment...")
            
            try:
                # Load via LangChain HuggingFace
                cls._instance = HuggingFaceEmbeddings(
                    model_name=model_name,
                    # We can specify cpu/cuda parameters internally if required, standard defaults work best
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info(f"Embedding model '{model_name}' loaded and cached.")
            except Exception as e:
                logger.error(f"Error initializing Hf embeddings '{model_name}': {e}")
                raise RuntimeError(f"Could not load embedding model: {e}")
        else:
            logger.debug(f"Reusing already loaded embedding model class for '{model_name}'")

        return cls._instance
