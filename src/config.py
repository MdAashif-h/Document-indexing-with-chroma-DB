import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory layout
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class Config:
    """Application configuration container."""
    
    # API key
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Paths
    DATA_DIR: Path = BASE_DIR / "data"
    LOGS_DIR: Path = BASE_DIR / "logs"
    CHROMA_DB_PATH: Path = DATA_DIR / "chroma_db"
    LOG_FILE_PATH: Path = LOGS_DIR / "project.log"

    # Embedding settings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunker settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    @classmethod
    def create_required_dirs(cls) -> None:
        """Create necessary data and logs directories if they do not exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

# Ensure directory readiness immediately on import
Config.create_required_dirs()
