# RAG Document Indexing & Semantic Search System

A modular, SOLID-compliant Retrieval-Augmented Generation (RAG) and Semantic Document Indexing pipeline in Python 3.12. This tool extracts content page-by-page from PDFs, structures text into overlapping semantic window chunks, computes vector embeddings locally, checks for duplicate chunks, and utilizes OpenAI's chat interfaces for answering user questions based on document context.

## System Features
- **Deterministic Checksums**: Computes a message digest hash for each chunk. If the document is re-indexed, the database filters duplicate chunks before embedding generation, minimizing API overhead.
- **Strict Page Retention**: Preserves exact 1-based PDF page indices for source tracking.
- **Local Vectors**: Embeds content locally with `sentence-transformers/all-MiniLM-L6-v2` via CPU, caching the model loaded in-memory.
- **OpenAI Integration**: Synthesizes verified factual answers using LangChain Expression Language (LCEL) and GPT-4o.
- **Rich Terminal UI**: Displays index tasks, metrics tables, and RAG answer panels using `rich` and `typer`.
- **E-Book PDF Generator**: Generates letter-formatted PDF guides programmatically using `reportlab` to seed the database immediately.

---

## Directory Layout
```
d:\DocumentIndexing/
├── src/
│   ├── config.py            # Global variables, folders initialization
│   ├── logger.py            # Rich logging + file log configuration
│   ├── pdf_processor.py     # PDF reader page extractor (PyPDF)
│   ├── text_chunker.py      # Character splitting & metadata creation
│   ├── embedding_model.py   # Cached local SentenceTransformers
│   ├── vector_store.py      # Chroma DB CRUD integrations
│   ├── rag_service.py       # RAG pipeline using OpenAI GPT
│   ├── pdf_generator.py     # Programmatic ReportLab PDF guide creator
│   └── main.py              # CLI entry point (Typer Commands)
├── tests/
│   ├── test_pdf_processor.py
│   ├── test_text_chunker.py
│   └── test_vector_store.py
├── data/                    # Generated booklets and ChromaDB directories
├── logs/                    # Persistent application logs
├── .env                     # Api Credentials
└── requirements.txt         # Package dependencies
```

---

## Setup Instructions

### 1. Build Virtual Environment (Optional but recommended)
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Supply API Keys
Create a `.env` file in the root workspace directory (a `.env.example` template is provided):
```env
OPENAI_API_KEY=sk-svcacct-KsQJR...
```

---

## CLI Usage Guide

All interactions are managed using `src/main.py`.

### 1. Generate the Sample PDF Guide
Creates a structured 5-page PDF about semantic search and RAG inside the `data/` folder:
```bash
python src/main.py create-pdf
```

### 2. Index PDF Document(s)
Extracts page contents, split characters (size 800, overlap 100), hashes ids, checks duplicates, and inserts new records into ChromaDB:
```bash
# Index singular file
python src/main.py index data/semantic_search_guide.pdf

# Index entire directory content recursively
python src/main.py index data/
```

### 3. Semantic Search Lookups
Performs a cosine similarity search on persistent indices, printing matching references with coordinate ranks and scores:
```bash
python src/main.py search "Explain cosine similarity" --top-k 3
```

### 4. RAG QA Reasoning
Submits your prompt to the LLM alongside nearest context blocks retrieved from ChromaDB, printing answers and source pages:
```bash
python src/main.py query "What is the recommended chunk overlap in this codebase?"
```

### 5. Check Database Statistics
Lists the count of chunks and unique indexed document names in ChromaDB:
```bash
python src/main.py stats
```

### 6. Clear/Reset Collection
Resets database indices on disk:
```bash
python src/main.py reset
```

---

## Logging & Audying
Logs are written concurrently to:
- **Terminal console**: Filtered on `INFO` level using Rich formatting.
- **Log file `logs/project.log`**: Detailed traceback down to `DEBUG` levels.

---

## Running Unit Tests
Executes checks on files mapping, chunking, and database indexing logic (employing unittest mocks):
```bash
pytest tests/
```

## Streamlit Frontend
For the fastest browser-based UI, run:
```bash
streamlit run streamlit_app.py
```

Features:
- PDF upload and local path indexing
- Semantic search
- RAG question answering
- Database stats and reset controls
