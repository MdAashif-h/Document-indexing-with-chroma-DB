# RAG Document Indexing & Semantic Search System

A modular, SOLID-compliant Retrieval-Augmented Generation (RAG) and Semantic Document Indexing pipeline in Python 3.13. This tool extracts content page-by-page from PDFs, structures text into overlapping semantic window chunks, computes vector embeddings locally, checks for duplicate chunks, and utilizes OpenAI's chat interfaces for answering user questions based on document context.

## 🚀 Key Features

- **Deterministic Checksums**: Computes MD5 hash for each chunk. Re-indexing skips duplicates, minimizing API overhead.
- **Strict Page Retention**: Preserves exact 1-based PDF page indices for source tracking.
- **Local Embeddings**: Embeds content locally with `sentence-transformers/all-MiniLM-L6-v2` (CPU-based, cached in-memory).
- **OpenAI Integration**: Synthesizes factual answers using LangChain Expression Language (LCEL) and GPT-4o-mini.
- **Streamlit Frontend**: Beautiful, accessible web UI with guided workflows (recommended).
- **CLI Interface**: Typer-based commands with Rich terminal UI for power users.
- **E-Book Generator**: Generates sample PDFs with ReportLab for testing.
- **Production-Ready**: Fully tested (9 unit tests), error handling, logging, and SOLID architecture.

---

## 📂 Directory Layout

```
D:\DocumentIndexing/
├── src/
│   ├── config.py            # Configuration and environment setup
│   ├── logger.py            # Rich + file logging configuration
│   ├── pdf_processor.py     # PDF extraction (PyPDF)
│   ├── text_chunker.py      # Text chunking with metadata
│   ├── embedding_model.py   # Local embeddings (SentenceTransformers)
│   ├── vector_store.py      # ChromaDB integration & CRUD operations
│   ├── rag_service.py       # RAG pipeline (LangChain + OpenAI)
│   ├── pdf_generator.py     # PDF generation (ReportLab)
│   ├── main.py              # CLI entry point (Typer)
│   └── __init__.py
├── streamlit_app.py         # 🌟 Web UI (recommended)
├── tests/
│   ├── test_pdf_processor.py
│   ├── test_text_chunker.py
│   └── test_vector_store.py
├── data/
│   ├── chroma_db/           # Vector database
│   └── uploads/             # Temporary file storage
├── logs/                    # Application logs
├── requirements.txt         # Python dependencies
├── .env                     # API keys (git-ignored)
├── .gitignore
└── README.md
```

---

## 🔧 Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/MdAashif-h/Document-indexing-with-chroma-DB.git
cd Document-indexing-with-chroma-DB
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file in the root directory:
```
OPENAI_API_KEY=sk-your-api-key-here
```

---

## 🌐 Streamlit Web UI (Recommended)

The easiest way to use this system:

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`

### Features

#### 📊 Dashboard
- Overview of indexed documents
- Quick access statistics
- Getting started guide
- Quick navigation cards

#### 📁 Index PDFs
Three ways to add documents:
- **Upload PDFs**: Drag & drop or select files from your computer
- **Index Local Files**: Select files from your system or entire directories
- **Generate Sample**: Create demo PDFs for testing
- Progress tracking with detailed results

#### 🔎 Semantic Search
- Natural language queries
- Adjustable result count (1-10)
- Ranked by relevance score
- View full document text with expandable previews
- Direct access to source pages

#### 🤖 Ask Questions
- AI-powered Q&A powered by GPT-4o-mini
- Automatically retrieves relevant document context
- Source document references with expandable excerpts
- Click to view full original text from sources

#### ⚙️ Settings
- Real-time database statistics
- Clear all data (with confirmation dialog)
- Document management

### User Experience Features

✨ **User-Friendly Interface**
- Intuitive layout with emoji indicators
- Clear section headers and visual separators
- Helpful placeholder text and examples
- Progress spinners during processing

🎨 **Accessible Design**
- Semantic HTML structure
- High contrast info/success/warning boxes
- Keyboard navigation support
- Clear, descriptive labels

💡 **Guided Workflows**
- Step-by-step instructions
- Inline help and tooltips
- Examples for each feature
- Error messages with solutions

⚡ **Responsive Design**
- Mobile-ready interface
- Fast load times
- Smart caching
- Optimized for all screen sizes

---

## 💻 CLI Usage (Advanced)

For command-line interface:

```bash
# 1. Generate sample PDF
python src/main.py create-pdf

# 2. Index PDF(s)
python src/main.py index data/document.pdf      # Single file
python src/main.py index data/                  # Entire directory

# 3. Semantic search
python src/main.py search "Your query" --top-k 3

# 4. Ask a question (RAG)
python src/main.py query "Your question?"

# 5. View statistics
python src/main.py stats

# 6. Reset database
python src/main.py reset
```

---

## 🧪 Running Tests

```bash
pytest tests/
```

All 9 tests passing:
- ✅ 4 tests for PDF extraction
- ✅ 3 tests for text chunking  
- ✅ 2 tests for vector store operations

---

## 📊 Architecture

### Processing Pipeline

1. **PDF Extraction** → PyPDF extracts text page-by-page with metadata
2. **Text Chunking** → Split into semantic chunks (800 chars, 100 overlap)
3. **Embeddings** → Generate vectors with SentenceTransformers (local, CPU)
4. **Deduplication** → Check for duplicate chunks using MD5 hashes
5. **Storage** → Index in ChromaDB for fast retrieval
6. **Retrieval** → Semantic search with cosine similarity
7. **Generation** → AI-powered Q&A with source citations

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **RAG Framework** | LangChain (LCEL) |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) |
| **Vector Database** | ChromaDB 0.5.3 |
| **LLM** | OpenAI GPT-4o-mini |
| **Frontend** | Streamlit 1.60.0 |
| **PDF Processing** | PyPDF 4.2.0 |
| **CLI** | Typer 0.12.3 + Rich 13.7.1 |
| **Testing** | pytest 8.2.2 |
| **Python** | 3.13.5 |

---

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
CHUNK_SIZE = 800           # Characters per chunk
CHUNK_OVERLAP = 100        # Overlap between chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATABASE_PATH = "data/chroma_db"
LOG_LEVEL = "INFO"
```

---

## 📝 Logging

Logs are written to:

- **Terminal**: Rich-formatted INFO level messages
- **File**: `logs/project.log` with DEBUG level details

View live logs:
```bash
tail -f logs/project.log
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | `pip install streamlit` |
| `OpenAI API Error` | Check `.env` file has correct `OPENAI_API_KEY` with available credits |
| PDF not indexing | Ensure PDF is not corrupted; check `logs/project.log` for details |
| Database issues | Delete `data/chroma_db/` to reset; rebuild index |
| Import errors | Verify virtual environment is activated: `.venv\Scripts\Activate.ps1` |

---

## 📚 Example Workflows

### Workflow 1: Quick Demo
```bash
streamlit run streamlit_app.py
# → Click "Generate Sample PDF"
# → Click "Index PDFs" 
# → Try a search or question
```

### Workflow 2: Index Your Documents
```bash
# Copy your PDFs to data/
cp ~/Documents/*.pdf data/
streamlit run streamlit_app.py
# → Upload or index from local path
# → Search and query your documents
```

### Workflow 3: CLI Power User
```bash
python src/main.py create-pdf
python src/main.py index data/
python src/main.py search "machine learning" --top-k 5
python src/main.py query "What is deep learning?"
python src/main.py stats
```

---

## 🚀 Performance Tips

- **First Run**: Embedding model downloads (~60MB) on first use, then cached
- **Batch Indexing**: Index multiple PDFs at once for efficiency
- **Query Optimization**: More specific questions yield better results
- **Database Size**: Large vector databases may need increased chunk limits

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👨‍💻 Author

**Mohammed Aashif** - AI & RAG Systems Developer

For issues, suggestions, or contributions: [GitHub Issues](https://github.com/MdAashif-h/Document-indexing-with-chroma-DB/issues)
