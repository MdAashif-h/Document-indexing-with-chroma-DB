from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import streamlit as st

from src.config import Config
from src.pdf_generator import PDFGenerator
from src.pdf_processor import PDFProcessingError, PDFProcessor
from src.rag_service import RAGQAError, RAGService
from src.text_chunker import TextChunker
from src.vector_store import VectorStoreManager


st.set_page_config(
    page_title="RAG Document Indexing & Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better accessibility and styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1f77b4;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1976d2;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #388e3c;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f57c00;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource(show_spinner=False)
def get_rag_service() -> RAGService:
    return RAGService(get_vector_store())


def save_uploaded_pdfs(uploaded_files) -> list[Path]:
    upload_dir = Config.DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        target_path = upload_dir / uploaded_file.name
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target_path)
    return saved_paths


def collect_pdf_paths(local_path_text: str) -> list[Path]:
    if not local_path_text.strip():
        return []

    path = Path(local_path_text).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".pdf":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.pdf"))
    return []


def index_pdf_paths(pdf_paths: list[Path]) -> list[dict[str, str | int]]:
    if not pdf_paths:
        return []

    parser = PDFProcessor()
    chunker = TextChunker()
    db_manager = get_vector_store()

    summaries: list[dict[str, str | int]] = []
    for pdf_path in pdf_paths:
        pages = parser.extract_pages(pdf_path)
        if not pages:
            summaries.append(
                {
                    "file": pdf_path.name,
                    "pages": 0,
                    "chunks": 0,
                    "added": 0,
                    "status": "Skipped",
                }
            )
            continue

        chunks = chunker.chunk_pages(pages)
        if not chunks:
            summaries.append(
                {
                    "file": pdf_path.name,
                    "pages": len(pages),
                    "chunks": 0,
                    "added": 0,
                    "status": "Skipped",
                }
            )
            continue

        added = db_manager.add_documents(chunks)
        summaries.append(
            {
                "file": pdf_path.name,
                "pages": len(pages),
                "chunks": len(chunks),
                "added": added,
                "status": "Indexed",
            }
        )

    return summaries


def render_top_banner() -> None:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="main-header">🔍 Document Intelligence Hub</div>', unsafe_allow_html=True)
        st.caption("Intelligent RAG system for fast document indexing, semantic search, and AI-powered Q&A")
    with col2:
        try:
            metrics = get_vector_store().get_stats()
            st.metric("📚 Documents Indexed", metrics.get("num_files", 0), help="Total unique PDF files in database")
            st.metric("📄 Total Chunks", metrics.get("total_chunks", 0), help="Total text chunks indexed")
        except:
            pass


def render_metrics(metrics: dict[str, object]) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", metrics.get("total_chunks", 0))
    col2.metric("Unique Files", metrics.get("num_files", 0))
    indexed_files = metrics.get("indexed_files", []) or []
    col3.metric("Indexed File Names", len(indexed_files))

    if indexed_files:
        st.write("Indexed files:")
        st.write(", ".join(str(name) for name in indexed_files))


def render_dashboard() -> None:
    render_top_banner()
    
    st.markdown('<div class="section-header">📊 Database Overview</div>', unsafe_allow_html=True)
    
    try:
        metrics = get_vector_store().get_stats()
    except Exception as exc:
        st.error(f"❌ Could not load database stats: {exc}")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Total Indexed Chunks", metrics.get("total_chunks", 0), help="Semantic chunks stored in database")
    with col2:
        st.metric("📁 Unique Files", metrics.get("num_files", 0), help="Number of PDF documents indexed")
    with col3:
        st.metric("✅ Ready to Search", "Yes" if metrics.get("total_chunks", 0) > 0 else "No", help="Database status")

    if metrics.get("indexed_files"):
        with st.expander("📋 View Indexed Files", expanded=False):
            files_list = "\n".join([f"• {name}" for name in sorted(metrics.get("indexed_files", []))])
            st.markdown(files_list)
    else:
        st.info("💡 No documents indexed yet. Go to **Index PDFs** to get started!")
    
    st.divider()
    st.markdown('<div class="section-header">🚀 Getting Started</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("### 1️⃣ Index PDFs")
            st.caption("Upload or index local PDF files")
            if st.button("Go to Indexing", key="quick_index", use_container_width=True):
                st.switch_page("pages/1_index_pdfs.py")
    
    with col2:
        with st.container(border=True):
            st.markdown("### 2️⃣ Search Documents")
            st.caption("Find relevant content instantly")
            if st.button("Go to Search", key="quick_search", use_container_width=True):
                st.switch_page("pages/2_search.py")
    
    with col3:
        with st.container(border=True):
            st.markdown("### 3️⃣ Ask Questions")
            st.caption("Get AI-powered answers")
            if st.button("Go to Q&A", key="quick_qa", use_container_width=True):
                st.switch_page("pages/3_ask_question.py")


def render_index_page() -> None:
    render_top_banner()
    st.markdown('<div class="section-header">📁 Index PDF Documents</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>📌 How to use:</strong> Upload PDFs or link to a local folder. Your documents will be extracted, 
    split into semantic chunks, embedded, and stored in the database for instant searching.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Option 1: Upload PDF Files")
    uploaded_files = st.file_uploader(
        "Select one or more PDFs to upload",
        type=["pdf"],
        accept_multiple_files=True,
        help="Supported formats: PDF. Max size per file depends on your system."
    )

    st.markdown("#### Option 2: Index Local Files")
    local_path_text = st.text_input(
        "Or enter a local path to a PDF file or folder",
        placeholder="Example: D:\\data\\documents.pdf or D:\\data\\",
        help="Provide an absolute path to a PDF file or folder containing PDFs"
    )

    st.markdown("#### Option 3: Generate Sample PDF")
    st.caption("Create a sample PDF to test the system")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate Sample PDF", help="Creates a sample PDF with demo content", use_container_width=True):
            try:
                with st.spinner("✨ Creating sample PDF..."):
                    output_path = PDFGenerator.generate_sample_pdf(Config.DATA_DIR / "semantic_search_guide.pdf")
                    st.success(f"✅ Sample PDF created successfully!")
                    st.info(f"📍 Location: {output_path}")
            except Exception as exc:
                st.error(f"❌ Could not generate sample PDF: {exc}")

    with col2:
        index_uploaded = st.button("🚀 Index Uploaded PDFs", help="Index the uploaded files", use_container_width=True)

    with col3:
        index_local = st.button("📂 Index Local Files", help="Index files from the path above", use_container_width=True)

    if index_uploaded or index_local:
        paths: list[Path] = []
        
        if index_uploaded and uploaded_files:
            with st.spinner("📤 Processing uploaded files..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_paths = save_uploaded_pdfs(uploaded_files)
                    paths.extend(temp_paths)
                    try:
                        with st.spinner("⚙️ Indexing PDFs..."):
                            summaries = index_pdf_paths(paths)
                        st.success("✅ Indexing complete!")
                        if summaries:
                            st.dataframe(
                                summaries,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "file": st.column_config.TextColumn("📄 File Name"),
                                    "pages": st.column_config.NumberColumn("📑 Pages"),
                                    "chunks": st.column_config.NumberColumn("✂️ Chunks"),
                                    "added": st.column_config.NumberColumn("✨ Added"),
                                    "status": st.column_config.TextColumn("Status"),
                                }
                            )
                    except PDFProcessingError as exc:
                        st.error(f"❌ PDF processing error: {exc}")
                    except Exception as exc:
                        st.error(f"❌ Indexing failed: {exc}")
                    finally:
                        for temp_path in temp_paths:
                            if temp_path.exists():
                                temp_path.unlink(missing_ok=True)
        
        elif index_local and local_path_text.strip():
            paths = collect_pdf_paths(local_path_text)
            if not paths:
                st.error("❌ No PDF files found at the provided path. Please check the path and try again.")
                return

            try:
                with st.spinner("⚙️ Indexing local PDFs..."):
                    summaries = index_pdf_paths(paths)
                st.success(f"✅ Indexing complete! ({len(summaries)} files processed)")
                if summaries:
                    st.dataframe(
                        summaries,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "file": st.column_config.TextColumn("📄 File Name"),
                            "pages": st.column_config.NumberColumn("📑 Pages"),
                            "chunks": st.column_config.NumberColumn("✂️ Chunks"),
                            "added": st.column_config.NumberColumn("✨ Added"),
                            "status": st.column_config.TextColumn("Status"),
                        }
                    )
            except PDFProcessingError as exc:
                st.error(f"❌ PDF processing failed: {exc}")
            except Exception as exc:
                st.error(f"❌ Indexing failed: {exc}")
        else:
            st.warning("⚠️ Please upload PDFs or provide a local path first.")


def render_search_page() -> None:
    render_top_banner()
    st.markdown('<div class="section-header">🔎 Semantic Search</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>💡 How it works:</strong> Enter natural language queries to find relevant documents. 
    Results are ranked by relevance score (lower = more relevant).
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "Enter your search query",
        placeholder="Example: What is cosine similarity? How does semantic search work?",
        help="Ask any question about your documents. Natural language is supported."
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("Number of results to return", min_value=1, max_value=10, value=3, help="More results take longer to display")
    with col2:
        search_button = st.button("🔍 Search", help="Click to search", use_container_width=True)

    if search_button:
        if not query.strip():
            st.warning("⚠️ Please enter a search query first.")
            return

        try:
            with st.spinner("🔄 Searching indexed documents..."):
                results = get_vector_store().similarity_search(query, k=top_k)
        except Exception as exc:
            st.error(f"❌ Search failed: {exc}")
            return

        if not results:
            st.info("💭 No matches found. Try a different query or index more documents.")
            return

        st.success(f"✅ Found {len(results)} matching results")
        
        rows = []
        for idx, (doc, score) in enumerate(results, start=1):
            snippet = doc.page_content.replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            rows.append(
                {
                    "Rank": idx,
                    "Score": round(float(score), 4),
                    "Document": doc.metadata.get("document_name", "N/A"),
                    "Page": doc.metadata.get("page_number", "N/A"),
                    "Snippet": snippet,
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Score": st.column_config.NumberColumn("Score", help="Lower is better", width="small"),
                "Document": st.column_config.TextColumn("Document"),
                "Page": st.column_config.NumberColumn("Page", width="small"),
                "Snippet": st.column_config.TextColumn("Preview"),
            }
        )
        
        with st.expander("📖 View Full Text", expanded=False):
            selected_idx = st.selectbox("Select a result to view full text", range(len(results)), format_func=lambda i: f"{i+1}. {rows[i]['Document']} (Page {rows[i]['Page']})")
            if selected_idx is not None:
                full_text = results[selected_idx][0].page_content
                st.markdown(f"**Document:** {results[selected_idx][0].metadata.get('document_name')}")
                st.markdown(f"**Page:** {results[selected_idx][0].metadata.get('page_number')}")
                st.markdown("---")
                st.write(full_text)


def render_query_page() -> None:
    render_top_banner()
    st.markdown('<div class="section-header">🤖 Ask a Question (AI-Powered Q&A)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>✨ How it works:</strong> Ask any question about your documents. The AI finds relevant sections 
    and generates a comprehensive answer with source references.
    </div>
    """, unsafe_allow_html=True)

    question = st.text_area(
        "Enter your question",
        placeholder="Example: What is the recommended chunk overlap in this codebase? How does the RAG pipeline work?",
        height=120,
        help="Ask a question about your indexed documents"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        pass
    with col2:
        if st.button("🚀 Get Answer", help="Generate AI-powered answer", use_container_width=True):
            if not question.strip():
                st.warning("⚠️ Please enter a question first.")
                return

            try:
                with st.spinner("🤔 Analyzing question and searching documents..."):
                    rag_service = get_rag_service()
                    result = rag_service.query(question)
            except RAGQAError as exc:
                st.error(f"❌ RAG service error: {str(exc)}")
                return
            except Exception as exc:
                st.error(f"❌ Q&A failed: {exc}")
                return

            st.success("✅ Answer generated!")
            
            # Display answer in an attractive container
            st.markdown('<div class="section-header">💡 Answer</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(result["answer"])

            # Display sources
            source_docs = result.get("source_documents", [])
            if source_docs:
                st.markdown('<div class="section-header">📚 Source References</div>', unsafe_allow_html=True)
                st.caption(f"Answer is based on {len(source_docs)} source document(s)")
                
                seen = set()
                for idx, doc in enumerate(source_docs, 1):
                    doc_name = doc.metadata.get("document_name", "N/A")
                    page_num = doc.metadata.get("page_number", "N/A")
                    ref_key = f"{doc_name}_p{page_num}"
                    if ref_key in seen:
                        continue
                    seen.add(ref_key)
                    
                    preview = doc.page_content.replace("\n", " ")
                    if len(preview) > 220:
                        preview = preview[:217] + "..."
                    
                    with st.expander(f"📄 {doc_name} — Page {page_num}", expanded=(idx==1)):
                        st.caption("**Relevant Excerpt:**")
                        st.markdown(f"_{preview}_")
            else:
                st.info("💭 No source documents were returned for this query.")


def render_stats_page() -> None:
    render_top_banner()
    st.markdown('<div class="section-header">⚙️ Database Management & Settings</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Statistics", help="Reload database statistics", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Database", help="WARNING: This will delete all indexed data", use_container_width=True):
            st.session_state.show_delete_confirm = True
    
    if st.session_state.get("show_delete_confirm", False):
        st.warning("⚠️ **Are you sure?** This will delete all indexed documents and cannot be undone.")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Yes, Delete All", key="confirm_delete", use_container_width=True):
                try:
                    with st.spinner("🗑️ Clearing database..."):
                        get_vector_store().clear_database()
                    st.success("✅ Vector database cleared successfully.")
                    st.session_state.show_delete_confirm = False
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Reset failed: {exc}")
        
        with col2:
            if st.button("Cancel", key="cancel_delete", use_container_width=True):
                st.session_state.show_delete_confirm = False
                st.rerun()
    
    st.divider()
    st.markdown('<div class="section-header">📊 Current Database Status</div>', unsafe_allow_html=True)

    try:
        metrics = get_vector_store().get_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Total Documents", metrics.get("num_files", 0), help="Number of unique PDF files indexed")
        with col2:
            st.metric("📄 Total Chunks", metrics.get("total_chunks", 0), help="Total semantic chunks in database")
        with col3:
            status = "✅ Active" if metrics.get("total_chunks", 0) > 0 else "⏳ Empty"
            st.metric("Status", status)

        indexed_files = metrics.get("indexed_files", []) or []
        if indexed_files:
            st.markdown("#### 📋 Indexed Documents")
            for i, file in enumerate(sorted(indexed_files), 1):
                st.caption(f"{i}. {file}")
        else:
            st.info("💭 No documents indexed yet.")
            
    except Exception as exc:
        st.error(f"❌ Could not load statistics: {exc}")


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🗂️ Navigation")
        
        page = st.radio(
            "Select a section:",
            ["Dashboard", "Index PDFs", "Search", "Ask a Question", "Settings"],
            help="Choose what you'd like to do"
        )

        st.divider()
        
        st.markdown("### 📚 Quick Info")
        try:
            stats = get_vector_store().get_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Files", stats.get("num_files", 0))
            with col2:
                st.metric("Chunks", stats.get("total_chunks", 0))
        except:
            pass
        
        st.divider()
        
        st.markdown("### ℹ️ Help & Resources")
        with st.expander("💡 Tips & Tricks"):
            st.markdown("""
            **Indexing:**
            - Upload multiple PDFs at once
            - Use local folder paths for batch indexing
            
            **Searching:**
            - Use natural language queries
            - Lower score = higher relevance
            
            **Q&A:**
            - Ask specific questions
            - Answers show source references
            """)
        
        with st.expander("⚙️ How to Run"):
            st.code("streamlit run streamlit_app.py", language="bash")
        
        st.divider()
        st.caption("🔍 Document Intelligence Hub v1.0")
        st.caption("Powered by LangChain, ChromaDB, and OpenAI")
    
    return page


def main() -> None:
    # Initialize session state for UI state management
    if "show_delete_confirm" not in st.session_state:
        st.session_state.show_delete_confirm = False
    
    page = render_sidebar()

    if page == "Dashboard":
        render_dashboard()
    elif page == "Index PDFs":
        render_index_page()
    elif page == "Search":
        render_search_page()
    elif page == "Ask a Question":
        render_query_page()
    elif page == "Settings":
        render_stats_page()


if __name__ == "__main__":
    main()
