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
    page_title="RAG Document Indexing",
    page_icon="🔍",
    layout="wide",
)


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
    st.title("🔍 RAG Document Indexing")
    st.caption(
        "Fast Streamlit frontend for PDF indexing, semantic search, and Q&A over your ChromaDB collection."
    )


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
    st.info("Use the sidebar to switch between indexing, search, Q&A, and database tools.")

    try:
        metrics = get_vector_store().get_stats()
    except Exception as exc:
        st.error(f"Could not load database stats: {exc}")
        return

    render_metrics(metrics)


def render_index_page() -> None:
    render_top_banner()
    st.subheader("Index PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    local_path_text = st.text_input(
        "Or index a local file/folder path",
        placeholder=r"D:\\DocumentIndexing\\data\\semantic_search_guide.pdf",
    )

    col1, col2 = st.columns(2)
    generate_sample = col1.button("Generate sample PDF")
    index_uploaded = col2.button("Index uploaded/local PDFs")

    if generate_sample:
        try:
            output_path = PDFGenerator.generate_sample_pdf(Config.DATA_DIR / "semantic_search_guide.pdf")
            st.success(f"Sample PDF created at {output_path}")
        except Exception as exc:
            st.error(f"Could not generate sample PDF: {exc}")

    if index_uploaded:
        paths: list[Path] = []
        if uploaded_files:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_paths = save_uploaded_pdfs(uploaded_files)
                paths.extend(temp_paths)
                try:
                    with st.spinner("Indexing uploaded PDFs..."):
                        summaries = index_pdf_paths(paths)
                    st.success("Indexing complete.")
                    if summaries:
                        st.dataframe(summaries, use_container_width=True, hide_index=True)
                finally:
                    for temp_path in temp_paths:
                        if temp_path.exists():
                            temp_path.unlink(missing_ok=True)
        elif local_path_text.strip():
            paths = collect_pdf_paths(local_path_text)
            if not paths:
                st.error("No PDF files found at the provided path.")
                return

            try:
                with st.spinner("Indexing local PDFs..."):
                    summaries = index_pdf_paths(paths)
                st.success("Indexing complete.")
                if summaries:
                    st.dataframe(summaries, use_container_width=True, hide_index=True)
            except PDFProcessingError as exc:
                st.error(f"PDF processing failed: {exc}")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")
        else:
            st.warning("Upload PDFs or provide a local path first.")


def render_search_page() -> None:
    render_top_banner()
    st.subheader("Semantic Search")

    query = st.text_input("Search query", placeholder="Explain cosine similarity")
    top_k = st.slider("Top K results", min_value=1, max_value=10, value=3)

    if st.button("Run search"):
        if not query.strip():
            st.warning("Enter a search query first.")
            return

        try:
            with st.spinner("Searching indexed chunks..."):
                results = get_vector_store().similarity_search(query, k=top_k)
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            return

        if not results:
            st.warning("No matches found.")
            return

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

        st.dataframe(rows, use_container_width=True, hide_index=True)


def render_query_page() -> None:
    render_top_banner()
    st.subheader("Ask a Question")

    question = st.text_area(
        "Question",
        placeholder="What is the recommended chunk overlap in this codebase?",
        height=120,
    )

    if st.button("Get answer"):
        if not question.strip():
            st.warning("Enter a question first.")
            return

        try:
            with st.spinner("Generating answer..."):
                rag_service = get_rag_service()
                result = rag_service.query(question)
        except RAGQAError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Q&A failed: {exc}")
            return

        st.success("Answer generated.")
        st.markdown("### Answer")
        st.write(result["answer"])

        source_docs = result.get("source_documents", [])
        if source_docs:
            st.markdown("### Source Chunks")
            seen = set()
            for doc in source_docs:
                doc_name = doc.metadata.get("document_name", "N/A")
                page_num = doc.metadata.get("page_number", "N/A")
                ref_key = f"{doc_name}_p{page_num}"
                if ref_key in seen:
                    continue
                seen.add(ref_key)
                preview = doc.page_content.replace("\n", " ")
                if len(preview) > 220:
                    preview = preview[:217] + "..."
                with st.expander(f"{doc_name} — page {page_num}"):
                    st.write(preview)
        else:
            st.info("No source documents were returned.")


def render_stats_page() -> None:
    render_top_banner()
    st.subheader("Database Stats")

    col1, col2 = st.columns(2)
    if col1.button("Refresh stats"):
        st.cache_resource.clear()

    try:
        metrics = get_vector_store().get_stats()
        render_metrics(metrics)
    except Exception as exc:
        st.error(f"Could not load stats: {exc}")

    if col2.button("Reset database"):
        try:
            get_vector_store().clear_database()
            st.success("Vector database cleared.")
        except Exception as exc:
            st.error(f"Reset failed: {exc}")


def render_sidebar() -> str:
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Index PDFs", "Search", "Ask a Question", "Stats & Reset"],
    )

    st.sidebar.divider()
    st.sidebar.write("Run the app with:")
    st.sidebar.code("streamlit run streamlit_app.py")
    return page


def main() -> None:
    page = render_sidebar()

    if page == "Dashboard":
        render_dashboard()
    elif page == "Index PDFs":
        render_index_page()
    elif page == "Search":
        render_search_page()
    elif page == "Ask a Question":
        render_query_page()
    elif page == "Stats & Reset":
        render_stats_page()


if __name__ == "__main__":
    main()
