from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from src.logger import logger

class PDFGenerator:
    """Utility class to programmatically generate structured test PDFs with ReportLab."""

    @staticmethod
    def generate_sample_pdf(output_path: str | Path) -> Path:
        """Generates a multi-page structured test PDF file about Semantic Search and RAG.
        
        Args:
            output_path: Target path to write the PDF file.
            
        Returns:
            Path: Absolute path of the created PDF file.
        """
        dest_path = Path(output_path).resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating test PDF at: [bold]{dest_path}[/bold]")
        
        # Initialize Document
        doc = SimpleDocTemplate(
            str(dest_path),
            pagesize=letter,
            rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles for professional aesthetic
        title_style = ParagraphStyle(
            name="DocTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=30,
            textColor=colors.HexColor("#1A365D"),  # Navy Dark Accent
            spaceAfter=20
        )
        
        subtitle_style = ParagraphStyle(
            name="DocSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#4A5568"),  # Slate grey
            alignment=1, # Center
            spaceAfter=40
        )
        
        h1_style = ParagraphStyle(
            name="ChapterH1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#2B6CB0"),  # Medium Blue Accent
            spaceBefore=15,
            spaceAfter=15,
            keepWithNext=True
        )
        
        body_style = ParagraphStyle(
            name="ChapterBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#2D3748"),  # Dark charcoal body
            spaceAfter=12
        )
        
        code_style = ParagraphStyle(
            name="ChapterCode",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#805AD5"),  # Purple code
            leftIndent=20,
            spaceAfter=10
        )

        story = []

        # --- PAGE 1: TITLE PAGE ---
        story.append(Spacer(1, 100))
        story.append(Paragraph("A Complete Guide to Semantic Search and RAG Systems", title_style))
        story.append(Paragraph("An Architectural and Practical Reference Manual for Engineers", subtitle_style))
        story.append(Spacer(1, 120))
        
        brief_intro = (
            "<b>Overview:</b> This reference book is compiled to serve as an indexing target for "
            "evaluating Retrieval-Augmented Generation (RAG) pipelines. It details vector embeddings, "
            "ChromaDB persistence schemas, hierarchical semantic queries, and large language model "
            "synthesis workflows. The chapters are intentionally distributed across separated physical "
            "pages to rigorously test the page retention capabilities of PDF chunkers."
        )
        story.append(Paragraph(brief_intro, body_style))
        story.append(PageBreak())

        # --- PAGE 2: CHAPTER 1 ---
        story.append(Paragraph("Chapter 1: Embeddings and Vector Databases", h1_style))
        ch1_text1 = (
            "Semantic search shifts text processing from keyword matching to conceptual matching. "
            "At the heart of this paradigm are <b>vector embeddings</b>. An embedding model is a "
            "neural network trained to project strings of text into a high-dimensional mathematical space "
            "where semantic similarity corresponds to geometric distance. If two comments convey "
            "similar meanings (e.g., 'artificial intelligence safety' and 'machine learning ethics'), "
            "their vector coordinates will reside close to each other in this space, even if they share "
            "zero vocabulary terms."
        )
        story.append(Paragraph(ch1_text1, body_style))
        
        ch1_text2 = (
            "For embedding representations to be searched efficiently, we utilize <b>Vector Databases</b> "
            "such as ChromaDB. Typical databases index structured metadata fields like dates, strings, or numbers. "
            "In contrast, vector databases are engineered to run approximate nearest neighbor (ANN) indices on density-rich "
            "arrays (often with 384, 768, or 1536 floating points per vector). We measure distance between these "
            "points using similarity metrics, with cosine similarity and L2 (Euclidean) distance being the choice implementations. "
            "ChromaDB specifically exposes high-performance persistence APIs, enabling local prototyping with swift indexing."
        )
        story.append(Paragraph(ch1_text2, body_style))
        story.append(PageBreak())

        # --- PAGE 3: CHAPTER 2 ---
        story.append(Paragraph("Chapter 2: Retrieval-Augmented Generation (RAG)", h1_style))
        ch2_text1 = (
            "Large Language Models (LLMs) suffer from two primary limitations: a cutoff date representing "
            "when their training datasets were compiled, and a propensity to 'hallucinate' plausible-sounding "
            "but factually wrong answers. <b>Retrieval-Augmented Generation</b> (RAG) resolves this by acting "
            "as a dynamic context bridge. Under a RAG architecture, when a user asks a question, the system first "
            "embeds the question, searches the vector database for matching chunks, extracts those original text blocks, "
            "and injects them directly into the LLM system prompt as verified reference materials."
        )
        story.append(Paragraph(ch2_text1, body_style))

        ch2_text2 = (
            "By doing so, the LLM no longer relies solely on its internal weight memory. Instead, "
            "its role is restricted to that of an intelligent reader and synthesist, evaluating the "
            "retrieved resources and compiling a direct answer. If the vector indexes fail to find "
            "relevant records, a well-tuned system instructs the LLM to admit that it cannot answer. "
            "This reduces hallucinations significantly and increases factual lineage tracking."
        )
        story.append(Paragraph(ch2_text2, body_style))
        story.append(PageBreak())

        # --- PAGE 4: CHAPTER 3 ---
        story.append(Paragraph("Chapter 3: Optimal Chunking and Metadata Structure", h1_style))
        ch3_text1 = (
            "To split text efficiently, engineers must design strategies balancing search resolution against "
            "synthesis context window limitations. The <b>RecursiveCharacterTextSplitter</b> splits text recursively "
            "using a prioritized sequence of characters, typically starting with double newlines (paragraphs), "
            "then single newlines, spaces, and ultimately lone characters. The default configuration "
            "chosen for this system features a <b>Chunk Size of 800 characters</b> and a <b>Chunk Overlap of 100 characters</b>."
        )
        story.append(Paragraph(ch3_text1, body_style))

        ch3_text2 = (
            "Dividing text alone is insufficient for production RAG pipelines. Every chunk MUST be "
            "enriched with structured metadata. Essential attributes include: <i>document_name</i> (identifying target source), "
            "<i>page_number</i> (preserving precise pagination mapping), <i>chunk_id</i> (a deterministic hash of content in order "
            "to enforce duplicate detection checks), <i>timestamp</i> (creation time logs), and <i>source_path</i>."
        )
        story.append(Paragraph(ch3_text2, body_style))
        
        story.append(Paragraph("Example Chunk ID standard formulation:", body_style))
        story.append(Paragraph("chunk_id = hashlib.md5(content.encode('utf-8')).hexdigest()", code_style))
        story.append(Paragraph("By implementing duplicate checks prior to insertion, ChromaDB stays performant and small.", body_style))
        story.append(PageBreak())

        # --- PAGE 5: CHAPTER 4 ---
        story.append(Paragraph("Chapter 4: Implementation and Commands", h1_style))
        ch4_text1 = (
            "For administration, the CLI has several critical commands. The default "
            "commands mapped to index, browse, and execute semantic reasoning are as follows:\n"
            "1. <b>index</b>: Reads documents, chunks characters, embeds them using HuggingFace sentence representations, and builds the Chroma database.\n"
            "2. <b>search</b>: Runs a semantic query directly on Chroma, reporting top vector hits along with their page coordinates.\n"
            "3. <b>query</b>: Integrates query matching with the OpenAI Chat model to run full retrieval-augmented generation QA."
        )
        story.append(Paragraph(ch4_text1, body_style))
        
        ch4_text2 = (
            "To monitor database status, run the 'stats' command. This displays the total volume of chunks "
            "indexed in persistent storage and details the unique documents processed. Running 'reset' destroys the "
            "Chroma collection to permit fresh indexing. System logs are preserved continuously in 'logs/project.log'."
        )
        story.append(Paragraph(ch4_text2, body_style))
        
        # Build PDF
        doc.build(story)
        logger.info(f"Sample PDF successfully generated at {dest_path}")
        return dest_path
