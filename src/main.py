import os
import sys
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich import print as rprint

# Setup path imports for package environment compliance
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.logger import logger
from src.pdf_processor import PDFProcessor, PDFProcessingError
from src.text_chunker import TextChunker
from src.vector_store import VectorStoreManager
from src.rag_service import RAGService, RAGQAError
from src.pdf_generator import PDFGenerator

# Initialize components
app = typer.Typer(
    name="RAG Document Indexer",
    help="A SOLID, modular Command Line RAG & Semantic Search System."
)
console = Console()

def display_welcome_banner():
    """Prints a beautiful header panel for the CLI."""
    banner_content = (
        "[bold cyan]🔍 RAG Semantic Document Search CLI[/bold cyan]\n"
        "[dim]Powered by ChromaDB, Sentence-Transformers, and OpenAI GPT[/dim]"
    )
    console.print(Panel(banner_content, border_style="cyan", expand=False))

@app.command()
def create_pdf(
    output_filename: str = typer.Option(
        "semantic_search_guide.pdf",
        "--name", "-n",
        help="Name of the generated PDF file."
    )
):
    """Generates a rich, multi-page test PDF document with structured content."""
    display_welcome_banner()
    output_path = Config.DATA_DIR / output_filename
    
    with console.status("[bold yellow]Creating structured PDF guide...[/bold yellow]", spinner="dots"):
        try:
            pdf_path = PDFGenerator.generate_sample_pdf(output_path)
            console.print(
                f"\n[bold green]Success![/bold green] Sample PDF created at: "
                f"[underline cyan]{pdf_path}[/underline cyan]\n"
                f"[dim]You can now index this using: python src/main.py index data/{output_filename}[/dim]"
            )
        except Exception as e:
            console.print(f"\n[bold red]Error generating PDF:[/bold red] {e}")

@app.command()
def index(
    target_path: str = typer.Argument(
        ...,
        help="Path to a PDF file or a directory containing PDFs."
    )
):
    """Parses, chunks, embeds, and saves PDF document pages into persistent ChromaDB."""
    display_welcome_banner()
    
    path = Path(target_path).resolve()
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] Path does not exist: {path}")
        raise typer.Exit(code=1)

    pdf_files: list[Path] = []
    if path.is_file():
        if path.suffix.lower() == ".pdf":
            pdf_files.append(path)
        else:
            console.print(f"[bold red]Error:[/bold red] File is not a PDF: {path}")
            raise typer.Exit(code=1)
    elif path.is_dir():
        # Retrieve all PDFs in directory
        pdf_files = list(path.glob("**/*.pdf"))
        # Also check immediate level if deep glob empty
        if not pdf_files:
            pdf_files = list(path.glob("*.pdf"))

    if not pdf_files:
        console.print("[bold yellow]No PDF files found to index at the specified path.[/bold yellow]")
        raise typer.Exit()

    console.print(f"[bold blue]Indexing {len(pdf_files)} PDF matches...[/bold blue]\n")

    parser = PDFProcessor()
    chunker = TextChunker()
    
    try:
        db_manager = VectorStoreManager()
    except Exception as e:
        console.print(f"[bold red]Database Init Error: {e}[/bold red]")
        raise typer.Exit(code=1)

    # Walk files and process sequentially with Rich progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console
    ) as progress:
        
        file_task = progress.add_task("[cyan]Processing PDFs...", total=len(pdf_files))
        
        for pdf_file in pdf_files:
            progress.update(file_task, description=f"[cyan]Ingesting {pdf_file.name}...")
            
            try:
                # Extract
                pages = parser.extract_pages(pdf_file)
                if not pages:
                    logger.warning(f"File {pdf_file.name} yielded zero extractable pages. Skipping.")
                    progress.advance(file_task)
                    continue

                # Chunk
                chunks = chunker.chunk_pages(pages)
                if not chunks:
                    logger.warning(f"File {pdf_file.name} yielded zero chunks. Skipping.")
                    progress.advance(file_task)
                    continue
                
                # Add to Vector DB (duplicate checks executed internally)
                newly_indexed = db_manager.add_documents(chunks)
                
                logger.info(
                    f"Successfully indexed document '{pdf_file.name}': "
                    f"{len(pages)} pages split into {len(chunks)} chunks "
                    f"({newly_indexed} new)."
                )
                
            except PDFProcessingError as pe:
                console.print(f"\n[red]Skipped {pdf_file.name} due to structure errors: {pe}[/red]")
            except Exception as e:
                console.print(f"\n[bold red]Failed indexing {pdf_file.name}:[/bold red] {e}")
                
            progress.advance(file_task)

    console.print(f"\n[bold green]Indexing task finished![/bold green] Run [dim]python src/main.py stats[/dim] to see counts.")

@app.command()
def search(
    query: str = typer.Argument(..., help="Query to run semantic cosine matches against."),
    k: int = typer.Option(3, "--top-k", "-k", help="Number of retrieved matching items.")
):
    """Executes a semantic vector similarity database lookup on indexed records."""
    display_welcome_banner()
    
    try:
        db_manager = VectorStoreManager()
        results = db_manager.similarity_search(query, k=k)
    except Exception as e:
        console.print(f"[bold red]Search Execution Failure:[/bold red] {e}")
        raise typer.Exit(code=1)

    if not results:
        console.print("[yellow]No relevant chunks found in database. Is the DB empty?[/yellow]")
        return

    # Build Results Table
    table = Table(
        title=f"Semantic Search Matches for: '[italic]{query}[/italic]'",
        show_header=True,
        header_style="bold magenta",
        box=None
    )
    table.add_column("Rank", justify="center", style="dim", width=6)
    table.add_column("Score (Dist)", justify="center", width=12)
    table.add_column("Source Document", style="cyan", width=25)
    table.add_column("Page", justify="center", style="green", width=6)
    table.add_column("Text Snippet", style="white")

    for idx, (doc, score) in enumerate(results):
        snippet = doc.page_content.replace("\n", " ")
        if len(snippet) > 85:
            snippet = snippet[:82] + "..."
            
        doc_name = doc.metadata.get("document_name", "N/A")
        page_num = str(doc.metadata.get("page_number", "N/A"))
        
        # Format distance score to 4 decimals
        score_str = f"{score:.4f}"
        
        table.add_row(
            str(idx + 1),
            score_str,
            doc_name,
            page_num,
            snippet
        )

    console.print(table)

@app.command()
def query(
    user_question: str = typer.Argument(..., help="Question to submit to the RAG QA engine.")
):
    """Retrieves document chunks matching queries and synthesizes answers using OpenAI GPT-4o."""
    display_welcome_banner()
    
    try:
        db_manager = VectorStoreManager()
        rag_service = RAGService(db_manager)
    except RAGQAError as re:
        console.print(f"[bold red]RAG Pipeline Error:[/bold red] {re}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Initialization Failure:[/bold red] {e}")
        raise typer.Exit(code=1)

    with console.status("[bold green]Querying database and generating response from GPT...[/bold green]", spinner="dots"):
        try:
            result = rag_service.query(user_question)
            answer = result["answer"]
            sources = result["source_documents"]
        except Exception as e:
            console.print(f"\n[bold red]Reasoning Execution Failed:[/bold red] {e}")
            raise typer.Exit(code=1)

    # Display Answer
    console.print("\n[bold cyan]💡 Answer:[/bold cyan]")
    console.print(Panel(answer, border_style="green", padding=(1, 2)))

    # Display Sources Table
    if sources:
        sources_table = Table(
            title="Referenced Source Chunks",
            show_header=True,
            header_style="bold blue",
            box=None
        )
        sources_table.add_column("Source Document", style="cyan")
        sources_table.add_column("Page Number", justify="center", style="green")
        sources_table.add_column("Snippet Preview", style="dim")
        
        # Keep track of shown unique source references to prevent cluttering the sources log
        seen_refs = set()
        
        for doc in sources:
            doc_name = doc.metadata.get("document_name", "N/A")
            page_num = doc.metadata.get("page_number", "N/A")
            
            ref_key = f"{doc_name}_p{page_num}"
            if ref_key in seen_refs:
                continue
            seen_refs.add(ref_key)
            
            preview = doc.page_content.replace("\n", " ")
            if len(preview) > 65:
                preview = preview[:62] + "..."
                
            sources_table.add_row(doc_name, str(page_num), preview)
            
        console.print("\n")
        console.print(sources_table)
    else:
        console.print("\n[dim yellow]No source items were retrieved for context compilation.[/dim yellow]")

@app.command()
def stats():
    """Gathers collection metrics, listing unique documents and chunks indexed."""
    display_welcome_banner()
    
    try:
        db_manager = VectorStoreManager()
        metrics = db_manager.get_stats()
    except Exception as e:
        console.print(f"[bold red]Failed reading stats:[/bold red] {e}")
        raise typer.Exit(code=1)

    # Format Statistics Table
    stats_table = Table(title="Database Statistics", border_style="blue")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Total Document Chunks Indexed", str(metrics["total_chunks"]))
    stats_table.add_row("Total Unique Files Indexed", str(metrics["num_files"]))
    
    console.print(stats_table)
    
    if metrics["indexed_files"]:
        console.print("\n[bold]List of Indexed Files:[/bold]")
        for f in metrics["indexed_files"]:
            console.print(f" - [dim]{f}[/dim]")

@app.command()
def reset():
    """Destroys collection indices starting database from scratch."""
    display_welcome_banner()
    
    confirm = typer.confirm("[bold red]Are you absolutely sure you want to clear the entire database?[/bold red]")
    if not confirm:
        console.print("[yellow]Reset action cancelled.[/yellow]")
        raise typer.Exit()

    try:
        db_manager = VectorStoreManager()
        db_manager.clear_database()
        console.print("[bold green]Vector database collection has been cleared successfully.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Failed to clear database:[/bold red] {e}")

if __name__ == "__main__":
    app()
