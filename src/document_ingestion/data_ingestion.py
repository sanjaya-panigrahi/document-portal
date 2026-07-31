from __future__ import annotations
import os
import sys
import json
import uuid
import hashlib
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentIntelligenceError
from utils.file_io import generate_session_id, save_uploaded_files
from utils.document_ops import load_documents, concat_for_analysis, concat_for_comparison

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# FAISS Manager — handles creating, loading, and updating the vector index on disk.
class FaissManager:
    """
    Manages a FAISS vector store on disk for a single session.

    What it does:
        - On first run: creates a new FAISS index from the given text chunks.
        - On subsequent runs: loads the existing index and adds only new (unseen) documents.
        - Tracks which documents have already been ingested using a JSON metadata file
          (ingested_meta.json) so the same content is never embedded twice (idempotent).

    Args:
        index_dir:     Directory where the FAISS index files and metadata are stored.
        model_loader:  Optional ModelLoader instance. Created automatically if not provided.
    """
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}} ## this is dict of rows
        
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}} # load it if alrady there
            except Exception:
                self._meta = {"rows": {}} # init the empty one if dones not exists
        

        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None
        
    def _exists(self) -> bool:
        """Returns True if a valid FAISS index already exists on disk (index.faiss + index.pkl)."""
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()
    
    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        """
        Generates a unique fingerprint for a document chunk to detect duplicates.
        Uses source path + row_id if available; otherwise SHA-256 hashes the content.
        """
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def _save_meta(self):
        """Persists the ingested document fingerprints to ingested_meta.json on disk."""
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")
        
        
    def add_documents(self, docs: List[Document]):
        """
        Adds only new (previously unseen) documents to the FAISS index.
        Skips any document whose fingerprint already exists in ingested_meta.json.
        Saves the updated index and metadata to disk.

        Args:
            docs: List of LangChain Document chunks to add.

        Returns:
            Number of new documents actually added to the index.

        Raises:
            RuntimeError: If load_or_create() has not been called first.
        """
        
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents_idempotent().")
        
        new_docs: List[Document] = []
        
        for d in docs:
            
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)
            
        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)
    
    def load_or_create(self, texts: Optional[List[str]] = None, metadatas: Optional[List[dict]] = None):
        """
        Loads an existing FAISS index from disk, or creates a new one if none exists.

        Args:
            texts:     List of raw text strings used to create the index on first run.
            metadatas: Corresponding metadata dicts for each text (source, page, etc.).

        Returns:
            The FAISS vectorstore instance (loaded or newly created).

        Raises:
            DocumentIntelligenceError: If no index exists on disk and no texts are provided.
        """
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs
        
        
        if not texts:
            raise DocumentIntelligenceError("No existing FAISS index and no data to create one", sys)
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs
        
        
class ChatIngestor:
    """
    Handles end-to-end document ingestion for the chat feature.

    What it does:
        1. Saves uploaded files to a temp directory.
        2. Loads and parses the files into LangChain Documents.
        3. Splits documents into overlapping chunks for better retrieval.
        4. Stores the chunks in a FAISS vector index via FaissManager.
        5. Returns a retriever object ready to be used by ConversationalRAG.

    Each chat session gets its own subfolder (session_id) inside both
    the temp and FAISS base directories so multiple users don't conflict.

    Usage:
        ingestor = ChatIngestor(session_id="abc", embedding_key="openai")
        retriever = ingestor.built_retriver(uploaded_files, chunk_size=1000, k=5)
    """
    def __init__(
        self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
        embedding_key: Optional[str] = None,
    ):
        """
        Sets up directory paths and session state for this ingestion run.

        Args:
            temp_base:       Root folder for saving uploaded files (default "data").
            faiss_base:      Root folder for FAISS indexes (default "faiss_index").
            use_session_dirs: If True, files and indexes are stored in per-session subfolders.
                              Set False to use a single shared index.
            session_id:      Reuse an existing session ID to extend its index.
                             Auto-generated if not provided.
            embedding_key:   Which embedding model to use (e.g. "openai", "google").
                             Falls back to config.yaml default if not specified.
        """
        try:
            self.model_loader = ModelLoader()
            self.embedding_key = embedding_key
            
            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()
            
            self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)
            
            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            log.info("ChatIngestor initialized",
                      session_id=self.session_id,
                      temp_dir=str(self.temp_dir),
                      faiss_dir=str(self.faiss_dir),
                      sessionized=self.use_session)
        except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentIntelligenceError("Initialization error in ChatIngestor", e) from e
            
        
    def _resolve_dir(self, base: Path) -> Path:
        """
        Returns the correct working directory based on whether session dirs are enabled.
        If use_session_dirs=True, creates and returns base/<session_id>/.
        Otherwise returns base/ directly.
        """
        if self.use_session:
            d = base / self.session_id # e.g. "faiss_index/abc123"
            d.mkdir(parents=True, exist_ok=True) # creates dir if not exists
            return d
        return base # fallback: "faiss_index/"
        
    def _split(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        """
        Splits loaded documents into smaller overlapping chunks for better embedding and retrieval.

        Args:
            docs:          List of LangChain Documents to split.
            chunk_size:    Maximum characters per chunk (default 1000).
            chunk_overlap: Characters shared between consecutive chunks to preserve context (default 200).

        Returns:
            List of chunked LangChain Documents.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks
    
    def built_retriver(
        self,
        uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
    ):
        """
        Full ingestion pipeline: save → load → split → embed → index → return retriever.

        Args:
            uploaded_files: Iterable of file-like objects (FastAPIFileAdapter or open file handles).
            chunk_size:     Characters per text chunk (default 1000).
            chunk_overlap:  Overlap between chunks to avoid losing context at boundaries (default 200).
            k:              Number of top chunks the retriever returns per query (default 5).

        Returns:
            A FAISS-backed LangChain retriever ready to be passed to ConversationalRAG.
        """
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")
            
            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            ## FAISS manager very very important class for the docchat
            fm = FaissManager(self.faiss_dir, self.model_loader)
            fm.emb = self.model_loader.load_embeddings(embedding_key=self.embedding_key)
            
            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]
            
            try:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
            except Exception:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
                
            added = fm.add_documents(chunks)
            log.info("FAISS index updated", added=added, index=str(self.faiss_dir))
            
            return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
            
        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentIntelligenceError("Failed to build retriever", e) from e

            
        
            
class DocHandler:
    """
    Handles saving and reading a single PDF for document analysis.

    What it does:
        - Saves an uploaded PDF to a session-specific folder under data/document_analysis/.
        - Reads the saved PDF page-by-page using PyMuPDF (fitz) and returns the full text.

    Each call creates its own session folder so concurrent uploads don't overwrite each other.

    Usage:
        handler = DocHandler()
        path = handler.save_pdf(uploaded_file)
        text = handler.read_pdf(path)
    """
    def __init__(self, data_dir: Optional[str] = None, session_id: Optional[str] = None):
        """
        Args:
            data_dir:   Root directory for saving PDFs.
                        Defaults to DATA_STORAGE_PATH env var or data/document_analysis/.
            session_id: Unique ID for this upload session.
                        Auto-generated with a timestamp if not provided.
        """
        self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis"))
        self.session_id = session_id or generate_session_id("session")
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        log.info("DocHandler initialized", session_id=self.session_id, session_path=self.session_path)

    def save_pdf(self, uploaded_file) -> str:
        """
        Saves an uploaded PDF to the session directory.

        Args:
            uploaded_file: File-like object with .name attribute and either .read() or .getbuffer().
                           Only .pdf files are accepted.

        Returns:
            Absolute path to the saved PDF file.
        """
        try:
            filename = os.path.basename(uploaded_file.name)
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Invalid file type. Only PDFs are allowed.")
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            log.info("PDF saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            log.error("Failed to save PDF", error=str(e), session_id=self.session_id)
            raise DocumentIntelligenceError(f"Failed to save PDF: {str(e)}", e) from e

    def read_pdf(self, pdf_path: str) -> str:
        """
        Extracts all text from a PDF file, page by page, using PyMuPDF.

        Args:
            pdf_path: Absolute path to the saved PDF file.

        Returns:
            Full document text as a single string with page-number headers.
        """
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_chunks.append(f"\n--- Page {page_num + 1} ---\n{page.get_text()}")  # type: ignore
            text = "\n".join(text_chunks)
            log.info("PDF read successfully", pdf_path=pdf_path, session_id=self.session_id, pages=len(text_chunks))
            return text
        except Exception as e:
            log.error("Failed to read PDF", error=str(e), pdf_path=pdf_path, session_id=self.session_id)
            raise DocumentIntelligenceError(f"Could not process PDF: {pdf_path}", e) from e
class DocumentComparator:
    """
    Handles saving, reading, and combining two PDFs for LLM-based comparison.

    What it does:
        - Saves the reference and actual PDFs into a session-specific directory.
        - Reads each PDF page by page.
        - Combines both into a single text string with clear document labels so
          the LLM knows which content belongs to which file.
        - Optionally cleans up old session directories to save disk space.

    Usage:
        dc = DocumentComparator()
        dc.save_uploaded_files(ref_file, actual_file)
        combined = dc.combine_documents()
    """
    def __init__(self, base_dir: str = "data/document_compare", session_id: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.session_id = session_id or generate_session_id()
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        log.info("DocumentComparator initialized", session_path=str(self.session_path))

    def save_uploaded_files(self, reference_file, actual_file):
        """
        Saves the reference and actual PDF files into the session directory.

        Args:
            reference_file: The original/baseline PDF (file-like with .name and .read()/.getbuffer()).
            actual_file:    The modified PDF to compare against the reference.

        Returns:
            Tuple of (ref_path, act_path) — Path objects for the two saved files.
        """
        try:
            ref_path = self.session_path / reference_file.name
            act_path = self.session_path / actual_file.name
            for fobj, out in ((reference_file, ref_path), (actual_file, act_path)):
                if not fobj.name.lower().endswith(".pdf"):
                    raise ValueError("Only PDF files are allowed.")
                with open(out, "wb") as f:
                    if hasattr(fobj, "read"):
                        f.write(fobj.read())
                    else:
                        f.write(fobj.getbuffer())
            log.info("Files saved", reference=str(ref_path), actual=str(act_path), session=self.session_id)
            return ref_path, act_path
        except Exception as e:
            log.error("Error saving PDF files", error=str(e), session=self.session_id)
            raise DocumentIntelligenceError("Error saving files", e) from e

    def read_pdf(self, pdf_path: Path) -> str:
        """
        Reads a single PDF and returns its text content with page-number labels.
        Skips empty pages and raises an error if the PDF is encrypted.

        Args:
            pdf_path: Path to the PDF file to read.
        """
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted: {pdf_path.name}")
                parts = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()  # type: ignore
                    if text.strip():
                        parts.append(f"\n --- Page {page_num + 1} --- \n{text}")
            log.info("PDF read successfully", file=str(pdf_path), pages=len(parts))
            return "\n".join(parts)
        except Exception as e:
            log.error("Error reading PDF", file=str(pdf_path), error=str(e))
            raise DocumentIntelligenceError("Error reading PDF", e) from e

    def combine_documents(self) -> str:
        """
        Reads all PDFs in the session directory and combines them into one string.
        Each document is prefixed with its filename so the LLM can tell them apart.

        Returns:
            Combined text of all PDFs in the session, separated by blank lines.
        """
        try:
            doc_parts = []
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() == ".pdf":
                    content = self.read_pdf(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")
            combined_text = "\n\n".join(doc_parts)
            log.info("Documents combined", count=len(doc_parts), session=self.session_id)
            return combined_text
        except Exception as e:
            log.error("Error combining documents", error=str(e), session=self.session_id)
            raise DocumentIntelligenceError("Error combining documents", e) from e

    def clean_old_sessions(self, keep_latest: int = 3):
        """
        Deletes old session directories to free up disk space.

        Args:
            keep_latest: Number of most recent sessions to keep (default 3).
                         All older session folders are permanently deleted.
        """
        try:
            sessions = sorted([f for f in self.base_dir.iterdir() if f.is_dir()], reverse=True)
            for folder in sessions[keep_latest:]:
                shutil.rmtree(folder, ignore_errors=True)
                log.info("Old session folder deleted", path=str(folder))
        except Exception as e:
            log.error("Error cleaning old sessions", error=str(e))
            raise DocumentIntelligenceError("Error cleaning old sessions", e) from e

