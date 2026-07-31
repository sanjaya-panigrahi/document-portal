from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentIntelligenceError
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# ---------------------------------------------------------------------------
# Document loading helpers
# Supports PDF, DOCX, and TXT files using LangChain community loaders.
# ---------------------------------------------------------------------------


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """
    Loads one or more files from disk into LangChain Document objects.
    Each document is split by the loader into one Document per page (PDF)
    or one Document for the whole file (DOCX / TXT).

    Args:
        paths: Iterable of file paths to load.

    Returns:
        A flat list of LangChain Document objects ready for chunking.
    """
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
            elif ext == ".txt":
                loader = TextLoader(str(p), encoding="utf-8")
            else:
                log.warning("Unsupported extension skipped", path=str(p))
                continue
            docs.extend(loader.load())
        log.info("Documents loaded", count=len(docs))
        return docs
    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise DocumentIntelligenceError("Error loading documents", e) from e

def concat_for_analysis(docs: List[Document]) -> str:
    """
    Joins documents into a single string with source headers.
    Used by DocumentAnalyzer before sending text to the LLM.
    """
    parts = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        parts.append(f"\n--- SOURCE: {src} ---\n{d.page_content}")
    return "\n".join(parts)

def concat_for_comparison(ref_docs: List[Document], act_docs: List[Document]) -> str:
    """
    Combines reference and actual documents into one string with clear section markers.
    The comparison prompt uses <<REFERENCE_DOCUMENTS>> and <<ACTUAL_DOCUMENTS>> labels
    to tell the LLM which side is which.
    """
    left = concat_for_analysis(ref_docs)
    right = concat_for_analysis(act_docs)
    return f"<<REFERENCE_DOCUMENTS>>\n{left}\n\n<<ACTUAL_DOCUMENTS>>\n{right}"

# ---------- Helpers ----------
class FastAPIFileAdapter:
    """
    Wraps a FastAPI UploadFile so it looks like a Streamlit-style uploaded file.
    The rest of the codebase expects .name and .getbuffer() on uploaded file objects.
    This adapter bridges the difference between FastAPI and that interface.
    """
    def __init__(self, uf: UploadFile):
        """Wraps the FastAPI UploadFile and exposes .name for compatibility."""
        self._uf = uf
        self.name = uf.filename

    def getbuffer(self) -> bytes:
        """Reads and returns the full file content as bytes. Seeks to the start first."""
        self._uf.file.seek(0)
        return self._uf.file.read()

def read_pdf_via_handler(handler, path: str) -> str:
    """
    Calls the correct read method on a DocHandler instance.
    Tries read_pdf() first, then read_() as a fallback.
    Raises RuntimeError if neither method exists.
    """
    if hasattr(handler, "read_pdf"):
        return handler.read_pdf(path)  # type: ignore
    if hasattr(handler, "read_"):
        return handler.read_(path)  # type: ignore
    raise RuntimeError("DocHandler has neither read_pdf nor read_ method.")