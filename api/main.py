import os
import time
import threading
from collections import defaultdict
from datetime import date
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator,
    ChatIngestor,
)
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG
from utils.document_ops import FastAPIFileAdapter,read_pdf_via_handler
from utils.model_loader import ModelLoader
from utils.config_loader import load_config
from logger import GLOBAL_LOGGER as log

# ---------------------------------------------------------------------------
# Environment / config constants
# These can be overridden via environment variables without touching code.
# ---------------------------------------------------------------------------
FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")   # root folder for all FAISS vector indexes
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")         # root folder where uploaded files are saved
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")  # must match the name used in FAISS.save_local()

_cfg = load_config()
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", _cfg.get("upload", {}).get("max_file_size_bytes", 1048576)))  # default 1 MB

# ---------------------------------------------------------------------------
# POC usage restrictions — protects personal API keys from abuse
# ---------------------------------------------------------------------------
_r = _cfg.get("restrictions", {})

# Per-IP rate limit: max LLM requests per IP per minute.
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", _r.get("rate_limit_per_minute", 10)))

# Daily request cap: max total LLM calls across all users per calendar day.
DAILY_CAP = int(os.getenv("DAILY_REQUEST_CAP", _r.get("daily_request_cap", 100)))

# In-memory state (resets on server restart; sufficient for a POC)
_lock = threading.Lock()
_ip_timestamps: dict = defaultdict(list)   # ip -> [epoch_seconds, ...]
_daily: dict = {"date": str(date.today()), "count": 0}


def _enforce_restrictions(request: Request) -> None:
    """
    Applies two access controls to every LLM-backed endpoint:
      1. Rate limit  — max RATE_LIMIT requests per IP per 60-second window.
      2. Daily cap   — max DAILY_CAP total requests across all users today.
    Raises HTTP 429 / 503 on violation.
    """
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    with _lock:
        # -- 1. Per-IP rate limit (sliding 60-second window) --
        if RATE_LIMIT > 0:
            window = [t for t in _ip_timestamps[client_ip] if now - t < 60]
            if len(window) >= RATE_LIMIT:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Max {RATE_LIMIT} requests per minute per IP.",
                )
            window.append(now)
            _ip_timestamps[client_ip] = window

        # -- 2. Daily cap (resets at midnight) --
        if DAILY_CAP > 0:
            today = str(date.today())
            if _daily["date"] != today:
                _daily["date"] = today
                _daily["count"] = 0
            if _daily["count"] >= DAILY_CAP:
                raise HTTPException(
                    status_code=503,
                    detail=f"Daily request limit of {DAILY_CAP} reached. Try again tomorrow.",
                )
            _daily["count"] += 1

    log.info("Request allowed", ip=client_ip, daily_used=_daily["count"], daily_cap=DAILY_CAP)


def _check_file_size(file: UploadFile, label: str = "file") -> None:
    """
    Raises HTTP 413 if the uploaded file exceeds MAX_FILE_SIZE bytes.
    Call this at the start of every upload endpoint before processing.
    """
    file.file.seek(0, 2)          # seek to end
    size = file.file.tell()       # get byte position = file size
    file.file.seek(0)             # rewind for downstream readers
    if size > MAX_FILE_SIZE:
        mb = MAX_FILE_SIZE / 1_048_576
        raise HTTPException(
            status_code=413,
            detail=f"{label} exceeds the {mb:.0f} MB limit ({size:,} bytes uploaded).",
        )

# ---------------------------------------------------------------------------
# FastAPI application setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Document Intelligence Workspace", version="0.1")

BASE_DIR = Path(__file__).resolve().parent.parent
# Serve CSS / JS / images from the /static folder
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
# Jinja2 templates live in /templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Allow all origins so the UI (same server) and any external clients can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# UI route
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """
    Serves the main HTML page (index.html).
    This is the entry point for the browser-based UI.
    No input required — just open http://localhost:8080/ in a browser.
    """
    log.info("Serving UI homepage.")
    resp = templates.TemplateResponse(request=request, name="index.html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------------------------------------------------------------------
# Health & discovery routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    """
    Simple health-check endpoint.
    Returns a status message so load balancers / monitoring tools can verify
    the service is running.
    No input required.
    """
    log.info("Health check passed.")
    return {"status": "ok", "service": "document-intelligence-workspace"}


@app.get("/models")
def get_models() -> Dict[str, Any]:
    """
    Returns the list of available LLM models configured in config.yaml.
    The UI calls this on page load to populate the Model dropdown.

    Response shape:
        {
          "default": "google",          # key of the model used when none is selected
          "models": [
            { "key": "google", "provider": "google", "model_name": "gemini-3.6-flash", "label": "..." },
            ...
          ]
        }
    """
    try:
        loader = ModelLoader()
        models = loader.list_available_llms()
        default_key = os.getenv("LLM_PROVIDER", "google")
        if models and default_key not in {m["key"] for m in models}:
            default_key = models[0]["key"]
        return {"default": default_key, "models": models}
    except Exception as e:
        log.exception("Failed to fetch model list")
        raise HTTPException(status_code=500, detail=f"Unable to fetch models: {e}")


@app.get("/embeddings")
def get_embeddings() -> Dict[str, Any]:
    """
    Returns the list of available embedding models configured in config.yaml.
    The UI calls this on page load to populate the Embedding dropdown.

    Response shape:
        {
          "default": "openai",
          "embeddings": [
            { "key": "openai", "provider": "openai", "model_name": "text-embedding-3-small", "label": "..." },
            ...
          ]
        }
    """
    try:
        loader = ModelLoader()
        return loader.list_available_embeddings()
    except Exception as e:
        log.exception("Failed to fetch embedding list")
        raise HTTPException(status_code=500, detail=f"Unable to fetch embeddings: {e}")


# ---------------------------------------------------------------------------
# Document Analysis
# ---------------------------------------------------------------------------

@app.post("/analyze")
async def analyze_document(
    request: Request,
    file: UploadFile = File(...),
    model_key: Optional[str] = Form(None),
) -> Any:
    """
    Analyzes a single PDF and returns structured metadata extracted by an LLM.

    Required:
        file       — PDF file uploaded from the UI (multipart/form-data).

    Optional:
        model_key  — Which LLM to use (e.g. "google", "openai", "groq").
                     Falls back to the default LLM_PROVIDER env var if omitted.

    Response: JSON object with fields like Title, Author, Summary, PageCount, etc.
    """
    try:
        log.info(f"Received file for analysis: {file.filename}")
        _enforce_restrictions(request)
        _check_file_size(file, file.filename or "file")
        dh = DocHandler()
        saved_path = dh.save_pdf(FastAPIFileAdapter(file))
        text = read_pdf_via_handler(dh, saved_path)
        analyzer = DocumentAnalyzer(model_key=model_key)
        result = analyzer.analyze_document(text)
        log.info("Document analysis complete.")
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Error during document analysis")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


# ---------------------------------------------------------------------------
# Document Comparison
# ---------------------------------------------------------------------------

@app.post("/compare")
async def compare_documents(
    request: Request,
    reference: UploadFile = File(...),
    actual: UploadFile = File(...),
    model_key: Optional[str] = Form(None),
) -> Any:
    """
    Compares two PDFs page-by-page and returns a list of differences found by an LLM.

    Required:
        reference  — The original / baseline PDF (multipart/form-data).
        actual     — The updated / modified PDF to compare against the reference.

    Optional:
        model_key  — Which LLM to use. Falls back to default if omitted.

    Response:
        {
          "rows": [ { "Page": "1", "Changes": "..." }, ... ],
          "session_id": "<uuid>"
        }
    """
    try:
        log.info(f"Comparing files: {reference.filename} vs {actual.filename}")
        _enforce_restrictions(request)
        _check_file_size(reference, reference.filename or "reference")
        _check_file_size(actual, actual.filename or "actual")
        dc = DocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(
            FastAPIFileAdapter(reference), FastAPIFileAdapter(actual)
        )
        _ = ref_path, act_path
        combined_text = dc.combine_documents()
        comp = DocumentComparatorLLM(model_key=model_key)
        df = comp.compare_documents(combined_text)
        log.info("Document comparison completed.")
        return {"rows": df.to_dict(orient="records"), "session_id": dc.session_id}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")


# ---------------------------------------------------------------------------
# Document Chat — Step 1: Build / update the FAISS vector index
# ---------------------------------------------------------------------------

@app.post("/chat/index")
async def chat_build_index(
    request: Request,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
    embedding_key: Optional[str] = Form(None),
) -> Any:
    """
    Ingests one or more documents into a FAISS vector store so they can be
    queried by the chat endpoint.  Call this before /chat/query.

    Required:
        files          — One or more PDF / DOCX / TXT files (multipart/form-data).

    Optional:
        session_id     — Reuse an existing session to add more documents to its index.
                         Leave blank to auto-generate a new session ID.
        use_session_dirs — Store the FAISS index in a per-session subfolder (default True).
                           Set False to use a single shared index.
        chunk_size     — Number of characters per text chunk (default 1000).
        chunk_overlap  — Overlap between consecutive chunks to preserve context (default 200).
        k              — Number of top similar chunks to retrieve per query (default 5).
        embedding_key  — Which embedding model to use (e.g. "openai", "google").
                         Falls back to the default in config.yaml if omitted.

    Response:
        { "session_id": "<uuid>", "k": 5, "use_session_dirs": true }
    """
    try:
        log.info(f"Indexing chat session. Session ID: {session_id}, Files: {[f.filename for f in files]}")
        _enforce_restrictions(request)
        for f in files:
            _check_file_size(f, f.filename or "file")
        wrapped = [FastAPIFileAdapter(f) for f in files]
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
            embedding_key=embedding_key,
        )
        ci.built_retriver(
            wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k
        )
        log.info(f"Index created successfully for session: {ci.session_id}")
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat index building failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


# ---------------------------------------------------------------------------
# Document Chat — Step 2: Ask a question against the indexed documents
# ---------------------------------------------------------------------------

@app.post("/chat/query")
async def chat_query(
    request: Request,
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
    model_key: Optional[str] = Form(None),
) -> Any:
    """
    Answers a natural-language question using the documents previously indexed
    by /chat/index.  Uses a Conversational RAG (Retrieval-Augmented Generation)
    pipeline backed by FAISS.

    Required:
        question       — The question to ask about the uploaded documents.

    Optional:
        session_id     — The session ID returned by /chat/index.
                         Required when use_session_dirs=True (default).
        use_session_dirs — Must match what was used during indexing (default True).
        k              — How many document chunks to retrieve as context (default 5).
        model_key      — Which LLM to generate the answer with.
                         Falls back to default if omitted.

    Response:
        {
          "answer": "...",
          "session_id": "<uuid>",
          "k": 5,
          "engine": "LCEL-RAG"
        }
    """
    try:
        log.info(f"Received chat query: '{question}' | session: {session_id}")
        _enforce_restrictions(request)
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id, model_key=model_key)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)
        response = rag.invoke(question, chat_history=[])
        log.info("Chat query handled successfully.")

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG"
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

# ---------------------------------------------------------------------------
# Run locally:
#   uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
# ---------------------------------------------------------------------------