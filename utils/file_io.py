from __future__ import annotations
import re
import uuid
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid
from typing import Iterable, List
from logger import GLOBAL_LOGGER as log
import sys
from exception.custom_exception import DocumentIntelligenceError

# ---------------------------------------------------------------------------
# File I/O helpers shared by ingestion classes.
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def generate_session_id(prefix: str = "session") -> str:
    """
    Generates a unique session ID using the current IST timestamp + a short UUID suffix.
    Format: <prefix>_YYYYMMDD_HHMMSS_<8-char hex>

    Args:
        prefix: String prepended to the ID (default "session").
    """
    ist = ZoneInfo("Asia/Kolkata")
    return f"{prefix}_{datetime.now(ist).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> List[Path]:
    """
    Saves uploaded file objects to the target directory on disk.
    Accepts both Streamlit UploadedFile (.getbuffer()) and plain file handles (.read()).
    Unsupported extensions (not in SUPPORTED_EXTENSIONS) are silently skipped with a warning.

    Args:
        uploaded_files: Iterable of file-like objects with a .name attribute.
        target_dir:     Directory to save files into (created if it does not exist).

    Returns:
        List of Path objects pointing to the saved files.
    """
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for uf in uploaded_files:
            name = getattr(uf, "name", "file")
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                log.warning("Unsupported file skipped", filename=name)
                continue
            # Clean file name (only alphanum, dash, underscore)
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(name).stem).lower()
            fname = f"{safe_name}_{uuid.uuid4().hex[:6]}{ext}"
            fname = f"{uuid.uuid4().hex[:8]}{ext}"
            out = target_dir / fname
            with open(out, "wb") as f:
                if hasattr(uf, "read"):
                    f.write(uf.read())
                else:
                    f.write(uf.getbuffer())  # fallback
            saved.append(out)
            log.info("File saved for ingestion", uploaded=name, saved_as=str(out))
        return saved
    except Exception as e:
        log.error("Failed to save uploaded files", error=str(e), dir=str(target_dir))
        raise DocumentIntelligenceError("Failed to save uploaded files", sys) from e