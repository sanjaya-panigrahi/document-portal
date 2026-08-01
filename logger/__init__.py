# logger/__init__.py
# ---------------------------------------------------------------------------
# Package initialiser for the logger module.
# Creates a single application-wide structured logger (GLOBAL_LOGGER) that is
# shared across every module via:  from logger import GLOBAL_LOGGER as log
# All log output is JSON-formatted and written to both the console and a
# timestamped file under the /logs directory.
# ---------------------------------------------------------------------------
from .custom_logger import CustomLogger

# Instantiate once at import time so every module gets the same logger instance.
GLOBAL_LOGGER = CustomLogger().get_logger("document_intelligence_workspace")