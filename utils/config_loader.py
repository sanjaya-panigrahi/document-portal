from pathlib import Path
import os
import yaml

# ---------------------------------------------------------------------------
# Config loader — reads config/config.yaml into a plain Python dict.
# The path is resolved from the project root so it works regardless of
# which directory the process is started from.
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Returns the absolute path to the project root (two levels above this file)."""
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | None = None) -> dict:
    """
    Loads and returns config/config.yaml as a Python dict.

    Resolution order for the config file path:
      1. Explicit argument (config_path).
      2. CONFIG_PATH environment variable.
      3. Default: <project_root>/config/config.yaml.

    Args:
        config_path: Optional override path to a YAML config file.

    Returns:
        Dict containing all config values (llm, embedding_model, faiss_db, etc.).

    Raises:
        FileNotFoundError: If the resolved config file does not exist.
    """
    env_path = os.getenv("CONFIG_PATH")
    if config_path is None:
        config_path = env_path or str(_project_root() / "config" / "config.yaml")

    path = Path(config_path)
    if not path.is_absolute():
        path = _project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
