"""Workspace-relative paths shared by notebooks, extraction, and training."""

from pathlib import Path
import os

WORKSPACE_ROOT = Path(os.environ.get("EEG2IMAGE_ROOT", Path(__file__).resolve().parents[1])).resolve()


def workspace_path(value: str | Path) -> Path:
    """Resolve a relative path against EEG2IMAGE_ROOT, never the notebook's cwd."""
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path
