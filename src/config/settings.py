from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "")
_chroma_path = Path(os.getenv("CHROMA_DB_PATH", "vectorstore/chroma_db")).expanduser()
CHROMA_DB_PATH = (_chroma_path if _chroma_path.is_absolute() else PROJECT_ROOT / _chroma_path).resolve()