# src/cbb_predictor/config.py
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROCESSED_DIR / "cbb.duckdb"

load_dotenv(PROJECT_ROOT / ".env")
CBBD_API_KEY = os.getenv("CBBD_API_KEY")
CBBD_BASE_URL = "https://api.collegebasketballdata.com"