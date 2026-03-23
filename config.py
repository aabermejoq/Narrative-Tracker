"""
config.py - Configuración central del proyecto
Coloca tus API keys aquí antes de correr el sistema.
"""

import os
from pathlib import Path

# Cargar .env local si existe (para desarrollo local)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Leer secrets de Streamlit Cloud si estamos ahí
def _get_secret(key: str, default: str) -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)

# ─────────────────────────────────────────────
#  RUTAS DEL PROYECTO
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_CACHE = BASE_DIR / "data" / "cache"

for p in [DATA_RAW, DATA_PROCESSED, DATA_CACHE]:
    p.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  API KEYS  ← PON TUS CLAVES AQUÍ
# ─────────────────────────────────────────────
YOUTUBE_API_KEY = _get_secret("YOUTUBE_API_KEY", "TU_YOUTUBE_API_KEY_AQUI")
NEWS_API_KEY    = _get_secret("NEWS_API_KEY",    "TU_NEWS_API_KEY_AQUI")

# ─────────────────────────────────────────────
#  PARÁMETROS DE BÚSQUEDA
# ─────────────────────────────────────────────
POLITICIAN_NAME = "Gino Segura"
POLITICIAN_QUERY_VARIANTS = [
    "Gino Segura",
    "Gino Segura Quintana Roo",
    "Gino Segura político",
]

COMPARISON_POLITICIANS = ["Mara Lezama", "Claudia Sheinbaum"]

REGION = "MX"           # Google Trends región
TRENDS_START = "2022-01-01"

YOUTUBE_MAX_RESULTS = 50
YOUTUBE_MAX_COMMENTS = 100

NEWS_LOOKBACK_DAYS = 180   # Días hacia atrás para noticias

# Modelo de sentimiento (HuggingFace)
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
# Alternativa más rápida para producción:
# SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Topic Modeling
NUM_TOPICS = 6
BERTOPIC_MIN_CLUSTER = 10

# Instagram CSV (coloca tu archivo aquí)
INSTAGRAM_CSV = DATA_RAW / "instagram_comments.csv"

# Cache TTL en horas
CACHE_TTL_HOURS = 6
