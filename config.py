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

# Excel con datos reales (Google Trends + Instagram)
EXCEL_DATA = BASE_DIR / "data gino.xlsx"

# Cache TTL en horas
CACHE_TTL_HOURS = 6

# Horas máximas antes de re-correr el pipeline completo (fast CSV path)
PROCESSED_TTL_HOURS = 24

# ─────────────────────────────────────────────
#  RSS FEEDS — Medios mexicanos (político/local)
# ─────────────────────────────────────────────
RSS_FEEDS = [
    # ── Quintana Roo / Caribe ──────────────────────────────
    "https://www.poresto.net/feed/",
    "https://sipse.com/feed/",
    "https://www.novedadesqroo.com.mx/feed/",
    "https://quintanaroohoy.com/feed/",
    "https://cancun.corriente.news/feed/",
    "https://www.noticaribe.com.mx/feed/",
    "https://caribenuestro.mx/feed/",
    "https://laverdadnoticias.com/feed/",
    "https://www.cancunmio.com/feed/",
    "https://eju.tv/feed/",
    # ── Nacional ──────────────────────────────────────────
    "https://www.eluniversal.com.mx/arc/outboundfeeds/rss/",
    "https://www.jornada.com.mx/rss/index.xml",
    "https://proceso.com.mx/feed/",
    "https://www.milenio.com/rss",
    "https://www.excelsior.com.mx/rss/nacional",
    "https://aristeguinoticias.com/feed/",
    "https://www.sinembargo.mx/feed",
    "https://politico.mx/feed/",
    "https://www.radioformula.com.mx/rss.xml",
    "https://www.infobae.com/mexico/rss/",
    "https://mexico.quadratin.com.mx/feed/",
    "https://www.reporteindigo.com/feed/",
    "https://lopezdoriga.com/feed/",
    "https://www.sdpnoticias.com/rss.xml",
    "https://www.animal-politico.com/feed/",
    "https://expansion.mx/rss",
    "https://www.eleconomista.com.mx/rss/rss.xml",
    "https://www.cronica.com.mx/rss.xml",
    "https://www.24-horas.mx/feed/",
    "https://heraldodemexico.com.mx/feed/",
    "https://www.elsoldemexico.com.mx/rss.xml",
    "https://www.debate.com.mx/rss.xml",
    "https://www.zocalo.com.mx/rss.xml",
    "https://www.am.com.mx/rss.xml",
    "https://politicaparatodos.mx/feed/",
    "https://www.elmanana.com.mx/rss.xml",
    "https://cnnespanol.cnn.com/mexico/feed/",
    "https://noticieros.televisa.com/feed/",
]
