"""
ingestion.py - Recolección de datos desde todas las fuentes
Fuentes: YouTube, Google Trends, NewsAPI, Instagram CSV
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    YOUTUBE_API_KEY, NEWS_API_KEY, POLITICIAN_NAME,
    POLITICIAN_QUERY_VARIANTS, COMPARISON_POLITICIANS,
    REGION, TRENDS_START, YOUTUBE_MAX_RESULTS, YOUTUBE_MAX_COMMENTS,
    NEWS_LOOKBACK_DAYS, DATA_RAW, DATA_CACHE, CACHE_TTL_HOURS,
    INSTAGRAM_CSV,
)


# ─────────────────────────────────────────────────────────────
#  UTILIDAD DE CACHÉ
# ─────────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    return DATA_CACHE / f"{name}.json"


def _cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() < CACHE_TTL_HOURS * 3600


def _load_cache(name: str):
    p = _cache_path(name)
    if _cache_valid(p):
        logger.info(f"Cache hit: {name}")
        with open(p) as f:
            return json.load(f)
    return None


def _save_cache(name: str, data):
    p = _cache_path(name)
    with open(p, "w") as f:
        json.dump(data, f, default=str)
    logger.info(f"Cache saved: {name}")


# ─────────────────────────────────────────────────────────────
#  YOUTUBE
# ─────────────────────────────────────────────────────────────

def fetch_youtube_videos(query: str = POLITICIAN_NAME) -> pd.DataFrame:
    """Busca videos en YouTube y retorna un DataFrame con metadata."""
    cache_name = f"youtube_videos_{query.replace(' ', '_')}"
    cached = _load_cache(cache_name)
    if cached:
        return pd.DataFrame(cached)

    if YOUTUBE_API_KEY == "TU_YOUTUBE_API_KEY_AQUI":
        logger.warning("YouTube API key no configurada. Usando datos de ejemplo.")
        return _youtube_sample_data()

    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        video_ids, video_items = [], []
        page_token = None

        while len(video_ids) < YOUTUBE_MAX_RESULTS:
            params = {
                "q": query,
                "part": "id,snippet",
                "type": "video",
                "maxResults": min(50, YOUTUBE_MAX_RESULTS - len(video_ids)),
                "relevanceLanguage": "es",
                "regionCode": REGION,
                "order": "relevance",
            }
            if page_token:
                params["pageToken"] = page_token

            resp = youtube.search().list(**params).execute()
            for item in resp.get("items", []):
                if item["id"].get("kind") == "youtube#video":
                    video_ids.append(item["id"]["videoId"])

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        # Obtener estadísticas de los videos
        for i in range(0, len(video_ids), 50):
            chunk = ",".join(video_ids[i:i+50])
            stats_resp = youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=chunk
            ).execute()
            video_items.extend(stats_resp.get("items", []))

        rows = []
        for v in video_items:
            s = v.get("snippet", {})
            st = v.get("statistics", {})
            rows.append({
                "video_id": v["id"],
                "title": s.get("title", ""),
                "description": s.get("description", "")[:500],
                "channel": s.get("channelTitle", ""),
                "published_at": s.get("publishedAt", ""),
                "views": int(st.get("viewCount", 0)),
                "likes": int(st.get("likeCount", 0)),
                "comments_count": int(st.get("commentCount", 0)),
                "url": f"https://www.youtube.com/watch?v={v['id']}",
            })

        df = pd.DataFrame(rows)
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        _save_cache(cache_name, df.to_dict("records"))
        logger.info(f"YouTube: {len(df)} videos obtenidos")
        return df

    except Exception as e:
        logger.error(f"Error YouTube: {e}")
        return _youtube_sample_data()


def fetch_youtube_comments(video_ids: list, max_per_video: int = YOUTUBE_MAX_COMMENTS) -> pd.DataFrame:
    """Descarga comentarios de una lista de video IDs."""
    cache_name = f"youtube_comments_{len(video_ids)}"
    cached = _load_cache(cache_name)
    if cached:
        return pd.DataFrame(cached)

    if YOUTUBE_API_KEY == "TU_YOUTUBE_API_KEY_AQUI":
        logger.warning("YouTube API key no configurada. Retornando comentarios de ejemplo.")
        return _youtube_comments_sample()

    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        all_comments = []
        for vid in video_ids[:20]:  # limitar para no agotar cuota
            try:
                resp = youtube.commentThreads().list(
                    part="snippet",
                    videoId=vid,
                    maxResults=min(max_per_video, 100),
                    textFormat="plainText",
                    order="relevance",
                ).execute()
                for item in resp.get("items", []):
                    c = item["snippet"]["topLevelComment"]["snippet"]
                    all_comments.append({
                        "video_id": vid,
                        "text": c.get("textDisplay", ""),
                        "likes": c.get("likeCount", 0),
                        "date": c.get("publishedAt", ""),
                        "source": "youtube",
                    })
            except Exception as ve:
                logger.warning(f"Comentarios deshabilitados en {vid}: {ve}")
                continue

        df = pd.DataFrame(all_comments)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        _save_cache(cache_name, df.to_dict("records"))
        logger.info(f"YouTube: {len(df)} comentarios obtenidos")
        return df

    except Exception as e:
        logger.error(f"Error comentarios YouTube: {e}")
        return _youtube_comments_sample()


# ─────────────────────────────────────────────────────────────
#  GOOGLE TRENDS
# ─────────────────────────────────────────────────────────────

def fetch_google_trends(keywords: list = None) -> dict:
    """Obtiene interés en el tiempo desde Google Trends."""
    if keywords is None:
        keywords = [POLITICIAN_NAME] + COMPARISON_POLITICIANS

    cache_name = "google_trends_" + "_".join(k.replace(" ", "") for k in keywords)
    cached = _load_cache(cache_name)
    if cached:
        return {k: pd.DataFrame(v) for k, v in cached.items()}

    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="es-MX", tz=360, retries=3, backoff_factor=0.5)

        results = {}
        # pytrends acepta máx 5 keywords a la vez
        for i in range(0, len(keywords), 5):
            chunk = keywords[i:i+5]
            pytrends.build_payload(
                chunk,
                cat=0,
                timeframe=f"{TRENDS_START} {datetime.now().strftime('%Y-%m-%d')}",
                geo=REGION,
            )
            df = pytrends.interest_over_time()
            if not df.empty and "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            for kw in chunk:
                if kw in df.columns:
                    results[kw] = df[[kw]].reset_index().rename(columns={"date": "date", kw: "interest"})

            time.sleep(2)  # Evitar rate limit

        to_cache = {k: v.to_dict("records") for k, v in results.items()}
        _save_cache(cache_name, to_cache)
        logger.info(f"Google Trends: {len(results)} series obtenidas")
        return results

    except Exception as e:
        logger.error(f"Error Google Trends: {e}")
        return _trends_sample_data(keywords)


# ─────────────────────────────────────────────────────────────
#  NEWS API
# ─────────────────────────────────────────────────────────────

def fetch_news(query: str = POLITICIAN_NAME) -> pd.DataFrame:
    """Obtiene noticias desde NewsAPI."""
    cache_name = f"news_{query.replace(' ', '_')}"
    cached = _load_cache(cache_name)
    if cached:
        return pd.DataFrame(cached)

    if NEWS_API_KEY == "TU_NEWS_API_KEY_AQUI":
        logger.warning("NewsAPI key no configurada. Usando datos de ejemplo.")
        return _news_sample_data()

    try:
        from_date = (datetime.now() - timedelta(days=NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "es",
            "pageSize": 100,
            "apiKey": NEWS_API_KEY,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        articles = data.get("articles", [])
        rows = []
        for a in articles:
            rows.append({
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "content": (a.get("content") or "")[:800],
                "source": a.get("source", {}).get("name", ""),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
            })

        df = pd.DataFrame(rows)
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True).dt.tz_localize(None)
        df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
        _save_cache(cache_name, df.to_dict("records"))
        logger.info(f"NewsAPI: {len(df)} noticias obtenidas")
        return df

    except Exception as e:
        logger.error(f"Error NewsAPI: {e}")
        return _news_sample_data()


# ─────────────────────────────────────────────────────────────
#  INSTAGRAM CSV
# ─────────────────────────────────────────────────────────────

def load_instagram_data(csv_path: Path = INSTAGRAM_CSV) -> pd.DataFrame:
    """
    Carga el CSV de comentarios de Instagram.
    El CSV debe tener al menos una columna de texto de comentarios.
    Columnas esperadas (flexibles): text/comment/comentario, date/fecha, likes
    """
    if not csv_path.exists():
        logger.warning(f"Instagram CSV no encontrado en {csv_path}. Usando datos de ejemplo.")
        return _instagram_sample_data()

    try:
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")

        # Normalizar nombres de columnas
        df.columns = [c.lower().strip() for c in df.columns]

        # Detectar columna de texto
        text_col = next(
            (c for c in df.columns if c in ["text", "comment", "comentario", "texto", "body"]),
            df.columns[0]
        )
        df = df.rename(columns={text_col: "text"})

        # Detectar columna de fecha
        date_col = next(
            (c for c in df.columns if c in ["date", "fecha", "timestamp", "created_at", "time"]),
            None
        )
        if date_col:
            df = df.rename(columns={date_col: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["date"] = pd.NaT

        df["source"] = "instagram"
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 3]

        logger.info(f"Instagram CSV: {len(df)} comentarios cargados")
        return df[["text", "date", "source"]].copy()

    except Exception as e:
        logger.error(f"Error cargando Instagram CSV: {e}")
        return _instagram_sample_data()


# ─────────────────────────────────────────────────────────────
#  DATOS DE EJEMPLO (fallback cuando no hay API keys)
# ─────────────────────────────────────────────────────────────

def _youtube_sample_data() -> pd.DataFrame:
    import numpy as np
    dates = pd.date_range("2023-01-01", periods=20, freq="2W")
    return pd.DataFrame({
        "video_id": [f"vid_{i}" for i in range(20)],
        "title": [
            f"Gino Segura: {t}" for t in [
                "habla sobre seguridad en QRoo",
                "propuestas para Quintana Roo",
                "entrevista exclusiva 2024",
                "debate político",
                "agenda de gobierno",
                "visita municipios",
                "conferencia de prensa",
                "sobre turismo en QRoo",
                "infraestructura y desarrollo",
                "reunión con ciudadanos",
                "responde críticas",
                "plan de trabajo",
                "avances en obras",
                "diálogo con empresarios",
                "actos de campaña",
                "foro de seguridad",
                "medio ambiente",
                "educación en QRoo",
                "economía regional",
                "balance de gestión",
            ]
        ],
        "description": ["Descripción del video " + str(i) for i in range(20)],
        "channel": np.random.choice(["Canal Noticias QRoo", "Política Mx", "InfoCaribe", "Novedades QRoo"], 20),
        "published_at": dates,
        "views": np.random.randint(500, 50000, 20),
        "likes": np.random.randint(10, 2000, 20),
        "comments_count": np.random.randint(5, 500, 20),
        "url": [f"https://youtube.com/watch?v=vid_{i}" for i in range(20)],
    })


def _youtube_comments_sample() -> pd.DataFrame:
    import numpy as np
    comments = [
        "Gino Segura es una gran opción para QRoo",
        "No me convence su propuesta de seguridad",
        "Excelente trabajo en los municipios",
        "Necesitamos más transparencia",
        "Buen líder político para Quintana Roo",
        "Sus proyectos son interesantes",
        "Hay que ver sus resultados reales",
        "Apoyamos a Gino Segura",
        "Se necesitan más acciones concretas",
        "Muy buena propuesta para el turismo",
        "Hay que darle una oportunidad",
        "No cumple sus promesas",
        "Gran trabajo en infraestructura",
        "Esperamos ver más resultados",
        "El mejor candidato de la región",
        "Sus propuestas son vagas",
        "Siempre presente en la comunidad",
        "Necesita mejorar en educación",
        "Buen trabajo con el medio ambiente",
        "Un político de verdad",
    ] * 5
    return pd.DataFrame({
        "video_id": np.random.choice([f"vid_{i}" for i in range(20)], len(comments)),
        "text": comments,
        "likes": np.random.randint(0, 100, len(comments)),
        "date": pd.date_range("2023-01-01", periods=len(comments), freq="3D"),
        "source": "youtube",
    })


def _trends_sample_data(keywords: list) -> dict:
    import numpy as np
    dates = pd.date_range(TRENDS_START, datetime.now().strftime("%Y-%m-%d"), freq="W")
    results = {}
    for kw in keywords:
        base = np.random.randint(10, 40)
        trend = np.clip(base + np.cumsum(np.random.randn(len(dates)) * 3), 0, 100).astype(int)
        results[kw] = pd.DataFrame({"date": dates, "interest": trend})
    return results


def _news_sample_data() -> pd.DataFrame:
    import numpy as np
    titles = [
        "Gino Segura presenta plan de seguridad para Quintana Roo",
        "Propuesta de Gino Segura para el desarrollo turístico",
        "Gino Segura critica gestión municipal en QRoo",
        "Debate político: Gino Segura y sus propuestas",
        "Gino Segura encabeza foro de desarrollo económico",
        "Análisis: ¿Qué propone Gino Segura para QRoo?",
        "Gino Segura recorre municipios del estado",
        "Gino Segura habla sobre infraestructura en Cancún",
        "Reunión de Gino Segura con líderes empresariales",
        "Gino Segura y la agenda ambiental en QRoo",
    ] * 4
    dates = pd.date_range(
        datetime.now() - timedelta(days=NEWS_LOOKBACK_DAYS),
        periods=len(titles), freq="4D"
    )
    return pd.DataFrame({
        "title": titles,
        "description": ["Nota informativa sobre " + t[:30] + "..." for t in titles],
        "content": ["Contenido completo de la noticia. " * 5 for _ in titles],
        "source": np.random.choice(["Por Esto QRoo", "Novedades", "El Universal", "La Jornada", "Proceso"], len(titles)),
        "url": [f"https://example.com/news/{i}" for i in range(len(titles))],
        "published_at": dates,
        "text": titles,
    })


def _instagram_sample_data() -> pd.DataFrame:
    import numpy as np
    comments = [
        "Gran político para Quintana Roo 👏",
        "No estoy de acuerdo con sus propuestas",
        "Excelente trabajo en la región",
        "Gino Segura tiene mi apoyo total",
        "Necesitamos más transparencia en su gestión",
        "Muy buenas propuestas para el turismo",
        "Siempre pendiente de los ciudadanos",
        "Sus ideas son innovadoras",
        "Esperamos que cumpla sus promesas",
        "El mejor político de QRoo",
        "Hay que analizar mejor sus propuestas",
        "Apoyando a Gino desde Cancún",
        "Buen trabajo con la seguridad",
        "Me gustaría ver más acción",
        "Excelente liderazgo regional",
    ] * 6
    dates = pd.date_range("2023-06-01", periods=len(comments), freq="5D")
    return pd.DataFrame({
        "text": comments,
        "date": dates,
        "source": "instagram",
    })


# ─────────────────────────────────────────────────────────────
#  FUNCIÓN MAESTRA
# ─────────────────────────────────────────────────────────────

def ingest_all() -> dict:
    """Ejecuta toda la ingesta y retorna un diccionario con los DataFrames."""
    logger.info("=== Iniciando ingesta de datos ===")

    videos_df = fetch_youtube_videos()
    video_ids = videos_df["video_id"].tolist() if not videos_df.empty else []
    comments_df = fetch_youtube_comments(video_ids)
    trends_data = fetch_google_trends()
    news_df = fetch_news()
    instagram_df = load_instagram_data()

    # Guardar copias procesadas
    videos_df.to_csv(DATA_RAW / "youtube_videos.csv", index=False)
    comments_df.to_csv(DATA_RAW / "youtube_comments.csv", index=False)
    news_df.to_csv(DATA_RAW / "news.csv", index=False)
    instagram_df.to_csv(DATA_RAW / "instagram.csv", index=False)

    logger.info("=== Ingesta completada ===")

    return {
        "videos": videos_df,
        "comments": comments_df,
        "trends": trends_data,
        "news": news_df,
        "instagram": instagram_df,
    }


if __name__ == "__main__":
    data = ingest_all()
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            print(f"{k}: {len(v)} filas")
        elif isinstance(v, dict):
            print(f"{k}: {list(v.keys())}")
