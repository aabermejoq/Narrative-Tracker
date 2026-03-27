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
    INSTAGRAM_CSV, EXCEL_DATA, RSS_FEEDS,
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
#  NOTICIAS — RSS (primario) + NewsAPI (fallback)
# ─────────────────────────────────────────────────────────────

def fetch_rss_news(query: str = POLITICIAN_NAME) -> pd.DataFrame:
    """
    Obtiene noticias desde feeds RSS de medios mexicanos.
    Filtra artículos que mencionen al político o palabras clave relacionadas.
    """
    cache_name = f"rss_news_{query.replace(' ', '_')}"
    cached = _load_cache(cache_name)
    if cached:
        return pd.DataFrame(cached)

    try:
        import feedparser
        from time import mktime

        # Keywords para filtrar artículos relevantes
        search_terms = [v.lower() for v in POLITICIAN_QUERY_VARIANTS] + [
            "quintana roo", "qroo", "cancun", "cancún",
            "chetumal", "playa del carmen", "tulum",
        ]

        cutoff = datetime.now() - timedelta(days=NEWS_LOOKBACK_DAYS)
        articles = []
        feed_ok = 0

        for feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:100]:
                    title   = getattr(entry, "title",   "") or ""
                    summary = getattr(entry, "summary", "") or ""
                    combined = (title + " " + summary).lower()

                    if not any(kw in combined for kw in search_terms):
                        continue

                    # Fecha de publicación
                    pub = None
                    for attr in ("published_parsed", "updated_parsed"):
                        val = getattr(entry, attr, None)
                        if val:
                            try:
                                pub = datetime.fromtimestamp(mktime(val))
                            except Exception:
                                pass
                            break
                    if pub is None:
                        pub = datetime.now()

                    if pub < cutoff:
                        continue

                    articles.append({
                        "title":        title,
                        "description":  summary[:400],
                        "content":      summary[:800],
                        "source":       feed.feed.get("title", feed_url),
                        "url":          getattr(entry, "link", ""),
                        "published_at": pub,
                        "text":         title + " " + summary,
                    })
                feed_ok += 1
            except Exception as fe:
                logger.debug(f"Feed no disponible {feed_url}: {fe}")
                continue

        logger.info(f"RSS: {feed_ok}/{len(RSS_FEEDS)} feeds leídos, {len(articles)} artículos")

        if not articles:
            return _news_sample_data()

        df = pd.DataFrame(articles)
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df = df.drop_duplicates(subset=["title"]).sort_values("published_at", ascending=False)
        df = df.reset_index(drop=True)

        _save_cache(cache_name, df.to_dict("records"))
        return df

    except ImportError:
        logger.warning("feedparser no instalado. Intentando NewsAPI.")
        return _fetch_news_api(query)
    except Exception as e:
        logger.error(f"Error RSS: {e}")
        return _news_sample_data()


def _fetch_news_api(query: str = POLITICIAN_NAME) -> pd.DataFrame:
    """Obtiene noticias desde NewsAPI (fallback cuando RSS falla)."""
    if NEWS_API_KEY == "TU_NEWS_API_KEY_AQUI":
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

        rows = []
        for a in data.get("articles", []):
            rows.append({
                "title":        a.get("title", ""),
                "description":  a.get("description", ""),
                "content":      (a.get("content") or "")[:800],
                "source":       a.get("source", {}).get("name", ""),
                "url":          a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
            })

        df = pd.DataFrame(rows)
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True).dt.tz_localize(None)
        df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
        logger.info(f"NewsAPI: {len(df)} noticias obtenidas")
        return df

    except Exception as e:
        logger.error(f"Error NewsAPI: {e}")
        return _news_sample_data()


def fetch_news(query: str = POLITICIAN_NAME) -> pd.DataFrame:
    """Obtiene noticias: RSS primero, NewsAPI como fallback."""
    return fetch_rss_news(query)


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
#  EXCEL (strict OOXML — "data gino.xlsx")
# ─────────────────────────────────────────────────────────────

def _read_strict_xlsx_cells(filepath: Path, sheet_idx: int) -> list:
    """
    Lee todas las celdas de una hoja de un xlsx strict OOXML
    (namespace purl.oclc.org que openpyxl no soporta).
    Devuelve lista de (row_num, col_letter, value).
    """
    import zipfile
    import xml.etree.ElementTree as ET

    NS = "http://purl.oclc.org/ooxml/spreadsheetml/main"

    with zipfile.ZipFile(filepath, "r") as z:
        # Shared strings
        ss_raw = z.read("xl/sharedStrings.xml").decode("utf-8")
        ss_root = ET.fromstring(ss_raw)
        strings = []
        for si in ss_root:
            parts = [t.text for t in si.iter(f"{{{NS}}}t") if t.text]
            strings.append("".join(parts))

        sheet_file = f"xl/worksheets/sheet{sheet_idx + 1}.xml"
        s_raw = z.read(sheet_file).decode("utf-8")
        s_root = ET.fromstring(s_raw)

        cells = []
        for row_el in s_root.iter(f"{{{NS}}}row"):
            row_num = int(row_el.get("r", 0))
            for c in row_el:
                ref = c.get("r", "")
                col = "".join(ch for ch in ref if ch.isalpha())
                v_el = c.find(f"{{{NS}}}v")
                ctype = c.get("t", "")
                raw = v_el.text if v_el is not None else None
                if raw is None:
                    continue
                if ctype == "s":
                    val = strings[int(raw)]
                else:
                    try:
                        f = float(raw)
                        val = int(f) if f == int(f) else f
                    except (ValueError, TypeError):
                        val = raw
                cells.append((row_num, col, val))
        return cells


def load_excel_google_trends(filepath: Path = EXCEL_DATA) -> dict:
    """
    Lee la hoja 'data gino' del Excel:
      Fila 1: metadata (ignorar)
      Fila 2: Semana | Keyword1 | Keyword2 | ...
      Fila 3+: YYYY-MM-DD | valores
    Devuelve dict {keyword: DataFrame(date, interest)}.
    """
    if not filepath.exists():
        logger.warning(f"Excel no encontrado: {filepath}")
        return {}

    try:
        cells = _read_strict_xlsx_cells(filepath, sheet_idx=0)

        # Organizar en dict row → {col: value}
        from collections import defaultdict
        rows: dict = defaultdict(dict)
        for rn, col, val in cells:
            rows[rn][col] = val

        sorted_rows = sorted(rows.items())
        # Fila 2 = headers
        header_row = sorted_rows[1][1] if len(sorted_rows) > 1 else {}
        col_order = sorted(header_row.keys())
        headers = [header_row.get(c, c) for c in col_order]  # e.g. Semana, Morena:..., Gino:...

        results = {}
        for rn, row_dict in sorted_rows[2:]:  # skip fila 1 (metadata) y fila 2 (headers)
            date_val = row_dict.get(col_order[0], None)
            if date_val is None:
                continue
            for i, col in enumerate(col_order[1:], 1):
                kw_raw = headers[i]
                # Normalizar nombre: "Gino Segura: (Quintana Roo)" → "Gino Segura"
                kw = kw_raw.split(":")[0].strip() if isinstance(kw_raw, str) else str(kw_raw)
                interest = row_dict.get(col, 0)
                try:
                    interest = int(interest)
                except (ValueError, TypeError):
                    interest = 0
                if kw not in results:
                    results[kw] = []
                results[kw].append({"date": str(date_val), "interest": interest})

        final = {}
        for kw, records in results.items():
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            final[kw] = df

        logger.info(f"Excel Trends: {list(final.keys())}")
        return final

    except Exception as e:
        logger.error(f"Error leyendo Excel Trends: {e}")
        return {}


def load_excel_instagram(filepath: Path = EXCEL_DATA) -> pd.DataFrame:
    """
    Lee la hoja 'instagram' del Excel.
    Formato: columna A única, patrón alternado:
      Fila comment: texto del comentario
      Fila meta:    'Xw Y likesReply'  →  X semanas atrás, Y likes
    Devuelve DataFrame(text, date, likes, source).
    """
    if not filepath.exists():
        logger.warning(f"Excel no encontrado: {filepath}")
        return pd.DataFrame()

    try:
        import re
        cells = _read_strict_xlsx_cells(filepath, sheet_idx=1)

        # Solo columna A, ordenadas por fila
        col_a = [(rn, str(val)) for rn, col, val in cells if col == "A"]
        col_a.sort(key=lambda x: x[0])

        META_RE = re.compile(r"^(\d+)w\s*(\d+)\s*likes?Reply", re.IGNORECASE)
        EDIT_RE = re.compile(r"^Edited\s*[·•]\s*\d+w", re.IGNORECASE)

        records = []
        i = 0
        while i < len(col_a):
            _, text = col_a[i]
            text = text.strip()

            # Saltar líneas vacías, metadata de post y líneas de metadatos
            if not text or EDIT_RE.match(text) or META_RE.match(text):
                i += 1
                continue

            # Buscar metadata en la siguiente línea
            weeks_ago, likes = None, 0
            if i + 1 < len(col_a):
                _, nxt = col_a[i + 1]
                m = META_RE.match(nxt.strip())
                if m:
                    weeks_ago = int(m.group(1))
                    likes = int(m.group(2))
                    i += 2
                    records.append({"text": text, "weeks_ago": weeks_ago, "likes": likes})
                    continue

            records.append({"text": text, "weeks_ago": weeks_ago, "likes": likes})
            i += 1

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        ref_date = datetime.now()
        df["date"] = df["weeks_ago"].apply(
            lambda w: (ref_date - timedelta(weeks=int(w))).strftime("%Y-%m-%d")
            if pd.notna(w) and w is not None else None
        )
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["source"] = "instagram"
        df = df[df["text"].str.len() > 3].copy()

        logger.info(f"Excel Instagram: {len(df)} comentarios cargados")
        return df[["text", "date", "likes", "source"]].copy()

    except Exception as e:
        logger.error(f"Error leyendo Excel Instagram: {e}")
        return pd.DataFrame()


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
    news_df = fetch_news()

    # Preferir datos del Excel si existe
    if EXCEL_DATA.exists():
        logger.info(f"Usando Excel: {EXCEL_DATA}")
        trends_data = load_excel_google_trends()
        instagram_df = load_excel_instagram()
        if instagram_df.empty:
            instagram_df = load_instagram_data()
        if not trends_data:
            trends_data = fetch_google_trends()
    else:
        trends_data = fetch_google_trends()
        instagram_df = load_instagram_data()

    # Guardar copias en CSV (persistidas en el repositorio)
    videos_df.to_csv(DATA_RAW / "youtube_videos.csv", index=False)
    comments_df.to_csv(DATA_RAW / "youtube_comments.csv", index=False)
    news_df.to_csv(DATA_RAW / "news.csv", index=False)
    instagram_df.to_csv(DATA_RAW / "instagram.csv", index=False)

    # Guardar trends: un CSV por keyword
    trends_dir = DATA_RAW / "trends"
    trends_dir.mkdir(exist_ok=True)
    for kw, df_t in trends_data.items():
        safe_kw = kw.replace(" ", "_").replace("/", "_").replace(":", "")
        df_t.to_csv(trends_dir / f"{safe_kw}.csv", index=False)

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
