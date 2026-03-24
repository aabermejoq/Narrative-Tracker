"""
dashboard.py — Political Narrative Intelligence Dashboard
Streamlit + Plotly
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    POLITICIAN_NAME, COMPARISON_POLITICIANS, DATA_PROCESSED, DATA_CACHE,
)

# ─────────────────────────────────────────────────────────────
#  DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────

C = {
    "bg":        "#0d1117",
    "surface":   "#161b22",
    "surface2":  "#21262d",
    "border":    "#30363d",
    "positive":  "#3fb950",
    "negative":  "#f85149",
    "neutral":   "#d29922",
    "blue":      "#58a6ff",
    "purple":    "#bc8cff",
    "text":      "#e6edf3",
    "muted":     "#8b949e",
}

# Only keys that never conflict with per-chart overrides.
# xaxis/yaxis are applied separately via _theme() to avoid
# "multiple values for keyword argument" TypeError.
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,34,0.6)",
    font=dict(color=C["text"], family="Inter, Arial, sans-serif", size=12),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=C["border"], borderwidth=1),
    margin=dict(t=48, b=32, l=48, r=24),
)

_AXIS = dict(gridcolor=C["border"], zeroline=False, linecolor=C["border"])


def _theme(fig: go.Figure, height: int = 320, **extra_layout) -> go.Figure:
    """Apply the shared theme and axis style to any figure."""
    fig.update_layout(height=height, **PLOTLY_THEME, **extra_layout)
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    return fig

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title {
    font-size: 1.9rem; font-weight: 800; letter-spacing: -0.5px;
    color: #e6edf3; margin-bottom: 0.1rem;
}
.main-subtitle {
    font-size: 0.88rem; color: #8b949e; margin-bottom: 1.8rem;
    text-transform: uppercase; letter-spacing: 1px;
}
.section-title {
    font-size: 1.05rem; font-weight: 700; color: #e6edf3;
    text-transform: uppercase; letter-spacing: 1.5px;
    border-left: 3px solid #58a6ff; padding-left: 0.75rem;
    margin: 2rem 0 1rem 0;
}
.kpi-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    border-top: 2px solid #58a6ff;
}
.kpi-value { font-size: 2rem; font-weight: 800; color: #58a6ff; line-height: 1.1; }
.kpi-label { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 0.25rem; }
.kpi-sub   { font-size: 0.75rem; color: #8b949e; margin-top: 0.15rem; }
.kpi-delta { font-size: 0.75rem; font-weight: 600; margin-top: 0.35rem; }
.delta-pos { color: #3fb950; }
.delta-neg { color: #f85149; }
.delta-neu { color: #8b949e; }

.topic-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #bc8cff;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
}
.topic-id    { font-size: 0.72rem; font-weight: 700; color: #bc8cff; text-transform: uppercase; letter-spacing: 1px; }
.topic-words { font-size: 0.88rem; color: #e6edf3; margin-top: 0.3rem; }
.topic-count { font-size: 0.75rem; color: #8b949e; margin-top: 0.3rem; }

.status-row  { font-size: 0.82rem; color: #8b949e; line-height: 1.8; }
.status-ok   { color: #3fb950; font-weight: 600; }
.status-warn { color: #d29922; font-weight: 600; }

div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; }
</style>
"""


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Ingesting and processing data sources...")
def load_all_data():
    from ingestion import ingest_all
    from nlp_pipeline import run_pipeline
    raw_data = ingest_all()
    results = run_pipeline(raw_data)
    results["raw"] = raw_data
    return results


# ─────────────────────────────────────────────────────────────
#  CHART HELPERS
# ─────────────────────────────────────────────────────────────

def make_gauge(value: float, title: str, color: str = "#58a6ff") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 13, "color": C["muted"], "family": "Inter"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": C["border"],
                "tickfont": {"size": 10, "color": C["muted"]},
            },
            "bar": {"color": color, "thickness": 0.6},
            "bgcolor": C["surface"],
            "borderwidth": 1,
            "bordercolor": C["border"],
            "steps": [
                {"range": [0, 33],  "color": "rgba(248,81,73,0.12)"},
                {"range": [33, 66], "color": "rgba(210,153,34,0.12)"},
                {"range": [66, 100],"color": "rgba(63,185,80,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        },
        number={"suffix": "%", "font": {"size": 30, "color": color, "family": "Inter"}},
    ))
    fig.update_layout(
        height=210,
        margin=dict(t=36, b=8, l=16, r=16),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
    )
    return fig


def wordcloud_figure(word_freq: dict, title: str) -> go.Figure:
    if not word_freq:
        return go.Figure()
    words = list(word_freq.keys())[:50]
    freqs = [word_freq[w] for w in words]
    max_f = max(freqs) if freqs else 1

    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, len(words))
    y = rng.uniform(0, 10, len(words))
    sizes = [9 + 30 * (f / max_f) for f in freqs]
    palette = ["#58a6ff", "#bc8cff", "#3fb950", "#d29922", "#f85149", "#79c0ff", "#d2a8ff"]
    colors = [palette[i % len(palette)] for i in range(len(words))]

    fig = go.Figure()
    for word, xi, yi, sz, col in zip(words, x, y, sizes, colors):
        fig.add_annotation(
            x=xi, y=yi, text=word,
            font=dict(size=sz, color=col, family="Inter"),
            showarrow=False,
        )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=C["muted"])),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 10.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 10.5]),
        height=340,
        margin=dict(t=44, b=8, l=8, r=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=C["surface"],
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  HELPERS DE COMPARACIÓN TEMPORAL
# ─────────────────────────────────────────────────────────────

def _periodo_delta(
    df: pd.DataFrame,
    value_fn,
    date_col: str = "date",
    days: int = 30,
) -> tuple:
    """
    Calcula (valor_actual, pct_cambio) comparando los últimos `days` días
    contra el período anterior equivalente.
    value_fn recibe un DataFrame y devuelve un número.
    """
    if df.empty:
        return None, None
    fechas = pd.to_datetime(df[date_col], errors="coerce")
    # Eliminar timezone para comparación uniforme
    if hasattr(fechas, "dt") and fechas.dt.tz is not None:
        fechas = fechas.dt.tz_convert("UTC").dt.tz_localize(None)
    now = pd.Timestamp.now()
    reciente = df[fechas >= now - pd.Timedelta(days=days)]
    anterior = df[(fechas >= now - pd.Timedelta(days=days * 2)) &
                  (fechas < now - pd.Timedelta(days=days))]
    cur = value_fn(reciente) if not reciente.empty else 0
    prv = value_fn(anterior) if not anterior.empty else None
    if prv is None or prv == 0:
        return cur, None
    return cur, (cur - prv) / abs(prv) * 100


def _badge(pct: float | None, invertir: bool = False) -> tuple:
    """Devuelve (texto_badge, css_class) para una tarjeta KPI."""
    if pct is None:
        return "sin datos prev.", "delta-neu"
    positivo = pct > 0
    if invertir:
        css = "delta-neg" if positivo else "delta-pos"
    else:
        css = "delta-pos" if positivo else "delta-neg"
    arrow = "▲" if positivo else "▼"
    return f"{arrow} {abs(pct):.1f}% vs mes ant.", css


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────

def render_sidebar(data: dict):
    with st.sidebar:
        st.markdown(f"### {POLITICIAN_NAME}")
        st.markdown(
            '<div style="font-size:0.78rem;color:#8b949e;text-transform:uppercase;'
            'letter-spacing:1px;margin-bottom:1rem;">Panel de Narrativa Política</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        combined = data.get("combined_df", pd.DataFrame())
        if not combined.empty and "date" in combined.columns:
            valid_dates = (
                pd.Series(pd.DatetimeIndex(pd.to_datetime(combined["date"], errors="coerce")))
                .dropna()
            )
            if not valid_dates.empty:
                min_d = pd.Timestamp(valid_dates.min()).date()
                max_d = pd.Timestamp(valid_dates.max()).date()
                st.markdown(
                    '<div style="font-size:0.78rem;color:#8b949e;text-transform:uppercase;'
                    'letter-spacing:0.8px;margin-bottom:0.4rem;">Rango de fechas</div>',
                    unsafe_allow_html=True,
                )
                start_d, end_d = st.date_input(
                    "Rango de fechas",
                    value=(min_d, max_d),
                    min_value=min_d,
                    max_value=max_d,
                    label_visibility="collapsed",
                )
                st.session_state["date_range"] = (start_d, end_d)

        st.divider()
        st.markdown(
            '<div style="font-size:0.78rem;color:#8b949e;text-transform:uppercase;'
            'letter-spacing:0.8px;margin-bottom:0.4rem;">Fuentes de datos</div>',
            unsafe_allow_html=True,
        )
        sources = st.multiselect(
            "Fuentes",
            ["youtube", "instagram", "news"],
            default=["youtube", "instagram", "news"],
            label_visibility="collapsed",
        )
        st.session_state["selected_sources"] = sources

        st.divider()
        if st.button("Actualizar datos", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        from config import YOUTUBE_API_KEY, NEWS_API_KEY, EXCEL_DATA
        yt_ok   = YOUTUBE_API_KEY != "TU_YOUTUBE_API_KEY_AQUI"
        news_ok = NEWS_API_KEY    != "TU_NEWS_API_KEY_AQUI"
        excel_ok = EXCEL_DATA.exists()
        st.markdown(
            f"""<div class="status-row">
                YouTube API&nbsp;&nbsp;<span class="{'status-ok' if yt_ok else 'status-warn'}">
                    {'Conectado' if yt_ok else 'Demo'}</span><br>
                NewsAPI&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="{'status-ok' if news_ok else 'status-warn'}">
                    {'Conectado' if news_ok else 'Demo'}</span><br>
                Instagram&nbsp;&nbsp;&nbsp;&nbsp;<span class="{'status-ok' if excel_ok else 'status-warn'}">
                    {'Excel' if excel_ok else 'CSV demo'}</span><br>
                Google Trends&nbsp;<span class="{'status-ok' if excel_ok else 'status-warn'}">
                    {'Excel' if excel_ok else 'pytrends'}</span>
            </div>""",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
#  SECTION 1: OVERVIEW / KPIs
# ─────────────────────────────────────────────────────────────

def render_overview(data: dict):
    st.markdown('<div class="section-title">Resumen</div>', unsafe_allow_html=True)

    combined   = data.get("combined_df", pd.DataFrame())
    popularity = data.get("popularity_index", pd.DataFrame())
    videos     = data.get("raw", {}).get("videos", pd.DataFrame())

    total_mentions = len(combined)
    avg_pop    = popularity["popularity_index"].mean() if not popularity.empty and "popularity_index" in popularity.columns else 0
    pos_pct    = (combined["sentiment_label"] == "POSITIVE").mean() * 100 if not combined.empty and "sentiment_label" in combined.columns else 0
    neg_pct    = (combined["sentiment_label"] == "NEGATIVE").mean() * 100 if not combined.empty and "sentiment_label" in combined.columns else 0
    total_views = videos["views"].sum() if not videos.empty and "views" in videos.columns else 0

    # ── Deltas ──────────────────────────────────────────────
    if not combined.empty and "date" in combined.columns:
        _, d_mentions = _periodo_delta(combined, lambda df: len(df))
        _, d_pos  = _periodo_delta(
            combined[combined.get("sentiment_label", pd.Series(dtype=str)) == "POSITIVE"]
            if "sentiment_label" in combined.columns else pd.DataFrame(),
            lambda df: len(df) / max(len(combined), 1) * 100,
        )
        _, d_neg  = _periodo_delta(
            combined[combined["sentiment_label"] == "NEGATIVE"]
            if "sentiment_label" in combined.columns else pd.DataFrame(),
            lambda df: len(df) / max(len(combined), 1) * 100,
        )
    else:
        d_mentions = d_pos = d_neg = None

    # Popularidad: comparar últimas 2 semanas del índice
    d_pop = None
    if not popularity.empty and "popularity_index" in popularity.columns and len(popularity) >= 2:
        last = float(popularity["popularity_index"].iloc[-1])
        prev = float(popularity["popularity_index"].iloc[-2])
        d_pop = (last - prev) / abs(prev) * 100 if prev != 0 else None

    badge_mentions  = _badge(d_mentions)
    badge_pop       = _badge(d_pop)
    badge_pos       = _badge(d_pos)
    badge_neg       = _badge(d_neg, invertir=True)  # más negativo es malo

    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        (col1, "Menciones totales",     f"{total_mentions:,}",  "Todas las fuentes",       badge_mentions, C["blue"]),
        (col2, "Índice de popularidad", f"{avg_pop:.1f}",        "Compuesto / 100",         badge_pop,      C["purple"]),
        (col3, "Sentimiento positivo",  f"{pos_pct:.1f}%",       "Del total de menciones",  badge_pos,      C["positive"]),
        (col4, "Sentimiento negativo",  f"{neg_pct:.1f}%",       "Del total de menciones",  badge_neg,      C["negative"]),
        (col5, "Vistas en YouTube",     f"{total_views:,}",      "Acumulado",               (None, "delta-neu"), C["neutral"]),
    ]
    for col, label, value, sub, (badge_txt, badge_css), bc in kpis:
        with col:
            delta_html = (
                f'<div class="kpi-delta {badge_css}">{badge_txt}</div>'
                if badge_txt else ""
            )
            st.markdown(
                f"""<div class="kpi-card" style="border-top-color:{bc};">
                    <div class="kpi-value" style="color:{bc};">{value}</div>
                    <div class="kpi-label">{label}</div>
                    {delta_html}
                    <div class="kpi-sub">{sub}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Índice de popularidad en el tiempo
    if not popularity.empty and "popularity_index" in popularity.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=popularity["week"], y=popularity["popularity_index"],
            name="Índice de popularidad",
            line=dict(color=C["blue"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)",
            hovertemplate="%{x|%d %b}<br>Índice: %{y:.1f}<extra></extra>",
        ))
        _theme(fig, height=300,
            title=dict(text="Índice de popularidad compuesto — Semanal", font=dict(size=13, color=C["muted"])),
            xaxis_title="Fecha",
            yaxis_title="Índice (0–100)",
            yaxis=dict(range=[0, 100], gridcolor=C["border"], zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Distribución de sentimiento por fuente
    if not combined.empty and "sentiment_label" in combined.columns and "source" in combined.columns:
        sent_by_source = (
            combined.groupby(["source", "sentiment_label"])
            .size()
            .reset_index(name="count")
        )
        fig2 = px.bar(
            sent_by_source, x="source", y="count", color="sentiment_label",
            color_discrete_map={
                "POSITIVE": C["positive"],
                "NEGATIVE": C["negative"],
                "NEUTRAL":  C["neutral"],
            },
            title="Distribución de sentimiento por fuente",
            barmode="group",
            labels={"source": "Fuente", "count": "Menciones", "sentiment_label": "Sentimiento"},
        )
        _theme(fig2, height=290)
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  SECTION 2: SENTIMENT ANALYSIS
# ─────────────────────────────────────────────────────────────

def render_sentiment(data: dict):
    st.markdown('<div class="section-title">Análisis de Sentimiento</div>', unsafe_allow_html=True)

    combined = data.get("combined_df", pd.DataFrame())
    if combined.empty or "sentiment_label" not in combined.columns:
        st.info("No hay datos de sentimiento disponibles.")
        return

    col1, col2, col3 = st.columns(3)
    gauge_cfg = [
        ("POSITIVE", C["positive"], "Positivo"),
        ("NEUTRAL",  C["neutral"],  "Neutral"),
        ("NEGATIVE", C["negative"], "Negativo"),
    ]
    for col, (label, color, label_es) in zip([col1, col2, col3], gauge_cfg):
        pct = (combined["sentiment_label"] == label).mean() * 100
        with col:
            st.plotly_chart(make_gauge(pct, label_es, color), use_container_width=True)

    if "date" in combined.columns:
        combined_dt = combined.dropna(subset=["date"]).copy()
        combined_dt["week"] = pd.to_datetime(combined_dt["date"], errors="coerce").dt.to_period("W").dt.to_timestamp()
        combined_dt = combined_dt.dropna(subset=["week"])
        weekly_sent = (
            combined_dt.groupby(["week", "sentiment_label"])
            .size()
            .reset_index(name="count")
        )
        if not weekly_sent.empty:
            fig = px.area(
                weekly_sent, x="week", y="count", color="sentiment_label",
                color_discrete_map={
                    "POSITIVE": C["positive"],
                    "NEGATIVE": C["negative"],
                    "NEUTRAL":  C["neutral"],
                },
                title="Volumen de sentimiento — Tendencia semanal",
                labels={"week": "Semana", "count": "Menciones", "sentiment_label": "Sentimiento"},
            )
            _theme(fig, height=300)
            st.plotly_chart(fig, use_container_width=True)

    if "sentiment_score" in combined.columns:
        fig2 = px.histogram(
            combined, x="sentiment_score", color="sentiment_label",
            color_discrete_map={
                "POSITIVE": C["positive"],
                "NEGATIVE": C["negative"],
                "NEUTRAL":  C["neutral"],
            },
            title="Distribución de puntuación de sentimiento",
            nbins=40,
            labels={"sentiment_score": "Puntuación", "sentiment_label": "Categoría"},
            opacity=0.85,
        )
        _theme(fig2, height=270)
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  SECTION 3: NARRATIVE TOPICS
# ─────────────────────────────────────────────────────────────

def render_narratives(data: dict):
    st.markdown('<div class="section-title">Narrativas Dominantes</div>', unsafe_allow_html=True)

    topic_labels = data.get("topic_labels", {})
    combined     = data.get("combined_df", pd.DataFrame())

    if not topic_labels:
        st.info("Datos insuficientes para detección de narrativas. Se requieren más comentarios.")
        return

    topics_list = [(tid, info) for tid, info in topic_labels.items() if tid != -1]
    n_cols = min(3, len(topics_list))
    cols = st.columns(n_cols) if n_cols > 0 else [st.container()]

    for i, (tid, info) in enumerate(topics_list[:6]):
        with cols[i % n_cols]:
            words = info.get("words", [])[:8]
            word_str = "  ·  ".join(words)
            count = int((combined["topic"] == tid).sum()) if "topic" in combined.columns else 0
            st.markdown(
                f"""<div class="topic-card">
                    <div class="topic-id">Tópico {tid + 1}</div>
                    <div class="topic-words">{word_str}</div>
                    <div class="topic-count">{count:,} menciones</div>
                </div>""",
                unsafe_allow_html=True,
            )

    if topics_list:
        selected_topic_idx = st.selectbox(
            "Nube de palabras — selecciona tópico",
            options=[t[0] for t in topics_list],
            format_func=lambda x: f"Tópico {x + 1}: {', '.join(topic_labels[x]['words'][:3])}",
        )
        if selected_topic_idx in topic_labels:
            words = topic_labels[selected_topic_idx]["words"]
            word_freq = {w: max(10 - i * 1.5, 1) for i, w in enumerate(words)}
            st.plotly_chart(
                wordcloud_figure(word_freq, f"Términos clave — Tópico {selected_topic_idx + 1}"),
                use_container_width=True,
            )

    if "topic" in combined.columns:
        topic_counts = combined["topic"].value_counts().reset_index()
        topic_counts.columns = ["topic_id", "count"]
        topic_counts["label"] = topic_counts["topic_id"].apply(
            lambda x: f"T{x+1}: {', '.join(topic_labels.get(x, {}).get('words', ['?'])[:2])}"
        )
        fig = px.bar(
            topic_counts.head(10), x="label", y="count",
            title="Volumen de menciones por narrativa",
            color="count",
            color_continuous_scale="Blues",
            labels={"label": "Narrativa", "count": "Menciones"},
        )
        fig.update_coloraxes(showscale=False)
        _theme(fig, height=290)
        st.plotly_chart(fig, use_container_width=True)

    if "x_2d" in combined.columns and "y_2d" in combined.columns:
        sample = combined.sample(min(500, len(combined)), random_state=42)
        fig_scatter = px.scatter(
            sample, x="x_2d", y="y_2d",
            color=sample["topic"].astype(str) if "topic" in sample.columns else None,
            hover_data=["text"] if "text" in sample.columns else None,
            title="Mapa de comentarios — Proyección 2D",
            opacity=0.65,
            size_max=6,
        )
        fig_scatter.update_traces(marker=dict(size=4))
        _theme(fig_scatter, height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  SECTION 4: YOUTUBE ANALYTICS
# ─────────────────────────────────────────────────────────────

def render_youtube(data: dict):
    st.markdown('<div class="section-title">Analítica YouTube</div>', unsafe_allow_html=True)

    raw      = data.get("raw", {})
    videos   = raw.get("videos", pd.DataFrame())

    if videos.empty:
        st.info("No hay datos de YouTube disponibles.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Videos analizados",  len(videos))
    col2.metric("Vistas totales",     f"{videos['views'].sum():,}"          if "views"          in videos.columns else "N/A")
    col3.metric("Comentarios totales",f"{videos['comments_count'].sum():,}" if "comments_count" in videos.columns else "N/A")

    if "views" in videos.columns:
        top_vids = videos.nlargest(10, "views")
        fig = px.bar(
            top_vids, x="views", y="title",
            orientation="h",
            title="Top 10 videos por vistas",
            color="likes" if "likes" in top_vids.columns else None,
            color_continuous_scale="Blues",
            labels={"views": "Vistas", "title": "", "likes": "Likes"},
        )
        _theme(fig, height=400, yaxis={"categoryorder": "total ascending", "gridcolor": C["border"]})
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    if "likes" in videos.columns and "views" in videos.columns:
        videos_copy = videos.copy()
        videos_copy["tasa_engagement"] = (
            videos_copy["likes"] / videos_copy["views"].replace(0, 1) * 100
        ).round(2)
        fig2 = px.scatter(
            videos_copy, x="views", y="likes",
            hover_name="title",
            size="comments_count" if "comments_count" in videos_copy.columns else None,
            title="Engagement — Vistas vs. Likes",
            color="tasa_engagement",
            color_continuous_scale="Blues",
            labels={"views": "Vistas", "likes": "Likes", "tasa_engagement": "Tasa eng. (%)"},
            opacity=0.85,
        )
        _theme(fig2, height=340)
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Tabla completa de videos"):
        display_cols = [c for c in ["title", "channel", "published_at", "views", "likes", "comments_count", "url"] if c in videos.columns]
        col_cfg = {}
        if "url" in display_cols:
            col_cfg["url"] = st.column_config.LinkColumn("Enlace", display_text="Ver video")
        st.dataframe(
            videos[display_cols].sort_values("views", ascending=False),
            column_config=col_cfg,
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────
#  SECTION 5: GOOGLE TRENDS
# ─────────────────────────────────────────────────────────────

def render_trends(data: dict):
    st.markdown('<div class="section-title">Tendencias Google</div>', unsafe_allow_html=True)

    st.caption(
        "📊 **Nota:** Esta sección muestra búsquedas en Google de forma independiente. "
        "Los datos de cobertura mediática (noticias) se analizan por separado en la pestaña **Medios**."
    )

    trends = data.get("raw", {}).get("trends", {})
    if not trends:
        st.info("No hay datos de Google Trends disponibles.")
        return

    palette = [C["blue"], C["negative"], C["positive"], C["purple"], C["neutral"]]
    fig = go.Figure()

    for i, (kw, df) in enumerate(trends.items()):
        if isinstance(df, pd.DataFrame) and "interest" in df.columns:
            is_main = kw == POLITICIAN_NAME
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["interest"],
                name=kw,
                line=dict(
                    color=palette[i % len(palette)],
                    width=2.5 if is_main else 1.5,
                    dash="solid" if is_main else "dot",
                ),
                hovertemplate=f"{kw}<br>%{{x|%d %b}}: %{{y:.0f}}<extra></extra>",
            ))

    _theme(fig, height=380,
        title=dict(text=f"Interés de búsqueda: {POLITICIAN_NAME} vs. figuras comparables", font=dict(size=13, color=C["muted"])),
        xaxis_title="Fecha",
        yaxis_title="Interés relativo (0–100)",
    )
    st.plotly_chart(fig, use_container_width=True)

    avg_data = []
    for kw, df in trends.items():
        if isinstance(df, pd.DataFrame) and "interest" in df.columns:
            avg_data.append({
                "Figura política":     kw,
                "Interés promedio":    round(float(df["interest"].mean()), 1),
                "Pico máximo":         int(df["interest"].max()),
            })
    if avg_data:
        st.dataframe(
            pd.DataFrame(avg_data).sort_values("Interés promedio", ascending=False),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────
#  SECTION 6: MEDIA COVERAGE
# ─────────────────────────────────────────────────────────────

def render_news(data: dict):
    st.markdown('<div class="section-title">Cobertura Mediática</div>', unsafe_allow_html=True)

    news = data.get("news_sentiment", pd.DataFrame())

    # Mostrar error real de la API si existe
    if not news.empty and "_api_error" in news.columns:
        err_msg = news["_api_error"].iloc[0]
        st.error(f"**Error al conectar con NewsAPI:**\n\n{err_msg}")
        st.info(
            "💡 **Solución más común:** El plan gratuito de NewsAPI solo funciona en `localhost`. "
            "Para Streamlit Cloud necesitas el **plan Developer ($449/año)** o usar otra fuente de noticias."
        )
        return

    if news.empty:
        st.info("No hay datos de noticias disponibles.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de artículos", len(news))
    if "sentiment_label" in news.columns:
        col2.metric("Cobertura positiva", int((news["sentiment_label"] == "POSITIVE").sum()))
        col3.metric("Cobertura negativa", int((news["sentiment_label"] == "NEGATIVE").sum()))

    if "published_at" in news.columns:
        news_copy = news.copy()
        news_copy["week"] = pd.to_datetime(news_copy["published_at"], errors="coerce").dt.to_period("W").dt.to_timestamp()
        news_copy = news_copy.dropna(subset=["week"])

        if "sentiment_label" in news_copy.columns:
            weekly_news = news_copy.groupby(["week", "sentiment_label"]).size().reset_index(name="count")
            fig = px.bar(
                weekly_news, x="week", y="count", color="sentiment_label",
                color_discrete_map={
                    "POSITIVE": C["positive"],
                    "NEGATIVE": C["negative"],
                    "NEUTRAL":  C["neutral"],
                },
                title="Cobertura mediática — Volumen semanal por tono",
                barmode="stack",
                labels={"week": "Semana", "count": "Artículos", "sentiment_label": "Tono"},
            )
        else:
            weekly_news = news_copy.groupby("week").size().reset_index(name="count")
            fig = px.bar(weekly_news, x="week", y="count", title="Cobertura mediática — Volumen semanal")

        _theme(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

    if "source" in news.columns:
        source_counts = news["source"].value_counts().head(10).reset_index()
        source_counts.columns = ["source", "count"]
        fig2 = px.pie(
            source_counts, values="count", names="source",
            title="Cobertura por medio",
            hole=0.5,
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig2.update_traces(textfont_size=11)
        fig2.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C["text"], family="Inter"),
            margin=dict(t=44, b=8, l=8, r=8),
        )
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.plotly_chart(fig2, use_container_width=True)
        with col_r:
            st.markdown(
                '<div style="font-size:0.78rem;color:#8b949e;text-transform:uppercase;'
                'letter-spacing:0.8px;margin-bottom:0.5rem;">Artículos recientes</div>',
                unsafe_allow_html=True,
            )
            display_cols = [c for c in ["title", "source", "published_at", "sentiment_label", "url"] if c in news.columns]
            news_display = news[display_cols].sort_values("published_at", ascending=False).head(20).copy()
            col_cfg = {
                "title":         st.column_config.TextColumn("Título"),
                "source":        st.column_config.TextColumn("Medio"),
                "published_at":  st.column_config.DatetimeColumn("Fecha", format="DD/MM/YYYY"),
                "sentiment_label": st.column_config.TextColumn("Sentimiento"),
            }
            if "url" in news_display.columns:
                col_cfg["url"] = st.column_config.LinkColumn("Enlace", display_text="Ver artículo")
            st.dataframe(news_display, column_config=col_cfg, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  SECTION 7: INSTAGRAM
# ─────────────────────────────────────────────────────────────

def render_instagram(data: dict):
    st.markdown('<div class="section-title">Análisis Instagram</div>', unsafe_allow_html=True)

    combined = data.get("combined_df", pd.DataFrame())
    ig_data  = (
        combined[combined["source"] == "instagram"]
        if not combined.empty and "source" in combined.columns
        else pd.DataFrame()
    )

    if ig_data.empty:
        ig_raw = data.get("raw", {}).get("instagram", pd.DataFrame())
        if ig_raw.empty:
            st.info("Sin datos de Instagram. Coloca el Excel en la raíz del proyecto.")
            return
        ig_data = ig_raw

    col1, col2 = st.columns(2)
    col1.metric("Comentarios Instagram", len(ig_data))
    if "sentiment_label" in ig_data.columns:
        pos_pct = (ig_data["sentiment_label"] == "POSITIVE").mean() * 100
        col2.metric("Sentimiento positivo", f"{pos_pct:.1f}%")

        sent_counts = ig_data["sentiment_label"].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        fig = px.pie(
            sent_counts, values="count", names="sentiment",
            color="sentiment",
            color_discrete_map={
                "POSITIVE": C["positive"],
                "NEGATIVE": C["negative"],
                "NEUTRAL":  C["neutral"],
            },
            title="Distribución de sentimiento — Instagram",
            hole=0.5,
        )
        fig.update_layout(
            height=290,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C["text"], family="Inter"),
            margin=dict(t=44, b=8, l=8, r=8),
        )
        st.plotly_chart(fig, use_container_width=True)

    if "date" in ig_data.columns and not ig_data["date"].isna().all():
        ig_dt = ig_data.dropna(subset=["date"]).copy()
        ig_dt["week"] = pd.to_datetime(ig_dt["date"], errors="coerce").dt.to_period("W").dt.to_timestamp()
        ig_dt = ig_dt.dropna(subset=["week"])
        weekly = ig_dt.groupby("week").size().reset_index(name="count")
        fig2 = px.line(
            weekly, x="week", y="count",
            title="Volumen de comentarios — Instagram semanal",
            markers=True,
            labels={"week": "Semana", "count": "Comentarios"},
        )
        fig2.update_traces(line_color=C["purple"], marker=dict(size=5))
        _theme(fig2, height=270)
        st.plotly_chart(fig2, use_container_width=True)

    if "topic" in ig_data.columns:
        topic_labels = data.get("topic_labels", {})
        ig_topics = ig_data["topic"].value_counts().reset_index()
        ig_topics.columns = ["topic_id", "count"]
        ig_topics["label"] = ig_topics["topic_id"].apply(
            lambda x: f"T{x+1}: {', '.join(topic_labels.get(x, {}).get('words', ['?'])[:2])}"
        )
        fig3 = px.bar(
            ig_topics.head(8), x="label", y="count",
            title="Tópicos principales — Instagram",
            color="count",
            color_continuous_scale="Purples",
            labels={"label": "Tópico", "count": "Menciones"},
        )
        fig3.update_coloraxes(showscale=False)
        _theme(fig3, height=270)
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Muestra de comentarios"):
        display_cols = [c for c in ["text", "date", "likes", "sentiment_label", "topic"] if c in ig_data.columns]
        st.dataframe(ig_data[display_cols].head(100), use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title=f"{POLITICIAN_NAME} — Inteligencia Narrativa",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(f'<div class="main-title">{POLITICIAN_NAME}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Inteligencia de Narrativa Política &nbsp;|&nbsp; Quintana Roo, México</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Cargando y procesando fuentes de datos..."):
        data = load_all_data()

    render_sidebar(data)

    tabs = st.tabs([
        "Resumen",
        "Sentimiento",
        "Narrativas",
        "YouTube",
        "Tendencias Google",
        "Medios",
        "Instagram",
    ])

    with tabs[0]: render_overview(data)
    with tabs[1]: render_sentiment(data)
    with tabs[2]: render_narratives(data)
    with tabs[3]: render_youtube(data)
    with tabs[4]: render_trends(data)
    with tabs[5]: render_news(data)
    with tabs[6]: render_instagram(data)

    st.divider()
    st.markdown(
        f'<div style="font-size:0.75rem;color:#8b949e;">'
        f'Actualizado: {pd.Timestamp.now().strftime("%d/%m/%Y %H:%M UTC")}'
        f' &nbsp;|&nbsp; Plataforma de Inteligencia de Narrativa Política'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
