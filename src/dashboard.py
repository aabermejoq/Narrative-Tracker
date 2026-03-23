"""
dashboard.py - Dashboard interactivo de análisis de narrativas políticas
Construido con Streamlit + Plotly
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

COLORS = {
    "primary":  "#1a1a2e",
    "accent":   "#16213e",
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#f39c12",
    "blue":     "#3498db",
    "purple":   "#9b59b6",
}


# ─────────────────────────────────────────────────────────────
#  CARGA DE DATOS (CON CACHÉ STREAMLIT)
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Recopilando datos...")
def load_all_data():
    """Carga y procesa todos los datos (con caché de 1 hora)."""
    from ingestion import ingest_all
    from nlp_pipeline import run_pipeline
    raw_data = ingest_all()
    results = run_pipeline(raw_data)
    results["raw"] = raw_data
    return results


# ─────────────────────────────────────────────────────────────
#  HELPERS DE VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────

def sentiment_color(label: str) -> str:
    return {"POSITIVE": COLORS["positive"], "NEGATIVE": COLORS["negative"]}.get(label, COLORS["neutral"])


def make_gauge(value: float, title: str, color: str = "#3498db") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 33], "color": "#e74c3c22"},
                {"range": [33, 66], "color": "#f39c1222"},
                {"range": [66, 100], "color": "#2ecc7122"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "value": value},
        },
        number={"suffix": "%", "font": {"size": 28}},
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def wordcloud_figure(word_freq: dict, title: str) -> go.Figure:
    """Simula wordcloud con scatter plot de texto escalado."""
    if not word_freq:
        return go.Figure()
    words = list(word_freq.keys())[:50]
    freqs = [word_freq[w] for w in words]
    max_f = max(freqs) if freqs else 1

    np.random.seed(42)
    x = np.random.uniform(0, 10, len(words))
    y = np.random.uniform(0, 10, len(words))
    sizes = [8 + 32 * (f / max_f) for f in freqs]
    colors = [f"rgb({np.random.randint(50,200)},{np.random.randint(100,230)},{np.random.randint(150,255)})" for _ in words]

    fig = go.Figure()
    for i, (word, xi, yi, sz, col) in enumerate(zip(words, x, y, sizes, colors)):
        fig.add_annotation(
            x=xi, y=yi, text=word,
            font=dict(size=sz, color=col, family="Arial Black"),
            showarrow=False,
        )
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=350, margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor="rgba(26,26,46,0.4)",
        plot_bgcolor="rgba(26,26,46,0.4)",
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────

def render_sidebar(data: dict):
    with st.sidebar:
        st.markdown(f"## 🏛️ {POLITICIAN_NAME}")
        st.markdown("**Dashboard de Narrativas Políticas**")
        st.divider()

        combined = data.get("combined_df", pd.DataFrame())
        if not combined.empty and "date" in combined.columns:
            valid_dates = combined["date"].dropna()
            if not valid_dates.empty:
                min_d = valid_dates.min().date()
                max_d = valid_dates.max().date()
                start_d, end_d = st.date_input(
                    "Rango de fechas",
                    value=(min_d, max_d),
                    min_value=min_d,
                    max_value=max_d,
                )
                st.session_state["date_range"] = (start_d, end_d)

        st.divider()
        sources = st.multiselect(
            "Fuentes",
            ["youtube", "instagram", "news"],
            default=["youtube", "instagram", "news"],
        )
        st.session_state["selected_sources"] = sources

        st.divider()
        st.markdown("### 🔄 Actualizar datos")
        if st.button("Refrescar ahora", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.markdown("### ⚙️ API Status")
        from config import YOUTUBE_API_KEY, NEWS_API_KEY
        st.markdown(f"YouTube: {'✅ Configurada' if YOUTUBE_API_KEY != 'TU_YOUTUBE_API_KEY_AQUI' else '⚠️ Demo mode'}")
        st.markdown(f"NewsAPI: {'✅ Configurada' if NEWS_API_KEY != 'TU_NEWS_API_KEY_AQUI' else '⚠️ Demo mode'}")
        st.markdown("Instagram: ✅ CSV")
        st.markdown("Google Trends: ✅ pytrends")


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 1: OVERVIEW / KPIs
# ─────────────────────────────────────────────────────────────

def render_overview(data: dict):
    st.markdown('<div class="section-header">📊 Overview</div>', unsafe_allow_html=True)

    combined = data.get("combined_df", pd.DataFrame())
    popularity = data.get("popularity_index", pd.DataFrame())
    videos = data.get("raw", {}).get("videos", pd.DataFrame())
    news = data.get("news_sentiment", pd.DataFrame())

    # KPIs
    total_mentions = len(combined)
    avg_pop = popularity["popularity_index"].mean() if not popularity.empty and "popularity_index" in popularity.columns else 0
    pos_pct = (combined["sentiment_label"] == "POSITIVE").mean() * 100 if not combined.empty and "sentiment_label" in combined.columns else 0
    neg_pct = (combined["sentiment_label"] == "NEGATIVE").mean() * 100 if not combined.empty and "sentiment_label" in combined.columns else 0
    total_views = videos["views"].sum() if not videos.empty and "views" in videos.columns else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        (col1, "💬 Menciones", f"{total_mentions:,}", "Total comentarios"),
        (col2, "📈 Popularidad", f"{avg_pop:.0f}/100", "Índice compuesto"),
        (col3, "😊 Positivo", f"{pos_pct:.0f}%", "Del total"),
        (col4, "😤 Negativo", f"{neg_pct:.0f}%", "Del total"),
        (col5, "▶️ Vistas YT", f"{total_views:,}", "Total YouTube"),
    ]
    for col, icon_title, value, subtitle in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{icon_title}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{subtitle}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Índice de popularidad en el tiempo
    if not popularity.empty and "popularity_index" in popularity.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=popularity["week"], y=popularity["popularity_index"],
            name="Índice de Popularidad",
            line=dict(color="#3498db", width=3),
            fill="tozeroy",
            fillcolor="rgba(52,152,219,0.15)",
        ))
        fig.update_layout(
            title="Índice de Popularidad Compuesto (Semanal)",
            xaxis_title="Fecha", yaxis_title="Índice (0-100)",
            height=320, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,26,46,0.3)",
            font=dict(color="#ccc"),
            yaxis=dict(range=[0, 100]),
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
            color_discrete_map={"POSITIVE": COLORS["positive"], "NEGATIVE": COLORS["negative"], "NEUTRAL": COLORS["neutral"]},
            title="Distribución de Sentimiento por Fuente",
            barmode="group",
        )
        fig2.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,26,46,0.3)", font=dict(color="#ccc"))
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 2: SENTIMIENTO
# ─────────────────────────────────────────────────────────────

def render_sentiment(data: dict):
    st.markdown('<div class="section-header">😊 Análisis de Sentimiento</div>', unsafe_allow_html=True)

    combined = data.get("combined_df", pd.DataFrame())
    if combined.empty or "sentiment_label" not in combined.columns:
        st.info("No hay datos de sentimiento disponibles.")
        return

    col1, col2, col3 = st.columns(3)
    for col, label in zip([col1, col2, col3], ["POSITIVE", "NEUTRAL", "NEGATIVE"]):
        pct = (combined["sentiment_label"] == label).mean() * 100
        with col:
            color = {"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "NEUTRAL": "#f39c12"}[label]
            st.plotly_chart(make_gauge(pct, label, color), use_container_width=True)

    # Evolución temporal
    if "date" in combined.columns:
        combined_dt = combined.dropna(subset=["date"]).copy()
        combined_dt["week"] = pd.to_datetime(combined_dt["date"]).dt.to_period("W").dt.to_timestamp()
        weekly_sent = (
            combined_dt.groupby(["week", "sentiment_label"])
            .size()
            .reset_index(name="count")
        )
        if not weekly_sent.empty:
            fig = px.area(
                weekly_sent, x="week", y="count", color="sentiment_label",
                color_discrete_map={"POSITIVE": COLORS["positive"], "NEGATIVE": COLORS["negative"], "NEUTRAL": COLORS["neutral"]},
                title="Evolución del Sentimiento (Semanal)",
            )
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,26,46,0.3)", font=dict(color="#ccc"))
            st.plotly_chart(fig, use_container_width=True)

    # Score distribution
    if "sentiment_score" in combined.columns:
        fig2 = px.histogram(
            combined, x="sentiment_score", color="sentiment_label",
            color_discrete_map={"POSITIVE": COLORS["positive"], "NEGATIVE": COLORS["negative"], "NEUTRAL": COLORS["neutral"]},
            title="Distribución de Scores de Sentimiento",
            nbins=30,
        )
        fig2.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,26,46,0.3)", font=dict(color="#ccc"))
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 3: NARRATIVAS / TÓPICOS
# ─────────────────────────────────────────────────────────────

def render_narratives(data: dict):
    st.markdown('<div class="section-header">🗣️ Narrativas Dominantes</div>', unsafe_allow_html=True)

    topic_labels = data.get("topic_labels", {})
    combined = data.get("combined_df", pd.DataFrame())

    if not topic_labels:
        st.info("No hay suficientes datos para detectar narrativas. Se necesitan más comentarios.")
        return

    # Mostrar tópicos como cards
    topics_list = [(tid, info) for tid, info in topic_labels.items() if tid != -1]
    n_cols = min(3, len(topics_list))
    cols = st.columns(n_cols) if n_cols > 0 else [st.container()]

    for i, (tid, info) in enumerate(topics_list[:6]):
        with cols[i % n_cols]:
            words = info.get("words", [])[:8]
            word_str = " · ".join(words)
            count = (combined["topic"] == tid).sum() if "topic" in combined.columns else 0
            st.markdown(f"""
            <div style="background:rgba(26,26,46,0.6);border-radius:10px;padding:1rem;border-left:3px solid #9b59b6;margin-bottom:1rem;">
                <div style="color:#9b59b6;font-weight:700;font-size:1rem;">🎯 Tópico {tid + 1}</div>
                <div style="color:#ccc;font-size:0.85rem;margin-top:0.3rem;">{word_str}</div>
                <div style="color:#aaa;font-size:0.78rem;margin-top:0.3rem;">{count} menciones</div>
            </div>
            """, unsafe_allow_html=True)

    # Wordcloud del tópico seleccionado
    if topics_list:
        selected_topic_idx = st.selectbox(
            "Ver nube de palabras del tópico:",
            options=[t[0] for t in topics_list],
            format_func=lambda x: f"Tópico {x + 1}: {', '.join(topic_labels[x]['words'][:3])}",
        )
        if selected_topic_idx in topic_labels:
            words = topic_labels[selected_topic_idx]["words"]
            word_freq = {w: max(10 - i * 1.5, 1) for i, w in enumerate(words)}
            st.plotly_chart(wordcloud_figure(word_freq, f"Palabras clave - Tópico {selected_topic_idx + 1}"), use_container_width=True)

    # Distribución de menciones por tópico
    if "topic" in combined.columns:
        topic_counts = combined["topic"].value_counts().reset_index()
        topic_counts.columns = ["topic_id", "count"]
        topic_counts["label"] = topic_counts["topic_id"].apply(
            lambda x: f"T{x+1}: {', '.join(topic_labels.get(x, {}).get('words', ['?'])[:2])}"
        )
        fig = px.bar(
            topic_counts.head(10), x="label", y="count",
            title="Volumen de Menciones por Narrativa",
            color="count", color_continuous_scale="viridis",
        )
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,26,46,0.3)", font=dict(color="#ccc"))
        st.plotly_chart(fig, use_container_width=True)

    # Scatter 2D de embeddings (si disponible)
    if "x_2d" in combined.columns and "y_2d" in combined.columns:
        sample = combined.sample(min(500, len(combined)), random_state=42)
        fig_scatter = px.scatter(
            sample, x="x_2d", y="y_2d",
            color=sample["topic"].astype(str) if "topic" in sample.columns else None,
            hover_data=["text"] if "text" in sample.columns else None,
            title="Mapa de Comentarios (Embeddings 2D)",
            opacity=0.7,
        )
        fig_scatter.update_layout(
            height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,26,46,0.3)", font=dict(color="#ccc")
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 4: YOUTUBE ANALYTICS
# ─────────────────────────────────────────────────────────────

def render_youtube(data: dict):
    st.markdown('<div class="section-header">▶️ YouTube Analytics</div>', unsafe_allow_html=True)

    raw = data.get("raw", {})
    videos = raw.get("videos", pd.DataFrame())
    comments = raw.get("comments", pd.DataFrame())

    if videos.empty:
        st.info("No hay datos de YouTube disponibles.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Videos analizados", len(videos))
    col2.metric("Vistas totales", f"{videos['views'].sum():,}" if "views" in videos.columns else "N/A")
    col3.metric("Comentarios totales", f"{videos['comments_count'].sum():,}" if "comments_count" in videos.columns else "N/A")

    # Top videos por vistas
    if "views" in videos.columns:
        top_vids = videos.nlargest(10, "views")
        fig = px.bar(
            top_vids, x="views", y="title",
            orientation="h",
            title="Top 10 Videos por Vistas",
            color="likes" if "likes" in top_vids.columns else None,
            color_continuous_scale="blues",
        )
        fig.update_layout(
            height=400, yaxis={"categoryorder": "total ascending"},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,26,46,0.3)", font=dict(color="#ccc"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Engagement (likes/views ratio)
    if "likes" in videos.columns and "views" in videos.columns:
        videos_copy = videos.copy()
        videos_copy["engagement_rate"] = (videos_copy["likes"] / videos_copy["views"].replace(0, 1) * 100).round(2)

        fig2 = px.scatter(
            videos_copy, x="views", y="likes",
            hover_name="title", size="comments_count" if "comments_count" in videos_copy.columns else None,
            title="Engagement: Vistas vs Likes",
            color="engagement_rate", color_continuous_scale="viridis",
        )
        fig2.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,26,46,0.3)", font=dict(color="#ccc"))
        st.plotly_chart(fig2, use_container_width=True)

    # Tabla de videos
    with st.expander("Ver todos los videos"):
        display_cols = [c for c in ["title", "channel", "published_at", "views", "likes", "comments_count", "url"] if c in videos.columns]
        st.dataframe(videos[display_cols].sort_values("views", ascending=False), use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 5: GOOGLE TRENDS
# ─────────────────────────────────────────────────────────────

def render_trends(data: dict):
    st.markdown('<div class="section-header">🔍 Google Trends</div>', unsafe_allow_html=True)

    trends = data.get("raw", {}).get("trends", {})
    if not trends:
        st.info("No hay datos de Google Trends disponibles.")
        return

    fig = go.Figure()
    palette = [COLORS["blue"], "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]

    for i, (kw, df) in enumerate(trends.items()):
        if isinstance(df, pd.DataFrame) and "interest" in df.columns:
            color = palette[i % len(palette)]
            width = 3 if kw == POLITICIAN_NAME else 1.5
            dash = "solid" if kw == POLITICIAN_NAME else "dot"
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["interest"],
                name=kw,
                line=dict(color=color, width=width, dash=dash),
            ))

    fig.update_layout(
        title=f"Interés en Google Trends: {POLITICIAN_NAME} vs comparables",
        xaxis_title="Fecha", yaxis_title="Interés relativo (0-100)",
        height=380, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,26,46,0.3)", font=dict(color="#ccc"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla comparativa de promedios
    avg_data = []
    for kw, df in trends.items():
        if isinstance(df, pd.DataFrame) and "interest" in df.columns:
            avg_data.append({"Político": kw, "Interés Promedio": round(df["interest"].mean(), 1), "Máximo": df["interest"].max()})

    if avg_data:
        st.dataframe(
            pd.DataFrame(avg_data).sort_values("Interés Promedio", ascending=False),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 6: NOTICIAS
# ─────────────────────────────────────────────────────────────

def render_news(data: dict):
    st.markdown('<div class="section-header">📰 Cobertura Mediática</div>', unsafe_allow_html=True)

    news = data.get("news_sentiment", pd.DataFrame())
    if news.empty:
        st.info("No hay datos de noticias disponibles.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total noticias", len(news))
    if "sentiment_label" in news.columns:
        pos = (news["sentiment_label"] == "POSITIVE").sum()
        neg = (news["sentiment_label"] == "NEGATIVE").sum()
        col2.metric("Cobertura positiva", pos)
        col3.metric("Cobertura negativa", neg)

    # Cobertura en el tiempo
    if "published_at" in news.columns:
        news_copy = news.copy()
        news_copy["week"] = pd.to_datetime(news_copy["published_at"]).dt.to_period("W").dt.to_timestamp()
        weekly_news = news_copy.groupby(["week", "sentiment_label"]).size().reset_index(name="count") if "sentiment_label" in news_copy.columns else news_copy.groupby("week").size().reset_index(name="count")

        if "sentiment_label" in weekly_news.columns:
            fig = px.bar(
                weekly_news, x="week", y="count", color="sentiment_label",
                color_discrete_map={"POSITIVE": COLORS["positive"], "NEGATIVE": COLORS["negative"], "NEUTRAL": COLORS["neutral"]},
                title="Cobertura Mediática por Semana y Tono",
                barmode="stack",
            )
        else:
            fig = px.bar(weekly_news, x="week", y="count", title="Cobertura Mediática por Semana")

        fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,26,46,0.3)", font=dict(color="#ccc"))
        st.plotly_chart(fig, use_container_width=True)

    # Distribución por fuente
    if "source" in news.columns:
        source_counts = news["source"].value_counts().head(10).reset_index()
        source_counts.columns = ["source", "count"]
        fig2 = px.pie(source_counts, values="count", names="source", title="Distribución por Medio", hole=0.4)
        fig2.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"))
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.plotly_chart(fig2, use_container_width=True)

        # Noticias recientes
        with col_r:
            st.markdown("**Noticias recientes:**")
            display_cols = [c for c in ["title", "source", "published_at", "sentiment_label"] if c in news.columns]
            st.dataframe(
                news[display_cols].sort_values("published_at", ascending=False).head(15),
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 7: INSTAGRAM
# ─────────────────────────────────────────────────────────────

def render_instagram(data: dict):
    st.markdown('<div class="section-header">📸 Análisis Instagram</div>', unsafe_allow_html=True)

    combined = data.get("combined_df", pd.DataFrame())
    ig_data = combined[combined["source"] == "instagram"] if not combined.empty and "source" in combined.columns else pd.DataFrame()

    if ig_data.empty:
        ig_raw = data.get("raw", {}).get("instagram", pd.DataFrame())
        if ig_raw.empty:
            st.info("No hay datos de Instagram. Sube tu CSV a data/raw/instagram_comments.csv")
            return
        ig_data = ig_raw

    col1, col2 = st.columns(2)
    col1.metric("Comentarios Instagram", len(ig_data))

    if "sentiment_label" in ig_data.columns:
        pos_pct = (ig_data["sentiment_label"] == "POSITIVE").mean() * 100
        col2.metric("Sentimiento positivo", f"{pos_pct:.0f}%")

        # Pie de sentimiento
        sent_counts = ig_data["sentiment_label"].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        fig = px.pie(
            sent_counts, values="count", names="sentiment",
            color="sentiment",
            color_discrete_map={"POSITIVE": COLORS["positive"], "NEGATIVE": COLORS["negative"], "NEUTRAL": COLORS["neutral"]},
            title="Sentimiento en Instagram",
            hole=0.45,
        )
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"))
        st.plotly_chart(fig, use_container_width=True)

    # Evolución temporal
    if "date" in ig_data.columns and not ig_data["date"].isna().all():
        ig_dt = ig_data.dropna(subset=["date"]).copy()
        ig_dt["week"] = pd.to_datetime(ig_dt["date"]).dt.to_period("W").dt.to_timestamp()
        weekly = ig_dt.groupby("week").size().reset_index(name="count")
        fig2 = px.line(weekly, x="week", y="count", title="Volumen de Comentarios Instagram por Semana", markers=True)
        fig2.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,26,46,0.3)", font=dict(color="#ccc"))
        st.plotly_chart(fig2, use_container_width=True)

    # Topics en Instagram
    if "topic" in ig_data.columns:
        topic_labels = data.get("topic_labels", {})
        ig_topics = ig_data["topic"].value_counts().reset_index()
        ig_topics.columns = ["topic_id", "count"]
        ig_topics["label"] = ig_topics["topic_id"].apply(
            lambda x: f"T{x+1}: {', '.join(topic_labels.get(x, {}).get('words', ['?'])[:2])}"
        )
        fig3 = px.bar(ig_topics.head(8), x="label", y="count", title="Temas principales en Instagram", color="count", color_continuous_scale="purples")
        fig3.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,26,46,0.3)", font=dict(color="#ccc"))
        st.plotly_chart(fig3, use_container_width=True)

    # Muestra de comentarios
    with st.expander("Ver comentarios"):
        display_cols = [c for c in ["text", "date", "sentiment_label", "topic"] if c in ig_data.columns]
        st.dataframe(ig_data[display_cols].head(100), use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title=f"Narrativas: {POLITICIAN_NAME}",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #1a1a2e, #3498db);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #666; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .kpi-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px; padding: 1.2rem 1.5rem;
        border-left: 4px solid #3498db; color: white;
    }
    .kpi-value { font-size: 2.2rem; font-weight: 800; color: #3498db; }
    .kpi-label { font-size: 0.85rem; color: #aaa; }
    .section-header { font-size: 1.4rem; font-weight: 700; margin-top: 2rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)
    st.markdown(f'<div class="main-header">🏛️ {POLITICIAN_NAME} — Dashboard de Narrativas</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Análisis político en tiempo real · Quintana Roo, México</div>', unsafe_allow_html=True)

    with st.spinner("Cargando y procesando datos... (primera carga puede tardar ~2 minutos)"):
        data = load_all_data()

    render_sidebar(data)

    tabs = st.tabs([
        "📊 Overview",
        "😊 Sentimiento",
        "🗣️ Narrativas",
        "▶️ YouTube",
        "🔍 Google Trends",
        "📰 Noticias",
        "📸 Instagram",
    ])

    with tabs[0]:
        render_overview(data)
    with tabs[1]:
        render_sentiment(data)
    with tabs[2]:
        render_narratives(data)
    with tabs[3]:
        render_youtube(data)
    with tabs[4]:
        render_trends(data)
    with tabs[5]:
        render_news(data)
    with tabs[6]:
        render_instagram(data)

    st.divider()
    st.caption(f"Última actualización: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} · Dashboard de Narrativas Políticas")


if __name__ == "__main__":
    main()
