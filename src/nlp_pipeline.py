"""
nlp_pipeline.py - Pipeline NLP completo para análisis de narrativas políticas
Incluye: limpieza, sentimiento, topic modeling, embeddings, clustering
"""

import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SENTIMENT_MODEL, EMBEDDING_MODEL, NUM_TOPICS,
    BERTOPIC_MIN_CLUSTER, DATA_PROCESSED,
)

# ─────────────────────────────────────────────────────────────
#  STOPWORDS EN ESPAÑOL
# ─────────────────────────────────────────────────────────────

STOPWORDS_ES = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
    "contra", "cual", "cuando", "de", "del", "desde", "donde", "durante",
    "e", "el", "ella", "ellas", "ellos", "en", "entre", "era", "es",
    "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estado",
    "estar", "este", "esto", "estos", "fue", "fuera", "fui", "gran",
    "ha", "hace", "hacia", "han", "hasta", "hay", "he", "hemos",
    "her", "him", "his", "hizo", "https", "i", "igual", "in",
    "ja", "jaja", "jajaja", "la", "las", "le", "les", "lo", "los",
    "mas", "más", "me", "mi", "mis", "mismo", "muy", "ni", "no",
    "nos", "nosotros", "o", "of", "on", "otro", "para", "pero",
    "poco", "por", "porque", "que", "qué", "quien", "se", "ser",
    "si", "sin", "sobre", "son", "su", "sus", "también", "te",
    "tenemos", "tienen", "todo", "todos", "tu", "tú", "un", "una",
    "uno", "unos", "usted", "ve", "ya", "yo", "ver", "cada",
    "así", "tan", "bien", "solo", "puede", "tiene", "hacer",
    "the", "and", "is", "are", "was", "http", "www", "com",
}


# ─────────────────────────────────────────────────────────────
#  LIMPIEZA DE TEXTO
# ─────────────────────────────────────────────────────────────

def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Limpia texto en español: normaliza, elimina ruido y stopwords."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Eliminar URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Eliminar menciones y hashtags (conservar la palabra)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)

    # Normalizar acentos comunes (opcional, ayuda a unificar tokens)
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "ü": "u", "ñ": "n",
    }
    for accented, plain in replacements.items():
        text = text.replace(accented, plain)

    # Eliminar caracteres especiales y emojis
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)

    # Tokenizar y filtrar
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_ES and len(t) > 2]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Aplica limpieza a un DataFrame."""
    df = df.copy()
    df[text_col] = df[text_col].astype(str)
    df["clean_text"] = df[text_col].apply(clean_text)
    df["text_length"] = df["clean_text"].str.split().str.len()
    df = df[df["text_length"] >= 2].copy()
    return df


# ─────────────────────────────────────────────────────────────
#  ANÁLISIS DE SENTIMIENTO
# ─────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Análisis de sentimiento usando modelo multilingüe de HuggingFace.
    Modelo por defecto: nlptown/bert-base-multilingual-uncased-sentiment
    (1-5 estrellas → convierte a -1/0/+1)
    """

    def __init__(self, model_name: str = SENTIMENT_MODEL):
        self.model_name = model_name
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"Cargando modelo de sentimiento: {self.model_name}")
            self.pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                max_length=512,
            )
            logger.info("Modelo de sentimiento cargado")
        except Exception as e:
            logger.error(f"Error cargando modelo transformer: {e}")
            logger.warning("Usando análisis de sentimiento basado en léxico como fallback.")
            self.pipeline = None

    def _lexicon_sentiment(self, text: str) -> dict:
        """Fallback: sentimiento basado en léxico en español."""
        positive_words = {
            "bien", "bueno", "excelente", "gran", "mejor", "apoyo", "apoya",
            "favorito", "lider", "liderazgo", "logro", "trabajo", "avance",
            "progreso", "exitoso", "positivo", "desarrollo", "beneficio",
            "triunfo", "ganador", "victoria", "esperanza", "confianza",
        }
        negative_words = {
            "mal", "malo", "peor", "critica", "falla", "corrupcion", "fraude",
            "mentira", "problema", "negativo", "error", "fracaso", "critico",
            "peligro", "riesgo", "desastre", "robo", "ladrón", "corrupto",
        }
        words = set(clean_text(text).split())
        pos = len(words & positive_words)
        neg = len(words & negative_words)

        if pos > neg:
            return {"label": "POSITIVE", "score": 0.7, "stars": 4}
        elif neg > pos:
            return {"label": "NEGATIVE", "score": 0.7, "stars": 2}
        else:
            return {"label": "NEUTRAL", "score": 0.6, "stars": 3}

    def analyze(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Analiza sentimiento de una lista de textos. Retorna lista de dicts."""
        results = []

        if self.pipeline is None:
            for text in texts:
                r = self._lexicon_sentiment(text)
                results.append(r)
            return results

        try:
            # Procesar en batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                # Truncar textos muy largos
                batch = [t[:512] if len(t) > 512 else t for t in batch]
                batch_results = self.pipeline(batch)
                for r in batch_results:
                    label = r["label"]  # "1 star" a "5 stars"
                    score = r["score"]

                    # Convertir estrellas a categorías
                    try:
                        stars = int(label.split()[0])
                    except (ValueError, IndexError):
                        # Para modelos que retornan POSITIVE/NEGATIVE/NEUTRAL
                        stars = 3 if "neutral" in label.lower() else (5 if "pos" in label.lower() else 1)

                    if stars <= 2:
                        sentiment = "NEGATIVE"
                    elif stars == 3:
                        sentiment = "NEUTRAL"
                    else:
                        sentiment = "POSITIVE"

                    results.append({"label": sentiment, "score": score, "stars": stars})

        except Exception as e:
            logger.error(f"Error en análisis de sentimiento: {e}")
            results = [self._lexicon_sentiment(t) for t in texts]

        return results

    def analyze_dataframe(self, df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
        """Agrega columnas de sentimiento al DataFrame."""
        df = df.copy()
        texts = df[text_col].fillna("").tolist()
        sentiments = self.analyze(texts)

        df["sentiment_label"] = [s["label"] for s in sentiments]
        df["sentiment_score"] = [s["score"] for s in sentiments]
        df["sentiment_stars"] = [s.get("stars", 3) for s in sentiments]

        # Valor numérico: -1, 0, 1
        label_map = {"NEGATIVE": -1, "NEUTRAL": 0, "POSITIVE": 1}
        df["sentiment_value"] = df["sentiment_label"].map(label_map)

        return df


# ─────────────────────────────────────────────────────────────
#  EMBEDDINGS Y CLUSTERING
# ─────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """Genera embeddings multilingüe y agrupa comentarios similares."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Cargando sentence-transformer: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Modelo de embeddings cargado")
        except Exception as e:
            logger.error(f"Error cargando sentence-transformer: {e}")
            self.model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        """Retorna embeddings para una lista de textos."""
        if self.model is None:
            # Fallback: TF-IDF simple
            return self._tfidf_embeddings(texts)

        try:
            return self.model.encode(texts, show_progress_bar=True, batch_size=32)
        except Exception as e:
            logger.error(f"Error generando embeddings: {e}")
            return self._tfidf_embeddings(texts)

    def _tfidf_embeddings(self, texts: list[str]) -> np.ndarray:
        """Fallback usando TF-IDF."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        vectorizer = TfidfVectorizer(max_features=500)
        matrix = vectorizer.fit_transform(texts)
        n_components = min(50, matrix.shape[1] - 1, matrix.shape[0] - 1)
        if n_components < 2:
            return matrix.toarray()
        svd = TruncatedSVD(n_components=n_components)
        return svd.fit_transform(matrix)

    def cluster(self, embeddings: np.ndarray, n_clusters: int = NUM_TOPICS) -> np.ndarray:
        """Agrupa embeddings con K-Means."""
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=min(n_clusters, len(embeddings) - 1), random_state=42, n_init=10)
        return km.fit_predict(embeddings)

    def reduce_2d(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce embeddings a 2D con UMAP (fallback: PCA)."""
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            return reducer.fit_transform(embeddings)
        except ImportError:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            return pca.fit_transform(embeddings)


# ─────────────────────────────────────────────────────────────
#  TOPIC MODELING
# ─────────────────────────────────────────────────────────────

class TopicModeler:
    """
    Detecta temas dominantes usando LDA (con scikit-learn) o BERTopic.
    Incluye extracción de keywords por tópico.
    """

    def __init__(self, n_topics: int = NUM_TOPICS, use_bertopic: bool = True):
        self.n_topics = n_topics
        self.use_bertopic = use_bertopic
        self.model = None
        self.vectorizer = None
        self.topic_labels = {}

    def fit(self, texts: list[str], embeddings: np.ndarray = None):
        """Entrena el modelo de tópicos."""
        if self.use_bertopic:
            self._fit_bertopic(texts, embeddings)
        else:
            self._fit_lda(texts)
        return self

    def _fit_bertopic(self, texts: list[str], embeddings: np.ndarray = None):
        try:
            from bertopic import BERTopic
            from sklearn.feature_extraction.text import CountVectorizer

            vectorizer_model = CountVectorizer(
                stop_words=list(STOPWORDS_ES),
                ngram_range=(1, 2),
                min_df=2,
            )
            topic_model = BERTopic(
                vectorizer_model=vectorizer_model,
                nr_topics=self.n_topics,
                min_topic_size=BERTOPIC_MIN_CLUSTER,
                language="multilingual",
                verbose=False,
            )

            if embeddings is not None:
                topics, probs = topic_model.fit_transform(texts, embeddings)
            else:
                topics, probs = topic_model.fit_transform(texts)

            self.model = topic_model
            self.topic_assignments = topics
            self._extract_topic_labels_bertopic()
            logger.info(f"BERTopic: {len(set(topics))} tópicos detectados")

        except Exception as e:
            logger.error(f"Error BERTopic: {e}. Usando LDA como fallback.")
            self._fit_lda(texts)

    def _fit_lda(self, texts: list[str]):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        self.vectorizer = CountVectorizer(
            max_features=1000,
            stop_words=list(STOPWORDS_ES),
            ngram_range=(1, 2),
            min_df=2,
        )
        X = self.vectorizer.fit_transform(texts)

        n = min(self.n_topics, X.shape[0] // 2)
        lda = LatentDirichletAllocation(
            n_components=n,
            random_state=42,
            max_iter=20,
            learning_method="online",
        )
        lda.fit(X)
        self.model = lda

        topic_assignments = lda.transform(X).argmax(axis=1)
        self.topic_assignments = topic_assignments
        self._extract_topic_labels_lda()
        logger.info(f"LDA: {n} tópicos ajustados")

    def _extract_topic_labels_lda(self):
        feature_names = self.vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
            self.topic_labels[topic_idx] = {
                "words": top_words,
                "label": f"Tópico {topic_idx + 1}: {', '.join(top_words[:3])}",
            }

    def _extract_topic_labels_bertopic(self):
        try:
            topic_info = self.model.get_topic_info()
            for _, row in topic_info.iterrows():
                tid = row["Topic"]
                if tid == -1:  # ruido/outliers de BERTopic — ignorar
                    continue
                words_scores = self.model.get_topic(tid)
                if words_scores:
                    # Filtrar palabras vacías o muy cortas
                    words = [w for w, _ in words_scores[:10] if len(w) > 2]
                    if not words:
                        continue
                    self.topic_labels[tid] = {
                        "words": words,
                        "label": f"Tópico {tid + 1}: {', '.join(words[:3])}",
                    }
        except Exception as e:
            logger.error(f"Error extrayendo labels BERTopic: {e}")

    def get_topics(self) -> dict:
        return self.topic_labels

    def transform(self, texts: list[str]) -> np.ndarray:
        """Asigna tópicos a nuevos textos."""
        if hasattr(self, "topic_assignments"):
            return self.topic_assignments
        return np.zeros(len(texts), dtype=int)


# ─────────────────────────────────────────────────────────────
#  MÉTRICAS AGREGADAS
# ─────────────────────────────────────────────────────────────

def compute_popularity_index(
    sentiment_df: pd.DataFrame,
    trends_data: dict,
    news_df: pd.DataFrame,
    politician: str,
) -> pd.DataFrame:
    """
    Construye un índice de popularidad compuesto por semana:
    - Sentimiento promedio (normalizado 0-100)
    - Volumen de menciones
    - Interés en Google Trends
    """
    # Sentimiento semanal
    if "date" in sentiment_df.columns and not sentiment_df["date"].isna().all():
        _parsed = pd.to_datetime(sentiment_df["date"], errors="coerce")
        _parsed = pd.Series(pd.DatetimeIndex(_parsed), index=sentiment_df.index)
        sentiment_df["week"] = _parsed.dt.to_period("W").dt.to_timestamp()
        weekly_sentiment = (
            sentiment_df.groupby("week")["sentiment_value"]
            .mean()
            .reset_index()
            .rename(columns={"sentiment_value": "avg_sentiment"})
        )
        weekly_volume = (
            sentiment_df.groupby("week")
            .size()
            .reset_index(name="mention_volume")
        )
        weekly = weekly_sentiment.merge(weekly_volume, on="week", how="outer")
    else:
        weekly = pd.DataFrame(columns=["week", "avg_sentiment", "mention_volume"])

    # Google Trends (normalizado 0-100)
    if politician in trends_data:
        trends_df = trends_data[politician].copy()
        _parsed = pd.to_datetime(trends_df["date"], errors="coerce")
        trends_df["week"] = pd.Series(pd.DatetimeIndex(_parsed), index=trends_df.index).dt.to_period("W").dt.to_timestamp()
        trends_weekly = trends_df.groupby("week")["interest"].mean().reset_index()
        weekly = weekly.merge(trends_weekly, on="week", how="outer")
    else:
        weekly["interest"] = 50  # neutro

    # Noticias por semana
    if not news_df.empty and "published_at" in news_df.columns:
        news_df = news_df.copy()
        _parsed = pd.to_datetime(news_df["published_at"], errors="coerce")
        news_df["week"] = pd.Series(pd.DatetimeIndex(_parsed), index=news_df.index).dt.to_period("W").dt.to_timestamp()
        news_weekly = news_df.groupby("week").size().reset_index(name="news_count")
        weekly = weekly.merge(news_weekly, on="week", how="outer")
    else:
        weekly["news_count"] = 0

    weekly = weekly.fillna(0).sort_values("week")

    # Normalizar a 0-100
    def normalize(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series([50.0] * len(s), index=s.index)
        return (s - mn) / (mx - mn) * 100

    if not weekly.empty:
        w_sent = normalize(weekly["avg_sentiment"].clip(-1, 1)) if "avg_sentiment" in weekly.columns else 50
        w_vol = normalize(weekly["mention_volume"]) if "mention_volume" in weekly.columns else 50
        w_trend = normalize(weekly["interest"]) if "interest" in weekly.columns else 50
        w_news = normalize(weekly["news_count"]) if "news_count" in weekly.columns else 50

        weekly["popularity_index"] = (
            0.35 * w_sent +
            0.30 * w_vol +
            0.25 * w_trend +
            0.10 * w_news
        )

    return weekly


# ─────────────────────────────────────────────────────────────
#  PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────

def run_pipeline(data: dict) -> dict:
    """
    Ejecuta el pipeline NLP completo sobre todos los datos.
    Retorna un diccionario con todos los resultados procesados.
    """
    logger.info("=== Iniciando pipeline NLP ===")
    results = {}

    # 1. Combinar fuentes de texto (comentarios)
    all_texts = []
    for key in ["comments", "instagram"]:
        df = data.get(key, pd.DataFrame())
        if not df.empty and "text" in df.columns:
            df_clean = preprocess_dataframe(df)
            all_texts.append(df_clean)

    if all_texts:
        combined = pd.concat(all_texts, ignore_index=True)
    else:
        combined = pd.DataFrame({"text": [], "clean_text": [], "source": [], "date": []})

    logger.info(f"Textos combinados: {len(combined)} registros")

    # 2. Sentimiento
    logger.info("Analizando sentimiento...")
    sentiment_analyzer = SentimentAnalyzer()

    if not combined.empty:
        combined = sentiment_analyzer.analyze_dataframe(combined)
        results["sentiment_df"] = combined
    else:
        results["sentiment_df"] = pd.DataFrame()

    # Sentimiento en noticias
    news_df = data.get("news", pd.DataFrame())
    if not news_df.empty:
        news_df = preprocess_dataframe(news_df, text_col="text" if "text" in news_df.columns else "title")
        news_df = sentiment_analyzer.analyze_dataframe(news_df, text_col="clean_text")
        results["news_sentiment"] = news_df
    else:
        results["news_sentiment"] = pd.DataFrame()

    # 3. Embeddings y clustering (solo si hay suficientes textos)
    min_texts_for_ml = 10
    if not combined.empty and len(combined) >= min_texts_for_ml:
        logger.info("Generando embeddings...")
        embedding_engine = EmbeddingEngine()
        texts_list = combined["clean_text"].tolist()
        embeddings = embedding_engine.encode(texts_list)

        clusters = embedding_engine.cluster(embeddings)
        combined["cluster"] = clusters
        results["embeddings"] = embeddings

        try:
            coords_2d = embedding_engine.reduce_2d(embeddings)
            combined["x_2d"] = coords_2d[:, 0]
            combined["y_2d"] = coords_2d[:, 1]
        except Exception as e:
            logger.warning(f"No se pudo reducir a 2D: {e}")

        # 4. Topic Modeling
        logger.info("Detectando tópicos...")
        topic_modeler = TopicModeler(n_topics=NUM_TOPICS, use_bertopic=True)
        topic_modeler.fit(texts_list, embeddings)
        combined["topic"] = topic_modeler.topic_assignments
        results["topic_labels"] = topic_modeler.get_topics()
        results["topic_modeler"] = topic_modeler
    else:
        logger.warning(f"Pocos textos ({len(combined)}). Saltando embeddings/tópicos.")
        results["embeddings"] = None
        results["topic_labels"] = {}

    results["combined_df"] = combined

    # 5. Índice de popularidad
    logger.info("Calculando índice de popularidad...")
    from config import POLITICIAN_NAME
    results["popularity_index"] = compute_popularity_index(
        combined if not combined.empty else pd.DataFrame({"date": [], "sentiment_value": []}),
        data.get("trends", {}),
        news_df if not news_df.empty else pd.DataFrame(),
        POLITICIAN_NAME,
    )

    # Guardar resultados (CSV persistidos en el repositorio)
    if not combined.empty:
        combined.to_csv(DATA_PROCESSED / "combined_analysis.csv", index=False)
    if not results["popularity_index"].empty:
        results["popularity_index"].to_csv(DATA_PROCESSED / "popularity_index.csv", index=False)
    if not results.get("news_sentiment", pd.DataFrame()).empty:
        results["news_sentiment"].to_csv(DATA_PROCESSED / "news_sentiment.csv", index=False)

    # Guardar topic_labels como JSON para carga rápida
    if results.get("topic_labels"):
        import json
        serializable = {str(k): v for k, v in results["topic_labels"].items()}
        with open(DATA_PROCESSED / "topic_labels.json", "w", encoding="utf-8") as fj:
            json.dump(serializable, fj, ensure_ascii=False)

    logger.info("=== Pipeline NLP completado ===")
    return results


if __name__ == "__main__":
    from ingestion import ingest_all
    data = ingest_all()
    results = run_pipeline(data)
    print("\n=== Resultados del Pipeline ===")
    print(f"Textos analizados: {len(results.get('combined_df', []))}")
    print(f"Tópicos detectados: {len(results.get('topic_labels', {}))}")
    if not results["popularity_index"].empty:
        print(f"Semanas en índice de popularidad: {len(results['popularity_index'])}")
