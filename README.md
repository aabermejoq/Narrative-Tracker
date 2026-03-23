# 🏛️ Narrative Tracker — Dashboard de Análisis Político

Dashboard de análisis de narrativas y popularidad para **Gino Segura** en Quintana Roo, México.

---

## Estructura del proyecto

```
Narrative-Tracker/
├── config.py                  ← ⚠️ PON TUS API KEYS AQUÍ
├── run.py                     ← Punto de entrada principal
├── requirements.txt
├── src/
│   ├── ingestion.py           ← Recolección de datos (YT, Trends, News, IG)
│   ├── nlp_pipeline.py        ← Sentimiento, tópicos, embeddings
│   └── dashboard.py           ← Dashboard Streamlit + Plotly
└── data/
    ├── raw/                   ← Datos crudos (CSV, JSON)
    │   └── instagram_comments.csv  ← ⬅ Sube tu CSV aquí
    ├── processed/             ← Datos procesados
    └── cache/                 ← Caché de APIs (TTL: 6h)
```

---

## Setup rápido

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota:** Si no tienes GPU, el build de CPU de torch funciona bien. Si tienes GPU NVIDIA, reemplaza `torch` por `torch[cuda]` en requirements.txt.

### 2. Configurar API Keys

Edita `config.py` y reemplaza los placeholders:

```python
YOUTUBE_API_KEY = "TU_YOUTUBE_API_KEY_AQUI"   # ← Aquí
NEWS_API_KEY    = "TU_NEWS_API_KEY_AQUI"       # ← Aquí
```

**Cómo obtener las keys:**
- **YouTube Data API v3**: Google Cloud Console → Crear proyecto → Habilitar YouTube Data API v3 → Credenciales → API Key
- **NewsAPI**: newsapi.org → Registro gratuito → Tu API Key

> Sin API keys el sistema corre en **modo demo** con datos realistas de ejemplo.

### 3. Subir CSV de Instagram

Copia tu CSV a `data/raw/instagram_comments.csv`.
Columnas reconocidas automáticamente: `text/comment/comentario` y `date/fecha/timestamp`.

### 4. Lanzar el dashboard

```bash
python run.py                  # Dashboard (default)
streamlit run src/dashboard.py # Alternativa directa
python run.py --all            # Ingesta + NLP + Dashboard
python run.py --ingest         # Solo recolectar datos
python run.py --nlp            # Solo pipeline NLP
```

Abre: **http://localhost:8501**

---

## Secciones del Dashboard

| Pestaña | Contenido |
|---------|-----------|
| 📊 Overview | KPIs principales, índice de popularidad compuesto |
| 😊 Sentimiento | Gauges, evolución temporal, distribución por fuente |
| 🗣️ Narrativas | Tópicos LDA/BERTopic, wordclouds, mapa de embeddings 2D |
| ▶️ YouTube | Top videos, engagement, tabla completa |
| 🔍 Google Trends | Comparativa con Mara Lezama y Claudia Sheinbaum |
| 📰 Noticias | Cobertura por tono, evolución semanal, fuentes |
| 📸 Instagram | Sentimiento, temas, volumen temporal |

---

## Arquitectura técnica

### Pipeline
```
Fuentes → ingestion.py → data/raw/ → nlp_pipeline.py → data/processed/ → dashboard.py
```

### Modelos NLP
- **Sentimiento**: `nlptown/bert-base-multilingual-uncased-sentiment` — modelo 1-5 estrellas multilingüe, fallback léxico automático
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — soporta español nativo
- **Topic Modeling**: BERTopic (preferido) con fallback a LDA scikit-learn
- **Reducción 2D**: UMAP (si disponible) o PCA

### Índice de Popularidad Compuesto
```
Popularidad = 0.35×Sentimiento + 0.30×Volumen + 0.25×Google Trends + 0.10×Noticias
```
Todo normalizado a escala 0–100.

### Caché
- TTL: 6 horas (ajustable en `config.py`)
- Ubicación: `data/cache/*.json`
- Forzar actualización: botón en el sidebar o `python run.py --ingest`

---

## Variables de entorno (alternativa a editar config.py)

```bash
export YOUTUBE_API_KEY="tu_key_aqui"
export NEWS_API_KEY="tu_key_aqui"
python run.py
```

---

## Sugerencias de mejora

1. **Twitter/X** — Agregar tweets para mayor cobertura en redes sociales
2. **Alertas automáticas** — Notificación cuando el sentimiento cae bruscamente
3. **Comparación multi-político** — Expandir para comparar varios candidatos
4. **spaCy español** — `es_core_news_lg` para lematización y NER avanzado
5. **PostgreSQL** — Reemplazar cache JSON por BD para histórico de largo plazo
6. **APScheduler/Celery** — Actualización automática cada N horas
7. **GDELT** — Cobertura de prensa internacional