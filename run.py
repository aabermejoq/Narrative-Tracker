"""
run.py - Script de entrada principal
Opciones:
  python run.py --ingest     Solo recolectar datos
  python run.py --nlp        Solo correr pipeline NLP
  python run.py --dashboard  Lanzar el dashboard (por defecto)
  python run.py --all        Todo en orden
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Narrative Tracker - Análisis Político")
    parser.add_argument("--ingest",    action="store_true", help="Ejecutar ingesta de datos")
    parser.add_argument("--nlp",       action="store_true", help="Ejecutar pipeline NLP")
    parser.add_argument("--dashboard", action="store_true", help="Lanzar dashboard Streamlit")
    parser.add_argument("--all",       action="store_true", help="Ingesta + NLP + Dashboard")
    args = parser.parse_args()

    base = Path(__file__).parent

    if args.all or args.ingest:
        print("\n[1/3] Iniciando ingesta de datos...")
        from src.ingestion import ingest_all
        data = ingest_all()
        print(f"  ✓ YouTube videos: {len(data['videos'])}")
        print(f"  ✓ YouTube comentarios: {len(data['comments'])}")
        print(f"  ✓ Noticias: {len(data['news'])}")
        print(f"  ✓ Instagram: {len(data['instagram'])}")
        print(f"  ✓ Google Trends: {list(data['trends'].keys())}")

    if args.all or args.nlp:
        print("\n[2/3] Ejecutando pipeline NLP...")
        if "data" not in dir():
            from src.ingestion import ingest_all
            data = ingest_all()
        from src.nlp_pipeline import run_pipeline
        results = run_pipeline(data)
        print(f"  ✓ Textos analizados: {len(results.get('combined_df', []))}")
        print(f"  ✓ Tópicos: {len(results.get('topic_labels', {}))}")

    # Por defecto o con --dashboard: lanzar Streamlit
    if args.all or args.dashboard or not any([args.ingest, args.nlp]):
        print("\n[3/3] Lanzando dashboard...")
        dashboard_path = base / "src" / "dashboard.py"
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path), "--server.port=8501"],
            check=True,
        )


if __name__ == "__main__":
    main()
