# Image runtime unique — sert l'API, la démo Streamlit et la boucle quotidienne.
FROM python:3.12-slim

# libgomp1 : requis par LightGBM (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) dépendances (couche cache séparée)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) code
COPY src/ ./src/
COPY streamlit_app/ ./streamlit_app/
COPY config/ ./config/

# utilisateur non-root
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501

# défaut : l'API (surchargé par docker-compose pour les autres services)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
