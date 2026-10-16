FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts
COPY models ./models
COPY .streamlit ./.streamlit
ENV PORT=8501 API_URL=http://127.0.0.1:8000 PYTHONPATH=/app
EXPOSE 8501 8000
CMD ["python", "scripts/start_services.py"]