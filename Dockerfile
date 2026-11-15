# Образ для RAG-сервиса и веб-интерфейса; сама модель работает в отдельном контейнере Ollama
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# PyTorch только для процессора: энкодеру e5 и реранкеру видеокарта не нужна
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app
COPY data/chunks.jsonl ./data/chunks.jsonl

# Стоп-слова, реранкер и векторный индекс готовятся при сборке, чтобы сервис запускался быстро и без интернета
RUN python -m src.retrieval build --encoder e5 \
    && python -c "from src.hybrid_retrieval import make_reranker; make_reranker()"
