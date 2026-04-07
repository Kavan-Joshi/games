FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev

COPY . .

EXPOSE 7860

CMD [".venv/bin/uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
