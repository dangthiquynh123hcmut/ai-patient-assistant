# Backend: FastAPI + LangGraph. Build: docker build -t patient-assistant-backend .
# EXTRAS: nhóm phụ thuộc tùy chọn trong pyproject.toml. Bật STT: --build-arg EXTRAS=postgres,stt (image lớn hơn nhiều).

# ---- build: cài package vào virtualenv riêng
FROM python:3.13-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
ARG EXTRAS=postgres
RUN pip install ".[${EXTRAS}]"

# ---- runtime: chỉ venv + migration, chạy bằng user thường
FROM python:3.13-slim
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home --uid 1000 app
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY alembic.ini ./
COPY migrations ./migrations
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
# Sau nginx: tin X-Forwarded-* để log đúng IP/scheme. Số worker chỉnh bằng UVICORN_WORKERS.
CMD ["sh", "-c", "exec uvicorn patient_assistant.interface.api.app:app --factory --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2} --proxy-headers --forwarded-allow-ips='*'"]
