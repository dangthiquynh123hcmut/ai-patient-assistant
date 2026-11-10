# Backend: FastAPI + LangGraph. Build: docker build -t patient-assistant-backend .
# EXTRAS: nhóm phụ thuộc tùy chọn trong pyproject.toml. Bật STT: --build-arg EXTRAS=stt (image lớn hơn nhiều;
# với compose: đặt BACKEND_EXTRAS=stt trong .env);
# image chạy test: EXTRAS=dev (service `test` trong docker-compose.yml).

# ---- build: cài package vào virtualenv riêng
FROM python:3.13-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
# 1) Thư viện: chỉ cài lại khi pyproject.toml đổi (package rỗng tạm thời để pip đọc được metadata).
#    Cache của pip giữ lại giữa các lần build, nên đổi phụ thuộc cũng không phải tải lại toàn bộ.
COPY pyproject.toml README.md ./
RUN mkdir -p src/patient_assistant && touch src/patient_assistant/__init__.py
ARG EXTRAS=
RUN --mount=type=cache,target=/root/.cache/pip pip install ".${EXTRAS:+[$EXTRAS]}"
# 2) Mã nguồn: sửa code chỉ cài lại package của mình (vài giây), không đụng tới thư viện.
COPY src ./src
RUN pip install --no-cache-dir --no-deps --force-reinstall .

# ---- runtime: chỉ venv + migration, chạy bằng user thường
FROM python:3.13-slim
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home --uid 1000 app \
    # Chỗ lưu model Whisper (volume `stt-models`); tạo sẵn để volume mới thuộc user app, ghi được.
    && mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
# Migration nằm trong package (đã cài vào venv); alembic.ini chỉ để chạy lệnh `alembic` ở /app.
COPY alembic.ini ./
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
# Sau nginx: tin X-Forwarded-* để log đúng IP/scheme. Số worker chỉnh bằng UVICORN_WORKERS.
CMD ["sh", "-c", "exec uvicorn patient_assistant.interface.api.app:app --factory --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2} --proxy-headers --forwarded-allow-ips='*'"]
