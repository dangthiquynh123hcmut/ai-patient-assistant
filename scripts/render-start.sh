#!/bin/sh
# Lệnh khởi động trên Render (render.yaml -> dockerCommand). Viết thành script vì Render không xử lý đúng
# chuỗi có dấu ngoặc trong dockerCommand.
# Mỗi lần khởi động: migration + bổ sung dữ liệu demo (idempotent), rồi chạy API. Render đặt cổng qua PORT.
set -e
alembic upgrade head
python -m patient_assistant.interface.cli.seed
exec uvicorn patient_assistant.interface.api.app:app --factory \
    --host 0.0.0.0 --port "${PORT:-8000}" --workers "${UVICORN_WORKERS:-1}" \
    --proxy-headers --forwarded-allow-ips='*'
