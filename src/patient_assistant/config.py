from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Cấu hình qua biến môi trường tiền tố `PA_` (hoặc file .env)."""

    model_config = SettingsConfigDict(env_prefix="PA_", env_file=".env", extra="ignore")

    timezone: str = "Asia/Ho_Chi_Minh"

    # LLM
    llm_provider: Literal["bedrock", "offline"] = "offline"
    # Để trống thì SDK tự lấy theo chuỗi credential chuẩn của AWS: AWS_REGION / AWS_PROFILE / ~/.aws/config.
    aws_region: str | None = None
    aws_profile: str | None = None
    # Inference profile trên bedrock-runtime; ID dạng `anthropic.claude-opus-5-5` sẽ đi qua Mantle.
    llm_model_id: str = "global.anthropic.claude-opus-4-6-v1"
    llm_effort_extract: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_effort_light: Literal["low", "medium", "high", "xhigh", "max"] = "low"

    # Lưu trữ: database_url rỗng -> in-memory (mất khi tắt server). SQLite: sqlite+aiosqlite:///./data/app.db
    # Postgres:
    # postgresql+asyncpg://user:pass@localhost:5432/patient_assistant
    database_url: str | None = None
    checkpointer: Literal["memory", "postgres"] = "memory"
    checkpointer_url: str | None = None  # postgresql://user:pass@localhost:5432/patient_assistant
    # Dev/test: server tự tạo bảng và nạp dữ liệu demo lúc khởi động. Production (docker compose): tắt cả hai —
    # schema do `alembic upgrade head` quản lý, dữ liệu demo do `python -m patient_assistant.interface.cli.seed`.
    auto_create_schema: bool = True
    checkpointer_pool_size: int = 10
    seed_demo_data: bool = True

    # Đăng nhập. Tài khoản demo (demo_data/users.json) chỉ được tạo khi DB chưa có tài khoản nào.
    session_ttl_hours: int = 12
    seed_password: str = "demo1234"
    cookie_secure: bool = False  # bật khi chạy sau HTTPS

    # STT
    stt_enabled: bool = False
    stt_model: str = "small"
    stt_compute_type: str = "int8"
    stt_confidence_threshold: float = 0.6

    # Giới hạn agent
    max_corrections: int = 2
    max_clarify_rounds: int = 3
    max_slot_search_rounds: int = 3
    slot_options: int = 3
