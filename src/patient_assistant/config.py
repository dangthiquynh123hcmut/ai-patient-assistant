from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Cấu hình qua biến môi trường tiền tố `PA_` (hoặc file .env)."""

    model_config = SettingsConfigDict(env_prefix="PA_", env_file=".env", extra="ignore")

    timezone: str = "Asia/Ho_Chi_Minh"

    # LLM
    llm_provider: Literal["bedrock", "offline"] = "offline"
    aws_region: str = "us-east-1"
    llm_model_id: str = "anthropic.claude-opus-5-5"
    llm_effort_extract: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_effort_light: Literal["low", "medium", "high", "xhigh", "max"] = "low"

    # Lưu trữ: database_url rỗng -> in-memory (demo). Ví dụ Postgres:
    # postgresql+asyncpg://user:pass@localhost:5432/patient_assistant
    database_url: str | None = None
    checkpointer: Literal["memory", "postgres"] = "memory"
    checkpointer_url: str | None = None  # postgresql://user:pass@localhost:5432/patient_assistant

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
