from __future__ import annotations

from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Cấu hình qua biến môi trường tiền tố `PA_` (hoặc file .env)."""

    model_config = SettingsConfigDict(env_prefix="PA_", env_file=".env", extra="ignore")

    timezone: str = "Asia/Ho_Chi_Minh"

    # Nhà cung cấp LLM: bedrock (Claude qua IAM, mặc định) | openai (API key). Server không khởi động nếu không gọi
    # được LLM của nhà cung cấp đã chọn.
    llm_provider: Literal["bedrock", "openai"] = "bedrock"
    # OpenAI: key trống thì SDK đọc OPENAI_API_KEY. base_url chỉ đặt khi dùng dịch vụ tương thích OpenAI.
    # reasoning: gửi reasoning_effort hay không; trống = tự nhận theo tên model (gpt-5*, o1/o3/o4*).
    openai_api_key: str | None = None
    openai_model: str = "gpt-5-mini"
    openai_base_url: str | None = None
    openai_reasoning: bool | None = None

    # LLM: Claude trên Amazon Bedrock (khi llm_provider=bedrock).
    # Để trống thì SDK tự lấy theo chuỗi credential chuẩn của AWS: AWS_REGION / AWS_PROFILE / ~/.aws/config.
    aws_region: str | None = None
    aws_profile: str | None = None
    # Inference profile trên bedrock-runtime; ID dạng `anthropic.claude-opus-5-5` sẽ đi qua Mantle.
    llm_model_id: str = "global.anthropic.claude-opus-4-6-v1"
    llm_effort_extract: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_effort_light: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    # Effort của agent (PA_AGENT_MODE=booking|full). Đo thật: low cho kết quả như medium mà nhanh gấp ~3 lần.
    llm_effort_agent: Literal["low", "medium", "high", "xhigh", "max"] = "low"

    # Lưu trữ: chỉ PostgreSQL (dữ liệu nghiệp vụ + state hội thoại của LangGraph), bắt buộc.
    # postgresql+asyncpg://user:pass@localhost:5432/patient_assistant — checkpointer dùng cùng database này.
    database_url: str | None = None
    # Dev/test: server tự chạy migration Alembic (`upgrade head`) và nạp dữ liệu demo lúc khởi động.
    # Production (docker compose): tắt cả hai —
    # schema do `alembic upgrade head` quản lý, dữ liệu demo do `python -m patient_assistant.interface.cli.seed`.
    auto_create_schema: bool = True
    checkpointer_pool_size: int = 10
    seed_demo_data: bool = True

    # Đăng nhập bằng JWT. Tài khoản demo (demo_data/users.json) còn thiếu theo username được tạo khi seed.
    # Khóa ký HS256, tối thiểu 32 byte, giống nhau ở mọi worker/instance. Tạo: python -c "import secrets;
    # print(secrets.token_urlsafe(48))". Để trống: mỗi tiến trình tự sinh khóa tạm (chỉ hợp cho dev một worker —
    # khởi động lại là mọi người phải đăng nhập lại).
    jwt_secret: str | None = None
    access_token_ttl_minutes: int = 15
    session_ttl_hours: int = 12  # hạn phiên = hạn refresh token (tuyệt đối, tính từ lúc đăng nhập)
    seed_password: str = "demo1234"
    cookie_secure: bool = False  # bật khi chạy sau HTTPS
    allow_registration: bool = True  # người dùng tự tạo tài khoản bệnh nhân ở trang đăng ký
    # Swagger UI /docs, ReDoc /redoc, /openapi.json. Chỉ bật khi phát triển: liệt kê toàn bộ API ra ngoài.
    api_docs: bool = False

    # STT
    stt_enabled: bool = False
    stt_model: str = "small"
    stt_compute_type: str = "int8"
    # Ngôn ngữ mặc định khi client không gửi: "en", "vi" hoặc "auto" (whisper tự nhận ngôn ngữ).
    # Ép sai ngôn ngữ làm whisper giải mã lại nhiều lần: chậm gấp nhiều lần và dễ "dịch" bừa.
    stt_language: Literal["en", "vi", "auto"] = "en"
    stt_confidence_threshold: float = 0.6

    # Cách đặt lịch: "workflow" = luồng cứng (hỏi khoa -> ngày -> giờ -> chọn slot); "agent" = LLM tự hiểu ý và tra
    # dữ liệu bằng tool (hiểu "giống lịch trước, cách 7 ngày"), code chỉ giữ thẻ xác nhận trước khi ghi.
    booking_mode: Literal["workflow", "agent"] = "workflow"
    # Phạm vi agent: workflow | booking (chỉ đặt/đổi/hủy lịch) | full (mọi việc: đặt lịch, báo cáo, tra cứu, hỏi
    # tài liệu, nhiều việc trong một câu). Để trống: theo `booking_mode` (agent -> booking) cho tương thích.
    agent_mode: Literal["workflow", "booking", "full"] | None = None
    max_agent_tool_calls: int = 10

    @field_validator(
        "agent_mode",
        "openai_api_key",
        "openai_base_url",
        "openai_reasoning",
        "aws_region",
        "aws_profile",
        mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def active_llm_model(self) -> str:
        """Model đang dùng theo nhà cung cấp đã chọn (để log / báo lỗi)."""
        return self.openai_model if self.llm_provider == "openai" else self.llm_model_id

    @property
    def effective_agent_mode(self) -> Literal["workflow", "booking", "full"]:
        if self.agent_mode:
            return self.agent_mode
        return "booking" if self.booking_mode == "agent" else "workflow"

    # Giới hạn agent
    max_corrections: int = 2
    max_clarify_rounds: int = 3
    max_questions: int = 10
    max_slot_search_rounds: int = 3
    slot_options: int = 3
