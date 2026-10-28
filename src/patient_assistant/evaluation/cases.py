"""Định dạng case đánh giá (JSONL, mỗi dòng một case). Xem docs/EVALUATION.md."""

from __future__ import annotations

from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from patient_assistant.domain.common import SttFlag

WRITE_TOOLS = ("save_patient_report", "create_appointment")
DEFAULT_NOW = "2026-10-28T10:00:00+07:00"  # Thứ Tư — khớp các case trong dataset


class UserTurn(BaseModel):
    """Lượt mở đầu: người dùng gửi tin nhắn (text hoặc transcript sau STT)."""

    model_config = ConfigDict(extra="forbid")

    text: str
    input_source: Literal["text", "voice"] = "text"
    stt_flags: list[SttFlag] = Field(default_factory=list)


class Reply(BaseModel):
    """Câu trả lời cho một interrupt, dùng lần lượt theo thứ tự interrupt xuất hiện."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["reply", "confirm", "edit", "cancel", "choose", "other_time"]
    text: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    option_no: int | None = None


class Faults(BaseModel):
    """Lỗi được tiêm có chủ đích để đo khả năng không báo thành công giả."""

    model_config = ConfigDict(extra="forbid")

    llm_unavailable: bool = False
    save_unavailable: bool = False
    booking_unavailable: bool = False
    slot_taken_after_choice: bool = False  # người khác đặt mất slot ngay trước khi xác nhận
    prebook_window: dict[str, str] | None = None  # {"department": "PHCN", "date": "2026-11-03", "until": "12:00"}


class LLMScript(BaseModel):
    """Đầu ra LLM tiêm sẵn (cố ý sai) để đo validator và self-correction. Hết kịch bản thì dùng LLM thật."""

    model_config = ConfigDict(extra="forbid")

    reports: list[dict[str, Any]] = Field(default_factory=list)
    bookings: list[dict[str, Any]] = Field(default_factory=list)


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str | None = None
    outcome: str  # OutcomeKind, hoặc "waiting" nếu kịch bản kết thúc khi agent còn đang hỏi
    should_clarify: bool = False
    fields: dict[str, Any] = Field(default_factory=dict)  # giá trị cuối cùng (bản sẽ được lưu/đặt)
    tools: list[str] = Field(default_factory=list)  # tool bắt buộc phải gọi
    must_not_write: bool = False
    booked_option: int | None = None  # số thứ tự slot (trong lần đề xuất ĐẦU TIÊN) phải được đặt
    response_contains: list[str] = Field(default_factory=list)


class Labels(BaseModel):
    """Nhãn cho đánh giá validator ở lần trích xuất đầu tiên."""

    model_config = ConfigDict(extra="forbid")

    injected_error: bool = False
    first_attempt_valid: bool | None = None  # None -> suy ra: not (should_clarify or injected_error)
    issue_codes: list[str] = Field(default_factory=list)  # mã lỗi validator phải bắt được ở lần đầu


class Concurrency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessions: int = Field(ge=2, le=20)


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    suite: Literal["report", "booking", "conversation"]
    category: str
    description: str = ""
    now: datetime = Field(default_factory=lambda: datetime.fromisoformat(DEFAULT_NOW))
    patient_id: str = "patient-demo-001"
    message: UserTurn
    replies: list[Reply] = Field(default_factory=list)
    llm_script: LLMScript = Field(default_factory=LLMScript)
    faults: Faults = Field(default_factory=Faults)
    concurrency: Concurrency | None = None
    expected: Expected
    labels: Labels = Field(default_factory=Labels)

    def first_attempt_should_pass(self) -> bool:
        if self.labels.first_attempt_valid is not None:
            return self.labels.first_attempt_valid
        return not (self.expected.should_clarify or self.labels.injected_error)


def load_cases(paths: list[Path] | None = None) -> list[EvalCase]:
    """Đọc case từ các file JSONL; mặc định là bộ dataset đóng gói kèm."""
    if paths is None:
        root = resources.files("patient_assistant.evaluation").joinpath("datasets")
        texts = [
            (f.name, f.read_text("utf-8"))
            for f in sorted(root.iterdir(), key=lambda f: f.name)
            if f.name.endswith(".jsonl")
        ]
    else:
        texts = [(p.name, p.read_text("utf-8")) for p in paths]
    cases: list[EvalCase] = []
    for name, text in texts:
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.strip() and not line.lstrip().startswith("//"):
                try:
                    cases.append(EvalCase.model_validate_json(line))
                except ValueError as exc:
                    raise ValueError(f"{name}:{lineno}: case không hợp lệ: {exc}") from exc
    ids = [c.id for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Trùng id case: {sorted(duplicates)}")
    return cases
