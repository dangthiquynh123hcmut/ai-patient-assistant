"""Luật hội thoại thuần code: phân loại intent, an toàn, hiểu câu xác nhận, gắn cờ STT."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from patient_assistant.domain.common import SttFlag
from patient_assistant.domain.text.normalize import contains_phrase, normalize
from patient_assistant.domain.text.vi_numbers import DIGITS, digitize


class Intent(StrEnum):
    CREATE_REPORT = "create_report"
    GET_REPORT_HISTORY = "get_report_history"
    BOOK_APPOINTMENT = "book_appointment"
    LIST_APPOINTMENTS = "list_appointments"
    ANSWER_FAQ = "answer_faq"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------- intent (rule trước, LLM sau)

_INTENT_PHRASES: dict[Intent, tuple[str, ...]] = {
    Intent.LIST_APPOINTMENTS: ("lịch hẹn của tôi", "xem lịch hẹn", "lịch đã đặt", "tôi có lịch hẹn", "các lịch hẹn"),
    Intent.GET_REPORT_HISTORY: ("báo cáo trước", "lịch sử báo cáo", "các báo cáo", "xem báo cáo", "báo cáo cũ"),
    Intent.BOOK_APPOINTMENT: (
        "đặt lịch",
        "hẹn khám",
        "đặt hẹn",
        "đặt khám",
        "hẹn tái khám",
        "lấy lịch khám",
        "book lịch",
    ),
    Intent.CREATE_REPORT: (
        "đau mức",
        "mức đau",
        "đau khoảng",
        "đã tập",
        "tập được",
        "chưa tập",
        "không tập",
        "báo cáo tuần",
        "báo cáo theo dõi",
    ),
    Intent.ANSWER_FAQ: (
        "có nên",
        "làm sao",
        "làm thế nào",
        "bao lâu",
        "là gì",
        "như thế nào",
        "có được không",
        "chuẩn bị gì",
        "cần mang",
        "bao nhiêu",
        "ở đâu",
    ),
}


# "đau 5/10", "tập 3 ngày": có số mới tính là báo cáo — "gối tôi đau, đặt lịch" không phải báo cáo.
_REPORT_PATTERN = re.compile(r"\bđau (?:mức |khoảng )?\d|\btập\D{0,10}\d ?(?:ngày|buổi|hôm)")


@dataclass
class IntentGuess:
    primary: Intent | None
    secondary: list[Intent] = field(default_factory=list)


def classify_by_rules(text: str) -> IntentGuess:
    """Trả intent theo thứ tự xuất hiện trong câu. `primary=None` nghĩa là luật không kết luận được."""
    norm = f" {normalize(text)} "
    positions: dict[Intent, int] = {}
    for intent, phrases in _INTENT_PHRASES.items():
        hits = [norm.find(f" {normalize(p)} ") for p in phrases]
        hits = [h for h in hits if h >= 0]
        if hits:
            positions[intent] = min(hits)
    if m := _REPORT_PATTERN.search(digitize(text)):
        positions[Intent.CREATE_REPORT] = min(positions.get(Intent.CREATE_REPORT, m.start()), m.start())
    # "xem lịch hẹn" chứa "hẹn" nhưng không phải đặt lịch; báo cáo cũ không phải báo cáo mới.
    if Intent.LIST_APPOINTMENTS in positions:
        positions.pop(Intent.BOOK_APPOINTMENT, None)
    if Intent.GET_REPORT_HISTORY in positions:
        positions.pop(Intent.CREATE_REPORT, None)
    if Intent.ANSWER_FAQ in positions and len(positions) > 1:
        positions.pop(Intent.ANSWER_FAQ)
    ordered = sorted(positions, key=positions.__getitem__)
    if not ordered:
        return IntentGuess(None)
    return IntentGuess(ordered[0], ordered[1:])


# ---------------------------------------------------------------- an toàn (lớp 6)


class SafetyKind(StrEnum):
    OK = "ok"
    EMERGENCY = "emergency"
    MEDICAL_ADVICE = "medical_advice"


_EMERGENCY = (
    "khó thở",
    "không thở được",
    "đau ngực",
    "tức ngực",
    "ngất",
    "co giật",
    "chảy máu nhiều",
    "cấp cứu",
    "méo miệng",
    "yếu nửa người",
    "bị liệt",
    "liệt nửa người",
    "muốn tự tử",
    "bất tỉnh",
)
_MEDICAL_ADVICE = (
    "chẩn đoán",
    "tôi bị bệnh gì",
    "bị bệnh gì",
    "uống thuốc gì",
    "liều",
    "kê đơn",
    "kê thuốc",
    "đổi thuốc",
    "ngưng thuốc",
    "bỏ thuốc",
    "tăng thuốc",
    "giảm thuốc",
    "nên khám khoa nào",
    "khám khoa nào",
    "có phải bị",
)


@dataclass
class SafetyVerdict:
    kind: SafetyKind
    matched: str | None = None


def check_input_safety(text: str) -> SafetyVerdict:
    hit = contains_phrase(text, _EMERGENCY)
    if hit:
        return SafetyVerdict(SafetyKind.EMERGENCY, hit)
    hit = contains_phrase(text, _MEDICAL_ADVICE)
    if hit:
        return SafetyVerdict(SafetyKind.MEDICAL_ADVICE, hit)
    return SafetyVerdict(SafetyKind.OK)


_UNSAFE_OUTPUT = ("bạn bị", "chẩn đoán", "nên uống", "liều dùng", "ngưng thuốc", "tăng liều", "giảm liều", "kê đơn")


def output_is_safe(text: str) -> bool:
    return contains_phrase(text, _UNSAFE_OUTPUT) is None


# ---------------------------------------------------------------- xác nhận bằng lời


class ConfirmIntent(StrEnum):
    CONFIRM = "confirm"
    CANCEL = "cancel"
    UNCLEAR = "unclear"


_CONFIRM = ("đồng ý", "xác nhận", "đặt đi", "lưu đi", "chốt", "ok đặt", "đúng rồi lưu", "được đặt đi", "đồng ý lưu")
# Không dùng "thôi" đứng một mình: "tập 5 ngày thôi" không phải hủy.
_CANCEL = ("hủy", "huỷ", "không đặt", "không lưu", "thôi khỏi", "thôi không", "dừng lại", "bỏ qua")


def interpret_confirmation(text: str) -> ConfirmIntent:
    """Chỉ chấp nhận cụm xác nhận rõ ràng; "ừ", "ờ" lẫn tiếng ồn bị coi là chưa rõ (FR-09.7)."""
    cancel = contains_phrase(text, _CANCEL) is not None
    confirm = contains_phrase(text, _CONFIRM) is not None
    if cancel == confirm:  # không có, hoặc có cả hai ("thôi được, đặt đi") -> hỏi lại
        return ConfirmIntent.UNCLEAR
    return ConfirmIntent.CONFIRM if confirm else ConfirmIntent.CANCEL


# ---------------------------------------------------------------- cờ STT

_DIGIT_TOKEN = re.compile(r"\d")
_IMPORTANT_WORDS = set(DIGITS) | {
    "mười",
    "mươi",
    "rưỡi",
    "giờ",
    "thứ",
    "ngày",
    "tháng",
    "mai",
    "mốt",
    "sáng",
    "chiều",
    "khoa",
    "bác",
    "sĩ",
    "đau",
    "tập",
}


def stt_flags_from_words(words: list[tuple[str, float]], threshold: float = 0.6) -> list[SttFlag]:
    """Gắn cờ từ có xác suất thấp và quan trọng (số, ngày giờ, đau, khoa...)."""
    flags: list[SttFlag] = []
    for word, prob in words:
        norm = normalize(word)
        if prob >= threshold or not norm:
            continue
        if _DIGIT_TOKEN.search(norm) or norm in _IMPORTANT_WORDS:
            flags.append(SttFlag(text=word.strip(), probability=round(prob, 3), reason="low_confidence_key_token"))
    return flags
