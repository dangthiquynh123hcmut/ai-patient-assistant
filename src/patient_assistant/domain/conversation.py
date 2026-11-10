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
    GET_PATIENT_INFO = "get_patient_info"
    ANSWER_FAQ = "answer_faq"
    GET_DOCTOR_INFO = "get_doctor_info"  # hỏi phòng khám có những khoa / bác sĩ nào (danh mục, không phải hồ sơ)
    SMALL_TALK = "small_talk"  # chào hỏi, cảm ơn, hỏi trợ lý là ai — trả lời tự nhiên, không chạy nghiệp vụ
    UNKNOWN = "unknown"


# ---------------------------------------------------------------- intent (rule trước, LLM sau)

_INTENT_PHRASES: dict[Intent, tuple[str, ...]] = {
    Intent.GET_PATIENT_INFO: (
        "thông tin bệnh nhân",
        "thông tin của tôi",
        "thông tin cá nhân",
        "tra cứu thông tin",
        "hồ sơ",
        "kế hoạch theo dõi",
        "kế hoạch điều trị",
        "tình hình của tôi",
        "tổng quan",
        # Hỏi về bệnh nhân khác: vẫn vào đây, nhưng chỉ trả dữ liệu của chính người đang đăng nhập.
        "những bệnh nhân",
        "các bệnh nhân",
        "danh sách bệnh nhân",
        "bệnh nhân nào",
        "bao nhiêu bệnh nhân",
        "bệnh nhân khác",
    ),
    Intent.GET_DOCTOR_INFO: (
        "thông tin bác sĩ",
        "danh sách bác sĩ",
        "các bác sĩ",
        "những bác sĩ",
        "bác sĩ nào",
        "bác sĩ khoa",
        "bác sĩ của khoa",
        "danh sách khoa",
        "các khoa",
        "những khoa",
        "khoa nào có",
        "có khoa",
    ),
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
        "đặt thêm",  # "đặt thêm một lịch như vậy"
        "đặt một lịch",
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
    Intent.SMALL_TALK: (
        "xin chào",
        "chào",
        "hello",
        "hi",
        "hey",
        "alo",
        "cảm ơn",
        "cám ơn",
        "thanks",
        "thank you",
        "tạm biệt",
        "bye",
        "bạn là ai",
        "bạn tên gì",
        "bạn làm được gì",
        "bạn giúp được gì",
        "bạn có thể làm gì",
        "bạn có thể giúp gì",
    ),
}


# Từ khóa lỏng để XẾP HẠNG intent khi luật cụm từ không kết luận được (gợi ý cho người dùng chọn).
_INTENT_KEYWORDS: dict[Intent, tuple[str, ...]] = {
    Intent.GET_PATIENT_INFO: (
        "thông tin",
        "hồ sơ",
        "cá nhân",
        "kế hoạch",
        "tra cứu",
        "bệnh nhân",
        "tình hình",
        "tóm tắt",
        "của tôi",
    ),
    Intent.BOOK_APPOINTMENT: ("đặt", "hẹn", "khám", "bác sĩ", "tái khám", "gặp", "lịch"),
    Intent.LIST_APPOINTMENTS: ("lịch hẹn", "đã đặt", "sắp tới", "lịch khám", "lịch"),
    Intent.GET_REPORT_HISTORY: ("báo cáo", "lịch sử", "trước", "cũ", "tiến triển"),
    Intent.CREATE_REPORT: ("đau", "tập", "mức", "triệu chứng", "báo cáo", "gửi"),
    Intent.ANSWER_FAQ: ("hỏi", "cách", "nên", "hướng dẫn", "chườm", "bài tập", "tại sao", "gì"),
    Intent.GET_DOCTOR_INFO: ("bác sĩ", "khoa", "chuyên khoa"),
}


# Các việc nhiều bước, có hỏi lại và có thể dang dở giữa chừng.
FLOW_INTENTS = frozenset({Intent.CREATE_REPORT, Intent.BOOK_APPOINTMENT})
# Các việc thật sự (không phải trò chuyện / hỏi đáp chung).
TASK_INTENTS = frozenset(
    {
        Intent.CREATE_REPORT,
        Intent.BOOK_APPOINTMENT,
        Intent.LIST_APPOINTMENTS,
        Intent.GET_PATIENT_INFO,
        Intent.GET_REPORT_HISTORY,
    }
)

# Người dùng hỏi về chính cuộc trò chuyện: trợ lý có biết họ là ai, dữ liệu lưu ở đâu, sao lại hỏi.
_ASSISTANT_CONTEXT = (
    "tôi là ai",
    "biết tôi là ai",
    "biết tôi không",
    "lưu vào đâu",
    "lưu ở đâu",
    "lưu dữ liệu",
    "dữ liệu lưu",
    "dữ liệu của tôi",
    "ai xem được",
    "có bảo mật",
    "bảo mật không",
    "sao lại hỏi",
    "hỏi làm gì",
    "tại sao hỏi",
    "sao bạn hỏi",
)


def asks_about_assistant_context(text: str) -> bool:
    return contains_phrase(text, _ASSISTANT_CONTEXT) is not None


# "lưu đi", "làm tiếp", "lấy thông tin cũ": nối lại việc dang dở ở lượt trước, dùng lại những gì đã nói.
# Không dùng "vừa rồi" (hay gặp trong báo cáo: "tuần vừa rồi tôi đau...").
_CONTINUE = (
    "lưu đi",
    "lưu lại",
    "lưu giúp",
    "lưu báo cáo",
    "lưu giùm",
    "tiếp tục",
    "làm tiếp",
    "tiếp đi",
    "đặt tiếp",
    "đặt đi",
    "quay lại",
    "lúc nãy",
    "hồi nãy",
    "khi nãy",
    "ban nãy",
    "vừa nãy",
    "thông tin cũ",
    "thông tin trước",
    "đã trả lời",
    "đã nói",
    "phía trên",
    "ở trên",
    "như trên",
    "như cũ",
)


_QUESTION_WORDS = ("nào", "gì", "sao", "bao nhiêu", "mấy", "ai", "ở đâu", "khi nào", "được không", "có không")


def looks_like_question(text: str) -> bool:
    return "?" in text or contains_phrase(text, _QUESTION_WORDS) is not None


def is_continuation(text: str) -> bool:
    return contains_phrase(text, _CONTINUE) is not None


def rank_intents(text: str) -> list[tuple[Intent, float]]:
    """Xếp hạng intent theo số từ khóa xuất hiện (cụm nhiều từ nặng hơn). Rỗng nếu không có từ khóa nào."""
    norm = f" {normalize(text)} "
    scores: dict[Intent, float] = {}
    for intent, words in _INTENT_KEYWORDS.items():
        score = sum(1.0 + 0.5 * (len(w.split()) - 1) for w in words if f" {w} " in norm)
        if score:
            scores[intent] = score
    return sorted(scores.items(), key=lambda kv: -kv[1])


_OTHER_PATIENT = (
    "bệnh nhân khác",
    "những bệnh nhân",
    "các bệnh nhân",
    "danh sách bệnh nhân",
    "bệnh nhân nào",
    "bao nhiêu bệnh nhân",
    "người khác",
    "của mẹ tôi",
    "của bố tôi",
    "của chồng tôi",
    "của vợ tôi",
    "của con tôi",
    "mã bệnh nhân",
)


def asks_about_other_patient(text: str) -> bool:
    return contains_phrase(text, _OTHER_PATIENT) is not None


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
    # "chào bạn, cho tôi đặt lịch": lời chào đi kèm yêu cầu thì làm yêu cầu.
    if Intent.SMALL_TALK in positions and len(positions) > 1:
        positions.pop(Intent.SMALL_TALK)
    if Intent.ANSWER_FAQ in positions and len(positions) > 1:
        positions.pop(Intent.ANSWER_FAQ)
    # "đặt lịch với bác sĩ nào" là đặt lịch; chỉ hỏi về bác sĩ/khoa khi không kèm việc khác.
    if Intent.GET_DOCTOR_INFO in positions and len(positions) > 1:
        positions.pop(Intent.GET_DOCTOR_INFO)
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
