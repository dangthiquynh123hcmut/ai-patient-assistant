"""Câu chữ hiển thị cho người dùng, dựng bằng template từ dữ liệu đã xác minh.

Tách khỏi node để đổi câu chữ (hoặc đa ngôn ngữ) mà không đụng logic workflow.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from patient_assistant.application.agent.state import Choice, FieldExplanation, Outcome, OutcomeKind
from patient_assistant.domain.booking.models import BookingRequest, Department, Slot, SlotOption
from patient_assistant.domain.common import Origin, Resolution, ValidationIssue, evidence_for
from patient_assistant.domain.conversation import Intent, SafetyKind
from patient_assistant.domain.report.models import REPORT_FIELDS, ReportExtraction
from patient_assistant.domain.report.rules import current_period
from patient_assistant.domain.report.validators import NUMERIC_BOUNDS
from patient_assistant.domain.text.vi_dates import WEEKDAY_NAMES, DateWindow

LABELS = {
    "reporting_period": "Kỳ báo cáo",
    "pain_level": "Mức đau (0–10)",
    "exercise_days": "Số ngày tập",
    "symptom_change": "Thay đổi triệu chứng",
    "triggering_activity": "Hoạt động liên quan",
    "department_code": "Khoa",
    "doctor_name": "Bác sĩ",
    "date_expression": "Ngày khám",
    "time_expression": "Giờ khám",
    "reason": "Lý do khám",
    "slot": "Lịch khám",
}

INTENT_LABELS = {
    Intent.CREATE_REPORT: "gửi báo cáo theo dõi",
    Intent.BOOK_APPOINTMENT: "đặt lịch khám",
    Intent.LIST_APPOINTMENTS: "xem lịch hẹn",
    Intent.GET_PATIENT_INFO: "xem thông tin của bạn",
    Intent.GET_REPORT_HISTORY: "xem các báo cáo trước",
    Intent.ANSWER_FAQ: "hỏi thông tin",
}

_MISSING_QUESTIONS = {
    "pain_level": "Mức đau của bạn trong kỳ này là bao nhiêu trên thang 0–10?",
    "exercise_days": "Trong kỳ này bạn đã tập được bao nhiêu ngày?",
    "date_expression": "Bạn muốn khám vào ngày nào?",
}

_DATE_RULES = {
    "explicit_date": "ngày được nêu cụ thể",
    "tomorrow": "“ngày mai” tính từ hôm nay",
    "day_after_tomorrow": "“ngày kia” tính từ hôm nay",
    "today": "hôm nay",
    "next_week_weekday": "thứ được nêu của tuần kế tiếp",
    "this_week_weekday": "thứ được nêu của tuần này",
    "upcoming_weekday": "lần gần nhất sắp tới của thứ được nêu",
    "next_week": "cả tuần kế tiếp",
    "this_week": "những ngày còn lại của tuần này",
    "earliest_available": "các ngày gần nhất có lịch",
    "morning": "buổi sáng",
    "afternoon": "buổi chiều",
    "exact_time": "khoảng ±1 giờ quanh giờ được nêu",
    "hour_assumed_afternoon": "giờ 1–5 được hiểu là buổi chiều",
}


# ---------------------------------------------------------------- câu hỏi lại


def clarify_question(issues: list[ValidationIssue], departments: list[Department] | None = None) -> str:
    lines: list[str] = []
    for issue in issues:
        if issue.resolution != Resolution.ASK_USER:
            continue
        if issue.code == "missing_field" and issue.field == "department_code":
            names = ", ".join(d.name for d in departments or [])
            line = f"Bạn muốn khám khoa nào? Các khoa hiện có: {names}."
        elif issue.code == "missing_field":
            line = _MISSING_QUESTIONS.get(
                issue.field or "", f"Bạn bổ sung giúp mình {LABELS.get(issue.field or '', '')}?"
            )
        elif issue.code == "stt_uncertain":
            line = f"{issue.message} Bạn nói lại hoặc gõ giúp mình đoạn này nhé?"
        elif issue.code == "out_of_range":
            line = f"{issue.message} Bạn kiểm tra lại giúp mình con số này?"
        elif issue.code == "contradiction":
            line = "Bạn có nhắc là chưa tập, nhưng cũng có số ngày tập. Kỳ này bạn đã tập bao nhiêu ngày?"
        elif issue.field in ("date_expression", "time_expression"):
            line = f"{issue.message} Bạn muốn khám vào ngày/giờ nào?"
        elif issue.code == "doctor_not_found":
            line = f"{issue.message} Bạn muốn chọn bác sĩ khác, hay để hệ thống chọn bác sĩ còn trống?"
        else:
            line = issue.message
        if line not in lines:
            lines.append(line)
    return " ".join(lines[:2]) or "Bạn nói rõ hơn giúp mình được không?"


# Gợi ý trả lời nhanh cho từng intent: (nhãn nút, câu gửi đi — câu này khớp luật phân loại).
INTENT_CHOICES: dict[Intent, tuple[str, str]] = {
    Intent.GET_PATIENT_INFO: ("Xem thông tin của tôi", "xem thông tin của tôi"),
    Intent.BOOK_APPOINTMENT: ("Đặt lịch khám", "tôi muốn đặt lịch khám"),
    Intent.LIST_APPOINTMENTS: ("Xem lịch hẹn", "xem lịch hẹn của tôi"),
    Intent.GET_REPORT_HISTORY: ("Xem báo cáo trước", "xem các báo cáo trước"),
    Intent.CREATE_REPORT: ("Gửi báo cáo theo dõi", "tôi muốn gửi báo cáo theo dõi"),
}


def intent_choices(ranked: list[Intent]) -> list[Choice]:
    """Các intent đoán được xếp trước, rồi tới các lựa chọn còn lại."""
    order = [*ranked, *(i for i in INTENT_CHOICES if i not in ranked)]
    return [Choice(label=INTENT_CHOICES[i][0], text=INTENT_CHOICES[i][1]) for i in order if i in INTENT_CHOICES]


def intent_question(user_text: str, ranked: list[Intent]) -> str:
    if ranked:
        guesses = " hoặc ".join(INTENT_LABELS[i] for i in ranked[:2])
        return (
            f"Mình chưa chắc ý bạn trong câu “{user_text}”. Có phải bạn muốn {guesses}? "
            "Bấm một lựa chọn bên dưới, hoặc nói rõ hơn — ví dụ hỏi về bài tập, chăm sóc sau tập."
        )
    return (
        f"Mình chưa rõ ý bạn trong câu “{user_text}”. Bạn muốn làm việc nào dưới đây? Bạn cũng có thể "
        "hỏi mình về bài tập hay cách chăm sóc sau tập."
    )


CHAT_FALLBACK = (
    "Chào bạn! Mình là trợ lý ảo của phòng khám. Mình có thể giúp bạn gửi báo cáo theo dõi, "
    "xem lại báo cáo, đặt hoặc xem lịch khám, và trả lời câu hỏi từ tài liệu hướng dẫn. Bạn cần gì nào?"
)


ASIDE_FALLBACK = "Mình ghi nhận câu của bạn rồi."


def assistant_context_text(display_name: str | None, patient_id: str, task: str | None) -> str:
    """Trả lời bằng code (không qua LLM) câu hỏi "bạn có biết tôi là ai", "dữ liệu lưu ở đâu"."""
    who = (
        f"Bạn đang đăng nhập với tài khoản bệnh nhân {display_name} (mã {patient_id})"
        if display_name
        else f"Bạn đang đăng nhập với mã bệnh nhân {patient_id}"
    )
    doing = f" Thông tin cho việc {task} sẽ" if task else " Báo cáo và lịch hẹn sẽ"
    return (
        f"{who}, nên mình chỉ xem và ghi dữ liệu của chính tài khoản này.{doing} được lưu vào hồ sơ theo dõi "
        "của bạn trong hệ thống phòng khám, và chỉ sau khi bạn xem bản nháp và bấm xác nhận. Hiện chưa có gì "
        "được lưu. (Bản demo: dữ liệu là giả lập.)"
    )


INTENT_QUESTION = (
    "Mình có thể giúp bạn: gửi báo cáo theo dõi, xem báo cáo trước, đặt lịch khám, "
    "xem lịch hẹn, hoặc hỏi thông tin hướng dẫn. Bạn muốn làm gì?"
)


# ---------------------------------------------------------------- giải thích từng trường


def report_explanations(
    draft: ReportExtraction, issues: list[ValidationIssue], edited: list[str], now: datetime
) -> list[FieldExplanation]:
    out: list[FieldExplanation] = []
    for name in REPORT_FIELDS:
        value: Any = getattr(draft, name)
        field_issues = [i for i in issues if i.field == name]
        quote = None if name in edited else evidence_for(draft.evidence, name)  # giá trị sửa tay không còn trích dẫn
        origin = Origin.USER_EDIT if name in edited else Origin.USER_SAID
        if name == "reporting_period" and value is None:
            out.append(
                FieldExplanation(
                    field=name,
                    label=LABELS[name],
                    value=current_period(now),
                    evidence=None,
                    origin=Origin.SYSTEM_DEFAULT,
                    validation="passed",
                    explanation="Bạn không nêu kỳ báo cáo; hệ thống mặc định tuần hiện tại.",
                )
            )
            continue
        if field_issues:
            issue = field_issues[0]
            status = "uncertain" if issue.resolution == Resolution.ASK_USER else "failed"
            out.append(
                FieldExplanation(
                    field=name,
                    label=LABELS[name],
                    value=value,
                    evidence=quote,
                    origin=origin,
                    validation=status,
                    explanation=issue.message,
                )
            )
        elif value is None:
            out.append(
                FieldExplanation(
                    field=name,
                    label=LABELS[name],
                    value=None,
                    evidence=None,
                    origin=origin,
                    validation="missing",
                    explanation="Không có trong lời bạn nói.",
                )
            )
        else:
            if name in edited:
                why = "Bạn đã sửa trực tiếp giá trị này."
            elif name in NUMERIC_BOUNDS:
                lo, hi = NUMERIC_BOUNDS[name]
                why = f"Giá trị {value} được nêu trực tiếp trong lời bạn nói và nằm trong phạm vi {lo}–{hi}."
            else:
                why = "Khớp với trích dẫn nguyên văn từ lời bạn nói."
            out.append(
                FieldExplanation(
                    field=name,
                    label=LABELS[name],
                    value=_plain(value),
                    evidence=quote,
                    origin=origin,
                    validation="passed",
                    explanation=why,
                )
            )
    return out


def _plain(value: Any) -> Any:
    return getattr(value, "value", value)


def describe_window_rules(window: DateWindow) -> str:
    parts = [_DATE_RULES[r] for r in window.rules if r in _DATE_RULES]
    return "Quy đổi theo luật: " + ", ".join(parts) + "." if parts else ""


def booking_explanations(
    req: BookingRequest,
    window: DateWindow | None,
    department: Department | None,
    from_plan: bool,
    slot: Slot | None = None,
) -> list[FieldExplanation]:
    out = [
        FieldExplanation(
            field="department_code",
            label=LABELS["department_code"],
            value=department.name if department else req.department_code,
            evidence=None if from_plan else (department.name if department else None),
            origin=Origin.BACKEND if from_plan else Origin.USER_SAID,
            validation="passed",
            explanation=(
                "Bạn không nêu khoa; lấy theo kế hoạch theo dõi hiện tại của bạn."
                if from_plan
                else "Khoa do bạn nêu, có trong danh mục."
            ),
        )
    ]
    if window:
        out.append(
            FieldExplanation(
                field="date_expression",
                label=LABELS["date_expression"],
                value=window.describe(),
                evidence=" ".join(x for x in (req.date_expression, req.time_expression) if x) or None,
                origin=Origin.USER_SAID,
                validation="passed",
                explanation=describe_window_rules(window),
            )
        )
    if slot:
        out.append(
            FieldExplanation(
                field="slot",
                label=LABELS["slot"],
                value=slot.label(department.name if department else None),
                evidence=None,
                origin=Origin.BACKEND,
                validation="passed",
                explanation=f"Slot do hệ thống lịch trả về (mã {slot.slot_id}); bạn đã chọn slot này.",
            )
        )
    if req.reason:
        out.append(
            FieldExplanation(
                field="reason",
                label=LABELS["reason"],
                value=req.reason,
                evidence=req.reason,
                origin=Origin.USER_SAID,
                validation="passed",
                explanation="Ghi chú nguyên văn, không dùng để chẩn đoán.",
            )
        )
    return out


def slot_options_message(options: list[SlotOption], relaxed: bool, department_name: str | None) -> str:
    head = (
        "Không còn lịch đúng khoảng bạn chọn. Đây là các lịch gần nhất (đã nới thêm ±7 ngày, "
        "không giới hạn giờ và bác sĩ):"
        if relaxed
        else "Các lịch còn trống:"
    )
    lines = [f"{o.option_no}. {o.slot.label(department_name)}" for o in options]
    tail = "Bạn chọn số mấy? Bạn cũng có thể nói “giờ khác” hoặc “hủy”."
    return "\n".join([head, *lines, tail])


def booking_confirm_message(slot: Slot, department_name: str | None) -> str:
    return (
        f"Bạn xác nhận đặt lịch: {slot.label(department_name)}? "
        "Nói “xác nhận” để đặt, “sửa” để đổi thông tin, hoặc “hủy”."
    )


REPORT_CONFIRM_MESSAGE = (
    "Đây là bản nháp báo cáo. Bạn kiểm tra giúp mình: “xác nhận” để lưu, sửa trường sai, hoặc “hủy”."
)


# ---------------------------------------------------------------- câu trả lời cuối


_SAFETY_TEXT = {
    SafetyKind.EMERGENCY: (
        "Những dấu hiệu bạn mô tả có thể cần được xử lý ngay. Hãy gọi 115 hoặc đến cơ sở y tế "
        "gần nhất. Mình đã dừng các thao tác khác."
    ),
    SafetyKind.MEDICAL_ADVICE: (
        "Mình không thể chẩn đoán, tư vấn thuốc hay chọn khoa dựa trên triệu chứng — việc "
        "này cần bác sĩ. Mình có thể giúp bạn gửi báo cáo theo dõi, đặt lịch khám theo khoa "
        "bạn chọn, hoặc tìm thông tin trong tài liệu hướng dẫn đã được duyệt."
    ),
}

_HANDOFF_TEXT = {
    "too_many_corrections": "Mình chưa trích xuất được thông tin hợp lệ sau số lần thử cho phép.",
    "too_many_rounds": "Mình chưa thu thập đủ thông tin sau nhiều lượt hỏi lại.",
    "no_slots": "Hiện không còn lịch trống phù hợp trong thời gian cho phép đặt.",
    "blocked": "Yêu cầu có nội dung nằm ngoài phạm vi hỗ trợ.",
    "conflict": "Slot bạn chọn vừa được người khác đặt và không còn phương án thay thế.",
}


def _error_text(outcome: Outcome) -> str:
    code = outcome.code or ""
    if code.startswith("llm_"):
        return "Hệ thống AI tạm thời không xử lý được yêu cầu. Chưa có dữ liệu nào được lưu, bạn thử lại sau nhé."
    if code == "save_failed":
        details = "; ".join(outcome.data or []) if isinstance(outcome.data, list) else ""
        return f"Lưu báo cáo KHÔNG thành công{': ' + details if details else ''}. Bạn có thể thử lại an toàn."
    if code == "booking_failed":
        details = "; ".join(outcome.data or []) if isinstance(outcome.data, list) else ""
        return f"Đặt lịch KHÔNG thành công{': ' + details if details else ''}."
    if code == "verify_failed":
        return "Hệ thống chưa xác nhận được kết quả ghi, nên mình chưa thể báo thành công. Bạn kiểm tra lại sau nhé."
    return "Không kết nối được hệ thống dữ liệu. Chưa có thao tác nào được thực hiện, bạn thử lại sau nhé."


def final_text(outcome: Outcome, safety: SafetyKind, pending: list[Intent]) -> str:
    kind = outcome.kind
    data = outcome.data
    if kind == OutcomeKind.SAFETY:
        text = _SAFETY_TEXT.get(safety, _SAFETY_TEXT[SafetyKind.MEDICAL_ADVICE])
    elif kind == OutcomeKind.REPORT_SAVED:
        text = f"Đã lưu báo cáo kỳ {data['reporting_period']} (mã báo cáo {data['id']})."
    elif kind == OutcomeKind.APPOINTMENT_BOOKED:
        slot = Slot.model_validate(data["slot"])
        text = f"Đã đặt lịch: {slot.label()}. Mã lịch hẹn: {data['id']}."
    elif kind == OutcomeKind.HISTORY:
        text = (
            "Các báo cáo gần đây:\n"
            + "\n".join(
                f"- {r['reporting_period']}: đau {r['pain_level']}/10, tập {r['exercise_days']} ngày (mã {r['id']})"
                for r in data
            )
            if data
            else "Bạn chưa có báo cáo nào."
        )
    elif kind == OutcomeKind.APPOINTMENTS:
        text = (
            "Lịch hẹn sắp tới:\n"
            + "\n".join(f"- {Slot.model_validate(a['slot']).label()} (mã {a['id']})" for a in data)
            if data
            else "Bạn chưa có lịch hẹn nào sắp tới."
        )
    elif kind == OutcomeKind.PATIENT_INFO:
        text = patient_info_text(data, own_only=outcome.code == "own_data_only")
    elif kind == OutcomeKind.ANSWER:
        text = str(data)
        if outcome.source_ids:
            text += f"\n\nNguồn: {', '.join(outcome.source_ids)}"
    elif kind == OutcomeKind.CHAT:
        text = str(data)
    elif kind == OutcomeKind.CANCELLED:
        text = "Đã dừng theo yêu cầu của bạn. Chưa có dữ liệu nào được lưu."
    elif kind == OutcomeKind.HANDOFF:
        text = (
            _HANDOFF_TEXT.get(outcome.code or "", "Mình chưa thể hoàn tất yêu cầu.")
            + " Yêu cầu đã được đánh dấu để nhân viên xem lại. Chưa có dữ liệu nào được lưu."
        )
    else:
        text = _error_text(outcome)
    if pending:
        todo = ", ".join(INTENT_LABELS.get(i, i.value) for i in pending)
        text += f"\n\nBạn cũng nhắc tới việc {todo} — nhắn tiếp để mình làm nhé."
    return text


_TREND = {"down": "giảm", "up": "tăng", "flat": "không đổi"}


def pain_trend(reports: list[dict[str, Any]]) -> str | None:
    """Mô tả xu hướng mức đau từ các báo cáo (cũ -> mới). Chỉ mô tả số liệu, không diễn giải y khoa."""
    if len(reports) < 2:
        return None
    ordered = sorted(reports, key=lambda r: r["reporting_period"])
    first, last = ordered[0]["pain_level"], ordered[-1]["pain_level"]
    direction = "down" if last < first else "up" if last > first else "flat"
    return (
        f"mức đau {_TREND[direction]} từ {first}/10 ({ordered[0]['reporting_period']}) xuống {last}/10"
        if direction == "down"
        else f"mức đau {_TREND[direction]}: {first}/10 → {last}/10"
    )


def patient_info_text(data: dict[str, Any], own_only: bool = False) -> str:
    lines = []
    if own_only:
        lines.append("Mình chỉ có thể hiển thị thông tin của chính tài khoản đang đăng nhập.")
    profile = data["profile"]
    lines.append(f"Thông tin của {profile['display_name']} (mã {profile['id']}):")
    plan = data.get("plan")
    if plan:
        lines.append(
            f"- Kế hoạch theo dõi: {plan['department_name'] or plan['department_code']} — {plan['summary']}"
            f" (mã {plan['plan_id']})"
        )
    else:
        lines.append("- Chưa có kế hoạch theo dõi.")
    reports = data.get("recent_reports") or []
    if reports:
        latest = max(reports, key=lambda r: r["reporting_period"])
        lines.append(
            f"- Báo cáo gần nhất {latest['reporting_period']}: đau {latest['pain_level']}/10, "
            f"tập {latest['exercise_days']} ngày."
        )
        trend = pain_trend(reports)
        if trend:
            lines.append(f"- {len(reports)} báo cáo gần đây: {trend}.")
    else:
        lines.append("- Chưa có báo cáo nào.")
    upcoming = data.get("upcoming_appointments") or []
    if upcoming:
        lines.append(
            "- Lịch hẹn sắp tới: "
            + "; ".join(f"{Slot.model_validate(a['slot']).label()} (mã {a['id']})" for a in upcoming)
        )
    else:
        lines.append("- Chưa có lịch hẹn sắp tới.")
    lines.append("Số liệu lấy trực tiếp từ hệ thống (dữ liệu giả lập); đây không phải đánh giá y khoa.")
    return "\n".join(lines)


def weekday_name(dt: datetime) -> str:
    return WEEKDAY_NAMES[dt.weekday()]
