"""Diễn giải từng bước graph đã chạy thành câu dễ đọc — để người dùng thấy agent quyết định thế nào.

Đầu vào là update mà mỗi node trả về (lấy từ `graph.astream(stream_mode="updates")`); hàm ở đây thuần,
không đọc thêm dữ liệu nào ngoài update đó.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

StepKind = Literal["code", "llm", "hitl", "tool", "stop"]

NODES: dict[str, tuple[str, StepKind]] = {
    "ingest": ("Nhận đầu vào", "code"),
    "guard_input": ("Kiểm tra an toàn", "code"),
    "safety_response": ("Trả lời an toàn cố định", "stop"),
    "classify": ("Phân loại yêu cầu", "code"),
    "clarify_intent": ("Hỏi rõ yêu cầu", "hitl"),
    "fetch_history": ("Lấy báo cáo trước", "tool"),
    "fetch_appointments": ("Lấy lịch hẹn", "tool"),
    "rag_answer": ("Trả lời từ tài liệu đã duyệt", "llm"),
    "respond": ("Soạn câu trả lời", "code"),
    # report_flow
    "extract": ("LLM trích xuất báo cáo", "llm"),
    "validate": ("Validator báo cáo", "code"),
    "correct": ("LLM sửa theo lỗi validator", "llm"),
    "clarify": ("Hỏi lại người dùng", "hitl"),
    "handoff": ("Dừng an toàn, chuyển người xử lý", "stop"),
    "present_draft": ("Người dùng duyệt bản nháp", "hitl"),
    "apply_edits": ("Áp chỉnh sửa của người dùng", "code"),
    "save": ("Gọi tool lưu báo cáo", "tool"),
    "verify_save": ("Đọc lại bản ghi để xác minh", "code"),
    # booking_flow
    "extract_booking": ("LLM trích xuất yêu cầu đặt lịch", "llm"),
    "resolve_datetime": ("Code quy đổi ngày giờ", "code"),
    "validate_booking": ("Validator đặt lịch", "code"),
    "search_slots": ("Tìm slot trống", "tool"),
    "offer_alternatives": ("Nới điều kiện tìm slot", "tool"),
    "present_slot_options": ("Người dùng chọn slot", "hitl"),
    "match_selection": ("Ghép câu chọn với danh sách slot", "code"),
    "present_booking_draft": ("Người dùng xác nhận lịch hẹn", "hitl"),
    "create_booking": ("Gọi tool đặt lịch", "tool"),
    "verify_booking": ("Đọc lại lịch hẹn để xác minh", "code"),
}

_RESOLUTION = {"llm_fixable": "LLM sửa", "ask_user": "hỏi người dùng", "block": "chặn"}
_INTENT_SOURCE = {"rule": "theo luật (code)", "llm": "bởi LLM vì luật không kết luận được"}
_REPORT_LABELS = {
    "pain_level": "đau",
    "exercise_days": "số ngày tập",
    "symptom_change": "triệu chứng",
    "triggering_activity": "hoạt động",
    "reporting_period": "kỳ",
}


class StepTrace(BaseModel):
    index: int
    flow: str  # main | report_flow | booking_flow
    node: str
    title: str
    kind: StepKind
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


def _issues(issues: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{i['code']} ({i.get('field') or '—'}) → {_RESOLUTION.get(i['resolution'], i['resolution'])}" for i in issues
    )


def _route_hint(issues: list[dict[str, Any]]) -> str:
    resolutions = {i["resolution"] for i in issues}
    if "block" in resolutions:
        return "Có lỗi bị chặn → dừng an toàn."
    if "ask_user" in resolutions:
        return "Có lỗi cần thông tin từ người dùng → hỏi lại (không để LLM tự đoán)."
    return "Toàn bộ là lỗi do LLM → gửi lỗi cụ thể cho LLM sửa nếu còn lượt, hết lượt thì dừng."


def _outcome(update: dict[str, Any]) -> str | None:
    outcome = update.get("outcome")
    if not outcome:
        return None
    return f"Kết quả: {outcome['kind']}" + (f" ({outcome['code']})" if outcome.get("code") else "")


def _decision(update: dict[str, Any]) -> str:
    d = update.get("decision") or {}
    parts = [f"action={d.get('action')}"]
    if d.get("text"):
        parts.append(f"“{d['text']}”")
    if d.get("option_no"):
        parts.append(f"option {d['option_no']}")
    if d.get("fields"):
        parts.append(f"sửa {d['fields']}")
    return "Người dùng trả lời: " + ", ".join(parts)


def _summary(node: str, u: dict[str, Any]) -> str:  # noqa: C901 - một nhánh cho mỗi loại node
    outcome = _outcome(u)
    if node == "ingest":
        return "Chuẩn hóa văn bản đầu vào (text hoặc transcript đã duyệt)."
    if node == "guard_input":
        kind = u.get("safety", "ok")
        return (
            "Không phát hiện cấp cứu / yêu cầu chẩn đoán."
            if kind == "ok"
            else f"Phát hiện '{kind}' bằng luật từ khóa → không gọi LLM, trả lời an toàn cố định."
        )
    if node == "classify":
        if outcome:
            return outcome
        text = f"Intent = {u.get('intent')} — {_INTENT_SOURCE.get(u.get('intent_source') or '', '')}."
        if u.get("pending_intents"):
            text += f" Việc khác để sau: {', '.join(u['pending_intents'])}."
        return text
    if node in ("extract", "correct", "extract_booking"):
        if outcome:
            return f"LLM lỗi, không đoán kết quả. {outcome}"
        prefix = f"Lần sửa thứ {u['correction_count']}. " if "correction_count" in u else ""
        draft = u.get("report_draft") or u.get("booking_request")
        if draft is None:
            return prefix + f"Đầu ra sai schema: {_issues(u.get('issues', []))}"
        filled = {k: v for k, v in draft.items() if v not in (None, [], "") and k not in ("evidence", "missing_fields")}
        text = prefix + "Trích được: " + ", ".join(f"{_REPORT_LABELS.get(k, k)}={v}" for k, v in filled.items())
        if draft.get("missing_fields"):
            text += f". Thiếu: {', '.join(draft['missing_fields'])}"
        if u.get("department_from_plan"):
            text += ". Khoa không được nêu → code lấy từ kế hoạch theo dõi (backend)."
        return text
    if node == "resolve_datetime":
        w = u.get("resolved_window")
        if w:
            when = f"{w['date_from']}" + (f" → {w['date_to']}" if w["date_to"] != w["date_from"] else "")
            if w.get("time_from"):
                when += f", {w['time_from'][:5]}–{w['time_to'][:5]}"
            return f"Quy đổi bằng code: {when} (luật: {', '.join(w.get('rules', []))})."
        if u.get("date_issues"):
            return f"Không quy đổi được: {_issues(u['date_issues'])}"
        return "Chưa có cụm ngày để quy đổi."
    if node in ("validate", "validate_booking"):
        issues = u.get("issues", [])
        if not issues:
            return "Đạt mọi lớp kiểm tra (schema, nghiệp vụ, grounding, nhất quán) → đi tiếp."
        return f"Phát hiện: {_issues(issues)}. {_route_hint(issues)}"
    if node in ("clarify", "clarify_intent", "present_draft", "present_slot_options", "present_booking_draft"):
        return outcome or _decision(u)
    if node == "apply_edits":
        return f"Trường người dùng sửa: {', '.join(u.get('edited_fields', []))} → chạy lại validator."
    if node in ("save", "create_booking"):
        if outcome:
            return f"Chốt chặn/tool báo lỗi. {outcome}"
        w = u.get("write_result") or {}
        err = (w.get("error") or {}).get("code")
        return f"Tool trả status={w.get('status')}" + (f", lỗi={err}" if err else "") + " → đi xác minh, chưa báo gì."
    if node in ("verify_save", "verify_booking"):
        if u.get("notice"):
            return f"Slot bị người khác đặt mất (conflict) → tìm lại slot. Lần thử lại: {u.get('conflict_retries')}."
        verified = outcome and "error" not in outcome
        return (outcome or "") + (" — đã đọc lại được bản ghi từ backend." if verified else "")
    if node in ("search_slots", "offer_alternatives"):
        if outcome:
            return outcome
        slots = u.get("offered_slots", [])
        relaxed = " (đã nới ±7 ngày, bỏ ràng buộc giờ/bác sĩ)" if node == "offer_alternatives" else ""
        return f"Backend trả {len(slots)} slot{relaxed}." if slots else "Không có slot đúng điều kiện → nới điều kiện."
    if node == "match_selection":
        if u.get("selected_slot_id"):
            return f"Khớp đúng 1 slot trong danh sách đã đưa: {u['selected_slot_id']}."
        return f"Chưa xác định được: {u.get('notice')} → hỏi lại."
    if node == "rag_answer":
        if outcome and u["outcome"].get("source_ids"):
            return f"Trả lời dựa trên tài liệu: {', '.join(u['outcome']['source_ids'])}."
        return outcome or ""
    if node == "respond":
        response = u.get("response") or {}
        return f"Câu trả lời dựng bằng template từ kết quả đã xác minh (kind={response.get('kind')})."
    if node == "handoff":
        return outcome or "Dừng an toàn."
    return outcome or ""


def describe_step(index: int, namespace: tuple[str, ...], node: str, update: Any) -> StepTrace:
    data: dict[str, Any] = to_jsonable_python(update or {}, fallback=str)
    title, kind = NODES.get(node, (node, "code"))
    flow = namespace[0].split(":")[0] if namespace else "main"
    # Không lặp lại các trường lớn/không cần cho việc giải thích.
    details = {k: v for k, v in data.items() if k not in ("attempts", "tool_calls", "last_output")}
    return StepTrace(
        index=index, flow=flow, node=node, title=title, kind=kind, summary=_summary(node, data), details=details
    )


def describe_interrupt(index: int, value: dict[str, Any]) -> StepTrace:
    return StepTrace(
        index=index,
        flow="main",
        node="interrupt",
        title="Dừng chờ người dùng",
        kind="hitl",
        summary=f"interrupt({value.get('kind')}): {value.get('message', '')[:160]}",
        details={"allowed_actions": value.get("allowed_actions", [])},
    )
