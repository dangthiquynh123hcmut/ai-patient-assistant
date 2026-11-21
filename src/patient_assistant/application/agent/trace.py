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
    "fetch_patient_info": ("Tổng hợp thông tin bệnh nhân", "tool"),
    "fetch_doctors": ("Danh sách khoa và bác sĩ", "tool"),
    "rag_answer": ("Trả lời từ tài liệu đã duyệt", "llm"),
    "converse": ("Trò chuyện tự nhiên", "llm"),
    "side_reply": ("Trả lời câu hỏi phụ", "llm"),
    "switch_task": ("Đổi sang việc mới", "code"),
    "reuse_context": ("Dùng lại thông tin đã nói", "code"),
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
    # agent_flow
    "agent": ("Agent (LLM tự tra dữ liệu bằng tool)", "llm"),
    "agent_confirm": ("Người dùng xác nhận việc agent đề xuất", "hitl"),
    "agent_wrapup": ("Ghi kết quả vào lịch sử agent", "code"),
    "agent_after_write": ("Đã ghi xong → xem còn việc nào chưa làm", "code"),
    "change_appointment": ("Gọi tool đổi/hủy lịch", "tool"),
    "verify_change": ("Đọc lại lịch hẹn để xác minh", "code"),
}

_RESOLUTION = {"llm_fixable": "LLM sửa", "ask_user": "hỏi người dùng", "block": "chặn"}
_INTENT_SOURCE = {
    "rule": "theo luật (code)",
    "llm": "bởi LLM vì luật không kết luận được",
    "context": "nối lại việc dang dở ở lượt trước (người dùng nói tiếp/lưu đi)",
}
_REPORT_LABELS = {
    "pain_level": "đau",
    "exercise_days": "số ngày tập",
    "symptom_change": "triệu chứng",
    "triggering_activity": "hoạt động",
    "reporting_period": "kỳ",
}


class StepTrace(BaseModel):
    index: int
    flow: str  # main | report_flow | booking_flow | agent_flow
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
    text = f"Kết quả: {outcome['kind']}" + (f" ({outcome['code']})" if outcome.get("code") else "")
    return text + (f" — Chi tiết: {outcome['detail']}" if outcome.get("detail") else "")


def _decision(update: dict[str, Any]) -> str:
    decision = update.get("decision") or {}
    parts = [f"action={decision.get('action')}"]
    if decision.get("text"):
        parts.append(f"“{decision['text']}”")
    if decision.get("option_no"):
        parts.append(f"option {decision['option_no']}")
    if decision.get("fields"):
        parts.append(f"sửa {decision['fields']}")
    return "Người dùng trả lời: " + ", ".join(parts)


def _summary(node: str, update: dict[str, Any]) -> str:  # noqa: C901 - một nhánh cho mỗi loại node
    outcome = _outcome(update)
    if node == "ingest":
        return "Chuẩn hóa văn bản đầu vào (text hoặc transcript từ giọng nói)."
    if node == "guard_input":
        kind = update.get("safety", "ok")
        return (
            "Không phát hiện cấp cứu / yêu cầu chẩn đoán."
            if kind == "ok"
            else f"Phát hiện '{kind}' bằng luật từ khóa → không gọi LLM, trả lời an toàn cố định."
        )
    if node == "classify":
        if outcome:
            return outcome
        text = f"Intent = {update.get('intent')} — {_INTENT_SOURCE.get(update.get('intent_source') or '', '')}."
        if update.get("pending_intents"):
            text += f" Việc khác để sau: {', '.join(update['pending_intents'])}."
        return text
    if node in ("extract", "correct", "extract_booking"):
        if outcome:
            return f"LLM lỗi, không đoán kết quả. {outcome}"
        prefix = f"Lần sửa thứ {update['correction_count']}. " if "correction_count" in update else ""
        draft = update.get("report_draft") or update.get("booking_request")
        if draft is None:
            return prefix + f"Đầu ra sai schema: {_issues(update.get('issues', []))}"
        filled = {k: v for k, v in draft.items() if v not in (None, [], "") and k not in ("evidence", "missing_fields")}
        text = prefix + "Trích được: " + ", ".join(f"{_REPORT_LABELS.get(k, k)}={v}" for k, v in filled.items())
        if draft.get("missing_fields"):
            text += f". Thiếu: {', '.join(draft['missing_fields'])}"
        if update.get("department_from_plan"):
            text += ". Khoa không được nêu → code lấy từ kế hoạch theo dõi (backend)."
        return text
    if node == "resolve_datetime":
        window = update.get("resolved_window")
        if window:
            when = f"{window['date_from']}" + (
                f" → {window['date_to']}" if window["date_to"] != window["date_from"] else ""
            )
            if window.get("time_from"):
                when += f", {window['time_from'][:5]}–{window['time_to'][:5]}"
            return f"Quy đổi bằng code: {when} (luật: {', '.join(window.get('rules', []))})."
        if update.get("date_issues"):
            return f"Không quy đổi được: {_issues(update['date_issues'])}"
        return "Chưa có cụm ngày để quy đổi."
    if node in ("validate", "validate_booking"):
        issues = update.get("issues", [])
        if not issues:
            return "Đạt mọi lớp kiểm tra (schema, nghiệp vụ, grounding, nhất quán) → đi tiếp."
        return f"Phát hiện: {_issues(issues)}. {_route_hint(issues)}"
    if node in ("clarify", "clarify_intent", "present_draft", "present_slot_options", "present_booking_draft"):
        return outcome or _decision(update)
    if node == "apply_edits":
        return f"Trường người dùng sửa: {', '.join(update.get('edited_fields', []))} → chạy lại validator."
    if node in ("save", "create_booking"):
        if outcome:
            return f"Chốt chặn/tool báo lỗi. {outcome}"
        write_result = update.get("write_result") or {}
        error = (write_result.get("error") or {}).get("code")
        return (
            f"Tool trả status={write_result.get('status')}"
            + (f", lỗi={error}" if error else "")
            + " → đi xác minh, chưa báo gì."
        )
    if node in ("verify_save", "verify_booking"):
        if update.get("notice"):
            return (
                f"Slot bị người khác đặt mất (conflict) → tìm lại slot. Lần thử lại: {update.get('conflict_retries')}."
            )
        verified = outcome and "error" not in outcome
        return (outcome or "") + (" — đã đọc lại được bản ghi từ backend." if verified else "")
    if node in ("search_slots", "offer_alternatives"):
        if outcome:
            return outcome
        slots = update.get("offered_slots", [])
        relaxed = " (đã nới ±7 ngày, bỏ ràng buộc giờ/bác sĩ)" if node == "offer_alternatives" else ""
        return f"Backend trả {len(slots)} slot{relaxed}." if slots else "Không có slot đúng điều kiện → nới điều kiện."
    if node == "match_selection":
        if update.get("selected_slot_id"):
            return f"Khớp đúng 1 slot trong danh sách đã đưa: {update['selected_slot_id']}."
        return f"Chưa xác định được: {update.get('notice')} → hỏi lại."
    if node == "fetch_patient_info":
        patient_info_outcome = update.get("outcome") or {}
        if patient_info_outcome.get("code") == "other_patient":
            return "Người dùng hỏi dữ liệu của người khác → từ chối, không tra cứu gì (bằng code)."
        if patient_info_outcome.get("kind") != "patient_info":
            return _outcome(update) or ""
        return "Ghép hồ sơ + kế hoạch + báo cáo gần đây + lịch hẹn từ backend (bằng code)."
    if node == "rag_answer":
        if outcome and update["outcome"].get("source_ids"):
            return f"Trả lời dựa trên tài liệu: {', '.join(update['outcome']['source_ids'])}."
        return outcome or ""
    if node == "agent":
        if update.get("outcome") and update["outcome"].get("kind") == "error":
            return f"LLM lỗi, không đoán kết quả. {outcome}"
        calls = update.get("agent_run_tools", [])
        used = f"LLM gọi tool: {', '.join(calls)}. " if calls else "LLM không cần gọi tool. "
        proposal = update.get("agent_proposal")
        if proposal:
            what = {"booking": "đặt lịch", "reschedule": "dời lịch", "cancel": "hủy lịch", "report": "lưu báo cáo"}
            return (
                used + f"Đề xuất {what.get(proposal['kind'], proposal['kind'])} → code đã kiểm tra, hiện thẻ xác nhận."
            )
        if update.get("agent_followup") is not None or (update.get("outcome") or {}).get("kind") not in (None, "chat"):
            return (
                used + "Không còn việc nào chưa làm."
                if not update.get("agent_followup")
                else used + "Nói thêm với người dùng."
            )
        return used + "Trả lời/hỏi lại người dùng bằng lời tự nhiên."
    if node == "agent_confirm":
        if outcome:
            return outcome
        action = (update.get("decision") or {}).get("action")
        if action == "confirm":
            return "Người dùng xác nhận → ghi."
        if action == "edit" and update.get("agent_proposal"):
            return "Người dùng sửa trên thẻ → code áp giá trị, chạy lại validator, hiện lại thẻ."
        return "Người dùng muốn đổi → quay lại agent."
    if node == "agent_after_write":
        if update.get("agent_note"):
            return "Kết quả đã xác minh → cho agent làm tiếp việc còn lại người dùng đã yêu cầu (nếu có)."
        return "Kết quả đã xác minh → kết thúc lượt."
    if node == "change_appointment":
        write_result = update.get("write_result") or {}
        return outcome or f"Tool trả status={write_result.get('status')} → đi xác minh, chưa báo gì."
    if node == "verify_change":
        if update.get("notice"):
            return "Slot mới vừa bị đặt mất → agent tìm slot khác, lịch cũ giữ nguyên."
        if not outcome or "error" in outcome:
            return outcome or ""
        cancelled = (update.get("outcome") or {}).get("kind") == "appointment_cancelled"
        return outcome + (" — đã đọc lại, lịch đã hủy." if cancelled else " — đã đọc lại cả lịch cũ lẫn lịch mới.")
    if node == "agent_wrapup":
        return "Ghi kết quả vào lịch sử hội thoại của agent cho các lượt sau."
    if node == "converse":
        return "Câu chào hỏi/trò chuyện → LLM trả lời tự nhiên, không gọi tool, không đọc dữ liệu bệnh nhân."
    if node == "respond":
        response = update.get("response") or {}
        return f"Câu trả lời dựng bằng template từ kết quả đã xác minh (kind={response.get('kind')})."
    if node == "handoff":
        return outcome or "Dừng an toàn."
    return outcome or ""


def describe_step(index: int, namespace: tuple[str, ...], node: str, update: Any) -> StepTrace:
    data: dict[str, Any] = to_jsonable_python(update or {}, fallback=str)
    title, kind = NODES.get(node, (node, "code"))
    flow = namespace[0].split(":")[0] if namespace else "main"
    # Không lặp lại các trường lớn/không cần cho việc giải thích.
    details = {
        k: v
        for k, v in data.items()
        # `attempts` giữ lại: tab "Kiểm tra & sửa lỗi" dựng lịch sử kiểm tra của cả hội thoại từ các bước đã lưu.
        if k not in ("tool_calls", "last_output", "agent_history", "agent_slots")
    }
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
