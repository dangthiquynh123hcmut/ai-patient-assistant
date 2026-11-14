"""Chấm điểm: hàm thuần từ (case, observation) -> metric. Định nghĩa từng metric ở docs/EVALUATION.md."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from patient_assistant.evaluation.cases import WRITE_TOOLS, EvalCase, OfferExpect
from patient_assistant.evaluation.harness import CaseObservation

SUCCESS_OUTCOMES = {
    "report_saved": "Đã lưu báo cáo",
    "appointment_booked": "Đã đặt lịch",
    "appointment_rescheduled": "Đã dời lịch",
    "appointment_cancelled": "Đã hủy lịch",
}
# Tool đọc lại sau khi ghi là bước kiểm tra bắt buộc, không tính là gọi thừa.
VERIFICATION_TOOLS = {"get_report", "get_appointment"}
# Chế độ có agent LLM + tool (hỏi lại bằng tin nhắn chat thay vì interrupt).
AGENT_MODES = {"booking", "full"}
# Chế độ full trả mọi câu trả lời đọc dữ liệu thành tin nhắn chat do agent soạn, không có kiểu outcome riêng.
READ_OUTCOMES = {"appointments", "history", "patient_info", "answer", "chat"}
# Tool của luồng workflow -> tool agent dùng cho cùng việc.
TOOL_ALIASES = {"list_my_appointments": "find_my_appointments"}
# Trường chỉ luồng workflow mới có (agent không trích xuất riêng) — bỏ qua ở chế độ agent, ghi chú lại.
WORKFLOW_ONLY_FIELDS = {"department_from_plan"}


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


class Rate(BaseModel):
    numerator: int
    denominator: int
    value: float | None

    @classmethod
    def of(cls, num: int, den: int) -> Rate:
        return cls(numerator=num, denominator=den, value=_ratio(num, den))


class CaseVerdict(BaseModel):
    case_id: str
    session: int
    suite: str
    category: str
    passed: bool
    failures: list[str] = Field(default_factory=list)
    # Kiểm tra viết cho luồng workflow, không áp dụng ở chế độ agent — chỉ ghi chú, không tính là trượt.
    notes: list[str] = Field(default_factory=list)
    latency_ms: int


class EvalReport(BaseModel):
    cases: int
    observations: int
    crashed: int
    case_pass_rate: Rate
    intent_accuracy: Rate
    outcome_accuracy: Rate
    field_accuracy: dict[str, Rate]
    field_accuracy_overall: Rate
    first_attempt_field_accuracy: Rate
    tool_selection_accuracy: Rate
    tool_recall: Rate
    unnecessary_tool_call_rate: Rate
    write_argument_accuracy: Rate
    validator: dict[str, Any]
    self_correction: dict[str, Any]
    clarification: dict[str, Any]
    false_success: Rate
    slot_hallucination: Rate
    double_booking: Rate
    own_schedule_overlap: Rate  # đề xuất slot trùng giờ lịch hẹn bệnh nhân đã có
    latency_ms: dict[str, int]
    by_suite: dict[str, Rate]
    verdicts: list[CaseVerdict]


def _values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return bool(expected == actual)


def outcome_ok(case: EvalCase, obs: CaseObservation) -> bool:
    actual = obs.outcome if obs.status == "done" else "waiting"
    expected = case.expected.outcome
    if actual == expected:
        return True
    if obs.agent_mode in AGENT_MODES and expected == "waiting" and obs.ended_with_question:
        return True  # agent hỏi lại bằng chat và kịch bản dừng ở đó = đang chờ người dùng
    return obs.agent_mode == "full" and expected in READ_OUTCOMES and actual == "chat"


def clarified(obs: CaseObservation) -> bool:
    """Có hỏi lại người dùng: interrupt `clarify`, hoặc (chế độ agent) câu hỏi dạng chat."""
    if "clarify" in obs.interrupts:
        return True
    return obs.agent_mode in AGENT_MODES and bool({"chat_question", "chat_question_open"} & set(obs.interrupts))


def over_asked(obs: CaseObservation) -> bool:
    """Hỏi lại khi không cần. Câu chat cuối lượt không ai trả lời ("mình giúp gì thêm?") không tính."""
    return "clarify" in obs.interrupts or (obs.agent_mode in AGENT_MODES and "chat_question" in obs.interrupts)


def tool_called(tool: str, called: set[str]) -> bool:
    return tool in called or TOOL_ALIASES.get(tool) in called


def field_ok(key: str, expected: Any, obs: CaseObservation) -> bool | None:
    """So một trường; None = không áp dụng ở chế độ này."""
    actual = obs.final_fields
    if obs.agent_mode in AGENT_MODES and key in WORKFLOW_ONLY_FIELDS:
        return None
    if key in ("date_from", "date_to") and key not in actual and "slot_date" in actual:
        # Chế độ agent không có khoảng ngày trích xuất: slot đã đặt phải nằm trong khoảng case mong đợi.
        day = str(actual["slot_date"])
        return day >= str(expected) if key == "date_from" else day <= str(expected)
    return _values_match(expected, actual.get(key))


def is_false_success(obs: CaseObservation) -> bool:
    """Agent nói thành công mà backend không có bản ghi tương ứng — hoặc nói thành công khi kết quả là lỗi."""
    claims = any(phrase in obs.response_text for phrase in SUCCESS_OUTCOMES.values())
    if obs.outcome == "report_saved":
        return not (obs.saved_report and obs.saved_report["id"] in obs.response_text)
    if obs.outcome in ("appointment_booked", "appointment_rescheduled"):
        return not obs.booked_slot_ids
    if obs.outcome == "appointment_cancelled":
        return obs.cancelled_count == 0
    return claims


def offer_failures(expect: OfferExpect, offers: list[dict[str, Any]]) -> list[str]:
    """So lần đề xuất slot đầu tiên với kỳ vọng của case."""
    if not offers:
        return ["offers: không có lần đề xuất slot nào"]
    failures: list[str] = []
    starts = [datetime.fromisoformat(o["start_at"]) for o in offers]
    hhmm = [s.strftime("%H:%M") for s in starts]
    if expect.on_date and any(s.date() != expect.on_date for s in starts):
        failures.append(f"offers: expected all on {expect.on_date}, got {[s.date().isoformat() for s in starts]}")
    if expect.department and any(o["department_code"] != expect.department for o in offers):
        failures.append(
            f"offers: expected department {expect.department}, got {[o['department_code'] for o in offers]}"
        )
    if expect.doctor and any(o["doctor_id"] != expect.doctor for o in offers):
        failures.append(f"offers: expected doctor {expect.doctor}, got {[o['doctor_id'] for o in offers]}")
    if expect.first_start and hhmm[0] != expect.first_start:
        failures.append(f"offers: expected first slot at {expect.first_start}, got {hhmm[0]}")
    if expect.min_start and any(h < expect.min_start for h in hhmm):
        failures.append(f"offers: expected all from {expect.min_start}, got {hhmm}")
    if expect.max_start and any(h > expect.max_start for h in hhmm):
        failures.append(f"offers: expected all until {expect.max_start}, got {hhmm}")
    if expect.hint is not None and any(bool(o.get("hint")) != expect.hint for o in offers):
        failures.append(f"offers: expected hint={expect.hint}, got {[bool(o.get('hint')) for o in offers]}")
    return failures


def _first_attempt_fields(obs: CaseObservation, fields: dict[str, Any]) -> tuple[int, int]:
    if not obs.attempts or obs.attempts[0].output is None:
        return 0, 0
    output = obs.attempts[0].output
    keys = [k for k in fields if k in output]
    return sum(_values_match(fields[k], output[k]) for k in keys), len(keys)


def verdict(case: EvalCase, obs: CaseObservation) -> CaseVerdict:
    exp = case.expected
    failures: list[str] = []
    notes: list[str] = []
    # Case viết cho workflow chạy ở chế độ full: câu chữ cố định của presenter workflow không áp dụng.
    workflow_text = obs.agent_mode == "full" and case.agent_mode is None
    if obs.status == "crashed":
        failures.append(f"crashed: {obs.error}")
    actual_outcome = obs.outcome if obs.status == "done" else "waiting"
    if case.concurrency is None and not outcome_ok(case, obs):
        failures.append(f"outcome: expected {exp.outcome}, got {actual_outcome} ({obs.outcome_code})")
    if exp.intent and obs.agent_mode != "full" and obs.intent != exp.intent:  # full không có bước phân loại
        failures.append(f"intent: expected {exp.intent}, got {obs.intent}")
    if case.concurrency is None:
        for key, value in exp.fields.items():
            ok = field_ok(key, value, obs)
            if ok is None:
                notes.append(f"field {key} không áp dụng ở chế độ {obs.agent_mode}")
            elif not ok:
                got = obs.final_fields.get(key, obs.final_fields.get("slot_date"))
                failures.append(f"field {key}: expected {value!r}, got {got!r}")
        called = set(obs.tool_calls)
        missing_tools = [t for t in exp.tools if not tool_called(t, called)]
        if missing_tools:
            failures.append(f"tools not called: {sorted(missing_tools)}")
        if exp.booked_option is not None:
            want = obs.first_offered_slot_ids[exp.booked_option - 1 : exp.booked_option]
            if obs.booked_slot_ids != want:
                failures.append(f"booked {obs.booked_slot_ids}, expected option {exp.booked_option} -> {want}")
    if exp.should_clarify and not clarified(obs):
        failures.append("clarify: expected True, got False")
    if not exp.should_clarify and over_asked(obs):
        failures.append("clarify: expected False, got True")
    if exp.must_not_write and (obs.reports_written or obs.booked_slot_ids or obs.cancelled_count):
        failures.append("wrote data although must_not_write")
    for text in exp.response_contains:
        if text not in obs.response_text:
            (notes if workflow_text else failures).append(f"response missing {text!r}")
    for text in exp.interrupt_contains:
        if not any(text in m for m in obs.interrupt_messages):
            failures.append(f"no question contains {text!r}")
    if exp.offers is not None and case.concurrency is None:
        offers = exp.offers
        if obs.agent_mode in AGENT_MODES and offers.hint is not None:
            notes.append("offers.hint không áp dụng ở chế độ agent (không có thẻ chọn slot)")
            offers = offers.model_copy(update={"hint": None})
        failures += offer_failures(offers, obs.first_offers)
    if obs.own_overlap_slot_ids:
        failures.append(f"offered slots overlapping own appointments {obs.own_overlap_slot_ids}")
    if is_false_success(obs):
        failures.append("FALSE SUCCESS")
    if obs.hallucinated_slot_ids:
        failures.append(f"hallucinated slots {obs.hallucinated_slot_ids}")
    if obs.double_booked_slot_ids:
        failures.append(f"double booked {obs.double_booked_slot_ids}")
    return CaseVerdict(
        case_id=case.id,
        session=obs.session,
        suite=case.suite,
        category=case.category,
        passed=not failures,
        failures=failures,
        notes=notes,
        latency_ms=obs.latency_ms,
    )


def _check_concurrency(pairs: list[tuple[EvalCase, CaseObservation]], verdicts: list[CaseVerdict]) -> None:
    """Case tranh chấp: đúng MỘT phiên đặt được slot, các phiên còn lại phải được báo slot đã bị đặt."""
    sessions: dict[str, list[int]] = defaultdict(list)
    for index, (case, _) in enumerate(pairs):
        if case.concurrency is not None:
            sessions[case.id].append(index)
    for indexes in sessions.values():
        winners = [i for i in indexes if pairs[i][1].booked_slot_ids]
        if len(winners) != 1:
            first = verdicts[indexes[0]]
            first.failures.append(f"concurrency: {len(winners)} sessions booked, expected exactly 1")
            first.passed = False


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def compute(cases: list[EvalCase], observations: list[CaseObservation]) -> EvalReport:
    by_id = {c.id: c for c in cases}
    pairs = [(by_id[o.case_id], o) for o in observations]
    singles = [(c, o) for c, o in pairs if c.concurrency is None]
    verdicts = [verdict(c, o) for c, o in pairs]
    _check_concurrency(pairs, verdicts)

    # -- field accuracy (giá trị cuối cùng) và lần trích xuất đầu tiên
    per_field: dict[str, list[bool]] = defaultdict(list)
    first_ok = first_total = 0
    for case, obs in singles:
        for key, value in case.expected.fields.items():
            matched = field_ok(key, value, obs)
            if matched is not None:
                per_field[key].append(matched)
        ok, total = _first_attempt_fields(obs, case.expected.fields)
        first_ok, first_total = first_ok + ok, first_total + total
    all_fields = [x for values in per_field.values() for x in values]

    # -- tool calls
    selection_ok = recall_hit = recall_total = unnecessary = total_calls = 0
    write_ok = write_total = 0
    for case, obs in singles:
        expected = set(case.expected.tools)
        called = set(obs.tool_calls)
        # Khi tiêm lỗi backend, việc THỬ gọi tool ghi là đúng; "không ghi dữ liệu" đã được kiểm ở verdict.
        must_not_try = case.expected.must_not_write and case.expected.outcome != "error"
        forbidden = set(WRITE_TOOLS) if must_not_try else set()
        hits = [t for t in expected if tool_called(t, called)]
        selection_ok += len(hits) == len(expected) and not (called & forbidden)
        recall_hit += len(hits)
        recall_total += len(expected)
        if "tools" in case.expected.model_fields_set:  # chỉ case khai báo tường minh tập tool hợp lệ
            allowed = expected | {TOOL_ALIASES.get(t, t) for t in expected} | VERIFICATION_TOOLS
            total_calls += len(obs.tool_calls)
            unnecessary += sum(1 for t in obs.tool_calls if t not in allowed)
        if case.expected.outcome == "report_saved" and case.expected.fields:
            write_total += 1
            saved = obs.saved_report or {}
            write_ok += all(
                _values_match(v, saved.get(k))
                for k, v in case.expected.fields.items()
                if k in ("pain_level", "exercise_days", "symptom_change", "triggering_activity")
            )
        if case.expected.booked_option is not None:
            write_total += 1
            want = obs.first_offered_slot_ids[case.expected.booked_option - 1 : case.expected.booked_option]
            write_ok += obs.booked_slot_ids == want

    # -- validator trên lần trích xuất đầu tiên
    tp = fp = fn = tn = code_hit = code_total = 0
    for case, obs in singles:
        if not obs.attempts:
            continue
        should_pass = case.first_attempt_should_pass()
        flagged = not obs.attempts[0].passed
        tp += flagged and not should_pass
        fn += (not flagged) and not should_pass
        fp += flagged and should_pass
        tn += (not flagged) and should_pass
        for code in case.labels.issue_codes:
            code_total += 1
            code_hit += code in obs.attempts[0].issue_codes

    # -- self-correction: case có lỗi LLM sửa được ở lần đầu
    needing = success = regression = 0
    attempt_counts: list[int] = []
    for case, obs in singles:
        if not case.labels.injected_error or not obs.attempts or obs.attempts[0].passed:
            continue
        needing += 1
        first_codes = set(obs.attempts[0].llm_fixable_codes)
        attempt_counts.append(len(obs.attempts))
        success += any(a.passed for a in obs.attempts[1:])
        # Hồi quy = lần sửa sinh ra lỗi LLM MỚI (không tính lỗi chuyển sang hỏi người dùng như missing_field).
        regression += any(set(a.llm_fixable_codes) - first_codes for a in obs.attempts[1:])

    # -- hỏi lại đúng lúc
    should = [(c, o) for c, o in singles if c.expected.should_clarify]
    should_not = [(c, o) for c, o in singles if not c.expected.should_clarify]
    asked = sum(clarified(o) for _, o in should)
    over = sum(over_asked(o) for _, o in should_not)
    guessed = sum(
        not clarified(o)
        and bool(o.reports_written or o.booked_slot_ids or {"confirm_report", "confirm_booking"} & set(o.interrupts))
        for _, o in should
    )

    suites: dict[str, list[bool]] = defaultdict(list)
    for v in verdicts:
        suites[v.suite].append(v.passed)
    latencies = [o.latency_ms for o in observations]
    booking_cases = {c.id for c, _ in pairs if c.suite == "booking"}
    double_cases = {o.case_id for o in observations if o.double_booked_slot_ids}

    return EvalReport(
        cases=len(cases),
        observations=len(observations),
        crashed=sum(o.status == "crashed" for o in observations),
        case_pass_rate=Rate.of(sum(v.passed for v in verdicts), len(verdicts)),
        intent_accuracy=Rate.of(  # chế độ full không có bước phân loại intent
            sum(o.intent == c.expected.intent for c, o in singles if c.expected.intent and o.agent_mode != "full"),
            sum(1 for c, o in singles if c.expected.intent and o.agent_mode != "full"),
        ),
        outcome_accuracy=Rate.of(sum(outcome_ok(c, o) for c, o in singles), len(singles)),
        field_accuracy={k: Rate.of(sum(v), len(v)) for k, v in sorted(per_field.items())},
        field_accuracy_overall=Rate.of(sum(all_fields), len(all_fields)),
        first_attempt_field_accuracy=Rate.of(first_ok, first_total),
        tool_selection_accuracy=Rate.of(selection_ok, len(singles)),
        tool_recall=Rate.of(recall_hit, recall_total),
        unnecessary_tool_call_rate=Rate.of(unnecessary, total_calls),
        write_argument_accuracy=Rate.of(write_ok, write_total),
        validator={
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "false_positive_rate": _ratio(fp, fp + tn),
            "false_negative_rate": _ratio(fn, fn + tp),
            "issue_code_recall": _ratio(code_hit, code_total),
        },
        self_correction={
            "cases": needing,
            "success_rate": _ratio(success, needing),
            "regression_rate": _ratio(regression, needing),
            "avg_attempts": round(sum(attempt_counts) / len(attempt_counts), 2) if attempt_counts else None,
        },
        clarification={
            "should_clarify": len(should),
            "clarified_when_needed": _ratio(asked, len(should)),
            "guess_rate": _ratio(guessed, len(should)),
            "unnecessary_clarification_rate": _ratio(over, len(should_not)),
        },
        false_success=Rate.of(sum(is_false_success(o) for o in observations), len(observations)),
        slot_hallucination=Rate.of(
            sum(bool(o.hallucinated_slot_ids) for o in observations),
            sum(1 for o in observations if o.offered_slot_ids or o.booked_slot_ids),
        ),
        double_booking=Rate.of(len(double_cases), len(booking_cases)),
        own_schedule_overlap=Rate.of(
            sum(bool(o.own_overlap_slot_ids) for o in observations),
            sum(1 for o in observations if o.offered_slot_ids),
        ),
        latency_ms={
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies, default=0),
        },
        by_suite={k: Rate.of(sum(v), len(v)) for k, v in sorted(suites.items())},
        verdicts=verdicts,
    )
