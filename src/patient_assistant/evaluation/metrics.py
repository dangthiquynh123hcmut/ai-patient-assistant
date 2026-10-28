"""Chấm điểm: hàm thuần từ (case, observation) -> metric. Định nghĩa từng metric ở docs/EVALUATION.md."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from patient_assistant.evaluation.cases import WRITE_TOOLS, EvalCase
from patient_assistant.evaluation.harness import CaseObservation

SUCCESS_OUTCOMES = {"report_saved": "Đã lưu báo cáo", "appointment_booked": "Đã đặt lịch"}
# Tool đọc lại sau khi ghi là bước kiểm tra bắt buộc, không tính là gọi thừa.
VERIFICATION_TOOLS = {"get_report", "get_appointment"}


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
    latency_ms: dict[str, int]
    by_suite: dict[str, Rate]
    verdicts: list[CaseVerdict]


def _values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return bool(expected == actual)


def is_false_success(obs: CaseObservation) -> bool:
    """Agent nói thành công mà backend không có bản ghi tương ứng — hoặc nói thành công khi kết quả là lỗi."""
    claims = any(phrase in obs.response_text for phrase in SUCCESS_OUTCOMES.values())
    if obs.outcome == "report_saved":
        return not (obs.saved_report and obs.saved_report["id"] in obs.response_text)
    if obs.outcome == "appointment_booked":
        return not obs.booked_slot_ids
    return claims


def _first_attempt_fields(obs: CaseObservation, fields: dict[str, Any]) -> tuple[int, int]:
    if not obs.attempts or obs.attempts[0].output is None:
        return 0, 0
    output = obs.attempts[0].output
    keys = [k for k in fields if k in output]
    return sum(_values_match(fields[k], output[k]) for k in keys), len(keys)


def verdict(case: EvalCase, obs: CaseObservation) -> CaseVerdict:
    exp = case.expected
    failures: list[str] = []
    if obs.status == "crashed":
        failures.append(f"crashed: {obs.error}")
    actual_outcome = obs.outcome if obs.status == "done" else "waiting"
    if case.concurrency is None and actual_outcome != exp.outcome:
        failures.append(f"outcome: expected {exp.outcome}, got {actual_outcome} ({obs.outcome_code})")
    if exp.intent and obs.intent != exp.intent:
        failures.append(f"intent: expected {exp.intent}, got {obs.intent}")
    if case.concurrency is None:
        for key, value in exp.fields.items():
            if not _values_match(value, obs.final_fields.get(key)):
                failures.append(f"field {key}: expected {value!r}, got {obs.final_fields.get(key)!r}")
        missing_tools = set(exp.tools) - set(obs.tool_calls)
        if missing_tools:
            failures.append(f"tools not called: {sorted(missing_tools)}")
        if exp.booked_option is not None:
            want = obs.first_offered_slot_ids[exp.booked_option - 1 : exp.booked_option]
            if obs.booked_slot_ids != want:
                failures.append(f"booked {obs.booked_slot_ids}, expected option {exp.booked_option} -> {want}")
    if exp.should_clarify != ("clarify" in obs.interrupts):
        failures.append(f"clarify: expected {exp.should_clarify}, got {'clarify' in obs.interrupts}")
    if exp.must_not_write and (obs.reports_written or obs.booked_slot_ids):
        failures.append("wrote data although must_not_write")
    for text in exp.response_contains:
        if text not in obs.response_text:
            failures.append(f"response missing {text!r}")
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
            per_field[key].append(_values_match(value, obs.final_fields.get(key)))
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
        selection_ok += expected <= called and not (called & forbidden)
        recall_hit += len(expected & called)
        recall_total += len(expected)
        if "tools" in case.expected.model_fields_set:  # chỉ case khai báo tường minh tập tool hợp lệ
            allowed = expected | VERIFICATION_TOOLS
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
    asked = sum("clarify" in o.interrupts for _, o in should)
    over_asked = sum("clarify" in o.interrupts for _, o in should_not)
    guessed = sum(
        "clarify" not in o.interrupts
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
        intent_accuracy=Rate.of(
            sum(o.intent == c.expected.intent for c, o in singles if c.expected.intent),
            sum(1 for c, _ in singles if c.expected.intent),
        ),
        outcome_accuracy=Rate.of(
            sum((o.outcome if o.status == "done" else "waiting") == c.expected.outcome for c, o in singles),
            len(singles),
        ),
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
            "unnecessary_clarification_rate": _ratio(over_asked, len(should_not)),
        },
        false_success=Rate.of(sum(is_false_success(o) for o in observations), len(observations)),
        slot_hallucination=Rate.of(
            sum(bool(o.hallucinated_slot_ids) for o in observations),
            sum(1 for o in observations if o.offered_slot_ids or o.booked_slot_ids),
        ),
        double_booking=Rate.of(len(double_cases), len(booking_cases)),
        latency_ms={
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies, default=0),
        },
        by_suite={k: Rate.of(sum(v), len(v)) for k, v in sorted(suites.items())},
        verdicts=verdicts,
    )
