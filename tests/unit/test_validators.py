from __future__ import annotations

from datetime import date, datetime, timedelta

from patient_assistant.domain.booking.models import BookingRequest, Slot, SlotOption
from patient_assistant.domain.booking.policy import BookingPolicy
from patient_assistant.domain.booking.selection import match_selection
from patient_assistant.domain.booking.validators import BookingValidationContext, validate_booking
from patient_assistant.domain.common import Evidence, Resolution, SttFlag
from patient_assistant.domain.report.models import ReportExtraction
from patient_assistant.domain.report.validators import ReportValidationContext, parse_report, validate_report
from patient_assistant.domain.text.vi_dates import resolve
from patient_assistant.infrastructure.persistence.seed import DEPARTMENTS, DOCTORS
from tests.conftest import NOW, TZ

SOURCE = "Tuần này tôi đau mức 5 trên 10, đã tập 4 ngày, thấy đau tăng khi gập gối"


def _report(**kw: object) -> ReportExtraction:
    base: dict[str, object] = {
        "pain_level": 5,
        "exercise_days": 4,
        "symptom_change": "worse_during_activity",
        "evidence": [
            Evidence(field="pain_level", quote="đau mức 5 trên 10"),
            Evidence(field="exercise_days", quote="đã tập 4 ngày"),
            Evidence(field="symptom_change", quote="thấy đau tăng khi gập gối"),
        ],
    }
    base.update(kw)
    return ReportExtraction.model_validate(base)


def _codes(draft: ReportExtraction, sources: list[str] | None = None, **ctx: object) -> dict[str, Resolution]:
    issues = validate_report(draft, ReportValidationContext(sources=sources or [SOURCE], now=NOW, **ctx))  # type: ignore[arg-type]
    return {i.code: i.resolution for i in issues}


def test_valid_report_passes() -> None:
    assert _codes(_report()) == {}


def test_schema_layer_reports_field_errors() -> None:
    draft, issues = parse_report({"pain_level": "nhiều", "exercise_days": 4})
    assert draft is None
    assert issues[0].field == "pain_level" and issues[0].resolution == Resolution.LLM_FIXABLE


def test_value_not_matching_evidence_is_llm_fixable() -> None:
    assert _codes(_report(exercise_days=5)) == {"value_evidence_mismatch": Resolution.LLM_FIXABLE}


def test_fabricated_evidence_is_llm_fixable() -> None:
    draft = _report(
        evidence=[
            Evidence(field="pain_level", quote="đau mức 5 trên 10"),
            Evidence(field="exercise_days", quote="tập đủ 4 ngày liên tục rất đều"),
        ]
    )
    assert _codes(draft)["evidence_not_found"] == Resolution.LLM_FIXABLE


def test_out_of_range_said_by_user_asks_user_instead_of_fixing() -> None:
    draft = _report(
        pain_level=12,
        evidence=[Evidence(field="pain_level", quote="đau 12/10"), Evidence(field="exercise_days", quote="tập 3 ngày")],
        exercise_days=3,
        symptom_change=None,
    )
    assert _codes(draft, ["tuần này đau 12/10, tập 3 ngày"]) == {"out_of_range": Resolution.ASK_USER}


def test_missing_and_contradiction() -> None:
    assert _codes(_report(exercise_days=None))["missing_field"] == Resolution.ASK_USER
    draft = _report(
        evidence=[
            Evidence(field="pain_level", quote="đau mức 5 trên 10"),
            Evidence(field="exercise_days", quote="tập 4 ngày"),
        ],
        symptom_change=None,
    )
    codes = _codes(draft, ["đau mức 5 trên 10, tuần trước chưa tập, tuần này tập 4 ngày"])
    assert codes == {"contradiction": Resolution.ASK_USER}


def test_stt_flag_on_evidence_asks_user() -> None:
    codes = _codes(_report(), stt_flags=[SttFlag(text="4", probability=0.3, reason="low")])
    assert codes == {"stt_uncertain": Resolution.ASK_USER}


def _booking_ctx(sources: list[str], **kw: object) -> BookingValidationContext:
    return BookingValidationContext(
        sources=sources, now=NOW, policy=BookingPolicy(), departments=DEPARTMENTS, doctors=DOCTORS, **kw
    )  # type: ignore[arg-type]


def test_booking_department_must_be_said_by_user() -> None:
    sources = ["tôi bị đau khớp gối, đặt lịch thứ ba tuần sau"]
    req = BookingRequest(department_code="CXK", date_expression="thứ ba tuần sau")
    issues = validate_booking(req, resolve(req.date_expression, None, NOW), _booking_ctx(sources))
    assert [i.code for i in issues] == ["department_not_in_source"]
    assert (
        validate_booking(req, resolve(req.date_expression, None, NOW), _booking_ctx(sources, department_from_plan=True))
        == []
    )


def test_booking_weekday_mismatch_and_sunday() -> None:
    sources = ["đặt lịch khoa phục hồi chức năng thứ hai ngày 5/11"]
    req = BookingRequest(department_code="PHCN", date_expression="thứ hai ngày 5/11")
    issues = validate_booking(req, resolve(req.date_expression, None, NOW), _booking_ctx(sources))
    assert [i.code for i in issues] == ["weekday_mismatch"]
    sunday = BookingRequest(department_code="PHCN", date_expression="chủ nhật tuần sau")
    issues = validate_booking(
        sunday, resolve(sunday.date_expression, None, NOW), _booking_ctx(["khoa phục hồi chức năng chủ nhật tuần sau"])
    )
    assert [i.code for i in issues] == ["non_working_day"]


def test_booking_expression_must_be_verbatim() -> None:
    req = BookingRequest(department_code="PHCN", date_expression="03/11/2026")
    issues = validate_booking(
        req, resolve(req.date_expression, None, NOW), _booking_ctx(["khoa phục hồi chức năng thứ ba tuần sau"])
    )
    assert [i.code for i in issues] == ["expression_not_in_source"]


def _options() -> list[SlotOption]:
    day = date(2026, 11, 3)
    out = []
    for i, (hour, minute, doctor) in enumerate(
        [(8, 0, "BS. Trần Văn An"), (9, 0, "BS. Trần Văn An"), (9, 0, "BS. Lê Thị Bình")], start=1
    ):
        start = datetime(day.year, day.month, day.day, hour, minute, tzinfo=TZ)
        out.append(
            SlotOption(
                option_no=i,
                slot=Slot(
                    slot_id=f"s{i}",
                    department_code="PHCN",
                    doctor_id=doctor,
                    doctor_name=doctor,
                    start_at=start,
                    end_at=start + timedelta(minutes=30),
                ),
            )
        )
    return out


def test_selection_rules() -> None:
    options = _options()
    assert match_selection("cái thứ hai", options).option_no == 2
    assert match_selection("số 3", options).option_no == 3
    assert match_selection("3", options).option_no == 3
    assert match_selection("lấy cái đầu tiên", options).option_no == 1
    assert match_selection("8 giờ", options).option_no == 1
    ambiguous = match_selection("chín giờ", options)
    assert ambiguous.option_no is None and ambiguous.candidates == [2, 3]
    assert match_selection("9 giờ bác sĩ Bình", options).option_no == 3
    assert match_selection("không biết nữa", options).option_no is None
    assert match_selection("số 7", options).option_no is None  # không có trong danh sách
