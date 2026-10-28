"""Dữ liệu giả lập: khoa, bác sĩ, kế hoạch theo dõi, slot, tài liệu hướng dẫn."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from patient_assistant.application.ports.repositories import KnowledgeDoc
from patient_assistant.domain.booking.models import Department, Doctor, PatientPlan, Slot

DEMO_PATIENT_ID = "patient-demo-001"

DEPARTMENTS = [
    Department(code="PHCN", name="Phục hồi chức năng", aliases=("phcn", "vật lý trị liệu", "tập phục hồi")),
    Department(code="CXK", name="Cơ xương khớp", aliases=("xương khớp", "khoa khớp")),
    Department(code="NOI", name="Nội tổng quát", aliases=("khoa nội", "nội khoa")),
]

DOCTORS = [
    Doctor(id="doc-an", name="BS. Trần Văn An", department_code="PHCN"),
    Doctor(id="doc-binh", name="BS. Lê Thị Bình", department_code="PHCN"),
    Doctor(id="doc-cuong", name="BS. Phạm Quốc Cường", department_code="CXK"),
    Doctor(id="doc-dung", name="BS. Nguyễn Thu Dung", department_code="NOI"),
]

PLANS = [
    PatientPlan(
        plan_id="plan-knee-01",
        patient_id=DEMO_PATIENT_ID,
        department_code="PHCN",
        summary="Phục hồi sau mổ dây chằng gối: tập gập/duỗi gối 5 ngày/tuần, báo cáo mức đau hằng tuần.",
    )
]

KNOWLEDGE = [
    KnowledgeDoc(
        doc_id="KB-001",
        title="Tập gập gối sau mổ dây chằng",
        content="Bài tập gập gối nên thực hiện chậm, mỗi lần 10 đến 15 lần, 2 đến 3 hiệp mỗi ngày. "
        "Nếu đau tăng khi tập, giảm biên độ và báo cho nhân viên y tế trong báo cáo tuần.",
    ),
    KnowledgeDoc(
        doc_id="KB-002",
        title="Chườm lạnh sau khi tập",
        content="Có thể chườm lạnh vùng gối 15 đến 20 phút sau khi tập để giảm sưng. Không chườm trực tiếp đá lên da.",
    ),
    KnowledgeDoc(
        doc_id="KB-003",
        title="Chuẩn bị trước buổi tái khám",
        content="Mang theo giấy hẹn, kết quả chụp chiếu cũ và ghi lại mức đau các tuần gần nhất "
        "để bác sĩ đánh giá tiến triển.",
    ),
]


def generate_slots(today: date, tz: ZoneInfo, days: int = 35) -> list[Slot]:
    """Mỗi bác sĩ: sáng 08:00–11:00 và chiều 14:00–16:00, mỗi slot 30 phút, Thứ Hai – Thứ Bảy."""
    starts = [
        time(8, 0),
        time(8, 30),
        time(9, 0),
        time(9, 30),
        time(10, 0),
        time(10, 30),
        time(14, 0),
        time(14, 30),
        time(15, 0),
        time(15, 30),
    ]
    slots: list[Slot] = []
    for offset in range(days):
        day = today + timedelta(days=offset)
        if day.weekday() == 6:
            continue
        for doctor in DOCTORS:
            for t in starts:
                start = datetime.combine(day, t, tz)
                slots.append(
                    Slot(
                        slot_id=f"{doctor.id}-{day:%Y%m%d}-{t:%H%M}",
                        department_code=doctor.department_code,
                        doctor_id=doctor.id,
                        doctor_name=doctor.name,
                        start_at=start,
                        end_at=start + timedelta(minutes=30),
                    )
                )
    return slots
