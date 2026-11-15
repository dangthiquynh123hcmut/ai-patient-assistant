"""Xuất dữ liệu đang có trong database ra JSON để kiểm tra.

python -m patient_assistant.interface.cli.dump                      # database theo PA_DATABASE_URL
python -m patient_assistant.interface.cli.dump --out demo_dump.json
python -m patient_assistant.interface.cli.dump --database-url postgresql+asyncpg://user:pass@localhost:5432/db
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from patient_assistant.config import Settings
from patient_assistant.container import sql_repositories
from patient_assistant.infrastructure.clock import SystemClock
from patient_assistant.infrastructure.persistence import seed
from patient_assistant.infrastructure.persistence.sql.setup import make_engine


async def collect(database_url: str, slot_limit: int) -> dict[str, Any]:
    now = SystemClock(Settings().timezone).now()
    engine = make_engine(database_url)  # chỉ đọc, không đụng schema
    try:
        return await _collect(sql_repositories(async_sessionmaker(engine), now.tzinfo), database_url, now, slot_limit)
    finally:
        await engine.dispose()


async def _collect(repositories: dict[str, Any], database_url: str, now: datetime, slot_limit: int) -> dict[str, Any]:
    departments = await repositories["catalog"].list_departments()
    doctors = await repositories["catalog"].list_doctors()
    patients = [patient.model_dump() for patient in seed.PATIENTS]
    plans = []
    for patient in patients:
        plan = await repositories["plans"].get_active_plan(patient["id"])
        if plan:
            plans.append(plan)
    reports = [
        report for patient in patients for report in await repositories["reports"].list_by_patient(patient["id"], 100)
    ]
    appointments = [
        appointment
        for patient in patients
        for appointment in await repositories["appointments"].list_for_patient(patient["id"])
    ]
    available = []
    for department in departments:
        available += await repositories["slots"].find_available(
            department.code, now, now.replace(year=now.year + 1), None, 10_000
        )
    available.sort(key=lambda slot: slot.start_at)
    return {
        "source": database_url.split("@")[-1],  # bỏ user/mật khẩu khỏi URL
        "generated_at": now.isoformat(),
        "departments": [department.model_dump() for department in departments],
        "doctors": [doctor.model_dump() for doctor in doctors],
        "patients": patients,
        "plans": [plan.model_dump() for plan in plans],
        "reports": [report.model_dump(mode="json") for report in reports],
        "appointments": [appointment.model_dump(mode="json") for appointment in appointments],
        "available_slots": {
            "total": len(available),
            "first": [slot.model_dump(mode="json") for slot in available[:slot_limit]],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Xuất dữ liệu database ra JSON")
    parser.add_argument("--database-url", default=Settings().database_url)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--slots", type=int, default=10, help="Số slot trống in ra (mặc định 10)")
    arguments = parser.parse_args()
    if not arguments.database_url:
        sys.exit("Chưa có database: đặt PA_DATABASE_URL hoặc --database-url")
    text = json.dumps(asyncio.run(collect(arguments.database_url, arguments.slots)), ensure_ascii=False, indent=2)
    if arguments.out:
        arguments.out.write_text(text, "utf-8")
        print(f"Đã ghi {arguments.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
