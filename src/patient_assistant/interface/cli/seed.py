"""Nạp dữ liệu demo vào database đã migrate (`alembic upgrade head` chạy trước).

    python -m patient_assistant.interface.cli.seed

Chạy lại an toàn (idempotent):
- danh mục khoa/bác sĩ/bệnh nhân/kế hoạch/tài liệu + lịch sử demo: chỉ nạp khi DB còn trống;
- tài khoản demo (`demo_data/users.json`, mật khẩu PA_SEED_PASSWORD): chỉ tạo khi chưa có tài khoản nào;
- slot khám: luôn bổ sung cho đủ `days_ahead` ngày tới, nên chạy mỗi lần deploy hoặc theo lịch hằng ngày;
- tạo/nâng cấp bảng checkpoint của LangGraph (một lần, trước khi backend chạy).
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack

from sqlalchemy.ext.asyncio import async_sessionmaker

from patient_assistant.config import Settings
from patient_assistant.container import build_auth, postgres_checkpointer, sql_repositories
from patient_assistant.infrastructure.clock import SystemClock
from patient_assistant.infrastructure.persistence import seed
from patient_assistant.infrastructure.persistence.sql.setup import make_engine, seed_demo_data


async def run(settings: Settings) -> None:
    if not settings.database_url:
        sys.exit("PA_DATABASE_URL chưa đặt — không có database để seed.")
    clock = SystemClock(settings.timezone)
    now = clock.now()
    engine = make_engine(settings.database_url)
    try:
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        # DB trống: nạp toàn bộ; đã có: bổ sung khoa/bác sĩ/... mới thêm vào JSON và slot còn thiếu.
        added = await seed_demo_data(sessions, now.date(), now.tzinfo, with_history=True)  # type: ignore[arg-type]
        auth = build_auth(settings, sql_repositories(sessions, clock.now().tzinfo), clock)
        await auth.seed(seed.demo_accounts(settings.seed_password))
        async with AsyncExitStack() as stack:
            await (await postgres_checkpointer(stack, settings)).setup()
        print(f"Seed xong: thêm {added} slot mới.")
    finally:
        await engine.dispose()


def main() -> None:
    asyncio.run(run(Settings()))


if __name__ == "__main__":
    main()
