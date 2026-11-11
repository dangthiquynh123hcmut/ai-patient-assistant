"""Dọn phiên đăng nhập đã hết hạn khỏi bảng `sessions`.

    python -m patient_assistant.interface.cli.purge_sessions              # chạy một lần rồi thoát
    python -m patient_assistant.interface.cli.purge_sessions --every 60   # chạy mãi, mỗi 60 phút một lần

Chạy một lần: dùng với crontab của máy chủ, Kubernetes CronJob, v.v.
Chạy `--every`: dùng cho service `session-cleaner` trong docker-compose (không cần cài cron trong image).

Đây chỉ là dọn dẹp cho bảng không phình ra: token hết hạn vốn đã bị từ chối ở mỗi request,
nên bỏ lỡ vài lần chạy (vd. container khởi động lại) cũng không ảnh hưởng bảo mật.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import timedelta

from sqlalchemy.ext.asyncio import async_sessionmaker

from patient_assistant.application.services.auth_service import AuthService
from patient_assistant.config import Settings
from patient_assistant.infrastructure.clock import SystemClock
from patient_assistant.infrastructure.persistence.sql import accounts
from patient_assistant.infrastructure.persistence.sql import repositories as sql
from patient_assistant.infrastructure.persistence.sql.setup import make_engine

log = logging.getLogger("patient_assistant.purge_sessions")


async def purge(auth: AuthService, every_minutes: float | None = None) -> None:
    """Một lần (every_minutes=None, lỗi thì ném ra) hoặc lặp mãi (lỗi tạm thời chỉ ghi log rồi thử lại lần sau)."""
    while True:
        try:
            removed = await auth.purge_expired_sessions()
            log.info("Đã xóa %d phiên hết hạn.", removed)
        except Exception:
            if every_minutes is None:
                raise
            log.exception("Dọn phiên thất bại, sẽ thử lại ở lần chạy sau.")
        if every_minutes is None:
            return
        await asyncio.sleep(every_minutes * 60)


async def run(settings: Settings, every_minutes: float | None = None) -> None:
    if not settings.database_url:
        sys.exit("PA_DATABASE_URL chưa đặt — không có database để dọn.")
    engine = make_engine(settings.database_url)
    try:
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        auth = AuthService(
            accounts.SqlUserRepository(sessions),
            accounts.SqlSessionRepository(sessions),
            sql.SqlPatientRepository(sessions),
            SystemClock(settings.timezone),
            timedelta(hours=settings.session_ttl_hours),
        )
        await purge(auth, every_minutes)
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Xóa phiên đăng nhập đã hết hạn.")
    parser.add_argument(
        "--every",
        type=float,
        metavar="PHÚT",
        help="Chạy lặp lại mỗi PHÚT phút (bỏ trống: chạy một lần rồi thoát).",
    )
    args = parser.parse_args(argv)
    if args.every is not None and args.every <= 0:
        parser.error("--every phải lớn hơn 0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(Settings(), args.every))


if __name__ == "__main__":
    main()
