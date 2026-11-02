"""Đăng nhập, phân quyền theo vai trò, quyền sở hữu hội thoại và lưu lịch sử câu hỏi — qua HTTP thật."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from patient_assistant.application.services.auth_service import AuthService, hash_password, verify_password
from patient_assistant.domain.account import NewUser
from patient_assistant.infrastructure.clock import FixedClock
from patient_assistant.infrastructure.persistence import seed
from patient_assistant.infrastructure.persistence.sql import accounts
from patient_assistant.infrastructure.persistence.sql import repositories as sql
from patient_assistant.infrastructure.persistence.sql.setup import create_schema, make_engine, seed_demo_data
from tests.conftest import NOW, PATIENT, TZ, api_client, login

BOOK = "Cho tôi đặt lịch khoa phục hồi chức năng thứ ba tuần sau buổi sáng"


def test_password_hash_roundtrip() -> None:
    stored = hash_password("mật-khẩu-1")
    assert stored.startswith("scrypt$") and "mật" not in stored
    assert verify_password("mật-khẩu-1", stored)
    assert not verify_password("mật-khẩu-2", stored)
    assert hash_password("x" * 8) != hash_password("x" * 8)  # mỗi lần một salt
    assert not verify_password("a", "rác")


async def test_login_logout_and_bad_credentials(make_app: Any) -> None:
    async with api_client(make_app()) as client:
        assert (await client.get("/v1/auth/me")).status_code == 401
        bad = await client.post("/v1/auth/login", json={"username": "benhnhan1", "password": "sai-mat-khau"})
        assert bad.status_code == 401 and "Sai tên đăng nhập" in bad.json()["detail"]
        unknown = await client.post("/v1/auth/login", json={"username": "khongco", "password": "x"})
        assert unknown.json()["detail"] == bad.json()["detail"]  # không lộ username nào có thật

        r = await client.post("/v1/auth/login", json={"username": "BenhNhan1", "password": "demo1234"})
        assert r.status_code == 200 and r.json()["role"] == "patient"
        cookie = r.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=lax" in cookie
        assert (await client.get("/v1/auth/me")).json()["username"] == "benhnhan1"  # cookie dùng được
        assert "password" not in r.text

        assert (await client.post("/v1/auth/logout")).status_code == 204
        client.cookies.clear()
        assert (await client.get("/v1/auth/me")).status_code == 401


async def test_logout_revokes_token_server_side(make_app: Any) -> None:
    async with api_client(make_app()) as client:
        headers = await login(client, "benhnhan1")
        assert (await client.post("/v1/auth/logout", headers=headers)).status_code == 204
        assert (await client.get("/v1/auth/me", headers=headers)).status_code == 401


async def test_lockout_after_repeated_failures(make_app: Any) -> None:
    async with api_client(make_app()) as client:
        for _ in range(5):
            await client.post("/v1/auth/login", json={"username": "bacsi", "password": "sai"})
        locked = await client.post("/v1/auth/login", json={"username": "bacsi", "password": "demo1234"})
        assert locked.status_code == 401 and "quá nhiều lần" in locked.json()["detail"]
        await login(client, "benhnhan1")  # tài khoản khác không bị ảnh hưởng


async def test_session_expires() -> None:
    from patient_assistant.infrastructure.persistence.memory import (
        InMemoryPatientRepository,
        InMemorySessionRepository,
        InMemoryUserRepository,
    )

    clock = FixedClock(NOW)
    auth = AuthService(
        InMemoryUserRepository(),
        InMemorySessionRepository(),
        InMemoryPatientRepository(seed.profiles()),
        clock,
        timedelta(hours=1),
    )
    await auth.seed(seed.demo_accounts("demo1234"))
    token, _ = await auth.login("bacsi", "demo1234")
    assert await auth.authenticate(token) is not None
    clock.now = lambda: NOW + timedelta(hours=2)  # type: ignore[method-assign]
    assert await auth.authenticate(token) is None


@pytest.mark.parametrize(
    ("username", "path", "method", "status"),
    [
        ("benhnhan1", "/v1/staff/conversations", "get", 403),
        ("benhnhan1", "/v1/admin/users", "get", 403),
        ("bacsi", "/v1/admin/users", "get", 403),
        ("bacsi", "/v1/chat/x/messages", "post", 403),  # nhân viên không chat thay bệnh nhân
        ("bacsi", "/v1/me/reports", "get", 403),
        ("bacsi", "/v1/staff/conversations", "get", 200),
        ("admin", "/v1/staff/conversations", "get", 200),
        ("admin", "/v1/admin/users", "get", 200),
    ],
)
async def test_role_permissions(make_app: Any, username: str, path: str, method: str, status: int) -> None:
    async with api_client(make_app()) as client:
        headers = await login(client, username)
        kwargs: dict[str, Any] = {"json": {"text": "xin chào"}} if method == "post" else {}
        assert (await client.request(method, path, headers=headers, **kwargs)).status_code == status


async def test_patient_cannot_touch_another_patients_conversation(make_app: Any) -> None:
    async with api_client(make_app()) as client:
        owner = await login(client, "benhnhan1")
        other = await login(client, "benhnhan2")
        r = await client.post("/v1/chat/c-1/messages", headers=owner, json={"text": BOOK})
        assert r.json()["interrupt"]["kind"] == "choose_slot"
        # Biết thread_id cũng không đọc / gửi / trả lời thay được.
        assert (await client.post("/v1/chat/c-1/messages", headers=other, json={"text": "hi"})).status_code == 404
        resume = await client.post("/v1/chat/c-1/resume", headers=other, json={"action": "choose", "option_no": 1})
        assert resume.status_code == 404
        assert (await client.get("/v1/chat/c-1/state", headers=other)).status_code == 404
        assert (await client.get("/v1/me/conversations/c-1", headers=other)).status_code == 404
        assert (await client.get("/v1/me/conversations", headers=other)).json() == []


async def test_every_question_and_answer_is_logged_for_staff(make_app: Any) -> None:
    async with api_client(make_app()) as client:
        patient = await login(client, "benhnhan1")
        await client.post("/v1/chat/log-1/messages", headers=patient, json={"text": "tuần này tôi đau mức 3"})
        await client.post("/v1/chat/log-1/resume", headers=patient, json={"action": "reply", "text": "tập 4 ngày"})
        await client.post("/v1/chat/log-1/resume", headers=patient, json={"action": "confirm"})

        mine = (await client.get("/v1/me/conversations", headers=patient)).json()
        assert [(c["thread_id"], c["title"], c["message_count"], c["last_kind"]) for c in mine] == [
            ("log-1", "tuần này tôi đau mức 3", 6, "report_saved")
        ]

        staff = await login(client, "bacsi")
        listed = (await client.get(f"/v1/staff/conversations?patient_id={PATIENT}", headers=staff)).json()
        assert [c["thread_id"] for c in listed] == ["log-1"]
        assert (await client.get("/v1/staff/conversations?patient_id=patient-demo-002", headers=staff)).json() == []
        messages = (await client.get("/v1/staff/conversations/log-1", headers=staff)).json()
        assert [(m["role"], m["action"], m["text"][:22]) for m in messages] == [
            ("user", "message", "tuần này tôi đau mức 3"),
            ("assistant", "clarify", "Trong kỳ này bạn đã tậ"),
            ("user", "reply", "tập 4 ngày"),
            ("assistant", "confirm_report", "Đây là bản nháp báo cá"),
            ("user", "confirm", "Xác nhận"),
            ("assistant", "report_saved", messages[5]["text"][:22]),
        ]
        assert messages[1]["intent"] == "create_report"
        assert (await client.get("/v1/staff/conversations/khong-co", headers=staff)).status_code == 404


async def test_admin_manages_accounts(make_app: Any) -> None:
    async with api_client(make_app()) as client:
        admin = await login(client, "admin")
        new = {
            "username": "nguyenvana",
            "display_name": "Nguyễn Văn A",
            "role": "patient",
            "patient_id": "patient-demo-002",
            "password": "matkhau123",
        }
        created = await client.post("/v1/admin/users", headers=admin, json=new)
        assert created.status_code == 201 and created.json()["patient_id"] == "patient-demo-002"
        assert (await client.post("/v1/admin/users", headers=admin, json=new)).status_code == 409
        bad_patient = {**new, "username": "x-bad", "patient_id": "khong-co"}
        assert (await client.post("/v1/admin/users", headers=admin, json=bad_patient)).status_code == 400
        staff_with_patient = {**new, "username": "x-staff", "role": "staff"}
        assert (await client.post("/v1/admin/users", headers=admin, json=staff_with_patient)).status_code == 422
        short = {**new, "username": "x-short", "password": "123"}
        assert (await client.post("/v1/admin/users", headers=admin, json=short)).status_code == 422

        user_id = created.json()["id"]
        session = await login(client, "nguyenvana", "matkhau123")
        assert (await client.get("/v1/me/reports", headers=session)).status_code == 200

        # Đổi mật khẩu -> phiên cũ bị đăng xuất, mật khẩu cũ hết dùng được.
        r = await client.patch(f"/v1/admin/users/{user_id}", headers=admin, json={"password": "matkhaumoi1"})
        assert r.status_code == 200
        assert (await client.get("/v1/auth/me", headers=session)).status_code == 401
        old = await client.post("/v1/auth/login", json={"username": "nguyenvana", "password": "matkhau123"})
        assert old.status_code == 401
        session = await login(client, "nguyenvana", "matkhaumoi1")

        # Khóa tài khoản -> phiên hiện tại mất hiệu lực, không đăng nhập lại được.
        assert (await client.patch(f"/v1/admin/users/{user_id}", headers=admin, json={"active": False})).json()[
            "active"
        ] is False
        assert (await client.get("/v1/auth/me", headers=session)).status_code == 401
        locked = await client.post("/v1/auth/login", json={"username": "nguyenvana", "password": "matkhaumoi1"})
        assert locked.status_code == 401

        me = (await client.get("/v1/auth/me", headers=admin)).json()
        self_lock = await client.patch(f"/v1/admin/users/{me['id']}", headers=admin, json={"active": False})
        assert self_lock.status_code == 400


async def test_sql_accounts_sessions_and_log(tmp_path: Any) -> None:
    from patient_assistant.domain.account import ChatMessage

    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'auth.db'}")
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    await seed_demo_data(sessions, NOW.date(), TZ)
    users = accounts.SqlUserRepository(sessions)
    auth = AuthService(
        users,
        accounts.SqlSessionRepository(sessions),
        sql.SqlPatientRepository(sessions),
        FixedClock(NOW),
        timedelta(hours=12),
    )
    await auth.seed(seed.demo_accounts("demo1234"))
    await auth.seed(seed.demo_accounts("khac-mat-khau"))  # chạy lại không tạo trùng / không đổi mật khẩu
    assert await users.count() == 5

    token, user = await auth.login("benhnhan1", "demo1234")
    assert (await auth.authenticate(token)) == user and user.patient_id == PATIENT
    with pytest.raises(Exception, match="Sai"):
        await auth.login("benhnhan1", "khac-mat-khau")
    with pytest.raises(Exception, match="bệnh nhân"):
        await auth.create_user(
            NewUser(username="yyy", display_name="Y", role="patient", patient_id="nope", password="12345678")
        )
    await auth.logout(token)
    assert await auth.authenticate(token) is None

    log = accounts.SqlConversationLog(sessions)
    assert await log.open("t1", user.id, PATIENT, "xin chào", NOW) == PATIENT
    assert await log.open("t1", "kẻ khác", "patient-demo-002", "x", NOW) == PATIENT  # chủ cũ giữ nguyên
    for i, (role, action) in enumerate([("user", "message"), ("assistant", "chat")]):
        await log.add_message(
            ChatMessage(id=f"m{i}", thread_id="t1", role=role, action=action, text=f"#{i}", created_at=NOW)
        )
    [conv] = await log.list_conversations(PATIENT, 10)
    assert (conv.message_count, conv.last_kind) == (2, "chat")
    assert [m.text for m in await log.list_messages("t1")] == ["#0", "#1"]  # cùng thời điểm vẫn đúng thứ tự
    await engine.dispose()
