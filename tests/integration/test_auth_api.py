"""Đăng nhập, phân quyền theo vai trò, quyền sở hữu hội thoại và lưu lịch sử câu hỏi — qua HTTP thật."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import timedelta
from typing import Any

import pytest

from patient_assistant.application.services.auth_service import AuthService, hash_password, verify_password
from patient_assistant.domain.account import NewUser
from patient_assistant.infrastructure.clock import FixedClock
from patient_assistant.infrastructure.persistence import seed
from patient_assistant.infrastructure.persistence.sql import accounts
from patient_assistant.infrastructure.persistence.sql import repositories as sql
from tests.conftest import NOW, PATIENT, ScriptedLLM, api_client, booking_out, done, login, report_out

BOOK = "Cho tôi đặt lịch khoa phục hồi chức năng thứ ba tuần sau buổi sáng"


def _booking_llm() -> ScriptedLLM:
    out = booking_out(department=("PHCN", "phục hồi chức năng"), date="thứ ba tuần sau", time="buổi sáng")
    return ScriptedLLM(bookings=[out])


def test_password_hash_roundtrip() -> None:
    stored = hash_password("mật-khẩu-1")
    assert stored.startswith("scrypt$") and "mật" not in stored
    assert verify_password("mật-khẩu-1", stored)
    assert not verify_password("mật-khẩu-2", stored)
    assert hash_password("x" * 8) != hash_password("x" * 8)  # mỗi lần một salt
    assert not verify_password("a", "rác")


async def test_login_logout_and_bad_credentials(make_app: Any) -> None:
    async with api_client(await make_app()) as client:
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


NEW_ACCOUNT = {"username": "Nguyen.Van_A", "display_name": "  Nguyễn   Văn A ", "password": "matkhau123"}


async def test_register_creates_patient_and_logs_in(make_app: Any) -> None:
    async with api_client(await make_app()) as client:
        assert (await client.get("/v1/auth/options")).json() == {"allow_registration": True}
        # Gửi kèm role/patient_id cũng không được: tài khoản tự đăng ký luôn là bệnh nhân với hồ sơ mới.
        r = await client.post(
            "/v1/auth/register", json={**NEW_ACCOUNT, "role": "admin", "patient_id": "patient-demo-001"}
        )
        assert r.status_code == 201, r.text
        user = r.json()
        assert (user["username"], user["display_name"], user["role"]) == ("nguyen.van_a", "Nguyễn Văn A", "patient")
        assert user["patient_id"].startswith("patient-") and user["patient_id"] != "patient-demo-001"
        assert "httponly" in r.headers["set-cookie"].lower() and "password" not in r.text

        me = (await client.get("/v1/auth/me")).json()  # đã đăng nhập ngay bằng cookie
        assert me["id"] == user["id"]
        reports = await client.get("/v1/me/reports")
        assert reports.status_code == 200 and reports.json() == []  # hồ sơ mới, chưa có dữ liệu
        client.cookies.clear()

        again = await client.post("/v1/auth/login", json={"username": "nguyen.van_a", "password": "matkhau123"})
        assert again.status_code == 200
        client.cookies.clear()

        # Nhân viên thấy bệnh nhân mới trong danh sách (lấy từ DB, không phải dữ liệu demo cố định).
        staff = await login(client, "bacsi")
        patients = (await client.get("/v1/meta", headers=staff)).json()["patients"]
        assert {"id": user["patient_id"], "label": "Nguyễn Văn A"} in [
            {"id": p["id"], "label": p["label"]} for p in patients
        ]
        assert any(p["id"] == "patient-demo-001" for p in patients)


async def test_register_rejects_taken_username_and_bad_input(make_app: Any) -> None:
    app = await make_app()
    async with api_client(app) as client:
        assert (await client.post("/v1/auth/register", json=NEW_ACCOUNT)).status_code == 201
        client.cookies.clear()
        taken = await client.post("/v1/auth/register", json={**NEW_ACCOUNT, "username": "NGUYEN.VAN_A"})
        assert taken.status_code == 409 and "Tên đăng nhập" in taken.json()["detail"]
        demo = await client.post("/v1/auth/register", json={**NEW_ACCOUNT, "username": "benhnhan1"})
        assert demo.status_code == 409
        for bad in ({"username": "a"}, {"password": "ngan"}, {"display_name": "   "}):
            r = await client.post("/v1/auth/register", json={**NEW_ACCOUNT, "username": "khac", **bad})
            assert r.status_code == 422, (bad, r.text)
    # Đăng ký trùng không để lại hồ sơ bệnh nhân mồ côi.
    profiles = await app.auth.list_patients()
    assert len(profiles) == len(seed.PATIENTS) + 1  # bệnh nhân demo + 1 tài khoản vừa đăng ký


async def test_register_can_be_disabled(make_app: Any) -> None:
    async with api_client(await make_app(allow_registration=False)) as client:
        assert (await client.get("/v1/auth/options")).json() == {"allow_registration": False}
        r = await client.post("/v1/auth/register", json=NEW_ACCOUNT)
        assert r.status_code == 403
        login_attempt = await client.post("/v1/auth/login", json={"username": "nguyen.van_a", "password": "x"})
        assert login_attempt.status_code == 401  # không có tài khoản nào được tạo


async def test_logout_revokes_token_server_side(make_app: Any) -> None:
    async with api_client(await make_app()) as client:
        headers = await login(client, "benhnhan1")
        assert (await client.post("/v1/auth/logout", headers=headers)).status_code == 204
        assert (await client.get("/v1/auth/me", headers=headers)).status_code == 401


async def test_logout_all_revokes_every_device(make_app: Any) -> None:
    async with api_client(await make_app()) as client:
        phone = await login(client, "benhnhan1")
        laptop = await login(client, "benhnhan1")
        other = await login(client, "benhnhan2")
        client.cookies.clear()
        assert (await client.post("/v1/auth/logout-all")).status_code == 401  # phải đăng nhập mới gọi được

        r = await client.post("/v1/auth/logout-all", headers=laptop)
        assert r.status_code == 204 and 'pa_session=""' in r.headers["set-cookie"]
        for headers in (phone, laptop):
            assert (await client.get("/v1/auth/me", headers=headers)).status_code == 401
        assert (await client.get("/v1/auth/me", headers=other)).status_code == 200  # tài khoản khác không bị ảnh hưởng


async def test_purge_removes_only_expired_sessions(make_storage: Any) -> None:
    repos = (await make_storage()).repos
    clock = FixedClock(NOW)
    auth = AuthService(repos["users"], repos["sessions"], repos["patients"], clock, timedelta(hours=1))
    stale, _ = await auth.login("bacsi", "demo1234")  # không bao giờ dùng lại -> không tự bị xóa
    clock.now = lambda: NOW + timedelta(hours=2)  # type: ignore[method-assign]
    fresh, _ = await auth.login("benhnhan1", "demo1234")
    assert await repos["sessions"].get(_hash(stale)) is not None  # đăng nhập không còn tự dọn

    assert await auth.purge_expired_sessions() == 1
    assert await repos["sessions"].get(_hash(stale)) is None
    assert await auth.authenticate(fresh) is not None  # phiên còn hạn giữ nguyên
    assert await auth.purge_expired_sessions() == 0


async def test_purge_job_once_and_on_a_schedule(make_storage: Any) -> None:
    from patient_assistant.interface.cli import purge_sessions

    repos = (await make_storage()).repos
    clock = FixedClock(NOW)
    auth = AuthService(repos["users"], repos["sessions"], repos["patients"], clock, timedelta(hours=1))
    first, _ = await auth.login("bacsi", "demo1234")
    clock.now = lambda: NOW + timedelta(hours=2)  # type: ignore[method-assign]
    await purge_sessions.purge(auth)  # không --every: chạy một lần rồi thoát
    assert await repos["sessions"].get(_hash(first)) is None

    # Chạy theo lịch: lỗi một lần (vd. DB khởi động lại) không làm chết job, lần sau vẫn dọn tiếp.
    second, _ = await auth.login("bacsi", "demo1234")
    clock.now = lambda: NOW + timedelta(hours=4)  # type: ignore[method-assign]
    real = auth.purge_expired_sessions
    calls = 0

    async def flaky() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("DB đang khởi động lại")
        if calls == 3:
            raise asyncio.CancelledError  # dừng vòng lặp của test
        return await real()

    auth.purge_expired_sessions = flaky  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        await purge_sessions.purge(auth, every_minutes=0.0001)
    assert calls == 3 and await repos["sessions"].get(_hash(second)) is None

    with pytest.raises(SystemExit):
        purge_sessions.main(["--every", "0"])


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def test_lockout_after_repeated_failures(make_app: Any) -> None:
    async with api_client(await make_app()) as client:
        for _ in range(5):
            await client.post("/v1/auth/login", json={"username": "bacsi", "password": "sai"})
        locked = await client.post("/v1/auth/login", json={"username": "bacsi", "password": "demo1234"})
        assert locked.status_code == 401 and "quá nhiều lần" in locked.json()["detail"]
        await login(client, "benhnhan1")  # tài khoản khác không bị ảnh hưởng


async def test_session_expires(make_storage: Any) -> None:
    repos = (await make_storage()).repos
    clock = FixedClock(NOW)
    auth = AuthService(repos["users"], repos["sessions"], repos["patients"], clock, timedelta(hours=1))
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
    async with api_client(await make_app()) as client:
        headers = await login(client, username)
        kwargs: dict[str, Any] = {"json": {"text": "xin chào"}} if method == "post" else {}
        assert (await client.request(method, path, headers=headers, **kwargs)).status_code == status


async def test_patient_cannot_touch_another_patients_conversation(make_app: Any) -> None:
    async with api_client(await make_app(llm=_booking_llm())) as client:
        owner = await login(client, "benhnhan1")
        other = await login(client, "benhnhan2")
        r = await client.post("/v1/chat/c-1/messages", headers=owner, json={"text": BOOK})
        assert r.json()["interrupt"]["kind"] == "choose_slot"
        # Biết conversation_id cũng không đọc / gửi / trả lời thay được.
        assert (await client.post("/v1/chat/c-1/messages", headers=other, json={"text": "hi"})).status_code == 404
        resume = await client.post("/v1/chat/c-1/resume", headers=other, json={"action": "choose", "option_no": 1})
        assert resume.status_code == 404
        assert (await client.get("/v1/chat/c-1/state", headers=other)).status_code == 404
        assert (await client.get("/v1/me/conversations/c-1", headers=other)).status_code == 404
        assert (await client.get("/v1/me/conversations", headers=other)).json() == []
        assert (await client.get("/v1/chat/c-1/pending", headers=other)).status_code == 404


async def test_patient_reopens_past_conversation_and_continues(make_app: Any) -> None:
    async with api_client(await make_app(llm=_booking_llm())) as client:
        owner = await login(client, "benhnhan1")
        await client.post("/v1/chat/c-2/messages", headers=owner, json={"text": BOOK})

        # Đăng nhập lại (phiên mới) vẫn thấy hội thoại cũ, đọc được tin nhắn và câu hỏi đang chờ.
        again = await login(client, "benhnhan1")
        [conv] = (await client.get("/v1/me/conversations", headers=again)).json()
        assert conv["conversation_id"] == "c-2" and conv["title"].startswith("Cho tôi đặt lịch")
        messages = (await client.get("/v1/me/conversations/c-2", headers=again)).json()
        assert [m["role"] for m in messages] == ["user", "assistant"]
        pending = (await client.get("/v1/chat/c-2/pending", headers=again)).json()
        assert pending["kind"] == "choose_slot" and pending["options"]

        # Trả lời tiếp ngay trên hội thoại cũ.
        r = await client.post("/v1/chat/c-2/resume", headers=again, json={"action": "choose", "option_no": 1})
        assert r.json()["interrupt"]["kind"] == "confirm_booking"
        r = await client.post("/v1/chat/c-2/resume", headers=again, json={"action": "confirm"})
        assert r.json()["status"] == "done"
        assert (await client.get("/v1/chat/c-2/pending", headers=again)).json() is None  # xong: không còn câu hỏi


async def test_every_question_and_answer_is_logged_for_staff(make_app: Any) -> None:
    pain = (3, "tuần này tôi đau mức 3")
    llm = ScriptedLLM(reports=[report_out(pain=pain), report_out(pain=pain, days=(4, "tập 4 ngày"))])
    async with api_client(await make_app(llm=llm)) as client:
        patient = await login(client, "benhnhan1")
        await client.post("/v1/chat/log-1/messages", headers=patient, json={"text": "tuần này tôi đau mức 3"})
        await client.post("/v1/chat/log-1/resume", headers=patient, json={"action": "reply", "text": "tập 4 ngày"})
        await client.post("/v1/chat/log-1/resume", headers=patient, json={"action": "confirm"})

        mine = (await client.get("/v1/me/conversations", headers=patient)).json()
        assert [(c["conversation_id"], c["title"], c["message_count"], c["last_kind"]) for c in mine] == [
            ("log-1", "tuần này tôi đau mức 3", 6, "report_saved")
        ]

        staff = await login(client, "bacsi")
        listed = (await client.get(f"/v1/staff/conversations?patient_id={PATIENT}", headers=staff)).json()
        assert [c["conversation_id"] for c in listed] == ["log-1"]
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
    async with api_client(await make_app()) as client:
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


async def test_sql_accounts_sessions_and_log(make_storage: Any) -> None:
    from patient_assistant.domain.account import ChatMessage

    sessions = (await make_storage()).sessions
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
    assert await users.count() == len(seed.demo_accounts("demo1234"))

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
            ChatMessage(id=f"m{i}", conversation_id="t1", role=role, action=action, text=f"#{i}", created_at=NOW)
        )
    [conv] = await log.list_conversations(PATIENT, 10)
    assert (conv.message_count, conv.last_kind) == (2, "chat")
    assert [m.text for m in await log.list_messages("t1")] == ["#0", "#1"]  # cùng thời điểm vẫn đúng thứ tự


async def test_patients_with_the_same_name_only_see_their_own_data(make_app: Any) -> None:
    """Ba bệnh nhân cùng tên "Nguyễn Văn An": mỗi người chỉ thấy hồ sơ, kế hoạch và hội thoại của chính mình."""
    async with api_client(await make_app()) as client:
        patient_ids = set()
        for username in ("nguyenvanan", "nguyenvanan2", "an.nguyen"):
            headers = await login(client, username)
            me = (await client.get("/v1/auth/me", headers=headers)).json()
            assert me["display_name"] == "Nguyễn Văn An"
            patient_ids.add(me["patient_id"])
            assert (await client.get("/v1/me/conversations", headers=headers)).json() == []
        assert len(patient_ids) == 3  # ba hồ sơ khác nhau dù trùng tên

        staff = await login(client, "bacsi")
        patients = (await client.get("/v1/meta", headers=staff)).json()["patients"]
        assert [p["label"] for p in patients].count("Nguyễn Văn An") == 3


async def test_register_with_birth_date_and_phone(make_app: Any) -> None:
    app = await make_app()
    async with api_client(app) as client:
        body = {**NEW_ACCOUNT, "date_of_birth": "1990-05-17", "phone": "+84 912 345 678"}
        user = (await client.post("/v1/auth/register", json=body)).json()
        profile = await app.toolbox._patients.get_profile(user["patient_id"])
        assert (str(profile.date_of_birth), profile.phone) == ("1990-05-17", "0912345678")  # +84 -> 0, bỏ khoảng trắng
        client.cookies.clear()
        for bad in ({"phone": "12345"}, {"date_of_birth": "2999-01-01"}):
            r = await client.post("/v1/auth/register", json={**NEW_ACCOUNT, "username": "khac", **bad})
            assert r.status_code == 422, (bad, r.text)


async def test_staff_can_tell_apart_patients_with_identical_details(make_app: Any) -> None:
    async with api_client(await make_app()) as client:
        staff = await login(client, "bacsi")
        patients = (await client.get("/v1/meta", headers=staff)).json()["patients"]
        binh = [p for p in patients if p["display_name"] == "Trần Thị Bình"]
        # Trùng cả họ tên, ngày sinh, số điện thoại: chỉ mã hồ sơ là khác nhau.
        assert len(binh) == 2 and len({(p["date_of_birth"], p["phone"]) for p in binh}) == 1
        assert len({p["id"] for p in binh}) == 2
        an = [p for p in patients if p["display_name"] == "Nguyễn Văn An"]
        assert len({p["date_of_birth"] for p in an}) == 2 and len({p["phone"] for p in an}) == 2


async def test_patient_info_shows_birth_date_and_phone_but_llm_tool_does_not(make_app: Any) -> None:
    app = await make_app()
    info = done(await app.chat.send_message("pi-1", PATIENT, "xem thông tin của tôi"), "patient_info")
    assert "Ngày sinh: 15/01/1980" in info.response.text and "Số điện thoại: 0901000001" in info.response.text
    summary = next(t for t in app.toolbox.read_only_specs(PATIENT, NOW) if t.name == "get_patient_summary")
    profile = (await summary.handler(None))["data"]["profile"]  # type: ignore[arg-type]
    assert profile["display_name"] == "Bệnh nhân Demo 1" and "phone" not in profile and "date_of_birth" not in profile
