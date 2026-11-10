"""Đăng nhập, phiên và quản lý tài khoản.

- Mật khẩu băm bằng scrypt (thư viện chuẩn), mỗi mật khẩu một salt; so sánh hằng thời gian.
- Token phiên ngẫu nhiên 256 bit; DB chỉ lưu SHA-256 của token, nên lộ DB không đăng nhập được.
- Sai mật khẩu nhiều lần liên tiếp thì khóa tạm tài khoản đó (trạng thái nằm ở DB, đúng cả khi chạy nhiều worker).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import uuid
from datetime import timedelta

from patient_assistant.application.ports.clock import Clock
from patient_assistant.application.ports.repositories import (
    PatientRepository,
    SessionRepository,
    UsernameTakenError,
    UserRepository,
)
from patient_assistant.domain.account import MIN_PASSWORD_LENGTH, NewUser, Registration, Role, User
from patient_assistant.domain.patient import PatientProfile

_SCRYPT = {"n": 2**14, "r": 8, "p": 1}
MAX_FAILED_LOGINS = 5
LOCKOUT = timedelta(minutes=5)


class AuthError(Exception):
    """Sai thông tin đăng nhập, tài khoản bị khóa, hoặc dữ liệu tài khoản không hợp lệ."""


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    b64 = base64.b64encode
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${b64(salt).decode()}${b64(digest).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt), dklen=len(expected), n=int(n), r=int(r), p=int(p)
        )
    except ValueError:
        return False
    return hmac.compare_digest(actual, expected)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# Dùng khi username không tồn tại: vẫn chạy scrypt để thời gian phản hồi không lộ username nào có thật.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


class AuthService:
    def __init__(
        self,
        users: UserRepository,
        sessions: SessionRepository,
        patients: PatientRepository,
        clock: Clock,
        session_ttl: timedelta,
    ) -> None:
        self._users = users
        self._sessions = sessions
        self._patients = patients
        self._clock = clock
        self._ttl = session_ttl

    @property
    def session_ttl(self) -> timedelta:
        return self._ttl

    # ------------------------------------------------------------------ đăng nhập

    async def login(self, username: str, password: str) -> tuple[str, User]:
        """Trả (token, user). Token chỉ xuất hiện ở đây — phía server không lưu bản rõ."""
        now = self._clock.now()
        found = await self._users.get_credentials(username.strip().lower())
        user = found[0] if found else None
        if user and user.locked_until and user.locked_until > now:
            raise AuthError("Đăng nhập sai quá nhiều lần, thử lại sau ít phút.")
        valid = verify_password(password, found[1] if found else _DUMMY_HASH)
        if user is None or not valid or not user.active:
            if user is not None:
                failed = user.failed_logins + 1
                locked = failed >= MAX_FAILED_LOGINS
                await self._users.set_login_state(user.id, 0 if locked else failed, now + LOCKOUT if locked else None)
            raise AuthError("Sai tên đăng nhập hoặc mật khẩu.")
        if user.failed_logins or user.locked_until:
            await self._users.set_login_state(user.id, 0, None)
        token = secrets.token_urlsafe(32)
        await self._sessions.add(_token_hash(token), user.id, now + self._ttl)
        return token, user

    async def authenticate(self, token: str) -> User | None:
        session = await self._sessions.get(_token_hash(token))
        if session is None:
            return None
        user_id, expires_at = session
        if expires_at <= self._clock.now():
            await self._sessions.delete(_token_hash(token))
            return None
        user = await self._users.get(user_id)
        return user if user and user.active else None

    async def logout(self, token: str) -> None:
        await self._sessions.delete(_token_hash(token))

    # ------------------------------------------------------------------ tự đăng ký

    async def register(self, registration: Registration) -> User:
        """Tạo hồ sơ bệnh nhân mới và tài khoản patient gắn với hồ sơ đó.

        Người tự đăng ký không chọn được vai trò hay hồ sơ có sẵn: chỉ admin tạo được staff/admin, hoặc
        gắn tài khoản vào hồ sơ bệnh nhân đã có.
        """
        if await self._users.get_credentials(registration.username):
            raise UsernameTakenError(registration.username)
        patient = PatientProfile(
            id=f"patient-{uuid.uuid4().hex[:12]}",
            display_name=registration.display_name,
            date_of_birth=registration.date_of_birth,
            phone=registration.phone,
        )
        await self._patients.add(patient)
        try:
            return await self.create_user(
                NewUser(
                    username=registration.username,
                    display_name=registration.display_name,
                    role=Role.PATIENT,
                    patient_id=patient.id,
                    password=registration.password,
                )
            )
        except Exception:
            await self._patients.remove(patient.id)  # vd. hai người đăng ký cùng username một lúc
            raise

    # ------------------------------------------------------------------ quản lý tài khoản (admin)

    async def list_patients(self) -> list[PatientProfile]:
        return await self._patients.list_profiles()

    async def create_user(self, new: NewUser) -> User:
        if new.patient_id and await self._patients.get_profile(new.patient_id) is None:
            raise AuthError(f"Không có bệnh nhân mã {new.patient_id}")
        user = User(
            id=str(uuid.uuid4()),
            username=new.username,
            display_name=new.display_name,
            role=new.role,
            patient_id=new.patient_id,
            created_at=self._clock.now(),
        )
        return await self._users.add(user, hash_password(new.password))

    async def list_users(self) -> list[User]:
        return await self._users.list()

    async def update_user(
        self, actor: User, user_id: str, *, active: bool | None = None, password: str | None = None
    ) -> User:
        if active is False and user_id == actor.id:
            raise AuthError("Không thể tự khóa tài khoản của chính mình.")
        if password is not None and len(password) < MIN_PASSWORD_LENGTH:
            raise AuthError(f"Mật khẩu tối thiểu {MIN_PASSWORD_LENGTH} ký tự.")
        updated = await self._users.update(
            user_id, active=active, password_hash=hash_password(password) if password is not None else None
        )
        if updated is None:
            raise AuthError("Không có tài khoản này.")
        if active is False or password is not None:
            await self._sessions.delete_for_user(user_id)  # đăng xuất mọi phiên cũ
        return updated

    async def seed(self, accounts: list[NewUser]) -> None:
        """Tạo các tài khoản demo còn thiếu (theo username). Tài khoản đã có — kể cả mật khẩu đã đổi — giữ nguyên,
        nên thêm người dùng vào `users.json` rồi chạy lại seed là có tài khoản mới mà không đụng tài khoản cũ."""
        for account in accounts:
            if await self._users.get_credentials(account.username) is None:
                await self.create_user(account)
