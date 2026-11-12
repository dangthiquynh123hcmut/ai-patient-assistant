"""Đăng nhập, phiên (JWT access + refresh token) và quản lý tài khoản.

- Mật khẩu băm bằng scrypt (thư viện chuẩn), mỗi mật khẩu một salt; so sánh hằng thời gian.
- Access token: JWT ngắn hạn (mặc định 15 phút), gửi qua `Authorization: Bearer`.
- Refresh token: JWT sống bằng hạn phiên, xoay vòng mỗi lần dùng. DB chỉ lưu SHA-256 của `jti` refresh token
  hiện hành, nên lộ DB không đăng nhập được. Dùng lại một refresh token đã bị xoay đi (dấu hiệu token bị đánh cắp)
  thì thu hồi cả phiên.
- Mỗi JWT mang `sid` (mã phiên). Access token chỉ hợp lệ khi phiên còn trong DB, nên đăng xuất / đổi mật khẩu /
  khóa tài khoản có hiệu lực ngay, không phải đợi access token hết hạn.
- Sai mật khẩu nhiều lần liên tiếp thì khóa tạm tài khoản đó (trạng thái nằm ở DB, đúng cả khi chạy nhiều worker).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from patient_assistant.application.ports.clock import Clock
from patient_assistant.application.ports.repositories import (
    PatientRepository,
    SessionRepository,
    UsernameTakenError,
    UserRepository,
)
from patient_assistant.application.ports.tokens import InvalidTokenError, TokenCodec
from patient_assistant.domain.account import MIN_PASSWORD_LENGTH, NewUser, Registration, Role, User
from patient_assistant.domain.patient import PatientProfile

_SCRYPT = {"n": 2**14, "r": 8, "p": 1}
MAX_FAILED_LOGINS = 5
LOCKOUT = timedelta(minutes=5)
# Hai tab cùng refresh bằng một token: request thua trong khoảng này chỉ bị từ chối (client thử lại với cookie mới),
# không bị coi là dùng lại token bị đánh cắp.
REFRESH_REUSE_GRACE = timedelta(seconds=30)
ACCESS = "access"
REFRESH = "refresh"


class AuthError(Exception):
    """Sai thông tin đăng nhập, tài khoản bị khóa, hoặc dữ liệu tài khoản không hợp lệ."""


class RefreshRaceError(AuthError):
    """Refresh token vừa được xoay bởi request khác (vd. hai tab cùng refresh): thử lại với cookie mới."""


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


def _token_hash(token_id: str) -> str:
    return hashlib.sha256(token_id.encode()).hexdigest()


# Dùng khi username không tồn tại: vẫn chạy scrypt để thời gian phản hồi không lộ username nào có thật.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))
_INVALID_SESSION = "Phiên đăng nhập không hợp lệ hoặc đã hết hạn."


@dataclass(frozen=True)
class IssuedTokens:
    """Kết quả đăng nhập / refresh. Token chỉ xuất hiện ở đây — phía server không lưu bản rõ."""

    access_token: str
    access_expires_in: int  # giây
    refresh_token: str
    refresh_expires_at: datetime
    user: User


class AuthService:
    def __init__(
        self,
        users: UserRepository,
        sessions: SessionRepository,
        patients: PatientRepository,
        clock: Clock,
        session_ttl: timedelta,
        *,
        tokens: TokenCodec,
        access_ttl: timedelta = timedelta(minutes=15),
    ) -> None:
        self._users = users
        self._sessions = sessions
        self._patients = patients
        self._clock = clock
        self._ttl = session_ttl
        self._tokens = tokens
        self._access_ttl = access_ttl

    @property
    def session_ttl(self) -> timedelta:
        return self._ttl

    # ------------------------------------------------------------------ đăng nhập

    async def login(self, username: str, password: str) -> IssuedTokens:
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
        session_id = str(uuid.uuid4())
        refresh_jti = secrets.token_urlsafe(32)
        expires_at = now + self._ttl  # hạn tuyệt đối của phiên: refresh không kéo dài
        await self._sessions.add(session_id, user.id, _token_hash(refresh_jti), expires_at)
        return self._issue(user, session_id, refresh_jti, expires_at, now)

    async def refresh(self, refresh_token: str) -> IssuedTokens:
        """Đổi refresh token lấy cặp token mới; refresh token cũ hết dùng được."""
        now = self._clock.now()
        claims = self._claims(refresh_token, REFRESH, now)
        if claims is None:
            raise AuthError(_INVALID_SESSION)
        session = await self._sessions.get(claims["sid"])
        if session is None or session.user_id != claims["sub"] or session.expires_at <= now:
            raise AuthError(_INVALID_SESSION)
        presented = _token_hash(claims["jti"])
        new_jti = secrets.token_urlsafe(32)
        rotated = presented == session.refresh_hash and await self._sessions.rotate(
            session.id, presented, _token_hash(new_jti), now
        )
        if not rotated:
            just_rotated = session.rotated_at is not None and now - session.rotated_at <= REFRESH_REUSE_GRACE
            if presented == session.refresh_hash or (presented == session.previous_refresh_hash and just_rotated):
                # Request khác vừa xoay đúng token này (vd. hai tab cùng refresh): từ chối, không thu hồi phiên.
                raise RefreshRaceError("Phiên vừa được làm mới ở nơi khác, hãy thử lại.")
            await self._sessions.delete(session.id)  # token cũ bị dùng lại: coi như bị lộ, thu hồi cả phiên
            raise AuthError("Phiên đăng nhập đã bị thu hồi.")
        user = await self._users.get(session.user_id)
        if user is None or not user.active:
            await self._sessions.delete(session.id)
            raise AuthError(_INVALID_SESSION)
        return self._issue(user, session.id, new_jti, session.expires_at, now)

    async def authenticate(self, access_token: str) -> User | None:
        now = self._clock.now()
        claims = self._claims(access_token, ACCESS, now)
        if claims is None:
            return None
        session = await self._sessions.get(claims["sid"])  # phiên bị thu hồi -> access token hết hiệu lực ngay
        if session is None or session.user_id != claims["sub"] or session.expires_at <= now:
            return None
        user = await self._users.get(session.user_id)
        return user if user and user.active else None

    async def logout(self, token: str) -> None:
        """Thu hồi phiên của token (access hoặc refresh, kể cả đã hết hạn). Token sai chữ ký thì bỏ qua."""
        try:
            claims = self._tokens.decode(token)
        except InvalidTokenError:
            return
        session = await self._sessions.get(str(claims["sid"]))
        if session is not None and session.user_id == claims["sub"]:
            await self._sessions.delete(session.id)

    async def logout_everywhere(self, user_id: str) -> None:
        """Thu hồi mọi phiên của người dùng (mọi thiết bị, kể cả phiên đang dùng)."""
        await self._sessions.delete_for_user(user_id)

    async def purge_expired_sessions(self) -> int:
        """Xóa mọi phiên đã hết hạn; trả số phiên đã xóa.

        Chỉ là dọn dẹp: phiên hết hạn vốn đã bị `authenticate` / `refresh` từ chối. Chạy theo lịch bởi
        `patient_assistant.interface.cli.purge_sessions` (service `session-cleaner` trong docker-compose).
        """
        return await self._sessions.delete_expired(self._clock.now())

    # ------------------------------------------------------------------ JWT

    def _issue(
        self, user: User, session_id: str, refresh_jti: str, expires_at: datetime, now: datetime
    ) -> IssuedTokens:
        access_exp = min(now + self._access_ttl, expires_at)  # access token không sống lâu hơn phiên
        base = {"sub": user.id, "sid": session_id, "iat": int(now.timestamp())}
        access = self._tokens.encode(
            {
                **base,
                "typ": ACCESS,
                "jti": secrets.token_urlsafe(16),
                "role": user.role.value,
                "exp": int(access_exp.timestamp()),
            }
        )
        refresh = self._tokens.encode({**base, "typ": REFRESH, "jti": refresh_jti, "exp": int(expires_at.timestamp())})
        return IssuedTokens(
            access_token=access,
            access_expires_in=max(0, int((access_exp - now).total_seconds())),
            refresh_token=refresh,
            refresh_expires_at=expires_at,
            user=user,
        )

    def _claims(self, token: str, typ: str, now: datetime) -> dict[str, Any] | None:
        """Claims nếu token đúng chữ ký, đúng loại và còn hạn (theo Clock của ứng dụng); ngược lại None."""
        try:
            claims = self._tokens.decode(token)
        except InvalidTokenError:
            return None
        if claims.get("typ") != typ or int(claims["exp"]) <= now.timestamp():
            return None
        return claims

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
