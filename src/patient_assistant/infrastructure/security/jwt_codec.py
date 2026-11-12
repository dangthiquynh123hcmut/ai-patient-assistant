"""JWT HS256 bằng PyJWT. Khóa bí mật lấy từ `PA_JWT_SECRET` (dùng chung cho mọi worker/instance)."""

from __future__ import annotations

from typing import Any

import jwt

from patient_assistant.application.ports.tokens import InvalidTokenError

ISSUER = "patient-assistant"
ALGORITHM = "HS256"
MIN_SECRET_BYTES = 32  # RFC 7518: khóa HS256 tối thiểu 256 bit


class JwtCodec:
    def __init__(self, secret: str, issuer: str = ISSUER) -> None:
        if len(secret.encode()) < MIN_SECRET_BYTES:
            raise ValueError(f"PA_JWT_SECRET phải dài tối thiểu {MIN_SECRET_BYTES} byte")
        self._secret = secret
        self._issuer = issuer

    def encode(self, claims: dict[str, Any]) -> str:
        return jwt.encode({**claims, "iss": self._issuer}, self._secret, algorithm=ALGORITHM)

    def decode(self, token: str) -> dict[str, Any]:
        try:
            return jwt.decode(
                token,
                self._secret,
                algorithms=[ALGORITHM],  # cố định thuật toán: chặn "alg: none" / đổi thuật toán
                issuer=self._issuer,
                # exp/iat được AuthService kiểm theo Clock của ứng dụng (test dùng FixedClock).
                options={
                    "require": ["iss", "sub", "sid", "jti", "typ", "iat", "exp"],
                    "verify_exp": False,
                    "verify_iat": False,
                },
            )
        except jwt.PyJWTError as exc:
            raise InvalidTokenError(str(exc)) from exc
