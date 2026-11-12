"""Port ký / kiểm chữ ký token (JWT). Implementation: `infrastructure.security.jwt_codec`."""

from __future__ import annotations

from typing import Any, Protocol


class InvalidTokenError(Exception):
    """Token sai định dạng, sai chữ ký hoặc sai issuer. Hết hạn do AuthService kiểm theo Clock."""


class TokenCodec(Protocol):
    def encode(self, claims: dict[str, Any]) -> str: ...

    def decode(self, token: str) -> dict[str, Any]:
        """Kiểm chữ ký + issuer rồi trả claims. KHÔNG kiểm `exp` (AuthService kiểm theo Clock để test được)."""
        ...
