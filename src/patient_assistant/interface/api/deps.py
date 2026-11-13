"""Dependency FastAPI dùng chung: xác thực và phân quyền.

Service lấy từ container bằng `Provide[Container.x]` + `@inject` (container được wire trong `create_app`).
"""

from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import Depends, HTTPException, Request

from patient_assistant.application.services.auth_service import AuthService
from patient_assistant.container import Container
from patient_assistant.domain.account import User


def bearer_token(request: Request) -> str | None:
    """Access token JWT trong `Authorization: Bearer`. Không đọc từ cookie, nên API không bị CSRF."""
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return None


@inject
async def current_user(
    request: Request,
    auth: Annotated[AuthService, Depends(Provide[Container.auth])],
) -> User:
    token = bearer_token(request)
    user = await auth.authenticate(token) if token else None
    if user is None:
        raise HTTPException(401, "Chưa đăng nhập hoặc phiên đã hết hạn")
    return user


U = Annotated[User, Depends(current_user)]


def _patient(user: U) -> User:
    if not user.can_chat:
        raise HTTPException(403, "Chức năng này chỉ dành cho tài khoản bệnh nhân")
    return user


def _staff(user: U) -> User:
    if not user.can_view_conversations:
        raise HTTPException(403, "Chức năng này chỉ dành cho nhân viên y tế")
    return user


def _admin(user: U) -> User:
    if not user.can_manage_users:
        raise HTTPException(403, "Chức năng này chỉ dành cho quản trị viên")
    return user


P = Annotated[User, Depends(_patient)]
S = Annotated[User, Depends(_staff)]
A = Annotated[User, Depends(_admin)]
