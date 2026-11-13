"""Đăng nhập / đăng ký / làm mới token / đăng xuất.

Refresh token nằm trong cookie HttpOnly `pa_refresh` (path `/v1/auth`, SameSite=Strict): JavaScript không đọc được,
trình duyệt chỉ gửi nó tới `/v1/auth/*`. `POST /v1/auth/refresh` đổi nó lấy access token mới (xoay vòng).
"""

from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from patient_assistant.application.ports.clock import Clock
from patient_assistant.application.ports.repositories import UsernameTakenError
from patient_assistant.application.services.auth_service import AuthError, AuthService, IssuedTokens, RefreshRaceError
from patient_assistant.config import Settings
from patient_assistant.container import Container
from patient_assistant.domain.account import Registration, User
from patient_assistant.interface.api.deps import U, bearer_token
from patient_assistant.interface.api.schemas import LoginIn, TokenOut

REFRESH_COOKIE = "pa_refresh"
REFRESH_COOKIE_PATH = "/v1/auth"  # chỉ gửi kèm request tới endpoint xác thực, không đi theo mọi API

router = APIRouter(prefix="/v1/auth", tags=["auth"])

AuthDep = Annotated[AuthService, Depends(Provide[Container.auth])]
SettingsDep = Annotated[Settings, Depends(Provide[Container.settings])]
ClockDep = Annotated[Clock, Depends(Provide[Container.clock])]


def _issue(response: Response, tokens: IssuedTokens, clock: Clock, settings: Settings) -> TokenOut:
    """Đặt refresh token vào cookie HttpOnly, trả access token trong body."""
    response.set_cookie(
        REFRESH_COOKIE,
        tokens.refresh_token,
        max_age=max(0, int((tokens.refresh_expires_at - clock.now()).total_seconds())),
        httponly=True,
        samesite="strict",
        secure=settings.cookie_secure,
        path=REFRESH_COOKIE_PATH,
    )
    response.headers["Cache-Control"] = "no-store"
    return TokenOut(access_token=tokens.access_token, expires_in=tokens.access_expires_in, user=tokens.user)


def _clear_refresh(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        REFRESH_COOKIE,
        path=REFRESH_COOKIE_PATH,
        httponly=True,
        samesite="strict",
        secure=settings.cookie_secure,
    )


@router.get("/options")
@inject
async def auth_options(settings: SettingsDep) -> dict[str, bool]:
    """Công khai (chưa đăng nhập): trang đăng nhập dùng để biết có hiện nút đăng ký không."""
    return {"allow_registration": settings.allow_registration}


@router.post("/login")
@inject
async def login(body: LoginIn, response: Response, auth: AuthDep, clock: ClockDep, settings: SettingsDep) -> TokenOut:
    try:
        tokens = await auth.login(body.username, body.password)
    except AuthError as exc:
        raise HTTPException(401, str(exc)) from exc
    return _issue(response, tokens, clock, settings)


@router.post("/refresh")
@inject
async def refresh(
    request: Request, response: Response, auth: AuthDep, clock: ClockDep, settings: SettingsDep
) -> TokenOut:
    """Đổi refresh token (cookie) lấy access token mới; cookie được thay bằng refresh token mới."""
    token = request.cookies.get(REFRESH_COOKIE)
    if not token:
        raise HTTPException(401, "Chưa đăng nhập hoặc phiên đã hết hạn")
    try:
        tokens = await auth.refresh(token)
    except RefreshRaceError as exc:
        raise HTTPException(409, str(exc)) from exc  # cookie đã có token mới: client gửi lại là được
    except AuthError as exc:
        raise HTTPException(401, str(exc)) from exc
    return _issue(response, tokens, clock, settings)


@router.post("/register", status_code=201)
@inject
async def register(
    body: Registration, response: Response, auth: AuthDep, clock: ClockDep, settings: SettingsDep
) -> TokenOut:
    """Tự tạo tài khoản bệnh nhân (kèm hồ sơ bệnh nhân mới) rồi đăng nhập luôn."""
    if not settings.allow_registration:
        raise HTTPException(403, "Hệ thống đang tắt chức năng tự đăng ký")
    try:
        await auth.register(body)
        tokens = await auth.login(body.username, body.password)
    except UsernameTakenError as exc:
        raise HTTPException(409, "Tên đăng nhập đã có người dùng") from exc
    except AuthError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _issue(response, tokens, clock, settings)


@router.post("/logout", status_code=204)
@inject
async def logout(request: Request, response: Response, auth: AuthDep, settings: SettingsDep) -> None:
    """Thu hồi phiên hiện tại (theo refresh cookie hoặc access token); không cần access token còn hạn."""
    for token in {request.cookies.get(REFRESH_COOKIE), bearer_token(request)} - {None}:
        await auth.logout(token)  # type: ignore[arg-type]
    _clear_refresh(response, settings)


@router.post("/logout-all", status_code=204)
@inject
async def logout_all(user: U, response: Response, auth: AuthDep, settings: SettingsDep) -> None:
    """Đăng xuất mọi thiết bị: thu hồi mọi phiên của tài khoản, kể cả phiên đang gọi."""
    await auth.logout_everywhere(user.id)
    _clear_refresh(response, settings)


@router.get("/me")
async def me(user: U) -> User:
    return user
