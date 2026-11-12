"""FastAPI: lớp mỏng chuyển HTTP <-> use case. Không chứa nghiệp vụ.

Chỉ phục vụ API (`/v1/...`, `/health`). Frontend React được build riêng (`frontend/`) và do nginx phục vụ,
nginx proxy `/v1/` về đây — cùng origin nên cookie refresh token dùng được mà không cần CORS.

Xác thực JWT:
- Access token (ngắn hạn) trả trong body, client giữ trong bộ nhớ và gửi `Authorization: Bearer <token>`.
- Refresh token nằm trong cookie HttpOnly `pa_refresh` (path `/v1/auth`, SameSite=Strict): JavaScript không đọc được,
  trình duyệt chỉ gửi nó tới `/v1/auth/*`. `POST /v1/auth/refresh` đổi nó lấy access token mới (xoay vòng).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile

from patient_assistant.application.agent.state import InterruptPayload, UserDecision
from patient_assistant.application.ports.repositories import UsernameTakenError
from patient_assistant.application.ports.stt import STTError
from patient_assistant.application.services.auth_service import AuthError, IssuedTokens, RefreshRaceError
from patient_assistant.application.use_cases.chat import NoPendingQuestion, TurnResult
from patient_assistant.application.use_cases.conversation import NotAllowed
from patient_assistant.application.use_cases.transcribe import TranscriptReview
from patient_assistant.config import Settings
from patient_assistant.container import Container, build_container
from patient_assistant.domain.account import ChatMessage, ConversationSummary, NewUser, Registration, User
from patient_assistant.interface.api.schemas import LoginIn, MessageIn, ResumeIn, TokenOut, UserUpdateIn

MAX_AUDIO_BYTES = 10 * 1024 * 1024
REFRESH_COOKIE = "pa_refresh"
REFRESH_COOKIE_PATH = "/v1/auth"  # chỉ gửi kèm request tới endpoint xác thực, không đi theo mọi API


def _container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


C = Annotated[Container, Depends(_container)]


def _token(request: Request) -> str | None:
    """Access token JWT trong `Authorization: Bearer`. Không đọc từ cookie, nên API không bị CSRF."""
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return None


async def _current_user(request: Request, c: C) -> User:
    token = _token(request)
    user = await c.auth.authenticate(token) if token else None
    if user is None:
        raise HTTPException(401, "Chưa đăng nhập hoặc phiên đã hết hạn")
    return user


U = Annotated[User, Depends(_current_user)]


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


def create_app(container_factory: Callable[[], AbstractAsyncContextManager[Container]] | None = None) -> FastAPI:
    factory = container_factory or (lambda: build_container(Settings()))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with factory() as container:
            app.state.container = container
            yield

    app = FastAPI(title="AI Patient Assistant", version="0.1.0", lifespan=lifespan)

    # ------------------------------------------------------------------ đăng nhập

    def issue(response: Response, c: Container, tokens: IssuedTokens) -> TokenOut:
        """Đặt refresh token vào cookie HttpOnly, trả access token trong body."""
        response.set_cookie(
            REFRESH_COOKIE,
            tokens.refresh_token,
            max_age=max(0, int((tokens.refresh_expires_at - c.clock.now()).total_seconds())),
            httponly=True,
            samesite="strict",
            secure=c.settings.cookie_secure,
            path=REFRESH_COOKIE_PATH,
        )
        response.headers["Cache-Control"] = "no-store"
        return TokenOut(access_token=tokens.access_token, expires_in=tokens.access_expires_in, user=tokens.user)

    def clear_refresh(response: Response, c: Container) -> None:
        response.delete_cookie(
            REFRESH_COOKIE,
            path=REFRESH_COOKIE_PATH,
            httponly=True,
            samesite="strict",
            secure=c.settings.cookie_secure,
        )

    @app.get("/v1/auth/options")
    async def auth_options(c: C) -> dict[str, bool]:
        """Công khai (chưa đăng nhập): trang đăng nhập dùng để biết có hiện nút đăng ký không."""
        return {"allow_registration": c.settings.allow_registration}

    @app.post("/v1/auth/login")
    async def login(body: LoginIn, response: Response, c: C) -> TokenOut:
        try:
            tokens = await c.auth.login(body.username, body.password)
        except AuthError as exc:
            raise HTTPException(401, str(exc)) from exc
        return issue(response, c, tokens)

    @app.post("/v1/auth/refresh")
    async def refresh(request: Request, response: Response, c: C) -> TokenOut:
        """Đổi refresh token (cookie) lấy access token mới; cookie được thay bằng refresh token mới."""
        token = request.cookies.get(REFRESH_COOKIE)
        if not token:
            raise HTTPException(401, "Chưa đăng nhập hoặc phiên đã hết hạn")
        try:
            tokens = await c.auth.refresh(token)
        except RefreshRaceError as exc:
            raise HTTPException(409, str(exc)) from exc  # cookie đã có token mới: client gửi lại là được
        except AuthError as exc:
            raise HTTPException(401, str(exc)) from exc
        return issue(response, c, tokens)

    @app.post("/v1/auth/register", status_code=201)
    async def register(body: Registration, response: Response, c: C) -> TokenOut:
        """Tự tạo tài khoản bệnh nhân (kèm hồ sơ bệnh nhân mới) rồi đăng nhập luôn."""
        if not c.settings.allow_registration:
            raise HTTPException(403, "Hệ thống đang tắt chức năng tự đăng ký")
        try:
            await c.auth.register(body)
            tokens = await c.auth.login(body.username, body.password)
        except UsernameTakenError as exc:
            raise HTTPException(409, "Tên đăng nhập đã có người dùng") from exc
        except AuthError as exc:
            raise HTTPException(400, str(exc)) from exc
        return issue(response, c, tokens)

    @app.post("/v1/auth/logout", status_code=204)
    async def logout(request: Request, response: Response, c: C) -> None:
        """Thu hồi phiên hiện tại (theo refresh cookie hoặc access token); không cần access token còn hạn."""
        for token in {request.cookies.get(REFRESH_COOKIE), _token(request)} - {None}:
            await c.auth.logout(token)  # type: ignore[arg-type]
        clear_refresh(response, c)

    @app.post("/v1/auth/logout-all", status_code=204)
    async def logout_all(user: U, response: Response, c: C) -> None:
        """Đăng xuất mọi thiết bị: thu hồi mọi phiên của tài khoản, kể cả phiên đang gọi."""
        await c.auth.logout_everywhere(user.id)
        clear_refresh(response, c)

    @app.get("/v1/auth/me")
    async def me(user: U) -> User:
        return user

    @app.get("/v1/meta")
    async def meta(c: C, user: U) -> dict[str, Any]:
        """Cấu hình đang chạy và dữ liệu giả lập — để UI hiển thị gợi ý."""
        departments, doctors = await c.toolbox.catalog()
        s = c.settings
        return {
            "user": user.model_dump(mode="json"),
            "llm_model": s.llm_model_id,
            "stt_enabled": s.stt_enabled,
            "stt_model": s.stt_model if s.stt_enabled else None,
            "stt_language": s.stt_language,
            "now": c.clock.now().isoformat(),
            "patients": [
                {
                    "id": p.id,
                    "label": c.patient_labels.get(p.id, p.display_name),
                    "display_name": p.display_name,
                    "date_of_birth": p.date_of_birth.isoformat() if p.date_of_birth else None,
                    "phone": p.phone,
                }
                for p in (await c.auth.list_patients() if user.can_view_conversations else [])
            ],
            "departments": [d.model_dump() for d in departments],
            "doctors": [d.model_dump() for d in doctors],
            "limits": {
                "max_corrections": s.max_corrections,
                "max_clarify_rounds": s.max_clarify_rounds,
                "max_questions": s.max_questions,
                "max_slot_search_rounds": s.max_slot_search_rounds,
            },
        }

    @app.get("/v1/chat/{conversation_id}/state")
    async def chat_state(conversation_id: str, c: C, user: P) -> Any:
        """State hiện tại của hội thoại (dữ liệu giả lập) — để xem chi tiết cách agent ra quyết định."""
        try:
            state = await c.conversations.state(user, conversation_id)
        except NotAllowed as exc:
            raise HTTPException(404, "Không có hội thoại này") from exc
        if state is None:
            raise HTTPException(404, "Không có hội thoại này")
        return state.model_dump(mode="json", exclude={"now"})

    @app.get("/v1/chat/{conversation_id}/pending")
    async def chat_pending(conversation_id: str, c: C, user: P) -> InterruptPayload | None:
        """Câu hỏi đang chờ trả lời của hội thoại (null nếu không có) — mở lại hội thoại cũ để trả lời tiếp."""
        try:
            return await c.conversations.pending(user, conversation_id)
        except NotAllowed as exc:
            raise HTTPException(404, "Không có hội thoại này") from exc

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/stt/transcribe", response_model=TranscriptReview)
    async def transcribe(
        c: C,
        _: P,
        audio: Annotated[UploadFile, File()],
        language: Annotated[Literal["en", "vi", "auto"] | None, Form()] = None,
    ) -> TranscriptReview:
        data = await audio.read(MAX_AUDIO_BYTES + 1)
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "Audio quá lớn")
        try:
            return await c.transcribe.transcribe(data, language)
        except STTError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/v1/chat/{conversation_id}/messages", response_model=TurnResult)
    async def send_message(conversation_id: str, body: MessageIn, c: C, user: P) -> TurnResult:
        try:
            return await c.conversations.send(user, conversation_id, body.text, body.input_source, body.stt_flags)
        except NotAllowed as exc:
            raise HTTPException(404, "Không có hội thoại này") from exc

    @app.post("/v1/chat/{conversation_id}/resume", response_model=TurnResult)
    async def resume(conversation_id: str, body: ResumeIn, c: C, user: P) -> TurnResult:
        try:
            return await c.conversations.resume(user, conversation_id, UserDecision(**body.model_dump()))
        except NotAllowed as exc:
            raise HTTPException(404, "Không có hội thoại này") from exc
        except NoPendingQuestion as exc:
            raise HTTPException(409, "Không có câu hỏi nào đang chờ trả lời trong cuộc hội thoại này") from exc

    @app.get("/v1/me/appointments")
    async def my_appointments(c: C, user: P) -> Any:
        assert user.patient_id
        result = await c.toolbox.list_my_appointments(user.patient_id, c.clock.now())
        if not result.ok:
            raise HTTPException(503, "Không đọc được lịch hẹn")
        return result.data

    @app.get("/v1/me/reports")
    async def my_reports(c: C, user: P) -> Any:
        assert user.patient_id
        result = await c.toolbox.get_report_history(user.patient_id, limit=20)
        if not result.ok:
            raise HTTPException(503, "Không đọc được báo cáo")
        return result.data

    @app.get("/v1/me/conversations")
    async def my_conversations(c: C, user: P) -> list[ConversationSummary]:
        return await c.log.list_conversations(user.patient_id, 50)

    @app.get("/v1/me/conversations/{conversation_id}")
    async def my_conversation(conversation_id: str, c: C, user: P) -> list[ChatMessage]:
        if await c.log.owner(conversation_id) != user.patient_id:
            raise HTTPException(404, "Không có hội thoại này")
        messages: list[ChatMessage] = await c.log.list_messages(conversation_id)
        return messages

    # ------------------------------------------------------------------ nhân viên y tế

    @app.get("/v1/staff/conversations")
    async def staff_conversations(
        c: C, _: S, patient_id: str | None = None, limit: int = 100
    ) -> list[ConversationSummary]:
        return await c.log.list_conversations(patient_id, min(max(limit, 1), 500))

    @app.get("/v1/staff/conversations/{conversation_id}")
    async def staff_conversation(conversation_id: str, c: C, _: S) -> list[ChatMessage]:
        if await c.log.owner(conversation_id) is None:
            raise HTTPException(404, "Không có hội thoại này")
        messages: list[ChatMessage] = await c.log.list_messages(conversation_id)
        return messages

    # ------------------------------------------------------------------ quản trị tài khoản

    @app.get("/v1/admin/users")
    async def list_users(c: C, _: A) -> list[User]:
        return await c.auth.list_users()

    @app.post("/v1/admin/users", status_code=201)
    async def create_user(body: NewUser, c: C, _: A) -> User:
        try:
            return await c.auth.create_user(body)
        except UsernameTakenError as exc:
            raise HTTPException(409, "Username đã tồn tại") from exc
        except AuthError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.patch("/v1/admin/users/{user_id}")
    async def update_user(user_id: str, body: UserUpdateIn, c: C, actor: A) -> User:
        try:
            return await c.auth.update_user(actor, user_id, active=body.active, password=body.password)
        except AuthError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app


def app() -> FastAPI:
    """Entry point cho uvicorn: `uvicorn patient_assistant.interface.api.app:app --factory`."""
    return create_app()
