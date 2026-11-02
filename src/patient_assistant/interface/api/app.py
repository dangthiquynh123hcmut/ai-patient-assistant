"""FastAPI: lớp mỏng chuyển HTTP <-> use case. Không chứa nghiệp vụ.

Chỉ phục vụ API (`/v1/...`, `/health`). Frontend React được build riêng (`frontend/`) và do nginx phục vụ,
nginx proxy `/v1/` về đây — cùng origin nên cookie phiên dùng được mà không cần CORS.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile

from patient_assistant.application.agent.state import UserDecision
from patient_assistant.application.ports.repositories import UsernameTakenError
from patient_assistant.application.ports.stt import STTError
from patient_assistant.application.services.auth_service import AuthError
from patient_assistant.application.use_cases.chat import NoPendingQuestion, TurnResult
from patient_assistant.application.use_cases.conversation import NotAllowed
from patient_assistant.application.use_cases.transcribe import TranscriptReview
from patient_assistant.config import Settings
from patient_assistant.container import Container, build_container
from patient_assistant.domain.account import ChatMessage, ConversationSummary, NewUser, User
from patient_assistant.interface.api.schemas import LoginIn, MessageIn, ResumeIn, UserUpdateIn

MAX_AUDIO_BYTES = 10 * 1024 * 1024
SESSION_COOKIE = "pa_session"


def _container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


C = Annotated[Container, Depends(_container)]


def _token(request: Request) -> str | None:
    """Token phiên: cookie HttpOnly (trình duyệt) hoặc `Authorization: Bearer` (client API, test)."""
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return request.cookies.get(SESSION_COOKIE)


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

    @app.post("/v1/auth/login")
    async def login(body: LoginIn, response: Response, c: C) -> User:
        try:
            token, user = await c.auth.login(body.username, body.password)
        except AuthError as exc:
            raise HTTPException(401, str(exc)) from exc
        response.set_cookie(
            SESSION_COOKIE,
            token,
            max_age=int(c.auth.session_ttl.total_seconds()),
            httponly=True,
            samesite="lax",
            secure=c.settings.cookie_secure,
            path="/",
        )
        return user

    @app.post("/v1/auth/logout", status_code=204)
    async def logout(request: Request, response: Response, c: C) -> None:
        if token := _token(request):
            await c.auth.logout(token)
        response.delete_cookie(SESSION_COOKIE, path="/")

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
            "llm_provider": s.llm_provider,
            "llm_model": s.llm_model_id if s.llm_provider == "bedrock" else "offline-rules",
            "stt_enabled": s.stt_enabled,
            "stt_model": s.stt_model if s.stt_enabled else None,
            "now": c.clock.now().isoformat(),
            "patients": c.patients if user.can_view_conversations else [],
            "departments": [d.model_dump() for d in departments],
            "doctors": [d.model_dump() for d in doctors],
            "limits": {
                "max_corrections": s.max_corrections,
                "max_clarify_rounds": s.max_clarify_rounds,
                "max_slot_search_rounds": s.max_slot_search_rounds,
            },
        }

    @app.get("/v1/chat/{thread_id}/state")
    async def chat_state(thread_id: str, c: C, user: P) -> Any:
        """State hiện tại của hội thoại (dữ liệu giả lập) — để xem chi tiết cách agent ra quyết định."""
        try:
            state = await c.conversations.state(user, thread_id)
        except NotAllowed as exc:
            raise HTTPException(404, "Không có hội thoại này") from exc
        if state is None:
            raise HTTPException(404, "Không có hội thoại này")
        return state.model_dump(mode="json", exclude={"now"})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/stt/transcribe", response_model=TranscriptReview)
    async def transcribe(c: C, _: P, audio: Annotated[UploadFile, File()]) -> TranscriptReview:
        data = await audio.read(MAX_AUDIO_BYTES + 1)
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "Audio quá lớn")
        try:
            return await c.transcribe.transcribe(data)
        except STTError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/v1/chat/{thread_id}/messages", response_model=TurnResult)
    async def send_message(thread_id: str, body: MessageIn, c: C, user: P) -> TurnResult:
        try:
            return await c.conversations.send(user, thread_id, body.text, body.input_source, body.stt_flags)
        except NotAllowed as exc:
            raise HTTPException(404, "Không có hội thoại này") from exc

    @app.post("/v1/chat/{thread_id}/resume", response_model=TurnResult)
    async def resume(thread_id: str, body: ResumeIn, c: C, user: P) -> TurnResult:
        try:
            return await c.conversations.resume(user, thread_id, UserDecision(**body.model_dump()))
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

    @app.get("/v1/me/conversations/{thread_id}")
    async def my_conversation(thread_id: str, c: C, user: P) -> list[ChatMessage]:
        if await c.log.owner(thread_id) != user.patient_id:
            raise HTTPException(404, "Không có hội thoại này")
        messages: list[ChatMessage] = await c.log.list_messages(thread_id)
        return messages

    # ------------------------------------------------------------------ nhân viên y tế

    @app.get("/v1/staff/conversations")
    async def staff_conversations(
        c: C, _: S, patient_id: str | None = None, limit: int = 100
    ) -> list[ConversationSummary]:
        return await c.log.list_conversations(patient_id, min(max(limit, 1), 500))

    @app.get("/v1/staff/conversations/{thread_id}")
    async def staff_conversation(thread_id: str, c: C, _: S) -> list[ChatMessage]:
        if await c.log.owner(thread_id) is None:
            raise HTTPException(404, "Không có hội thoại này")
        messages: list[ChatMessage] = await c.log.list_messages(thread_id)
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
