"""FastAPI: lớp mỏng chuyển HTTP <-> use case. Không chứa nghiệp vụ."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from patient_assistant.application.agent.state import UserDecision
from patient_assistant.application.ports.stt import STTError
from patient_assistant.application.use_cases.chat import NoPendingQuestion, TurnResult
from patient_assistant.application.use_cases.transcribe import TranscriptReview
from patient_assistant.config import Settings
from patient_assistant.container import Container, build_container
from patient_assistant.interface.api.schemas import MessageIn, ResumeIn

MAX_AUDIO_BYTES = 10 * 1024 * 1024
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
DEMO_PATIENTS = [
    {"id": "patient-demo-001", "label": "Bệnh nhân demo (có kế hoạch theo dõi khoa PHCN)"},
    {"id": "patient-no-plan", "label": "Bệnh nhân chưa có kế hoạch theo dõi"},
]


def _container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


def _patient_id(x_patient_id: Annotated[str, Header(min_length=1, max_length=64)]) -> str:
    """Auth giả lập: header X-Patient-Id. Thay bằng xác thực thật khi cần."""
    return x_patient_id


C = Annotated[Container, Depends(_container)]
P = Annotated[str, Depends(_patient_id)]


def create_app(container_factory: Callable[[], AbstractAsyncContextManager[Container]] | None = None) -> FastAPI:
    factory = container_factory or (lambda: build_container(Settings()))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with factory() as container:
            app.state.container = container
            yield

    app = FastAPI(title="AI Patient Assistant", version="0.1.0", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    async def index() -> RedirectResponse:
        return RedirectResponse("/ui/")

    @app.get("/v1/meta")
    async def meta(c: C) -> dict[str, Any]:
        """Cấu hình đang chạy và dữ liệu giả lập — để UI hiển thị gợi ý."""
        departments, doctors = await c.toolbox.catalog()
        s = c.settings
        return {
            "llm_provider": s.llm_provider,
            "llm_model": s.llm_model_id if s.llm_provider == "bedrock" else "offline-rules",
            "stt_enabled": s.stt_enabled,
            "stt_model": s.stt_model if s.stt_enabled else None,
            "now": c.clock.now().isoformat(),
            "patients": DEMO_PATIENTS,
            "departments": [d.model_dump() for d in departments],
            "doctors": [d.model_dump() for d in doctors],
            "limits": {
                "max_corrections": s.max_corrections,
                "max_clarify_rounds": s.max_clarify_rounds,
                "max_slot_search_rounds": s.max_slot_search_rounds,
            },
        }

    @app.get("/v1/chat/{thread_id}/state")
    async def chat_state(thread_id: str, c: C, pid: P) -> Any:
        """State hiện tại của hội thoại (dữ liệu giả lập) — để xem chi tiết cách agent ra quyết định."""
        state = await c.chat.inspect_state(thread_id)
        if state is None or state.patient_id != pid:
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
    async def send_message(thread_id: str, body: MessageIn, c: C, pid: P) -> TurnResult:
        return await c.chat.send_message(thread_id, pid, body.text, body.input_source, body.stt_flags)

    @app.post("/v1/chat/{thread_id}/resume", response_model=TurnResult)
    async def resume(thread_id: str, body: ResumeIn, c: C, pid: P) -> TurnResult:
        try:
            return await c.chat.resume(thread_id, pid, UserDecision(**body.model_dump()))
        except NoPendingQuestion as exc:
            raise HTTPException(409, "Không có câu hỏi nào đang chờ trả lời trong cuộc hội thoại này") from exc

    @app.get("/v1/me/appointments")
    async def my_appointments(c: C, pid: P) -> Any:
        result = await c.toolbox.list_my_appointments(pid, c.clock.now())
        if not result.ok:
            raise HTTPException(503, "Không đọc được lịch hẹn")
        return result.data

    @app.get("/v1/me/reports")
    async def my_reports(c: C, pid: P) -> Any:
        result = await c.toolbox.get_report_history(pid, limit=20)
        if not result.ok:
            raise HTTPException(503, "Không đọc được báo cáo")
        return result.data

    if WEB_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")
    return app


def app() -> FastAPI:
    """Entry point cho uvicorn: `uvicorn patient_assistant.interface.api.app:app --factory`."""
    return create_app()
