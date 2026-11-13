"""Chat với trợ lý (chỉ tài khoản bệnh nhân) và chuyển giọng nói thành văn bản."""

from typing import Annotated, Any, Literal

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from patient_assistant.application.agent.state import InterruptPayload, UserDecision
from patient_assistant.application.ports.stt import STTError
from patient_assistant.application.use_cases.chat import NoPendingQuestion, TurnResult
from patient_assistant.application.use_cases.conversation import ConversationService, NotAllowed
from patient_assistant.application.use_cases.transcribe import TranscribeUseCase, TranscriptReview
from patient_assistant.container import Container
from patient_assistant.interface.api.deps import P
from patient_assistant.interface.api.schemas import MessageIn, ResumeIn

MAX_AUDIO_BYTES = 10 * 1024 * 1024

router = APIRouter(prefix="/v1", tags=["chat"])

ConversationsDep = Annotated[ConversationService, Depends(Provide[Container.conversations])]
TranscribeDep = Annotated[TranscribeUseCase, Depends(Provide[Container.transcribe])]


@router.get("/chat/{conversation_id}/state")
@inject
async def chat_state(conversation_id: str, conversations: ConversationsDep, user: P) -> Any:
    """State hiện tại của hội thoại (dữ liệu giả lập) — để xem chi tiết cách agent ra quyết định."""
    try:
        state = await conversations.state(user, conversation_id)
    except NotAllowed as exc:
        raise HTTPException(404, "Không có hội thoại này") from exc
    if state is None:
        raise HTTPException(404, "Không có hội thoại này")
    return state.model_dump(mode="json", exclude={"now"})


@router.get("/chat/{conversation_id}/pending")
@inject
async def chat_pending(conversation_id: str, conversations: ConversationsDep, user: P) -> InterruptPayload | None:
    """Câu hỏi đang chờ trả lời của hội thoại (null nếu không có) — mở lại hội thoại cũ để trả lời tiếp."""
    try:
        return await conversations.pending(user, conversation_id)
    except NotAllowed as exc:
        raise HTTPException(404, "Không có hội thoại này") from exc


@router.post("/stt/transcribe", response_model=TranscriptReview)
@inject
async def transcribe(
    transcriber: TranscribeDep,
    _: P,
    audio: Annotated[UploadFile, File()],
    language: Annotated[Literal["en", "vi", "auto"] | None, Form()] = None,
) -> TranscriptReview:
    data = await audio.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio quá lớn")
    try:
        return await transcriber.transcribe(data, language)
    except STTError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/chat/{conversation_id}/messages", response_model=TurnResult)
@inject
async def send_message(conversation_id: str, body: MessageIn, conversations: ConversationsDep, user: P) -> TurnResult:
    try:
        return await conversations.send(user, conversation_id, body.text, body.input_source, body.stt_flags)
    except NotAllowed as exc:
        raise HTTPException(404, "Không có hội thoại này") from exc


@router.post("/chat/{conversation_id}/resume", response_model=TurnResult)
@inject
async def resume(conversation_id: str, body: ResumeIn, conversations: ConversationsDep, user: P) -> TurnResult:
    try:
        return await conversations.resume(user, conversation_id, UserDecision(**body.model_dump()))
    except NotAllowed as exc:
        raise HTTPException(404, "Không có hội thoại này") from exc
    except NoPendingQuestion as exc:
        raise HTTPException(409, "Không có câu hỏi nào đang chờ trả lời trong cuộc hội thoại này") from exc
