# AI Patient Assistant — Thiết kế kiến trúc (LangGraph)

**Dựa trên:** `AI_Patient_Assistant_Functional_Requirements_v2.md` (v0.3)
**Ngày:** 2026-10-28
**Trạng thái:** Draft v2 — bổ sung **đặt lịch khám bằng giọng nói** (dữ liệu lịch giả lập) vào phạm vi MVP, xem §7.

> **Thay đổi phạm vi so với FR v0.2 (đã cập nhật vào FR v0.3):** §2 của FR để "đặt/đổi/hủy lịch khám **thật**, tích hợp HIS/EMR" ngoài phạm vi. Bản kiến trúc này đưa vào **đặt lịch trên dữ liệu giả lập** (P0) và xem lịch đã đặt (P0); đổi/hủy lịch là P1. Tích hợp hệ thống bệnh viện thật vẫn nằm ngoài phạm vi. Tài liệu FR đã được cập nhật lên v0.3 (§2, FR-02.1, FR-03, FR-09) cho khớp.

---

## 0. Nguyên tắc thiết kế (rút ra từ FR)

| # | Nguyên tắc | Hệ quả trong kiến trúc | FR liên quan |
|---|---|---|---|
| P1 | **Code quyết định, LLM đề xuất** | Routing, validation, retry, quyền, ID, timestamp đều là code. LLM chỉ làm 3 việc: trích xuất có cấu trúc, sửa theo lỗi, soạn câu trả lời. | §5, FR-04.7 |
| P2 | **LLM không bao giờ cầm tool ghi** | `save_patient_report` không được expose cho LLM. Chỉ node `save` (code) gọi sau khi gate xác nhận đạt. | FR-03.5, FR-07.3 |
| P3 | **Định danh lấy từ context, không từ LLM** | `patient_id` được inject từ session/auth vào tool; schema tool mà LLM nhìn thấy không có trường này. | FR-03.2 |
| P4 | **Mọi vòng lặp đều có trần** | `max_correction_attempts`, `max_tool_calls`, `recursion_limit` của LangGraph. | FR-02.5, FR-06.3 |
| P5 | **Lỗi là dữ liệu, không phải exception lọt ra ngoài** | Mọi tool/LLM/STT trả `Result` chuẩn hóa; node đọc `status` rồi route. | FR-02.6, FR-03.7 |
| P6 | **Human-in-the-loop là trạng thái của graph** | Dùng `interrupt()` + checkpointer Postgres: graph dừng thật, resume bằng `Command(resume=...)`. | FR-01.3, FR-02.4, FR-07 |
| P7 | **Mọi thứ đều đo được** | Mỗi attempt, validation issue, tool call được ghi vào bảng riêng để chạy metric. | FR-06.6, FR-08 |
| P8 | **Ngày giờ và slot do code giải quyết** | LLM chỉ trích nguyên văn cụm thời gian ("thứ Ba tuần sau", "sáng mai"); code quy đổi ra ngày theo múi giờ `Asia/Ho_Chi_Minh`. `slot_id` chỉ được lấy từ danh sách backend vừa trả về, LLM không tự tạo. | FR-09.3, FR-09.4 |
| P9 | **Mỗi luồng là một subgraph** | `report_flow` và `booking_flow` là subgraph riêng, dùng chung validator pipeline, cơ chế interrupt và tool contract. Thêm luồng mới không phải sửa luồng cũ. | FR-02.3 |

---

## 1. Tech stack và thư viện

### 1.1 Lõi

| Thành phần | Thư viện | Lý do chọn |
|---|---|---|
| Ngôn ngữ / quản lý dự án | Python 3.12, **uv** | Lock file nhanh, tái lập môi trường. |
| API | **FastAPI** + **uvicorn** | Async, tự sinh OpenAPI từ Pydantic. |
| Agent orchestration | **langgraph** (StateGraph, `interrupt`, `Command`) | Graph có state, routing có điều kiện, HITL native. |
| Checkpointer | **langgraph-checkpoint-postgres** (`AsyncPostgresSaver`) | Lưu state theo `thread_id`, resume sau interrupt, xem lại lịch sử từng bước (time travel) để debug. |
| Schema / validation | **pydantic v2**, **pydantic-settings** | Lớp 1 validation (FR-04.1) + config qua env. |
| LLM | **anthropic** (`anthropic[bedrock]`, client `AnthropicBedrockMantle`) | SDK chính thức, xác thực bằng IAM của AWS, có `messages.parse(output_format=PydanticModel)` cho structured output và `strict: true` cho tool. |
| Database | **PostgreSQL 16**, **SQLAlchemy 2.0 (async)** + **asyncpg**, **Alembic** | Một Postgres dùng cho cả dữ liệu nghiệp vụ, checkpoint LangGraph và (sau này) pgvector. |
| STT | **faster-whisper** (CTranslate2, `int8` trên CPU) | Chạy local trên CPU (FR-01.6), trả xác suất theo từng từ để phát hiện đoạn nhận dạng kém (FR-01.4). |
| Xử lý audio | **ffmpeg** (hệ thống) / **soundfile** | Chuẩn hóa về 16kHz mono trước khi đưa vào STT. |
| So khớp chuỗi | **difflib** (stdlib) | Source-grounding: kiểm tra `evidence` có thật trong transcript (gần đúng, chịu được dấu câu/chính tả, nhưng con số phải giữ nguyên). |
| Logging | **structlog** (JSON) | Log có cấu trúc, gắn `thread_id`/`run_id`; có processor che PII (FR-08.8). |
| Ngày giờ | **zoneinfo** (stdlib) + **holidays** (lịch nghỉ lễ Việt Nam) | Quy đổi ngày tương đối theo múi giờ cố định; chặn đặt lịch vào ngày nghỉ lễ. Bộ phân tích ngày tiếng Việt (`vi_dates.py`) tự viết theo luật và có test riêng; không dựa vào thư viện parse ngày tổng quát vì độ phủ tiếng Việt cho các cụm tương đối ("thứ Ba tuần sau") không đáng tin. |

### 1.2 Test và đánh giá

| Mục đích | Thư viện |
|---|---|
| Unit/integration test | **pytest**, **pytest-asyncio**, **httpx** (`AsyncClient` test FastAPI) |
| DB thật trong test | **testcontainers[postgres]** |
| Property-based test cho validator | **hypothesis** (sinh `pain_level`, `exercise_days` biên/ngoài biên) |
| LLM giả lập trong test workflow | `FakeLLMProvider` tự viết (cài cùng interface với provider thật) |
| Evaluation runner | Tự viết trong `src/patient_assistant/evaluation/` (dataset JSONL + metric bằng code, xuất Markdown/JSON); không cần pandas |
| WER cho STT | **jiwer** |
| Chất lượng code | **ruff**, **mypy** (strict cho `schemas/`, `validators/`, `agent/`) |

### 1.3 Tùy chọn (P1 — chưa cần cho MVP)

| Mục đích | Lựa chọn | Ghi chú |
|---|---|---|
| Tracing LLM/graph | **Langfuse self-host** hoặc **OpenTelemetry** | Self-host để dữ liệu không ra dịch vụ ngoài (FR-08.8). LangSmith chỉ dùng khi chắc chắn toàn dữ liệu giả lập. |
| RAG vector | **pgvector** + embedding qua Bedrock | MVP bắt đầu bằng Postgres full-text search là đủ cho tập tài liệu nhỏ; chuyển sang vector khi cần. |
| UI demo | **Streamlit** hoặc trang HTML tĩnh gọi API | Cần 3 màn hình: chat, xem/sửa transcript, xem/sửa/xác nhận bản nháp. |

### 1.4 Vì sao dùng Anthropic SDK thay vì `langchain-aws`?

LangGraph không bắt buộc dùng model của LangChain — một node chỉ là một hàm Python. Đề xuất gọi Claude qua SDK `anthropic` (client Bedrock Mantle) phía sau một `LLMProvider` interface vì:

- Có `messages.parse(output_format=Model)` trả thẳng object Pydantic đã validate.
- Kiểm soát rõ `stop_reason` (`refusal`, `max_tokens`) — cần thiết cho FR-02.6 (không biến lỗi thành thành công giả).
- Vòng gọi tool đọc dữ liệu rất ngắn (tối đa vài lần), tự viết vòng lặp ~40 dòng dễ test và dễ giới hạn hơn `ToolNode`.

Nếu sau này muốn dùng hệ sinh thái LangChain (`ToolNode`, `with_structured_output`), chỉ cần thêm một implementation `LangChainBedrockProvider` dùng `ChatBedrockConverse` — workflow không đổi.

### 1.5 Ghi chú về model Claude trên Bedrock

- Model mặc định: **Claude Opus 5.5**, model ID trên Bedrock có tiền tố `anthropic.` → `anthropic.claude-opus-5-5`. Đặt qua env (`LLM_MODEL_ID`) và kiểm tra lại ID/region khả dụng trong Bedrock console của tài khoản.
- Có thể cấu hình model riêng theo từng node (`LLM_MODEL_EXTRACT`, `LLM_MODEL_RESPOND`…) để sau này thử model rẻ hơn cho bước đơn giản — chỉ đổi khi bộ eval chứng minh chất lượng giữ nguyên.
- Những điểm khác API cần biết khi viết code cho Opus 5.5:
  - Không gửi `temperature`/`top_p`/`top_k` (bị từ chối). Tính ổn định đến từ schema + validator, không đến từ temperature.
  - Không tắt được thinking; điều khiển bằng `output_config.effort` (mặc định `medium`). Gợi ý: `low` cho phân loại/soạn câu trả lời, `medium` cho trích xuất và sửa lỗi.
  - Không dùng `tool_choice` dạng `any`/`tool` (bị lỗi 400). Dùng `auto` + `strict: true`; khi chỉ cần JSON thì dùng structured output.
  - Không dùng assistant prefill.
  - Luôn kiểm tra `stop_reason` trước khi đọc nội dung. `refusal` → map thành `LLMError(code="refusal")` → route sang `handoff`. Trên Bedrock không có tham số `fallbacks` phía server; nếu muốn tự chuyển sang model dự phòng khi bị từ chối, dùng `BetaRefusalFallbackMiddleware` phía client của SDK.
  - System prompt + định nghĩa tool cố định → bật prompt caching để giảm chi phí.

---

## 2. Kiến trúc tổng thể

```text
┌──────────────┐   text / audio    ┌───────────────────────── FastAPI ──────────────────────────┐
│  Client/UI   │ ────────────────▶ │ POST /stt/transcribe   POST /chat/{thread_id}/messages      │
│ (Streamlit)  │ ◀──────────────── │ POST /chat/{thread_id}/resume   GET /reports                │
└──────────────┘  response/interrupt└───────┬──────────────────────┬─────────────────────────────┘
                                           │                      │
                                   ┌───────▼──────┐      ┌────────▼─────────────────────────┐
                                   │  stt/        │      │  agent/  (LangGraph StateGraph)  │
                                   │ faster-whisper│     │  nodes · routing · retry policy  │
                                   └──────────────┘      └──┬───────────┬───────────┬───────┘
                                                            │           │           │
                                               ┌────────────▼──┐ ┌──────▼─────┐ ┌───▼──────────┐
                                               │ llm/          │ │ validators/│ │ tools/       │
                                               │ LLMProvider   │ │ 6 lớp,     │ │ ToolResult   │
                                               │ (Bedrock)     │ │ thuần code │ │ chuẩn hóa    │
                                               └───────────────┘ └────────────┘ └──────┬───────┘
                                                                                       │
                                                                  ┌────────────────────▼───────┐
                                                                  │ services/ → repositories/  │
                                                                  │ quyền, nghiệp vụ, idempot. │
                                                                  └────────────┬───────────────┘
                                                                               │
                                                     ┌─────────────────────────▼──────────────────┐
                                                     │ PostgreSQL: patients, plans, reports,      │
                                                     │ knowledge_docs, agent_attempts, tool_calls,│
                                                     │ langgraph checkpoints                      │
                                                     └────────────────────────────────────────────┘
```

**STT nằm ngoài graph** (gọi qua endpoint riêng) để người dùng xem/sửa transcript trước khi bắt đầu một lượt agent (FR-01.3). Transcript đã duyệt cùng metadata độ tin cậy được gửi vào graph như text bình thường → hai loại đầu vào dùng chung một workflow (FR-01.5).

---

## 3. LangGraph workflow

### 3.1 Sơ đồ node

**Graph chính** chỉ làm việc chung: nhận đầu vào, chặn an toàn, phân loại, rồi chuyển sang subgraph phù hợp.

```text
                 START
                   │
             ┌─────▼──────┐
             │ ingest     │  chuẩn hóa input, gắn input_source, stt_flags, now (đồng hồ inject)
             └─────┬──────┘
             ┌─────▼──────┐
             │ guard_input│  safety: chẩn đoán/thuốc/cấp cứu → refuse/handoff (code + từ khóa)
             └─────┬──────┘   (cấp cứu luôn được ưu tiên hơn mọi intent, kể cả đặt lịch)
             ┌─────▼──────┐
             │ classify   │  rule trước, LLM (structured) khi rule không chắc
             └─────┬──────┘
   ┌───────────┬───┴─────────┬──────────────┬──────────────┬───────────┐
   ▼           ▼             ▼              ▼              ▼           ▼
create_report book_appointment list_appointments get_report_history answer_faq unknown
   │           │             │              │              │           │
┌──▼────────┐┌─▼──────────┐┌─▼────────────┐┌▼────────────┐┌▼─────────┐┌▼────────┐
│report_flow││booking_flow││fetch_appts   ││fetch_history││rag_answer││clarify  │
│(subgraph) ││(subgraph,§7)│└─┬────────────┘└┬────────────┘└┬─────────┘└─────────┘
└──┬────────┘└─┬──────────┘  │              │              │
   └───────────┴─────────────┴──────┬───────┴──────────────┘
                                    ▼
                                 respond ──▶ END
```

Mỗi lượt xử lý **một intent chính**. Nếu người dùng nói hai việc ("báo cáo đau mức 5 và đặt lịch tái khám"), `classify` trả `intent` + `pending_intents`; sau khi xong luồng thứ nhất, `respond` hỏi có muốn làm tiếp việc còn lại không. Không chạy song song hai luồng ghi dữ liệu trong cùng một lượt.

**Subgraph `report_flow`:**

```text
 extract ──▶ validate ──route_after_validate (code)──┐
                ▲  ├── pass ───────────────────────▶ present_draft ──interrupt (confirm/edit)
                │  ├── fixable & attempts < max ──▶ correct ──┘(quay lại validate)     │
                │  ├── cần thông tin từ user ─────▶ clarify ──interrupt──▶ extract    │
                │  └── hết lượt sửa ──────────────▶ handoff                            │
                │                                                                      ▼
                └──────────── apply_user_edits ◀── edit        confirm ──▶ save ──▶ verify_save
```

`booking_flow` có cấu trúc tương tự nhưng thêm bước tìm slot và chọn slot — xem §7.

### 3.2 Bảng node

| Node | Loại | Việc làm | Đầu ra vào state |
|---|---|---|---|
| `ingest` | code | Ghép `raw_input`, `transcript`, `stt_segments` (độ tin cậy từng từ). | `input`, `stt_flags` |
| `guard_input` | code | Phát hiện yêu cầu chẩn đoán/kê thuốc/cấp cứu bằng luật (từ khóa + pattern). Cấp cứu → trả hướng dẫn liên hệ cơ sở y tế cố định, không qua LLM. | `safety_flags` |
| `classify` | code → LLM | Rule-based trước (ví dụ có "đau", "mức", "tập … ngày" → `create_report`; "đặt lịch", "hẹn khám", "tái khám" → `book_appointment`; "lịch hẹn của tôi" → `list_appointments`). Chỉ gọi LLM khi rule không ra kết quả hoặc ra nhiều intent. | `intent`, `pending_intents`, `intent_source` |
| `extract` | LLM | `messages.parse(output_format=ReportExtraction)`. Prompt yêu cầu: mỗi trường kèm `evidence` là trích dẫn nguyên văn; không có bằng chứng → `null` + `missing_fields`. | `draft`, `attempts[]` |
| `validate` | code | Chạy 6 lớp validator (mục 5). Không gọi LLM. | `issues[]`, `validation_status` |
| `correct` | LLM | Gửi lại transcript + draft cũ + danh sách issue *chỉ loại LLM sửa được*. Tăng `correction_count`. | `draft` (phiên bản mới), `attempts[]` |
| `clarify` | code + interrupt | Sinh câu hỏi từ template theo `missing_fields`/issue (code, không phải LLM). `interrupt({"type":"clarify", ...})`. | `user_replies[]` |
| `present_draft` | code + interrupt | Dựng bản nháp + bảng giải thích từng trường (mục 6). `interrupt({"type":"confirm_draft", ...})`. | `confirmation` |
| `apply_user_edits` | code | Áp chỉnh sửa của người dùng; trường do người dùng sửa gắn `source="user_edit"` và vẫn validate lại. | `draft` |
| `save` | code → tool | Gọi `save_patient_report` với `idempotency_key`. Không chạy nếu `validation_status != passed` hoặc `confirmation != confirmed` (assert trong code). | `tool_results[]` |
| `verify_save` | code | Lớp 5: chỉ coi là thành công khi `status=ok` **và** `report_id` tồn tại khi đọc lại từ DB. | `save_outcome` |
| `fetch_history` | code → tool | Gọi `get_report_history` (patient_id từ context). | `tool_results[]` |
| `fetch_appts` | code → tool | Gọi `list_my_appointments` (patient_id từ context). | `tool_results[]` |
| `rag_answer` | LLM + tool đọc | Vòng tool-use giới hạn: LLM chỉ thấy `search_approved_knowledge`, `get_patient_plan`. Tối đa `max_tool_calls` lần. Câu trả lời bắt buộc dẫn `doc_id`. | `answer`, `citations[]` |
| `respond` | LLM (hoặc template) | Soạn câu trả lời từ dữ liệu *đã xác minh* trong state. Kết quả lưu, lỗi, từ chối → dùng template cố định; LLM chỉ diễn đạt phần tóm tắt. | `final_response` |
| `handoff` | code | Đánh dấu `needs_human_review`, trả thông báo an toàn. | `status` |

Các node `extract` → `verify_save` ở trên thuộc subgraph `report_flow`. Node của `booking_flow` liệt kê ở §7.3; các node `clarify`, `handoff`, `validate` dùng chung cơ chế (validator pipeline nhận loại draft khác nhau).

### 3.3 Routing là hàm thuần (dễ unit test)

```python
def route_after_validate(state: AgentState) -> Literal["present_draft", "correct", "clarify", "handoff"]:
    if not state.issues:
        return "present_draft"
    if any(i.resolution == "ask_user" for i in state.issues):
        return "clarify"
    if state.correction_count < settings.max_correction_attempts and all(
        i.resolution == "llm_fixable" for i in state.issues
    ):
        return "correct"
    return "handoff"
```

Mỗi `ValidationIssue` mang `resolution` do validator (code) quyết định:

| `resolution` | Ví dụ | Đi đâu |
|---|---|---|
| `llm_fixable` | Sai kiểu dữ liệu; `evidence` không có trong transcript; giá trị lệch với evidence ("bốn ngày" nhưng `exercise_days=5`) | `correct` |
| `ask_user` | Thiếu trường bắt buộc; người dùng nói "đau 12/10" (ngoài phạm vi nhưng đúng lời người dùng); STT nghi nhận sai số; mâu thuẫn trong chính lời người dùng | `clarify` |
| `block` | Nội dung y tế ngoài phạm vi | `handoff` / từ chối |

> Đây là điểm mấu chốt cho FR-06.5: nếu người dùng thật sự nói "12/10", LLM **không được** "sửa" thành 10 cho qua validator — đó là bịa dữ liệu. Validator phân biệt "LLM trích xuất sai" với "dữ liệu nguồn có vấn đề".

### 3.4 Giới hạn và an toàn vòng lặp

- `max_correction_attempts = 2` (sau lần tạo đầu tiên — FR-06.3).
- `max_tool_calls_per_turn = 4` cho `rag_answer`.
- `max_clarify_rounds = 3` → sau đó `handoff`.
- `max_slot_search_rounds = 3` (booking: tìm lại slot khi không còn trống hoặc người dùng muốn giờ khác) → sau đó `handoff`.
- `max_booking_conflict_retries = 1` (slot bị người khác đặt mất ngay lúc xác nhận).
- `graph.invoke(..., config={"recursion_limit": 25})` làm lưới an toàn cuối.

### 3.5 Human-in-the-loop với `interrupt`

```python
def present_draft(state: AgentState) -> Command:
    payload = build_draft_view(state)          # code, gồm explanation từng trường
    decision = interrupt({"type": "confirm_draft", "draft": payload})
    # decision = {"action": "confirm"} | {"action": "edit", "fields": {...}} | {"action": "cancel"}
    ...
```

API resume: `POST /chat/{thread_id}/resume` → `graph.ainvoke(Command(resume=decision), config={"configurable": {"thread_id": ...}})`. Checkpoint nằm trong Postgres nên server có restart giữa chừng vẫn resume được.

---

## 4. Agent State

```python
class AgentState(BaseModel):
    # định danh lượt (từ API/auth, không bao giờ từ LLM)
    thread_id: str
    patient_id: str
    turn_id: str

    # đầu vào
    input_source: Literal["text", "voice"]
    raw_input: str
    transcript: str | None = None
    stt_segments: list[SttWord] = []        # word, start, end, probability
    stt_flags: list[SttFlag] = []           # các đoạn xác suất thấp chứa số/ngày/mức đau

    # phân loại
    intent: Intent | None = None            # create_report | book_appointment | list_appointments
                                            # | get_report_history | answer_faq | unknown
    pending_intents: list[Intent] = []
    intent_source: Literal["rule", "llm"] | None = None
    safety_flags: list[str] = []

    now: datetime                           # đồng hồ inject (tz Asia/Ho_Chi_Minh) để test/eval tái lập được

    # trích xuất + validation (dùng chung cho mọi luồng)
    draft: ReportExtraction | BookingRequest | None = None
    draft_version: int = 0
    issues: list[ValidationIssue] = []
    validation_status: Literal["pending", "passed", "failed"] = "pending"
    correction_count: int = 0
    attempts: list[AttemptRecord] = []      # FR-06.6

    # HITL
    clarify_rounds: int = 0
    user_replies: list[str] = []
    confirmation: Literal["pending", "confirmed", "edited", "cancelled"] = "pending"

    # booking (§7)
    booking: BookingContext | None = None   # resolved_window, offered_slots, selected_slot_id, search_rounds

    # tool
    tool_calls_count: int = 0
    tool_results: list[ToolCallRecord] = []

    # đầu ra
    citations: list[Citation] = []
    final_response: AgentResponse | None = None
    status: Literal["running", "waiting_user", "done", "needs_human_review", "error"] = "running"
```

Dùng Pydantic cho state (LangGraph hỗ trợ) để validate cả state giữa các node. Các list dạng log (`attempts`, `tool_results`) dùng reducer `operator.add` nếu dùng `Annotated`.

---

## 5. Validation 6 lớp

Tất cả là code thuần trong `app/validators/`, mỗi lớp là một hàm `(draft, context) -> list[ValidationIssue]`, chạy theo thứ tự, lớp 1 fail thì dừng các lớp sau.

```python
class ValidationIssue(BaseModel):
    field: str | None
    layer: Literal["schema", "business", "grounding", "consistency", "tool_result", "safety"]
    code: str                    # "out_of_range", "evidence_not_found", "value_evidence_mismatch"...
    message: str                 # cho người đọc
    suggestion: str | None       # gợi ý sửa — đưa vào prompt correct
    resolution: Literal["llm_fixable", "ask_user", "block"]
```

| Lớp | Cách làm |
|---|---|
| 1. Schema | `ReportExtraction.model_validate(...)`; map `pydantic.ValidationError` → `ValidationIssue` (`llm_fixable`). |
| 2. Business | Luật trong `services/report_rules.py` (dùng chung với backend): `pain_level ∈ [0,10]`, `exercise_days ∈ [0,7]`, `reporting_period` đúng định dạng ISO week và không ở tương lai, enum `symptom_change`. Backend gọi lại chính hàm này khi lưu (FR-04.7). |
| 3. Source-grounding | (a) `evidence` phải xuất hiện trong transcript/user_replies — so khớp gần đúng (`difflib`, ngưỡng 0.85, con số phải giữ nguyên). (b) **Parse lại giá trị từ evidence bằng code**: bộ chuyển số tiếng Việt (`"bốn" → 4`, `"năm trên mười" → 5`) rồi so với giá trị LLM trả. Lệch → `value_evidence_mismatch`. (c) Evidence trùng đoạn bị STT gắn cờ → `ask_user`. |
| 4. Consistency | Luật chéo trường: "chưa tập lần nào" + `exercise_days > 0`; `triggering_activity` có nhưng `symptom_change = no_change`; so với kế hoạch (`get_patient_plan`) và báo cáo đã có cùng kỳ (trùng kỳ → hỏi ghi đè hay không). |
| 5. Tool/result | Áp cho mọi `ToolResult`: `status == ok`, có `report_id`, đọc lại được từ DB. Dùng trong `verify_save`. |
| 6. Safety | Chạy trên input (`guard_input`) và trên câu trả lời cuối (`respond`): chặn chẩn đoán, liều thuốc, đề xuất điều trị. Luật từ khóa/regex trước; có thể thêm LLM classifier ở P1 nhưng luật code vẫn là chốt chặn. |

> Không tin bất kỳ trường `validation_status` nào do LLM trả về — trường đó bị loại khỏi schema LLM sinh ra; `validation_status` chỉ được gán bởi node `validate`.

---

## 6. Schemas chính

```python
class FieldEvidence(BaseModel):
    quote: str | None            # trích nguyên văn; None = không có bằng chứng
    source: Literal["transcript", "user_reply", "user_edit"]

class ReportExtraction(BaseModel):        # thứ LLM được phép sinh ra
    reporting_period: str | None
    pain_level: int | None
    exercise_days: int | None
    symptom_change: SymptomChange | None
    triggering_activity: str | None
    missing_fields: list[str]
    evidence: dict[str, FieldEvidence]

class FieldExplanation(BaseModel):        # do CODE dựng, hiển thị cho user (FR-05)
    field: str
    value: Any
    evidence: str | None
    origin: Literal["user_said", "backend", "ai_summary"]   # FR-05.4
    validation: Literal["passed", "failed", "missing", "uncertain"]
    explanation: str             # từ template theo validator, vd "Nêu trực tiếp trong transcript, nằm trong 0–10."
```

`FieldExplanation.explanation` được sinh từ kết quả validator bằng template — không để LLM tự viết lý do "pass" (FR-05.6). LLM chỉ có thể thêm một câu tóm tắt có nhãn `ai_summary`.

### Tool contract

```python
class ToolError(BaseModel):
    code: Literal["not_found", "forbidden", "invalid_args", "conflict", "unavailable", "internal"]
    message: str
    retryable: bool

class ToolResult(BaseModel, Generic[T]):
    status: Literal["ok", "error"]
    data: T | None = None
    error: ToolError | None = None
    source_ids: list[str] = []   # record/doc ID để dẫn chứng (FR-05.2)
```

| Tool | LLM thấy? | Input LLM điền | Inject bởi code |
|---|---|---|---|
| `get_patient_plan` | Có (trong `rag_answer`) | — | `patient_id` |
| `get_report_history` | Có | `limit`, `period_from` | `patient_id` |
| `search_approved_knowledge` | Có | `query` | — |
| `validate_report` | Không — chạy trong node `validate` | — | toàn bộ |
| `save_patient_report` | **Không bao giờ** | — | `patient_id`, `idempotency_key`, `draft` |
| `list_departments` | Không — code nạp danh mục vào prompt/enum | — | — |
| `get_available_slots` | Có (trong `rag_answer`, để trả lời "tuần sau còn lịch không?") | `department_code`, `date_from`, `date_to` | `patient_id`, `limit` |
| `list_my_appointments` | Có | `upcoming_only` | `patient_id` |
| `create_appointment` | **Không bao giờ** | — | `patient_id`, `slot_id` (đã được người dùng chọn và xác nhận), `idempotency_key` |
| `cancel_appointment` / `reschedule_appointment` (P1) | **Không bao giờ** | — | toàn bộ |

Khi `get_available_slots` được LLM gọi trong `rag_answer`, `date_from`/`date_to` là ngày cụ thể. Nếu câu hỏi có cụm tương đối, code quy đổi trước bằng `vi_dates` rồi mới đưa ngày vào prompt — LLM không tự tính ngày. Trả lời "còn lịch" ở đây chỉ mang tính thông tin; muốn đặt thì phải chuyển sang `booking_flow`.

Tool cho LLM khai báo với `strict: true` và `additionalProperties: false`; tham số vẫn được validate lại bằng Pydantic trước khi chạy.

**Chống ghi trùng (FR-07.7):** `idempotency_key = sha256(thread_id + draft_version + canonical_json(draft))`, cột `UNIQUE` trong bảng `reports`. Retry trùng key → trả lại report đã có với `status=ok`, không tạo mới.

---

## 7. Đặt lịch khám bằng giọng nói (booking)

Booking dùng lại toàn bộ hạ tầng của luồng báo cáo: STT → transcript review → cùng graph chính → subgraph `booking_flow` → validator pipeline → `interrupt` xác nhận → tool ghi do code gọi → kiểm tra kết quả. Phần mới chủ yếu là **quy đổi ngày giờ**, **chọn slot từ danh sách backend** và **chống đặt trùng khi nhiều người tranh cùng slot**.

Toàn bộ khoa, bác sĩ, slot và lịch hẹn đều là **dữ liệu giả lập** trong Postgres; không tích hợp HIS.

### 7.1 Yêu cầu

Yêu cầu booking nằm ở **FR-09** trong `AI_Patient_Assistant_Functional_Requirements_v2.md` (v0.3); tài liệu FR là nguồn chuẩn, phần này chỉ mô tả cách đáp ứng. Bảng dưới ánh xạ yêu cầu sang thành phần kiến trúc:

| FR | Thành phần |
|---|---|
| FR-09.1 | `classify` → subgraph `booking_flow` (§3.1) |
| FR-09.2 | `extract_booking`, `BookingRequest` (§7.2) |
| FR-09.3 | `resolve_datetime`, `vi_dates.py` (§7.4) |
| FR-09.4, FR-09.5 | `search_slots`, `offer_alternatives`, validator lớp 5 (§7.3, §7.5) |
| FR-09.6 | `match_selection`, `selection.py` (§7.6) |
| FR-09.7 | `present_booking_draft` + interrupt (§7.3, §7.8) |
| FR-09.8 | `booking_rules.py`, dùng chung giữa validator và `booking_service` (§7.5, §7.7) |
| FR-09.9 | partial unique index + idempotency key (§7.7) |
| FR-09.10 | `verify_booking` (§7.3) |
| FR-09.11 | `guard_input`, validator lớp 6 (§7.5) |
| FR-09.12 | `fetch_appts`, `list_my_appointments` |
| FR-09.13 (P1) | §7.9 |
| FR-09.14 | Metric booking (§11) |

### 7.2 Schemas

```python
class BookingRequest(BaseModel):              # LLM được phép sinh ra
    department_code: str | None               # phải thuộc danh mục khoa (enum nạp từ DB)
    doctor_name: str | None
    date_expression: str | None               # NGUYÊN VĂN: "thứ Ba tuần sau", "ngày 5 tháng 11"
    time_expression: str | None               # NGUYÊN VĂN: "buổi sáng", "khoảng 9 giờ"
    reason: str | None                        # ngắn, chỉ để ghi chú; không dùng để phân loại bệnh
    missing_fields: list[str]
    evidence: dict[str, FieldEvidence]

class ResolvedWindow(BaseModel):              # do CODE tính từ date/time_expression
    date_from: date
    date_to: date
    time_from: time | None
    time_to: time | None
    rule_applied: str                         # "next_week_weekday", "explicit_dmy"... → dùng cho giải thích

class SlotOption(BaseModel):                  # do BACKEND trả
    option_no: int                            # 1..N, để người dùng chọn bằng giọng nói
    slot_id: str
    department_code: str
    doctor_id: str
    doctor_name: str
    start_at: datetime
    end_at: datetime

class BookingContext(BaseModel):
    resolved_window: ResolvedWindow | None = None
    offered_slots: list[SlotOption] = []
    selected_slot_id: str | None = None       # phải ∈ offered_slots
    search_rounds: int = 0
    conflict_retries: int = 0
```

Không có trường ngày dạng `date` trong `BookingRequest`: LLM không được tự quy đổi ngày. Đây là cách áp nguyên tắc P8 — lỗi tính ngày là lỗi rất dễ xảy ra với LLM, nhưng lại hoàn toàn giải quyết được bằng code có test.

### 7.3 Subgraph `booking_flow`

```text
 extract_booking ──▶ resolve_datetime ──▶ validate_booking ──route──┐
        ▲                 (code)              (code)               │
        │                                                          ├─ thiếu/mơ hồ ─▶ clarify ─interrupt─┐
        │                                                          ├─ llm_fixable ─▶ correct ──▶ validate_booking
        │                                                          └─ pass ─▶ search_slots (tool, code gọi)
        │                                                                          │
        │                                          không có slot ─▶ offer_alternatives┤ (tối đa N slot gần nhất)
        │                                                                          ▼
        │                                                               present_slot_options ─interrupt
        │                                                                          │
        │            "giờ khác/ngày khác" (search_rounds < max) ◀──────────────────┤
        └──────────────────────────────────────────────────────────────────────────┤
                                                                                   ▼ chọn option
                                                                        match_selection (code)
                                                                                   ▼
                                                                    present_booking_draft ─interrupt
                                                                     confirm │       │ edit/cancel
                                                                             ▼
                                                                     create_booking (tool, code gọi)
                                                                             ▼
                                                                     verify_booking ── conflict ─▶ search_slots
                                                                             ▼ ok
                                                                          respond
```

| Node | Loại | Việc làm |
|---|---|---|
| `extract_booking` | LLM | `messages.parse(output_format=BookingRequest)`. Danh mục khoa hợp lệ được đưa vào prompt (và enum trong schema). Không tự điền khoa. |
| `resolve_datetime` | code | `vi_dates.resolve(date_expression, time_expression, now)` → `ResolvedWindow` hoặc `ValidationIssue(ask_user)`. |
| `validate_booking` | code | Validator pipeline cho booking (§7.5). |
| `search_slots` | code → tool | `get_available_slots(department, window, doctor?)`, giới hạn N kết quả, sắp theo thời gian. |
| `offer_alternatives` | code → tool | Mở rộng khoảng tìm (±7 ngày, bỏ ràng buộc bác sĩ) theo luật cố định; ghi rõ đã nới điều kiện nào. |
| `present_slot_options` | code + interrupt | Danh sách đánh số, mỗi dòng: "1. Thứ Ba 03/11/2026, 09:00 – BS. Trần A – Khoa Phục hồi chức năng". |
| `match_selection` | code (LLM dự phòng) | Ghép câu trả lời ("cái thứ hai", "chín giờ", "bác sĩ An") với `offered_slots` bằng luật (§7.6). Chỉ gọi LLM khi luật không ra kết quả, và output LLM bắt buộc là `option_no` thuộc danh sách. |
| `present_booking_draft` | code + interrupt | Đọc lại đầy đủ + giải thích nguồn từng trường (khoa từ người dùng hay từ kế hoạch backend, ngày quy đổi theo luật nào). |
| `create_booking` | code → tool | `create_appointment(slot_id, idempotency_key)`. Assert: `validation_status == passed`, `confirmation == confirmed`, `selected_slot_id ∈ offered_slots`. |
| `verify_booking` | code | `status=ok` + `appointment_id` đọc lại được + `slot_id`/`start_at` khớp với bản đã xác nhận. Lỗi `conflict` → quay lại `search_slots` (tối đa `max_booking_conflict_retries`). |

### 7.4 Quy đổi ngày giờ tiếng Việt (`validators/vi_dates.py`)

Tham chiếu `now` được inject vào state (không gọi `datetime.now()` trong node) để test và eval luôn tái lập được.

| Cụm (ví dụ) | Luật | Kết quả với `now` = Thứ Tư 28/10/2026 |
|---|---|---|
| "ngày mai", "mai" | `today + 1` | 29/10/2026 |
| "thứ Ba tuần sau" | thứ Ba của tuần ISO kế tiếp | 03/11/2026 |
| "thứ Sáu này" | thứ Sáu tuần hiện tại nếu chưa qua | 30/10/2026 |
| "ngày 5 tháng 11", "5/11" | ngày tường minh, năm hiện tại; đã qua → **hỏi lại**, không tự cộng năm | 05/11/2026 |
| "buổi sáng" / "buổi chiều" | khung giờ cấu hình (07:30–11:30 / 13:30–16:30) | — |
| "khoảng 9 giờ" | 09:00 ± 60 phút | — |
| "cuối tuần", "đầu tháng", "dạo này" | mơ hồ hoặc ngoài giờ làm việc | `ask_user` |
| "thứ Hai ngày 5/11" | **xung đột**: 05/11/2026 là thứ Năm | `ask_user` (consistency) |

Bộ luật được phủ bằng test bảng + hypothesis (sinh `now` ngẫu nhiên, kiểm tra bất biến: kết quả luôn ≥ hôm nay, đúng thứ trong tuần).

### 7.5 Validator cho booking (cùng 6 lớp)

| Lớp | Luật |
|---|---|
| 1. Schema | `BookingRequest` hợp lệ; `department_code` thuộc enum. |
| 2. Business | Ngày ≥ hôm nay + `min_lead_time` (ví dụ 2 giờ); ≤ `max_horizon_days` (ví dụ 30); không phải Chủ nhật/ngày lễ (`holidays.VN`); giờ trong giờ làm việc; bệnh nhân chưa có lịch hiệu lực trùng khung giờ; tối đa `max_active_appointments`. |
| 3. Grounding | `date_expression`, `time_expression`, `department`, `doctor_name` phải xuất hiện trong transcript (so khớp gần đúng). Khoa lấy từ kế hoạch theo dõi → `origin="backend"`, kèm `plan_id`. Evidence trùng đoạn bị STT gắn cờ → `ask_user`. |
| 4. Consistency | Thứ trong tuần khớp với ngày; bác sĩ thuộc đúng khoa; "sáng" nhưng "3 giờ" → hỏi lại. |
| 5. Tool/result | `selected_slot_id ∈ offered_slots`; kết quả `create_appointment` có `appointment_id`, đọc lại được, `start_at` khớp với bản người dùng đã xác nhận. |
| 6. Safety | Lý do khám có dấu hiệu cấp cứu → dừng booking, chuyển sang hướng dẫn cấp cứu cố định. Agent không suy ra khoa từ triệu chứng (FR-09.11). |

### 7.6 Chọn slot bằng giọng nói

Người dùng trả lời bằng giọng nói sau khi nghe/xem danh sách slot. `match_selection` xử lý bằng luật theo thứ tự:

1. Số thứ tự: "số 2", "cái thứ hai", "phương án hai" → `option_no=2`.
2. Giờ: "chín giờ", "9 rưỡi" → các option có `start_at` khớp; đúng 1 option thì chọn, nhiều hơn thì hỏi lại.
3. Tên bác sĩ/thứ trong tuần → lọc theo cách tương tự.
4. Không khớp hoặc khớp nhiều → hỏi lại (giới hạn bởi `max_clarify_rounds`).

Kết quả luôn là một `option_no` thuộc danh sách đã đưa ra — dù luật hay LLM (dự phòng) chọn, giá trị đều được kiểm tra lại bằng code.

### 7.7 Đồng thời và chống đặt trùng

- Bảng `appointments`: `UNIQUE (slot_id) WHERE status = 'booked'` (partial unique index) → hai người cùng chốt một slot thì chỉ một người thành công; người kia nhận `ToolError(code="conflict")`.
- `idempotency_key = sha256(thread_id + slot_id + draft_version)`, `UNIQUE`. Retry do lỗi mạng → trả lại lịch hẹn đã tạo, `status=ok`.
- Kiểm tra lại toàn bộ business rule trong transaction lưu (`SELECT … FOR UPDATE` trên slot) — không dựa vào kết quả `search_slots` trước đó.
- Giữ chỗ tạm thời (hold slot có TTL trong lúc người dùng xác nhận) là **P1**; MVP dùng optimistic + xử lý `conflict`.

### 7.8 Rủi ro riêng của giọng nói và cách xử lý

| Rủi ro | Ví dụ | Biện pháp |
|---|---|---|
| Nhầm số ngày/giờ | "hai" ↔ "hai mươi", "bảy giờ" ↔ "bảy" | STT flag theo xác suất từ gần token số/ngày; UI tô sáng khi duyệt transcript; đọc lại dạng đầy đủ "Thứ Ba, 03/11/2026, 09:00". |
| Nhầm tên khoa | "khoa nội" ↔ "khoa ngoại" | Khoa chỉ nhận giá trị trong danh mục; điểm fuzzy match thấp hoặc gần hai khoa → hỏi lại. |
| Xác nhận mơ hồ | "ừ", "ờ được" lẫn tiếng ồn | Xác nhận đặt lịch bằng giọng nói chỉ chấp nhận cụm rõ ("đồng ý", "xác nhận", "đặt đi") hoặc nút bấm; còn lại hỏi lại. |
| Tên bác sĩ nhận sai | "BS. An" ↔ "BS. Anh" | So khớp với danh sách bác sĩ của khoa; không khớp chắc chắn → bỏ ràng buộc bác sĩ và báo rõ điều đó. |

### 7.9 Đổi/hủy lịch (P1)

`reschedule_appointment` và `cancel_appointment` dùng đúng khung trên: xác định lịch hẹn cần đổi từ `list_my_appointments` (người dùng chọn theo số thứ tự, không theo ID LLM tạo) → `present_*_draft` + interrupt → tool ghi do code gọi, có idempotency key → verify.

---

## 8. STT

```python
class STTProvider(Protocol):
    async def transcribe(self, audio: bytes, lang: str = "vi") -> SttResult: ...

class SttResult(BaseModel):
    text: str
    words: list[SttWord]          # word, start, end, probability
    avg_logprob: float
    no_speech_prob: float
    flags: list[SttFlag]          # do code tính
```

- Implementation mặc định: `FasterWhisperProvider(model_size=settings.stt_model, device="cpu", compute_type="int8")`, `word_timestamps=True`, `vad_filter=True`.
- Model cấu hình qua env (`STT_MODEL=small|medium|large-v3|<đường dẫn model đã convert>`). Có thể thử bản Whisper fine-tune tiếng Việt (ví dụ PhoWhisper) sau khi convert sang CTranslate2 — chỉ đổi config, không đổi agent (FR-01.6).
- **Gắn cờ (FR-01.4) bằng code:** từ có `probability < ngưỡng` nằm gần token số/ngày/giờ/thứ trong tuần/mức đau/tên hoạt động/tên khoa/tên bác sĩ → `SttFlag`. UI tô sáng các đoạn này khi người dùng duyệt transcript; nếu người dùng không sửa, validator grounding chuyển trường liên quan sang `ask_user`.
- Truyền `initial_prompt`/hotwords chứa danh mục khoa và tên bác sĩ giả lập để tăng độ chính xác nhận dạng tên riêng (đọc từ DB lúc khởi động; danh mục thay đổi thì khởi động lại).
- Model load một lần lúc khởi động (FastAPI lifespan); chạy trong threadpool (`run_in_threadpool`) để không chặn event loop.

---

## 9. Cấu trúc thư mục (clean architecture — đã hiện thực trong `src/`)

Code chia 4 lớp, phụ thuộc chỉ đi từ ngoài vào trong. Luật này được kiểm tra tự động bằng `tests/unit/test_architecture.py`: import sai lớp thì test fail.

```text
interface  ──▶  application  ──▶  domain
    │               ▲
    ▼               │ (implement ports)
container ──▶ infrastructure
```

| Lớp | Được phụ thuộc | Không được phụ thuộc |
|---|---|---|
| `domain` | pydantic, thư viện chuẩn | LangGraph, FastAPI, SQLAlchemy, Anthropic SDK, mọi lớp khác |
| `application` | `domain`, LangGraph | `infrastructure`, `interface`, FastAPI, SQLAlchemy, Anthropic SDK |
| `infrastructure` | `application.ports`, `domain`, thư viện ngoài | `interface`, `container` |
| `interface` / `container` | tất cả | — |

```text
src/patient_assistant/
  domain/                      # luật nghiệp vụ thuần, không I/O, mypy strict
    common.py                  # ValidationIssue, Resolution, Evidence, Origin, SttFlag
    conversation.py            # intent theo luật, safety, hiểu câu xác nhận, cờ STT
    text/  normalize.py  vi_numbers.py  vi_dates.py  grounding.py
    report/  models.py  rules.py  validators.py
    booking/ models.py  policy.py  validators.py  selection.py
  application/
    ports/                     # interface: llm, repositories, stt, clock
    services/                  # ReportService, BookingService (backend: kiểm tra lại, idempotency)
    tools/                     # ToolResult chuẩn hóa; Toolbox (tool đọc cho LLM, tool ghi chỉ cho code)
    agent/
      state.py                 # AgentState, Outcome, UserDecision, InterruptPayload, FieldExplanation
      routing.py               # hàm routing thuần
      presenters.py            # câu chữ: câu hỏi lại, giải thích từng trường, câu trả lời cuối
      deps.py                  # AgentDeps, AgentLimits (tiêm từ container)
      nodes/  common.py  report.py  booking.py  shared.py
      flows/  report_flow.py  booking_flow.py      # subgraph
      graph.py                 # graph chính
      trace.py                 # diễn giải từng node đã chạy (từ LangGraph stream) cho UI
    use_cases/  chat.py  transcribe.py             # ẩn LangGraph (Command, interrupt) khỏi API
  infrastructure/
    llm/  bedrock.py  offline.py  schema.py  prompts/*.md
    stt/  faster_whisper.py
    persistence/  memory.py  seed.py  sql/{models,repositories,setup}.py
    calendar/vn_holidays.py  clock.py  audit.py
  interface/api/  app.py  schemas.py
  interface/web/  index.html  styles.css  app.js     # UI tĩnh: chat + panel "cách agent ra quyết định"
  evaluation/                  # lớp ngoài cùng, giống interface: cases, harness, metrics, report, runner, datasets/
  config.py                    # Settings (env PA_*)
  container.py                 # composition root: nơi duy nhất chọn implementation
tests/
  unit/                        # vi_dates, vi_numbers, validators, selection, routing, schema, kiến trúc
  integration/                 # graph thật qua interrupt; SQL (SQLite) đặt trùng đồng thời; HTTP API;
                               # chính bộ eval (metric bắt được lỗi, mutation test tắt validator)
```

**Khác với bản thiết kế ban đầu:**
- Grounding dùng `difflib` của thư viện chuẩn thay cho `rapidfuzz` để domain không phụ thuộc thư viện ngoài. Khớp gần đúng còn bắt buộc **các con số giữ nguyên**: "đã tập 5 ngày" không khớp nguồn "đã tập 4 ngày".
- Có `OfflineRuleLLM`, một adapter theo luật, để chạy demo và test không cần AWS. Đầu ra của nó vẫn đi qua cùng validator với LLM thật.
- Node `respond` dựng câu trả lời bằng template từ dữ liệu đã xác minh. LLM chỉ viết câu trả lời ở `rag_answer`, và câu đó phải có `doc_id` nguồn.
- State của subgraph chỉ được gộp vào graph cha khi subgraph kết thúc. Trong lúc đang dừng chờ người dùng, muốn xem state thì đọc bằng `aget_state(..., subgraphs=True)`.

---

## 10. Dữ liệu (PostgreSQL)

| Bảng | Mục đích |
|---|---|
| `patients`, `patient_plans` | Dữ liệu giả lập. |
| `reports` | `id` (UUID do DB sinh), `patient_id`, `reporting_period`, các trường, `evidence` (JSONB), `idempotency_key UNIQUE`, `created_at` (DB). |
| `departments`, `doctors` | Danh mục giả lập (`code`, `name`, `aliases` cho fuzzy match). |
| `slots` | `slot_id`, `doctor_id`, `department_code`, `start_at`, `end_at` (timestamptz), `capacity=1`. Sinh sẵn bằng script seed. |
| `appointments` | `id` (UUID do DB sinh), `patient_id`, `slot_id`, `status` (booked/cancelled), `reason`, `evidence` (JSONB), `idempotency_key UNIQUE`, `created_at`; partial unique index `(slot_id) WHERE status='booked'`. |
| `knowledge_docs` | `doc_id`, `title`, `content`, `tsvector` (FTS); sau thêm cột `embedding vector`. |
| `agent_attempts` | `thread_id`, `turn_id`, `attempt_no`, `kind` (extract/correct), `draft_json`, `issues_json`, `model_id`, `latency_ms`, `tokens` (FR-06.6). |
| `tool_calls` | `tool_name`, `args_json` (đã che PII), `status`, `error_code`, `latency_ms`, `called_by` (llm/code). |
| `checkpoints*` | Do `AsyncPostgresSaver.setup()` tạo. |

---

## 11. Evaluation (FR-08) — đã hiện thực

Chi tiết ở [docs/EVALUATION.md](docs/EVALUATION.md). Tóm tắt:

- **Dataset JSONL** (`src/patient_assistant/evaluation/datasets/`): 60 case trong 3 suite (report, booking, conversation). Các nhóm case:
  - đủ thông tin; thiếu trường; ngoài phạm vi; mâu thuẫn; STT nghi nhận sai;
  - ngày tương đối, ngày đã qua, Chủ nhật, mơ hồ, quá xa, buổi tối; thiếu khoa; khoa lấy từ kế hoạch;
  - chọn slot mơ hồ hoặc theo bác sĩ; đổi giờ; sửa lúc xác nhận; xác nhận không rõ; hủy;
  - slot bị chiếm, hết slot trong khoảng đã chọn, 5 phiên tranh cùng một slot;
  - cấp cứu, xin chẩn đoán hoặc chọn khoa theo triệu chứng; FAQ có nguồn và không có nguồn; nhiều intent trong một câu.
- **Lỗi tiêm có chủ đích:**
  - Đầu ra LLM sai (`llm_script`) để đo validator và self-correction một cách độc lập với LLM được dùng.
  - Lỗi hạ tầng (`faults`) để đo việc không báo thành công giả.
- **Harness** chạy agent thật qua các interrupt; **metrics** là hàm thuần. Danh sách metric:
  - field accuracy (giá trị cuối và lần trích xuất đầu); tool selection, tool recall, unnecessary call, write argument;
  - validator TP/FP/FN/TN và tỷ lệ bắt đúng mã lỗi; self-correction success/regression;
  - hỏi lại khi cần, tự đoán, hỏi thừa; latency;
  - và ba ngưỡng cứng bằng 0: **false-success, slot hallucination, double-booking**.
- **CLI:** `python -m patient_assistant.evaluation.runner [--llm bedrock] [--fail-on-hard-zero]`, xuất `report.md` và `report.json`.
- **Chưa có:** STT WER trên audio thật, latency theo từng node, chi phí token.

## 12. Kế hoạch triển khai (khớp với §9 của FR v0.3)

| Bước | Việc chính | Thư viện mới thêm |
|---|---|---|
| 1 | Schemas + `extract` bằng `messages.parse` + script CLI | pydantic, anthropic |
| 2 | 6 lớp validator + `vi_numbers` + unit test/hypothesis | hypothesis |
| 3 | Tools + `ToolResult`, Postgres, repos | sqlalchemy, asyncpg, alembic |
| 4 | Dựng LangGraph: classify → extract → validate → present_draft(interrupt) → save → verify_save; FastAPI routes | langgraph, langgraph-checkpoint-postgres, fastapi |
| 5 | Node `correct` + routing theo `resolution` + ghi `agent_attempts` | — |
| 6 | `FieldExplanation` + `respond` + `rag_answer` có citation | — |
| 7 | Booking bằng text: dữ liệu seed khoa/bác sĩ/slot, `vi_dates`, `booking_rules`, subgraph `booking_flow`, `match_selection`, test đồng thời | holidays |
| 8 | Dataset + runner + metric (báo cáo và booking) | pandas, jiwer |
| 9 | STT endpoint + cờ độ tin cậy + UI duyệt transcript; bộ audio test cho cả báo cáo và booking (số, ngày, giờ, tên khoa) | faster-whisper, streamlit |
| 10 (P1) | Đổi/hủy lịch, giữ chỗ slot có TTL, TTS đọc lại xác nhận | — |

Booking được làm bằng text trước (bước 7) rồi mới thêm giọng nói (bước 9): vì giọng nói chỉ là STT đi vào cùng workflow, mọi lỗi logic đặt lịch được bắt bằng test text — rẻ và tái lập được — trước khi thêm biến số nhận dạng giọng nói.

LangGraph được dựng từ bước 4 (FR v0.3 §9 đã cập nhật theo thứ tự này): các node bước 1–3 vốn là hàm thuần nên ghép vào graph sớm không tốn thêm công, và tránh phải viết lại một orchestrator tạm.

---

## 13. Điểm cần làm rõ

1. **Tham số nghiệp vụ booking:** giờ làm việc, `min_lead_time`, `max_horizon_days`, số lịch hiệu lực tối đa, có làm việc thứ Bảy hay không — hiện là giá trị giả định, cấu hình qua env.
2. **Auth:** MVP dùng header/token giả lập để ra `patient_id`; cần quyết định có làm đa bệnh nhân trong demo hay không.
3. **Ngưỡng:** độ tin cậy STT, ngưỡng fuzzy grounding — đặt giá trị khởi điểm rồi chỉnh theo bộ eval.
