# AI Patient Assistant — Tóm tắt Tech Stack & Kiến trúc

> Tóm tắt từ source code hiện tại (`src/patient_assistant`, `frontend/`, `docker-compose.yml`, `ARCHITECTURE.md`).

## 1. Tổng quan tech stack

| Lớp | Công nghệ |
|---|---|
| **Frontend** | **React 19** + **TypeScript 5.8**, build bằng **Vite 6**, routing bằng **react-router-dom 7**. Không dùng UI lib / state lib ngoài (state bằng React Context + hooks). Phục vụ bằng **nginx** (static + reverse proxy `/v1/`). |
| **Backend** | **Python 3.12**, framework **FastAPI** + **uvicorn** (mặc định 2 worker). DI bằng **dependency-injector**. Validation / config bằng **Pydantic v2** + **pydantic-settings** (biến môi trường `PA_*`). Log bằng **structlog** (JSON). |
| **AI / Agent** | **LangGraph** (StateGraph, `interrupt()` cho human-in-the-loop), LLM **Claude qua Amazon Bedrock** bằng SDK `anthropic[bedrock]` (mặc định `global.anthropic.claude-opus-4-6-v1`, region `ap-southeast-1`, xác thực bằng IAM / AWS profile). |
| **STT (tùy chọn)** | **faster-whisper** chạy local trên CPU (bật bằng `PA_STT_ENABLED`). |
| **Database** | **PostgreSQL 17** (docker image `postgres:17-alpine`), ORM **SQLAlchemy 2.0 async** + driver **asyncpg**, migration bằng **Alembic**. |
| **Checkpoint agent** | **langgraph-checkpoint-postgres** (`AsyncPostgresSaver`, driver psycopg 3 + connection pool) — lưu state hội thoại trong cùng Postgres. |
| **Redis** | **Không dùng.** Session, trạng thái khóa đăng nhập, checkpoint agent đều nằm trong Postgres. |
| **Hạ tầng / triển khai** | **Docker Compose**: `db`, `migrate`, `backend`, `session-cleaner`, `frontend`, `test` (profile). |
| **Test / chất lượng** | pytest, pytest-asyncio, hypothesis, httpx; ruff, mypy (strict cho `domain`). Có bộ evaluation riêng (`evaluation/`, lệnh `pa-eval`). |

## 2. Frontend

- Thư mục: `frontend/src`
  - `api/client.ts` — HTTP client có kiểu, xử lý access token + tự refresh khi gặp 401.
  - `auth/` — `AuthProvider` (React Context), `LoginPage`, `RegisterPage`.
  - `features/chat` — màn chat của bệnh nhân (gửi tin, thẻ xác nhận interrupt, ghi âm STT, lịch sử).
  - `features/staff` — xem hội thoại (staff/admin), quản lý user (admin).
  - `features/inspector` — xem state agent để debug.
- Bảo vệ route phía client bằng `RequireRole` trong `App.tsx`:
  - `/` → chỉ `patient`; `/staff...` → `staff`, `admin`; trang quản lý user → chỉ `admin`.
  - Đây chỉ là UX; quyền thật được kiểm ở backend.
- nginx: cùng origin với API (proxy `/v1/` → `backend:8000`) nên **không cần CORS** và cookie refresh hoạt động; có header bảo mật (CSP chặt `default-src 'self'`, `X-Frame-Options: DENY`, `nosniff`, …).

## 3. Backend

- Ngôn ngữ **Python 3.12**, framework **FastAPI**; API version `/v1/...`, health check `/health`.
- Router (`interface/api/routers`):
  - `auth_router` — `/v1/auth/login|register|refresh|logout|logout-all|me|options`
  - `chat_router` — `/v1/chat/{id}/messages|resume|pending|state`, `/v1/stt/transcribe` (patient)
  - `me_router` — `/v1/meta`, `/v1/me/appointments|reports|conversations` (patient)
  - `staff_router` — `/v1/staff/conversations...` (staff/admin), `/v1/admin/users` (admin)
- CLI (`interface/cli`): `seed`, `purge_sessions`, `check_llm`, `dump`.
- Swagger/ReDoc chỉ bật khi `PA_API_DOCS=true`.

## 4. Authentication (JWT)

Code chính: `application/services/auth_service.py`, `infrastructure/security/jwt_codec.py`, `interface/api/routers/auth_router.py`, `interface/api/dependencies.py`, `frontend/src/api/client.ts`.

### 4.1 Mật khẩu & đăng nhập
- Hash mật khẩu bằng **scrypt** (stdlib `hashlib`, N=2^14, r=8, p=1), salt riêng mỗi mật khẩu, so sánh hằng thời gian (`hmac.compare_digest`).
- Username không tồn tại vẫn chạy scrypt với hash giả → không lộ username qua thời gian phản hồi.
- **Brute-force**: sai **5 lần** liên tiếp → khóa tài khoản **5 phút** (lưu ở DB: `failed_logins`, `locked_until`, đúng cả khi nhiều worker).
- Tự đăng ký (`/register`, bật/tắt bằng `PA_ALLOW_REGISTRATION`) luôn tạo role `patient` + hồ sơ bệnh nhân mới; staff/admin chỉ admin tạo được.

### 4.2 JWT
- Thư viện **PyJWT**, thuật toán **HS256**, secret `PA_JWT_SECRET` (bắt buộc ≥ 32 byte, dùng chung cho mọi worker), `iss = "patient-assistant"`.
- Decode cố định `algorithms=["HS256"]` (chặn `alg: none`/đổi thuật toán), bắt buộc các claim `iss, sub, sid, jti, typ, iat, exp`. `exp` do `AuthService` tự kiểm theo `Clock` của app (để test được).
- Claims:
  - `sub` = user id, `sid` = session id (khóa chính bảng `sessions`), `jti`, `iat`, `exp`
  - `typ` = `access` | `refresh` (không dùng lẫn được)
  - access token có thêm `role`

### 4.3 Access token
- TTL ngắn: mặc định **15 phút** (`PA_ACCESS_TOKEN_TTL_MINUTES`), không bao giờ sống lâu hơn phiên.
- Trả trong **response body**; FE giữ **trong bộ nhớ (biến JS)**, **không** lưu localStorage/sessionStorage.
- Gửi qua header `Authorization: Bearer <token>`; backend **không** đọc access token từ cookie → API không bị CSRF.
- Mỗi request, `authenticate()` kiểm: chữ ký + `typ=access` + còn hạn **và** phiên `sid` còn tồn tại trong DB, đúng user, chưa hết hạn, user còn `active`. → Access token **là stateful một phần**: thu hồi phiên thì access token mất hiệu lực **ngay**, không đợi hết 15 phút.

### 4.4 Refresh token
- Là JWT `typ=refresh`, hạn = **hạn tuyệt đối của phiên** (mặc định **12 giờ**, `PA_SESSION_TTL_HOURS`); refresh **không** kéo dài phiên → hết 12h phải đăng nhập lại.
- Lưu ở **cookie `pa_refresh`**: `HttpOnly`, `SameSite=Strict`, `Path=/v1/auth`, `Secure` theo `PA_COOKIE_SECURE` (nên bật khi chạy HTTPS). JS không đọc được.
- DB (`sessions`) **chỉ lưu SHA-256 của `jti`** refresh token hiện hành (`refresh_hash`), không lưu token rõ → lộ DB không dùng được.
- **Rotation** mỗi lần `/v1/auth/refresh`: cấp cặp access + refresh mới, `jti` cũ hết dùng (cập nhật `refresh_hash`, giữ `previous_refresh_hash`, `rotated_at`).
- **Phát hiện tái sử dụng (reuse detection)**: gửi lại refresh token đã bị xoay → coi như bị đánh cắp → **xóa cả phiên**.
  - Ngoại lệ race (2 tab cùng refresh): trong **30 giây** sau khi xoay, request thua nhận **HTTP 409** (không thu hồi phiên); FE đợi 300 ms rồi gửi lại với cookie mới.
- Luồng FE (`client.ts`):
  1. API trả 401 → gọi `/v1/auth/refresh` **một lần**, gộp mọi request 401 đồng thời vào cùng một promise.
  2. Thành công → gửi lại request gốc đúng 1 lần; thất bại → `onUnauthorized()` đưa về trang login.
  3. Tải lại trang (mất access token trong bộ nhớ) → `restoreSession()` dùng cookie refresh lấy access token mới.

### 4.5 Logout & xử lý token sau logout
- `POST /v1/auth/logout`: lấy cả refresh cookie lẫn Bearer token (nếu có), decode (chấp nhận token đã hết hạn nhưng phải đúng chữ ký) → **xóa dòng session `sid`** trong DB, rồi **xóa cookie** `pa_refresh`. Không cần access token còn hạn.
- `POST /v1/auth/logout-all`: xóa **mọi session** của user (mọi thiết bị).
- Sau logout, các token cũ **vô hiệu ngay** vì mọi lần dùng đều tra `sid` trong bảng `sessions` → **không cần blacklist / Redis**.
- Admin khóa tài khoản hoặc đổi mật khẩu → tự động xóa mọi session của user đó.
- FE chỉ coi là đã đăng xuất khi server xác nhận (hoặc trả 401); lỗi mạng thì báo lỗi vì phiên phía server vẫn còn. Sau đó xóa access token trong bộ nhớ.
- Dọn dẹp: service `session-cleaner` chạy `purge_sessions` mỗi 60 phút (`PA_SESSION_PURGE_EVERY_MINUTES`) xóa session hết hạn (chỉ là dọn rác — session hết hạn vốn đã bị từ chối).

## 5. Phân quyền (RBAC)

- 3 role (`domain/account.py`):
  | Role | Quyền |
  |---|---|
  | `patient` | Chat với trợ lý, xem lịch hẹn / báo cáo / hội thoại **của chính mình** (bắt buộc gắn `patient_id`). |
  | `staff` | Bác sĩ / nhân viên: xem hội thoại của bệnh nhân, **không** chat thay. |
  | `admin` | Như staff + quản lý tài khoản (tạo user, khóa/mở, đặt lại mật khẩu). |
- Quyền là **property trong domain**: `can_chat`, `can_view_conversations`, `can_manage_users` (đều yêu cầu `active`).
- Kiểm ở backend bằng FastAPI dependency (`dependencies.py`): `AuthenticatedUser` (401) → `PatientUser` / `StaffUser` / `AdminUser` (403).
- **Kiểm quyền sở hữu dữ liệu**: `ConversationService` kiểm `conversation_id` thuộc đúng `patient_id` của user; không phải → 404.
- Nguyên tắc thiết kế: **quyền và định danh do code quyết định, không bao giờ do LLM**. `patient_id` được inject từ phiên đăng nhập vào tool; schema tool mà LLM thấy không có trường này. Tool ghi dữ liệu không expose cho LLM (chỉ "propose" → người dùng xác nhận qua interrupt → code ghi).
- Role được lấy từ DB mỗi request (claim `role` trong access token chỉ mang tính thông tin) → đổi role/khóa có hiệu lực ngay.

## 6. Database

- **PostgreSQL 17** — một DB duy nhất cho:
  - Dữ liệu nghiệp vụ: `departments`, `doctors`, `patients`, `patient_plans`, `slots`, `appointments`, `reports`, `knowledge_docs`
  - Tài khoản & phiên: `users`, `sessions`
  - Lịch sử chat: `conversations`, `chat_messages`
  - Checkpoint của LangGraph (bảng do `AsyncPostgresSaver` quản lý) — lưu state agent theo `conversation_id`, cho phép resume sau `interrupt`.
- Truy cập: SQLAlchemy 2.0 async + asyncpg (nghiệp vụ), psycopg 3 pool (checkpointer). Schema quản lý bằng Alembic (service `migrate` chạy `alembic upgrade head` + seed dữ liệu demo).
- **Không dùng Redis** (hay cache ngoài nào khác). Dự kiến tương lai: pgvector cho RAG (hiện dùng full-text search Postgres).

## 7. Kiến trúc hệ thống

### 7.1 Kiến trúc triển khai

```
Browser (React SPA)
   │  HTTPS / HTTP :8080
   ▼
nginx (container frontend)
   ├── /           → file tĩnh React (build Vite)
   └── /v1/, /health, /docs → reverse proxy
                     ▼
            FastAPI + uvicorn (container backend, N worker, stateless)
               ├── PostgreSQL 17 (db)          — dữ liệu, session, checkpoint LangGraph
               ├── Amazon Bedrock (Claude)     — LLM qua IAM credentials (~/.aws mount read-only)
               └── faster-whisper (local CPU)  — STT tùy chọn
session-cleaner (cùng image backend) — job định kỳ xóa session hết hạn
migrate (chạy 1 lần)                 — Alembic migrate + seed
```

- Backend **stateless** (không giữ session trong RAM) → scale nhiều worker/instance được; mọi trạng thái nằm trong Postgres.
- Dạng **monolith modular** (một service backend), không phải microservices.

### 7.2 Kiến trúc code backend — Clean / Hexagonal Architecture (Ports & Adapters)

```
src/patient_assistant/
├── domain/          # Nghiệp vụ thuần, không I/O: account, patient, booking (policy, ranking, validators),
│                    # report (rules, validators), text (parse ngày/số tiếng Việt, grounding)
├── application/     # Use case & điều phối
│   ├── ports/       # Interface (Protocol): repositories, llm, stt, tokens, clock
│   ├── services/    # auth_service, booking_service, report_service
│   ├── use_cases/   # chat, conversation, transcribe
│   ├── tools/       # Toolbox cho agent (kết quả chuẩn hóa dạng Result)
│   └── agent/       # LangGraph: graph.py, flows (report/booking/agent), nodes, routing, state
├── infrastructure/  # Adapter: persistence/sql (SQLAlchemy), llm/bedrock, stt/faster_whisper,
│                    # security/jwt_codec, calendar/vn_holidays, audit, clock
├── interface/       # api (FastAPI routers, schemas, dependencies), cli
├── container.py     # DI container (dependency-injector) lắp ráp mọi thứ
├── config.py        # Settings (pydantic-settings, prefix PA_)
└── evaluation/      # Bộ đánh giá agent (dataset JSONL + metric)
```

- Phụ thuộc hướng vào trong: `interface → application → domain`; `infrastructure` hiện thực các `ports`.
- Test dùng `FakeLLMProvider`, `FixedClock`, Postgres thật với schema riêng mỗi test.

### 7.3 Kiến trúc agent (LangGraph)

```
START → ingest → guard_input ─(không an toàn)→ safety_response ──┐
                     │                                           │
                     ▼                                           │
                  classify ─┬→ report_flow   (subgraph)          │
                            ├→ booking_flow  (subgraph)          │
                            ├→ agent_flow    (subgraph, chế độ agent)
                            ├→ fetch_history / fetch_appointments / fetch_patient_info / fetch_doctors
                            ├→ rag_answer / converse / clarify_intent
                            ▼                                    │
                         respond  ←──────────────────────────────┘ → END
```

- Nguyên tắc: **"Code quyết định, LLM đề xuất"** — routing, validation, retry, quyền, ID, ngày giờ, slot đều do code; LLM chỉ trích xuất có cấu trúc, sửa lỗi theo phản hồi validator và soạn câu trả lời.
- **Human-in-the-loop**: `interrupt()` + checkpointer Postgres; FE hiện thẻ xác nhận, gửi quyết định qua `/v1/chat/{id}/resume`.
- Mọi vòng lặp có trần (`max_correction_attempts`, `max_tool_calls`, `recursion_limit`).
- Chế độ: `PA_BOOKING_MODE=workflow` (mặc định, luồng cứng) hoặc `PA_AGENT_MODE=booking|full` (LLM tự gọi tool đọc, tool ghi chỉ là "propose" cần xác nhận).

## 8. Biến cấu hình quan trọng

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `PA_DATABASE_URL` | — | `postgresql+asyncpg://...` |
| `PA_JWT_SECRET` | (bắt buộc, ≥ 32 byte) | Secret ký JWT HS256 |
| `PA_ACCESS_TOKEN_TTL_MINUTES` | 15 | Hạn access token |
| `PA_SESSION_TTL_HOURS` | 12 | Hạn tuyệt đối phiên / refresh token |
| `PA_COOKIE_SECURE` | false | Cờ `Secure` cho cookie refresh (bật khi HTTPS) |
| `PA_ALLOW_REGISTRATION` | true | Cho phép tự đăng ký |
| `PA_SESSION_PURGE_EVERY_MINUTES` | 60 | Chu kỳ dọn session hết hạn |
| `PA_LLM_MODEL_ID` / `PA_AWS_REGION` | Claude Opus trên Bedrock / `ap-southeast-1` | LLM |
| `PA_STT_ENABLED` / `PA_STT_MODEL` | false / small | STT local |
| `PA_API_DOCS` | false | Bật Swagger/ReDoc |
