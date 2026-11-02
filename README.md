# AI Patient Assistant

Agent LangGraph hỗ trợ bệnh nhân (dữ liệu giả lập, dự án học tập): gửi báo cáo theo dõi, đặt lịch khám bằng văn bản hoặc giọng nói, xem lịch hẹn và báo cáo, hỏi thông tin trong tài liệu đã duyệt.

- Yêu cầu: [AI_Patient_Assistant_Functional_Requirements_v2.md](AI_Patient_Assistant_Functional_Requirements_v2.md)
- Kiến trúc: [ARCHITECTURE.md](ARCHITECTURE.md), sơ đồ: [docs/diagrams.html](docs/diagrams.html)

## Kiến trúc triển khai

```
              ┌──────────────────────── docker compose ────────────────────────┐
 trình duyệt ─┼─► frontend :8080  nginx: React (TypeScript, Vite build)          │
              │      │  /v1/*  ──proxy──►  backend :8000  FastAPI + LangGraph   │
              │      │                       (uvicorn, UVICORN_WORKERS worker)  │
              │      │                          │            │                   │
              │      │                          ▼            ▼                   │
              │      │                    db: PostgreSQL 17 (volume pgdata)      │
              │      │                    · dữ liệu nghiệp vụ, tài khoản, phiên  │
              │      │                    · lịch sử hội thoại (chat_messages)    │
              │      │                    · checkpoint LangGraph (câu hỏi đang chờ)
              │   migrate (chạy 1 lần mỗi lần up): alembic upgrade head + seed   │
              └──────────────────────────────────────────────────────────────────┘
                                                 │ IAM (~/.aws mount chỉ đọc)
                                                 ▼
                                       Amazon Bedrock (Claude)
```

- **frontend/** — React 19 + TypeScript, build bằng Vite thành file tĩnh; nginx phục vụ và proxy `/v1/` về backend nên trình duyệt chỉ thấy một origin (cookie phiên HttpOnly, không cần CORS). nginx đặt CSP và các security header.
- **backend** — chỉ là API. Không giữ state trong RAM: hội thoại đang chờ người dùng trả lời nằm trong checkpoint Postgres, khóa đăng nhập nằm trong bảng `users`, nên chạy nhiều worker/instance được.
- **migrate** — schema do Alembic quản lý ([migrations/](migrations/)); seed idempotent (danh mục, tài khoản demo, bổ sung slot khám cho đủ số ngày tới, bảng checkpoint).
- Container backend chạy bằng user thường (uid 1000), Postgres không mở cổng ra ngoài.

## Chạy bằng Docker

```bash
cp .env.example .env
```

Sửa `.env`: đặt `POSTGRES_PASSWORD`; muốn dùng Claude thật thì `PA_LLM_PROVIDER=bedrock` (credential AWS lấy từ `~/.aws`, xem mục Bedrock bên dưới). Rồi:

```bash
docker compose up -d --build
```

Mở **http://localhost:8080** và đăng nhập (tài khoản ở mục **Đăng nhập và phân quyền**). Lệnh hữu ích:

```bash
docker compose logs -f backend
```

```bash
docker compose run --rm migrate
```

(chạy lại migration + seed, ví dụ sau khi thêm migration mới, hoặc hằng ngày để bổ sung slot khám)

```bash
docker compose exec db psql -U patient patient_assistant
```

Xóa sạch dữ liệu (kể cả tài khoản) để làm lại từ đầu: `docker compose down -v`.

Lần build đầu, `frontend/` chưa có `package-lock.json` nên Docker dùng `npm install`. Nên chạy `npm install` một lần trong `frontend/` và commit `package-lock.json` để các lần build sau dùng `npm ci` với đúng phiên bản đã khóa.

## Cài đặt (phát triển local)

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Tùy chọn: `.[postgres]` (Postgres + checkpointer Postgres), `.[stt]` (faster-whisper).

Frontend (cần Node 20+):

```bash
cd frontend && npm install
```

## Chạy test

```bash
.venv/bin/pytest -q
```

```bash
.venv/bin/ruff check src tests migrations && .venv/bin/mypy
```

```bash
cd frontend && npm run typecheck
```

`tests/integration/test_migrations.py` kiểm tra migration Alembic tạo đúng schema mà `sql/models.py` khai báo (bị bỏ qua nếu chưa cài `alembic`).

## Đánh giá (evaluation)

```bash
.venv/bin/python -m patient_assistant.evaluation.runner
```

Lệnh trên chạy 60 case JSONL qua agent thật, rồi xuất báo cáo vào `eval_reports/`. Báo cáo gồm field accuracy, tool call, FP/FN của validator, self-correction, tỷ lệ hỏi lại, và ba metric bắt buộc bằng 0: false-success, slot hallucination, double-booking. Định nghĩa từng metric ở [docs/EVALUATION.md](docs/EVALUATION.md). Thêm `--llm bedrock` để đo với Claude thật (tốn phí).

## Dữ liệu demo và database

Toàn bộ dữ liệu demo nằm trong các file JSON ở [src/patient_assistant/infrastructure/persistence/demo_data/](src/patient_assistant/infrastructure/persistence/demo_data/). Sửa các file này rồi khởi động lại server là đổi được dữ liệu, không cần sửa code.

| File | Nội dung |
|---|---|
| `departments.json` | Khoa và tên gọi khác của khoa (dùng để nhận ra khoa trong lời nói) |
| `doctors.json` | Bác sĩ thuộc từng khoa |
| `schedule.json` | Lịch làm việc dùng để sinh slot: ngày làm việc, giờ bắt đầu, độ dài slot, số ngày tới |
| `patients.json` | Bệnh nhân demo |
| `users.json` | Tài khoản đăng nhập demo, gắn với bệnh nhân (mật khẩu: `PA_SEED_PASSWORD`) |
| `plans.json` | Kế hoạch theo dõi: agent lấy khoa mặc định từ đây khi người dùng không nêu khoa |
| `knowledge.json` | Tài liệu hướng dẫn đã duyệt dùng cho FAQ (`doc_id` là nguồn được dẫn trong câu trả lời) |
| `history.json` | Báo cáo và lịch hẹn có sẵn, tính tương đối theo hôm nay (`weeks_ago`, `days_from_today`) |

File được kiểm tra khi khởi động: sai định dạng, hoặc tham chiếu sai (ví dụ bác sĩ thuộc khoa không tồn tại), thì server báo lỗi ngay.

**Database:**
- Không đặt `PA_DATABASE_URL`: in-memory, dữ liệu mất khi tắt server.
- `.env.example` đặt sẵn file SQLite `PA_DATABASE_URL=sqlite+aiosqlite:///./data/app.db`. Lần chạy đầu tạo bảng, nạp dữ liệu JSON và tạo tài khoản demo; các lần sau giữ nguyên dữ liệu đã có (tài khoản, mật khẩu đã đổi, lịch sử câu hỏi, báo cáo, lịch hẹn). Xóa `data/app.db` để làm lại từ đầu.
- Postgres: dùng URL `postgresql+asyncpg://...` và cài thêm `.[postgres]`. Với database dùng thật, đặt `PA_AUTO_CREATE_SCHEMA=false` và `PA_SEED_DEMO_DATA=false`, rồi dùng `alembic upgrade head` và `python -m patient_assistant.interface.cli.seed` (Docker compose làm sẵn hai bước này).
- Đổi `sql/models.py` thì sinh migration mới: `alembic revision --autogenerate -m "mô tả"`, đọc lại file sinh ra, rồi `alembic upgrade head`.

Test và eval luôn bắt đầu với database sạch (không nạp `history.json`) để kết quả tái lập được.

Xem dữ liệu **đang có trong database** dưới dạng JSON:

```bash
.venv/bin/python -m patient_assistant.interface.cli.dump --out demo_dump.json
```

Thêm `--database-url sqlite+aiosqlite:///./app.db` để xem dữ liệu trong file SQLite.

## Đăng nhập và phân quyền

Mở frontend (Docker: **http://localhost:8080**, dev: **http://localhost:5173**) sẽ vào trang đăng nhập. Tài khoản demo (mật khẩu mặc định `demo1234`, đặt bằng `PA_SEED_PASSWORD` **trước** lần chạy đầu):

| Tài khoản | Vai trò | Làm được gì |
|---|---|---|
| `benhnhan1`, `benhnhan2`, `benhnhan3` | `patient` | Chat với trợ lý; chỉ thấy dữ liệu và hội thoại của chính mình |
| `bacsi` | `staff` | Trang nhân viên: xem mọi hội thoại, từng câu bệnh nhân hỏi và câu trợ lý trả lời, lọc theo bệnh nhân. Không chat thay bệnh nhân |
| `admin` | `admin` | Như `staff`, thêm tab **Tài khoản**: tạo tài khoản, khóa/mở khóa, đặt lại mật khẩu |

- Mật khẩu băm bằng scrypt; phiên đăng nhập là token ngẫu nhiên trong cookie HttpOnly (DB chỉ lưu hash của token). Phiên hết hạn sau `PA_SESSION_TTL_HOURS` (mặc định 12 giờ).
- Sai mật khẩu 5 lần liên tiếp thì username đó bị khóa đăng nhập 5 phút.
- Khóa tài khoản hoặc đổi mật khẩu sẽ đăng xuất mọi phiên đang mở của tài khoản đó.
- Quyền được kiểm tra ở API cho từng request: bệnh nhân không đọc/gửi/trả lời được hội thoại của người khác, kể cả khi biết `thread_id`.
- Mọi câu bệnh nhân gửi (kể cả bấm nút xác nhận, chọn slot) và mọi câu trợ lý trả lời được lưu vào bảng `chat_messages`, nhóm theo `conversations`, kèm intent agent đã nhận ra.
- Chạy sau HTTPS thì đặt `PA_COOKIE_SECURE=true`.

## Giao diện (UI)

Đăng nhập bằng tài khoản bệnh nhân.

- **Bên trái — hội thoại.** Gõ hoặc chọn câu mẫu. Bấm 🎙 để ghi âm, hoặc ⤒ để tải file audio (cần bật STT). Transcript được hiện ra để bạn sửa trước khi gửi; các từ nhận dạng kém được tô vàng.
- **Khi agent hỏi lại hoặc cần xác nhận,** câu hỏi hiện thành thẻ có nút bấm: chọn slot, xác nhận, sửa bản nháp, hủy. Bạn cũng có thể trả lời tự do như khi nói, ví dụ "cái thứ hai" hay "đổi sang buổi chiều".
- **Khi agent chưa hiểu,** nó nhắc lại câu bạn vừa nói, đoán các việc gần nhất bằng điểm từ khóa, và đưa nút để bấm chọn (lựa chọn được đoán cao nhất được tô đậm).
- **Tra cứu thông tin:** "tôi muốn tra cứu thông tin bệnh nhân" trả về hồ sơ, kế hoạch theo dõi, các báo cáo gần đây (kèm xu hướng mức đau do code tính) và lịch hẹn sắp tới — chỉ của tài khoản đang đăng nhập.
- **Bên phải — "Cách agent ra quyết định":**
  - *Diễn biến:* từng node đã chạy trong mỗi lượt, tô màu theo loại (code, LLM, chờ người dùng, tool, dừng an toàn) kèm lý do; bấm vào một bước để xem dữ liệu chi tiết.
  - *Kiểm tra & sửa lỗi:* các lần trích xuất và sửa, lỗi validator, ai xử lý lỗi, bộ đếm giới hạn, cách quy đổi ngày giờ, cờ STT.
  - *Tool:* tool nào đã được gọi, do code hay LLM gọi, kết quả.
  - *State:* toàn bộ state của hội thoại.

Mã nguồn UI: [frontend/src/](frontend/src/) — `auth/` (đăng nhập, phân quyền route), `features/chat/` (hội thoại bệnh nhân), `features/inspector/` (panel quyết định), `features/staff/` (hội thoại cho nhân viên, quản lý tài khoản), `api/` (client có kiểu, khớp schema backend).

## Chạy local không dùng Docker

Backend — mặc định LLM offline (theo luật), không cần AWS; database theo `PA_DATABASE_URL` trong `.env` (SQLite file hoặc in-memory):

```bash
.venv/bin/uvicorn patient_assistant.interface.api.app:app --factory --reload
```

Frontend — Vite dev server ở http://localhost:5173, proxy `/v1` về backend `127.0.0.1:8000`:

```bash
cd frontend && npm run dev
```

Dùng Claude trên Bedrock (xác thực bằng IAM của AWS) và Postgres: đặt biến môi trường (hoặc file `.env`):

```text
PA_LLM_PROVIDER=bedrock
PA_AWS_REGION=us-east-1
PA_LLM_MODEL_ID=global.anthropic.claude-opus-4-6-v1
PA_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/patient_assistant
PA_CHECKPOINTER=postgres
PA_CHECKPOINTER_URL=postgresql://user:pass@localhost:5432/patient_assistant
PA_STT_ENABLED=true
PA_STT_MODEL=small
```

Kiểm tra model ID và region có dùng được trong tài khoản AWS. Với tiếng Việt, cần model Whisper đa ngôn ngữ (`small`, `medium`, `large-v3`…), không dùng bản `.en`.

## Dùng Claude thật trên Amazon Bedrock (IAM)

1. Credential AWS: `aws configure` (hoặc SSO / assumed role). Kiểm tra bằng `aws sts get-caller-identity`.
2. Quyền IAM: `bedrock:InvokeModel` + `bedrock:InvokeModelWithResponseStream` (model ID dạng inference profile như `global.anthropic.claude-opus-4-6-v1`, đi qua bedrock-runtime) hoặc `bedrock-mantle:CreateInference` (ID dạng `anthropic.claude-opus-5-5`, đi qua Mantle) trên model được dùng, và model đã được bật ở **AWS Console → Amazon Bedrock → Model access** đúng region.
3. Trong `.env`:

```text
PA_LLM_PROVIDER=bedrock
PA_AWS_REGION=ap-southeast-1
PA_LLM_MODEL_ID=global.anthropic.claude-opus-4-6-v1
```

`PA_AWS_PROFILE` để trống thì dùng `AWS_PROFILE` hoặc profile mặc định.

4. Kiểm tra kết nối từng bước (danh tính IAM → gọi Claude → trích xuất có validator → phân loại câu nói lỏng):

```bash
.venv/bin/python -m patient_assistant.interface.cli.check_llm
```

5. Chạy server như bình thường; badge "LLM" trên UI sẽ hiện `global.anthropic.claude-opus-4-6-v1`.

Test và eval mặc định **không** đọc `.env`, nên vẫn chạy offline và không tốn phí. Muốn đo với Claude thật thì chạy `--llm bedrock` cho eval, hoặc `PA_LLM_PROVIDER=bedrock .venv/bin/pytest -m llm` cho test.

## Luồng gọi API

0. `POST /v1/auth/login` `{"username": "benhnhan1", "password": "demo1234"}`: đặt cookie `pa_session`. Client không phải trình duyệt có thể gửi `Authorization: Bearer <giá trị cookie>`. `POST /v1/auth/logout` để đăng xuất, `GET /v1/auth/me` để xem tài khoản hiện tại.
1. (giọng nói) `POST /v1/stt/transcribe` (multipart `audio`): trả về transcript và `flags` (những từ nhận dạng kém) để người dùng duyệt.
2. `POST /v1/chat/{thread_id}/messages` `{"text": "...", "input_source": "voice", "stt_flags": [...]}`.
3. Nếu `status = "waiting_user"`: hiển thị `interrupt` (`clarify` / `confirm_report` / `choose_slot` / `confirm_booking`), rồi gọi `POST /v1/chat/{thread_id}/resume` với một trong các dạng:
   - `{"action": "reply", "text": "cái thứ hai"}` (câu trả lời bằng lời, kể cả giọng nói)
   - `{"action": "choose", "option_no": 2}`
   - `{"action": "confirm"}`
   - `{"action": "edit", "fields": {"pain_level": 4}}`
   - `{"action": "cancel"}`
   - `{"action": "other_time", "text": "buổi chiều"}`
4. Lặp bước 3 cho đến khi `status = "done"`, rồi đọc `response`.

Các API khác:

| API | Quyền | Nội dung |
|---|---|---|
| `GET /v1/me/conversations`, `GET /v1/me/conversations/{thread_id}` | patient | Hội thoại và tin nhắn của chính mình |
| `GET /v1/me/reports`, `GET /v1/me/appointments` | patient | Báo cáo, lịch hẹn của chính mình |
| `GET /v1/staff/conversations?patient_id=` , `GET /v1/staff/conversations/{thread_id}` | staff, admin | Mọi hội thoại / tin nhắn |
| `GET /v1/admin/users`, `POST /v1/admin/users`, `PATCH /v1/admin/users/{id}` `{"active": false}` / `{"password": "..."}` | admin | Quản lý tài khoản |

Chưa đăng nhập → `401`; sai vai trò → `403`; hội thoại không phải của mình → `404`.

## Cấu trúc

Bốn lớp theo clean architecture: `domain` → `application` → `infrastructure` / `interface`, lắp ráp ở `container.py`. Mô tả chi tiết ở [ARCHITECTURE.md §9](ARCHITECTURE.md). Luật phụ thuộc giữa các lớp được kiểm tra bằng `tests/unit/test_architecture.py`.
