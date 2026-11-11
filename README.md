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
- **migrate** — schema do Alembic quản lý ([sql/migrations/](src/patient_assistant/infrastructure/persistence/sql/migrations/)); seed idempotent (danh mục, tài khoản demo, bổ sung slot khám cho đủ số ngày tới, bảng checkpoint).
- Container backend chạy bằng user thường (uid 1000), Postgres không mở cổng ra ngoài.

## Chạy bằng Docker

```bash
cp .env.example .env
```

Sửa `.env`: đặt `POSTGRES_PASSWORD` và cấu hình Claude trên Bedrock (credential AWS lấy từ `~/.aws`, xem mục Bedrock bên dưới). Backend không khởi động nếu không gọi được LLM. Rồi:

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

**Nhận dạng giọng nói (STT) trong Docker:** image backend mặc định không có faster-whisper. Muốn bật, đặt trong `.env`:

```text
PA_STT_ENABLED=true
PA_STT_MODEL=small
BACKEND_EXTRAS=stt
```

rồi `docker compose up -d --build`. Model Whisper được tải ở lần nhận dạng đầu tiên (`small` khoảng 480 MB) và lưu trong volume `stt-models`, nên không phải tải lại khi build hay khởi động lại. Nên tải sẵn trước khi dùng, vì lần đầu có thể lâu hơn thời gian chờ 3 phút của nginx:

```bash
docker compose exec backend python -c "from faster_whisper import download_model; download_model('small')"
```

Backend đặt `HF_HUB_DISABLE_XET=1` để tải qua HTTP thường: trên một số mạng, giao thức Xet của HuggingFace bị treo ở 0 byte.

Lần build đầu, `frontend/` chưa có `package-lock.json` nên Docker dùng `npm install`. Nên chạy `npm install` một lần trong `frontend/` và commit `package-lock.json` để các lần build sau dùng `npm ci` với đúng phiên bản đã khóa.

## Cài đặt (phát triển local)

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Tùy chọn: `.[stt]` (faster-whisper). Driver Postgres (`asyncpg`, `psycopg`) là phụ thuộc bắt buộc.

Frontend (cần Node 20+):

```bash
cd frontend && npm install
```

## Chạy test

Test chạy với **Postgres thật**, bên trong container (cần Docker Desktop). Mỗi test dùng một schema riêng trong
database `<POSTGRES_DB>_test` và xóa schema khi xong, nên không đụng dữ liệu của app:

```bash
docker compose run --rm --build test
```

Mã nguồn được mount vào container, nên sửa code rồi chạy lại không cần `--build` (chỉ cần khi đổi dependency).
Chạy một file: `docker compose run --rm test pytest -q tests/integration/test_report_flow.py`.
Muốn chạy pytest trực tiếp trên máy thì đặt `PA_TEST_DATABASE_URL=postgresql+asyncpg://...` trỏ tới Postgres.

```bash
.venv/bin/ruff check src tests && .venv/bin/mypy
```

```bash
cd frontend && npm run typecheck
```

`tests/integration/test_migrations.py` kiểm tra migration Alembic tạo đúng schema mà `sql/models.py` khai báo (bị bỏ qua nếu chưa cài `alembic`).

## Đánh giá (evaluation)

```bash
.venv/bin/python -m patient_assistant.evaluation.runner
```

Lệnh trên chạy 87 case JSONL qua agent thật, rồi xuất báo cáo vào `eval_reports/`. Báo cáo gồm field accuracy, tool call, FP/FN của validator, self-correction, tỷ lệ hỏi lại, và bốn metric bắt buộc bằng 0: false-success, slot hallucination, double-booking, own-schedule overlap. Định nghĩa từng metric ở [docs/EVALUATION.md](docs/EVALUATION.md). Eval luôn gọi Claude thật trên Bedrock nên **tốn phí**.

## Dữ liệu demo và database

Toàn bộ dữ liệu demo nằm trong các file JSON ở [src/patient_assistant/infrastructure/persistence/demo_data/](src/patient_assistant/infrastructure/persistence/demo_data/). Sửa các file này là đổi được dữ liệu, không cần sửa code.

**Thêm khoa / bác sĩ:** thêm vào `departments.json` và `doctors.json` (mỗi khoa nên có ít nhất một bác sĩ để có slot khám), rồi build lại để chạy seed: `docker compose up -d --build` (service `migrate` chạy seed). Với database đã có dữ liệu, seed chỉ **bổ sung** khoa, bác sĩ, bệnh nhân, kế hoạch, tài liệu còn thiếu theo mã, kèm slot của bác sĩ mới; dữ liệu đã có (kể cả lịch hẹn, báo cáo) giữ nguyên. Sửa tên hay xóa một khoa đã có thì seed không tự áp dụng. Tên gọi khác (`aliases`) nên là cách gọi khoa rõ ràng ("khoa mắt", "khám mắt"), tránh từ trùng với triệu chứng ("mắt" trong "mắt cá chân"), vì hệ thống không được suy ra khoa từ triệu chứng.

| File | Nội dung |
|---|---|
| `departments.json` | Khoa và tên gọi khác của khoa (dùng để nhận ra khoa trong lời nói) |
| `doctors.json` | Bác sĩ thuộc từng khoa |
| `schedule.json` | Lịch làm việc dùng để sinh slot: ngày làm việc, giờ bắt đầu, độ dài slot, số ngày tới |
| `patients.json` | Bệnh nhân demo |
| `users.json` | Tài khoản đăng nhập demo, gắn với bệnh nhân (mật khẩu: `PA_SEED_PASSWORD`). Chạy lại seed sẽ tạo các tài khoản còn thiếu theo username; tài khoản đã có giữ nguyên |
| `plans.json` | Kế hoạch theo dõi: agent lấy khoa mặc định từ đây khi người dùng không nêu khoa |
| `knowledge.json` | Tài liệu hướng dẫn đã duyệt dùng cho FAQ (`doc_id` là nguồn được dẫn trong câu trả lời) |
| `history.json` | Báo cáo và lịch hẹn có sẵn, tính tương đối theo hôm nay (`weeks_ago`, `days_from_today`) |

File được kiểm tra khi khởi động: sai định dạng, hoặc tham chiếu sai (ví dụ bác sĩ thuộc khoa không tồn tại), thì server báo lỗi ngay.

**Database:**
- Chỉ dùng **PostgreSQL**, chạy bằng service `db` của Docker compose (`docker compose up -d db`); `PA_DATABASE_URL` là bắt buộc. Cả dữ liệu nghiệp vụ lẫn state hội thoại của LangGraph (checkpointer) đều nằm trong database này. Không còn chế độ in-memory hay SQLite.
- Cổng của `db` chỉ mở cho localhost (`127.0.0.1:${DB_PORT:-5432}`), để dùng `psql`/công cụ DB hoặc chạy backend ngoài Docker. Dữ liệu nằm trong volume `pgdata`; `docker compose down -v` để làm lại từ đầu.
- Bảng **chỉ** được tạo/đổi bằng migration Alembic, không dùng `metadata.create_all`. Migration nằm trong package ([sql/migrations/](src/patient_assistant/infrastructure/persistence/sql/migrations/)) nên được cài cùng code: server, test, eval và service `migrate` đều chạy cùng một lịch sử schema.
- Chạy backend trực tiếp (không qua compose): lần chạy đầu tự chạy `alembic upgrade head`, nạp dữ liệu JSON và tài khoản demo; các lần sau giữ nguyên dữ liệu. Trong compose, đặt `PA_AUTO_CREATE_SCHEMA=false` và `PA_SEED_DEMO_DATA=false`, schema do `alembic upgrade head` quản lý và dữ liệu do `python -m patient_assistant.interface.cli.seed` nạp (service `migrate` làm sẵn).
- Đổi `sql/models.py` thì sinh migration mới (chạy ở thư mục gốc, có `alembic.ini`, cần `PA_DATABASE_URL` trỏ tới DB đã ở bản mới nhất): `alembic revision --autogenerate -m "mô tả"`, đọc lại file sinh ra, rồi `alembic upgrade head`. Không sửa bảng bằng tay hay bằng `create_all`: `test_migrations.py` sẽ báo lệch giữa models và migration.

Test và eval chạy mỗi test/case trong một schema Postgres riêng, bắt đầu sạch (không nạp `history.json`) để kết quả tái lập được; schema bị xóa khi chạy xong.

Xem dữ liệu **đang có trong database** dưới dạng JSON:

```bash
.venv/bin/python -m patient_assistant.interface.cli.dump --out demo_dump.json
```

Mặc định đọc database theo `PA_DATABASE_URL`; đổi bằng `--database-url postgresql+asyncpg://...`.

## Đăng nhập và phân quyền

Mở frontend (Docker: **http://localhost:8080**, dev: **http://localhost:5173**) sẽ vào trang đăng nhập. Tài khoản demo (mật khẩu mặc định `demo1234`, đặt bằng `PA_SEED_PASSWORD` **trước** lần chạy đầu):

| Tài khoản | Vai trò | Làm được gì |
|---|---|---|
| `benhnhan1`, `benhnhan2`, `benhnhan3` | `patient` | Chat với trợ lý; chỉ thấy dữ liệu và hội thoại của chính mình |
| `bacsi` | `staff` | Trang nhân viên: xem mọi hội thoại, từng câu bệnh nhân hỏi và câu trợ lý trả lời, lọc theo bệnh nhân. Không chat thay bệnh nhân |
| `admin` | `admin` | Như `staff`, thêm tab **Tài khoản**: tạo tài khoản, khóa/mở khóa, đặt lại mật khẩu |

**Bệnh nhân trùng tên (để thử):** mỗi người một hồ sơ (mã khác nhau) và một tài khoản riêng, cùng mật khẩu demo.

| Họ tên | Tài khoản | Ghi chú |
|---|---|---|
| Nguyễn Văn An | `nguyenvanan`, `nguyenvanan2`, `an.nguyen` | 3 người: kế hoạch PHCN / kế hoạch Cơ xương khớp / chưa có kế hoạch |
| Trần Thị Bình | `tranthibinh`, `binh.tran` | 2 người: chưa có kế hoạch / kế hoạch Tim mạch |
| Lê Thị Hoa · Le Thi Hoa | `lethihoa`, `lethihoa.kd` | Chỉ khác nhau ở dấu |
| Trần Văn An | `tranvanan` | Bệnh nhân trùng tên bác sĩ BS. Trần Văn An (khoa PHCN) và tên hiển thị của tài khoản `bacsi` |
| Bệnh nhân Demo 1 | `benhnhan1b` | Trùng tên hiển thị với `benhnhan1` |

**Tự đăng ký:** ở trang đăng nhập bấm **Đăng ký** (trang `/register`), nhập họ tên, tên đăng nhập và mật khẩu (tối thiểu 8 ký tự). Hệ thống tạo một hồ sơ bệnh nhân mới cùng tài khoản `patient` gắn với hồ sơ đó, rồi đăng nhập luôn. Người tự đăng ký không chọn được vai trò hay hồ sơ bệnh nhân có sẵn: tài khoản `staff`/`admin`, hoặc tài khoản gắn vào hồ sơ đã có, vẫn chỉ admin tạo được. Bệnh nhân mới xuất hiện ngay trong danh sách bệnh nhân của trang nhân viên. Tắt chức năng này bằng `PA_ALLOW_REGISTRATION=false` (nút Đăng ký ẩn đi, API trả 403).

- Mật khẩu băm bằng scrypt; phiên đăng nhập là token ngẫu nhiên trong cookie HttpOnly (DB chỉ lưu hash của token). Phiên hết hạn sau `PA_SESSION_TTL_HOURS` (mặc định 12 giờ).
- Sai mật khẩu 5 lần liên tiếp thì username đó bị khóa đăng nhập 5 phút.
- Khóa tài khoản hoặc đổi mật khẩu sẽ đăng xuất mọi phiên đang mở của tài khoản đó.
- Quyền được kiểm tra ở API cho từng request: bệnh nhân không đọc/gửi/trả lời được hội thoại của người khác, kể cả khi biết `conversation_id`.
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

Backend — cần gọi được Claude trên Bedrock (xem mục Bedrock bên dưới), lúc khởi động server gọi thử LLM một lần và dừng nếu lỗi; cần Postgres đang chạy (`docker compose up -d db`) và `PA_DATABASE_URL` trong `.env` trỏ tới nó:

```bash
.venv/bin/uvicorn patient_assistant.interface.api.app:app --factory --reload
```

Frontend — Vite dev server ở http://localhost:5173, proxy `/v1` về backend `127.0.0.1:8000`:

```bash
cd frontend && npm run dev
```

Cấu hình đầy đủ: đặt biến môi trường (hoặc file `.env`):

```text
PA_AWS_REGION=us-east-1
PA_LLM_MODEL_ID=global.anthropic.claude-opus-4-6-v1
PA_DATABASE_URL=postgresql+asyncpg://patient:<POSTGRES_PASSWORD>@127.0.0.1:5432/patient_assistant
PA_STT_ENABLED=true
PA_STT_MODEL=small
```

Kiểm tra model ID và region có dùng được trong tài khoản AWS. Với tiếng Việt, cần model Whisper đa ngôn ngữ (`small`, `medium`, `large-v3`…), không dùng bản `.en`.

## Dùng Claude thật trên Amazon Bedrock (IAM)

1. Credential AWS: `aws configure` (hoặc SSO / assumed role). Kiểm tra bằng `aws sts get-caller-identity`.
2. Quyền IAM: `bedrock:InvokeModel` + `bedrock:InvokeModelWithResponseStream` (model ID dạng inference profile như `global.anthropic.claude-opus-4-6-v1`, đi qua bedrock-runtime) hoặc `bedrock-mantle:CreateInference` (ID dạng `anthropic.claude-opus-5-5`, đi qua Mantle) trên model được dùng, và model đã được bật ở **AWS Console → Amazon Bedrock → Model access** đúng region.
3. Trong `.env`:

```text
PA_AWS_REGION=ap-southeast-1
PA_LLM_MODEL_ID=global.anthropic.claude-opus-4-6-v1
```

`PA_AWS_PROFILE` để trống thì dùng `AWS_PROFILE` hoặc profile mặc định.

4. Kiểm tra kết nối từng bước (danh tính IAM → gọi Claude → trích xuất có validator → phân loại câu nói lỏng):

```bash
.venv/bin/python -m patient_assistant.interface.cli.check_llm
```

5. Chạy server như bình thường; badge "LLM" trên UI sẽ hiện `global.anthropic.claude-opus-4-6-v1`.

Test mặc định dùng LLM giả có kịch bản định sẵn (`ScriptedLLM` trong `tests/conftest.py`), không gọi Bedrock và không tốn phí. Test với Claude thật: `.venv/bin/pytest -m llm`. Eval luôn gọi Claude thật.

## Luồng gọi API

0. `POST /v1/auth/login` `{"username": "benhnhan1", "password": "demo1234"}`: đặt cookie `pa_session`. Client không phải trình duyệt có thể gửi `Authorization: Bearer <giá trị cookie>`. `POST /v1/auth/logout` để đăng xuất, `GET /v1/auth/me` để xem tài khoản hiện tại. Tự đăng ký: `POST /v1/auth/register` `{"username": "...", "display_name": "...", "password": "..."}` trả 201 và đặt cookie phiên như đăng nhập (409 nếu trùng tên đăng nhập, 403 nếu đang tắt); `GET /v1/auth/options` (không cần đăng nhập) cho biết có cho đăng ký không.
1. (giọng nói) `POST /v1/stt/transcribe` (multipart `audio`): trả về transcript và `flags` (những từ nhận dạng kém) để người dùng duyệt.
2. `POST /v1/chat/{conversation_id}/messages` `{"text": "...", "input_source": "voice", "stt_flags": [...]}`.
3. Nếu `status = "waiting_user"`: hiển thị `interrupt` (`clarify` / `confirm_report` / `choose_slot` / `confirm_booking`), rồi gọi `POST /v1/chat/{conversation_id}/resume` với một trong các dạng:
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
| `GET /v1/me/conversations`, `GET /v1/me/conversations/{conversation_id}` | patient | Hội thoại và tin nhắn của chính mình |
| `GET /v1/me/reports`, `GET /v1/me/appointments` | patient | Báo cáo, lịch hẹn của chính mình |
| `GET /v1/staff/conversations?patient_id=` , `GET /v1/staff/conversations/{conversation_id}` | staff, admin | Mọi hội thoại / tin nhắn |
| `GET /v1/admin/users`, `POST /v1/admin/users`, `PATCH /v1/admin/users/{id}` `{"active": false}` / `{"password": "..."}` | admin | Quản lý tài khoản |

Chưa đăng nhập → `401`; sai vai trò → `403`; hội thoại không phải của mình → `404`.

## Cấu trúc

Bốn lớp theo clean architecture: `domain` → `application` → `infrastructure` / `interface`, lắp ráp ở `container.py`. Mô tả chi tiết ở [ARCHITECTURE.md §9](ARCHITECTURE.md). Luật phụ thuộc giữa các lớp được kiểm tra bằng `tests/unit/test_architecture.py`.
