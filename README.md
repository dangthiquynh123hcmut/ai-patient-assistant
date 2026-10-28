# AI Patient Assistant

Agent LangGraph hỗ trợ bệnh nhân (dữ liệu giả lập, dự án học tập): gửi báo cáo theo dõi, đặt lịch khám bằng văn bản hoặc giọng nói, xem lịch hẹn và báo cáo, hỏi thông tin trong tài liệu đã duyệt.

- Yêu cầu: [AI_Patient_Assistant_Functional_Requirements_v2.md](AI_Patient_Assistant_Functional_Requirements_v2.md)
- Kiến trúc: [ARCHITECTURE.md](ARCHITECTURE.md), sơ đồ: [docs/diagrams.html](docs/diagrams.html)

## Cài đặt

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Tùy chọn: `.[postgres]` (Postgres + checkpointer Postgres), `.[stt]` (faster-whisper).

## Chạy test

```bash
.venv/bin/pytest -q
```

```bash
.venv/bin/ruff check src tests && .venv/bin/mypy
```

## Đánh giá (evaluation)

```bash
.venv/bin/python -m patient_assistant.evaluation.runner
```

Lệnh trên chạy 60 case JSONL qua agent thật, rồi xuất báo cáo vào `eval_reports/`. Báo cáo gồm field accuracy, tool call, FP/FN của validator, self-correction, tỷ lệ hỏi lại, và ba metric bắt buộc bằng 0: false-success, slot hallucination, double-booking. Định nghĩa từng metric ở [docs/EVALUATION.md](docs/EVALUATION.md). Thêm `--llm bedrock` để đo với Claude thật (tốn phí).

## Giao diện (UI)

Chạy server (mục **Chạy API** bên dưới), rồi mở **http://127.0.0.1:8000/**.

- **Bên trái — hội thoại.** Gõ hoặc chọn câu mẫu. Bấm 🎙 để ghi âm, hoặc ⤒ để tải file audio (cần bật STT). Transcript được hiện ra để bạn sửa trước khi gửi; các từ nhận dạng kém được tô vàng.
- **Khi agent hỏi lại hoặc cần xác nhận,** câu hỏi hiện thành thẻ có nút bấm: chọn slot, xác nhận, sửa bản nháp, hủy. Bạn cũng có thể trả lời tự do như khi nói, ví dụ "cái thứ hai" hay "đổi sang buổi chiều".
- **Bên phải — "Cách agent ra quyết định":**
  - *Diễn biến:* từng node đã chạy trong mỗi lượt, tô màu theo loại (code, LLM, chờ người dùng, tool, dừng an toàn) kèm lý do; bấm vào một bước để xem dữ liệu chi tiết.
  - *Kiểm tra & sửa lỗi:* các lần trích xuất và sửa, lỗi validator, ai xử lý lỗi, bộ đếm giới hạn, cách quy đổi ngày giờ, cờ STT.
  - *Tool:* tool nào đã được gọi, do code hay LLM gọi, kết quả.
  - *State:* toàn bộ state của hội thoại.

UI là file tĩnh trong `src/patient_assistant/interface/web/` (HTML/CSS/JS thuần, không cần build) và gọi các API `/v1/chat/...`, `/v1/meta`, `/v1/chat/{thread_id}/state`.

## Chạy API

Mặc định dùng LLM offline (theo luật) và dữ liệu in-memory, không cần AWS:

```bash
.venv/bin/uvicorn patient_assistant.interface.api.app:app --factory --reload
```

Dùng Claude trên Bedrock (xác thực bằng IAM của AWS) và Postgres: đặt biến môi trường (hoặc file `.env`):

```text
PA_LLM_PROVIDER=bedrock
PA_AWS_REGION=us-east-1
PA_LLM_MODEL_ID=anthropic.claude-opus-5-5
PA_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/patient_assistant
PA_CHECKPOINTER=postgres
PA_CHECKPOINTER_URL=postgresql://user:pass@localhost:5432/patient_assistant
PA_STT_ENABLED=true
PA_STT_MODEL=small
```

Kiểm tra model ID và region có dùng được trong tài khoản AWS. Với tiếng Việt, cần model Whisper đa ngôn ngữ (`small`, `medium`, `large-v3`…), không dùng bản `.en`.

## Luồng gọi API

Header `X-Patient-Id` là auth giả lập (bệnh nhân demo: `patient-demo-001`).

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

## Cấu trúc

Bốn lớp theo clean architecture: `domain` → `application` → `infrastructure` / `interface`, lắp ráp ở `container.py`. Mô tả chi tiết ở [ARCHITECTURE.md §9](ARCHITECTURE.md). Luật phụ thuộc giữa các lớp được kiểm tra bằng `tests/unit/test_architecture.py`.
