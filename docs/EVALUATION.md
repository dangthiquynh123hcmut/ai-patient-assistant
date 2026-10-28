# Evaluation — AI Patient Assistant

Bộ đánh giá chạy **agent thật**: LangGraph, validator, service và repository. Chỉ có dữ liệu là in-memory, đồng hồ là cố định, và một số lỗi được tiêm có chủ đích. Mỗi case dùng một container riêng nên các case không ảnh hưởng nhau.

Code nằm ở `src/patient_assistant/evaluation/`:

| File | Vai trò |
|---|---|
| `datasets/*.jsonl` | Dataset: `report.jsonl` (23 case), `booking.jsonl` (26), `conversation.jsonl` (11) |
| `cases.py` | Schema case (Pydantic) và hàm `load_cases` (báo lỗi theo `file:dòng`, chặn id trùng) |
| `harness.py` | Chạy case qua `ChatUseCase`, trả lời các interrupt theo kịch bản, tiêm lỗi, ghi `CaseObservation` |
| `metrics.py` | Hàm thuần: (case, observation) → metric và verdict từng case |
| `report.py` | Xuất `report.md` (để đọc) và `report.json` (để so sánh giữa các lần chạy) |
| `runner.py` | CLI `pa-eval` |

## Chạy

```bash
.venv/bin/python -m patient_assistant.evaluation.runner
```

```bash
.venv/bin/python -m patient_assistant.evaluation.runner --llm bedrock --suite booking --fail-on-hard-zero
```

Tùy chọn khác: `--case <id>` (lặp được), `--dataset file.jsonl ...`, `--out eval_reports`, `--concurrency 4`. Báo cáo được ghi vào `eval_reports/<thời điểm>-<llm>/`. Nếu đã cài lại package (`pip install -e .`), có thể gọi tắt bằng `pa-eval`.

`--llm bedrock` gọi Claude thật và **tốn phí**: mỗi case có từ 1 đến khoảng 5 lần gọi LLM. Nên chạy thử một suite nhỏ trước.

## Định dạng case

Mỗi dòng JSONL là một case. Các trường:

| Trường | Ý nghĩa |
|---|---|
| `id`, `suite`, `category`, `description` | Định danh và phân nhóm |
| `now` | Mốc thời gian cố định, mặc định `2026-10-28T10:00:00+07:00` (Thứ Tư) |
| `patient_id` | Mặc định `patient-demo-001` (có kế hoạch theo dõi ở khoa PHCN); `patient-no-plan` là bệnh nhân chưa có kế hoạch |
| `message` | Tin nhắn đầu tiên: `text`, `input_source` (`text`/`voice`), `stt_flags` (giả lập STT nhận dạng kém) |
| `replies` | Câu trả lời cho các interrupt, dùng lần lượt. Hết danh sách thì dừng, kết quả là `waiting` |
| `llm_script.reports` / `llm_script.bookings` | Đầu ra LLM **tiêm sẵn** (cố ý sai) cho các lần trích xuất đầu. Hết danh sách thì dùng LLM thật |
| `faults` | `llm_unavailable`, `save_unavailable`, `booking_unavailable`, `slot_taken_after_choice`, `prebook_window` |
| `concurrency.sessions` | Chạy N phiên (N bệnh nhân) song song, cùng chốt một slot vào cùng một thời điểm |
| `expected` | `intent`, `outcome`, `should_clarify`, `fields`, `tools`, `must_not_write`, `booked_option`, `response_contains` |
| `labels` | Nhãn cho validator: `injected_error`, `first_attempt_valid`, `issue_codes` |

Ví dụ một case tiêm lỗi:

```json
{"suite": "report", "id": "rep-inj-01", "category": "injected_value_mismatch",
 "message": {"text": "Tuần này tôi đau mức 5 trên 10, đã tập 4 ngày, thấy đau tăng khi gập gối"},
 "replies": [{"action": "confirm"}],
 "llm_script": {"reports": [{"pain_level": 5, "exercise_days": 5, "evidence": [
     {"field": "pain_level", "quote": "đau mức 5 trên 10"}, {"field": "exercise_days", "quote": "đã tập 4 ngày"}]}]},
 "expected": {"intent": "create_report", "outcome": "report_saved", "fields": {"pain_level": 5, "exercise_days": 4}},
 "labels": {"injected_error": true, "issue_codes": ["value_evidence_mismatch"]}}
```

## Định nghĩa metric

| Metric | Cách tính |
|---|---|
| **Case pass rate** | Case đạt khi mọi kiểm tra đều đúng: outcome, intent, fields, tool bắt buộc, slot được đặt, việc có hỏi lại hay không, không ghi khi `must_not_write`, nội dung bắt buộc trong câu trả lời. Ngoài ra không được báo thành công giả, không được đề xuất slot không có thật, không được đặt trùng |
| **Field accuracy** | So từng trường trong `expected.fields` với giá trị **cuối cùng**: bản nháp báo cáo, hoặc khoa và khoảng ngày giờ đã quy đổi cho booking |
| **First-attempt field accuracy** | Như trên nhưng tính trên đầu ra **lần trích xuất đầu tiên**, tức trước khi sửa và trước khi hỏi lại. So với field accuracy để thấy mức cải thiện nhờ self-correction và clarification |
| **Tool selection accuracy** | Đã gọi đủ `expected.tools`, và không gọi tool ghi khi case cấm ghi. Riêng case tiêm lỗi backend: thử gọi tool ghi vẫn được tính là đúng |
| **Tool recall** | Tỷ lệ tool bắt buộc đã thực sự được gọi |
| **Unnecessary tool call rate** | Chỉ tính trên các case khai báo `tools` tường minh: số lần gọi tool nằm ngoài `tools`, trừ bước đọc lại để kiểm chứng (`get_report`, `get_appointment`) |
| **Write argument accuracy** | Bản báo cáo đã lưu đúng các giá trị kỳ vọng, và slot đã đặt đúng `booked_option` trong danh sách được đề xuất đầu tiên |
| **Validator TP/FP/FN/TN** | Tính trên **lần trích xuất đầu tiên**. "Có lỗi" lấy từ `labels.first_attempt_valid`; nếu không khai báo thì coi là có lỗi khi `should_clarify` hoặc `injected_error`. FP: đầu ra hợp lệ nhưng bị gắn cờ. FN: có lỗi nhưng validator cho qua |
| **Issue code recall** | Tỷ lệ mã lỗi trong `labels.issue_codes` thực sự được validator báo ở lần đầu |
| **Self-correction** | Tính trên các case `injected_error` mà lần đầu bị gắn cờ. Success: có một lần sửa sau đó đạt. Regression: lần sửa sinh ra lỗi **LLM** mới (`llm_fixable`). Lỗi chuyển sang hỏi người dùng, như `missing_field`, không bị tính là regression |
| **Clarification** | Hỏi lại khi cần: tỷ lệ case `should_clarify` có interrupt `clarify`. Guess rate: case cần hỏi nhưng agent không hỏi mà đi thẳng tới bản nháp hoặc ghi dữ liệu. Hỏi lại thừa: case không cần hỏi nhưng agent vẫn hỏi |
| **False-success** (= 0) | Agent báo `report_saved` / `appointment_booked` nhưng backend không có bản ghi tương ứng, hoặc kết quả không phải thành công mà câu trả lời vẫn chứa "Đã lưu báo cáo" / "Đã đặt lịch" |
| **Slot hallucination** (= 0) | Slot được đề xuất không có trong kho slot, hoặc slot đã đặt không nằm trong danh sách vừa đề xuất cho người dùng |
| **Double-booking** (= 0) | Sau khi chạy xong case, có slot nào có hơn một lịch hẹn hiệu lực. Case `bk-conc-01` dùng 5 phiên chốt cùng lúc, và phải có đúng 1 phiên đặt được |
| **Latency** | p50 / p95 / max thời gian chạy một case (mỗi phiên tính riêng) |

Ba metric đánh dấu "= 0" là ngưỡng cứng. `--fail-on-hard-zero` trả exit code 1 nếu có metric nào khác 0, dùng được trong CI.

## Kết quả baseline (LLM offline, 2026-10-28)

| Metric | Kết quả |
|---|---|
| Case đạt | 64/64 (60 case; `bk-conc-01` có 5 phiên) |
| Field accuracy (cuối / lần đầu) | 100% (79/79) / 70.2% (33/47) |
| Tool selection / unnecessary calls / write args | 100% / 0% / 100% |
| Validator | TP 23, FP 0, FN 0, TN 25; bắt đúng 100% mã lỗi kỳ vọng |
| Self-correction | 8/9 sửa được, regression 0%, trung bình 2.22 lần thử. Case còn lại `rep-inj-06` được thiết kế để không sửa được, và agent dừng an toàn |
| Hỏi lại | Hỏi khi cần 100%, tự đoán 0%, hỏi thừa 0% |
| False-success / slot hallucination / double-booking | 0 / 0 / 0 |

**Cách đọc baseline này:** `OfflineRuleLLM` là bộ luật được viết song song với dataset, nên 100% ở đây chủ yếu chứng minh **workflow, validator và các chốt chặn** chạy đúng. Nó **không** đo chất lượng của một mô hình ngôn ngữ thật. Con số có ý nghĩa cho phần trích xuất là kết quả khi chạy `--llm bedrock`. Các case có `llm_script` tách riêng phần này: chúng đo validator và self-correction bằng đầu ra sai có chủ đích, độc lập với LLM được dùng.

## Bộ eval bắt được gì khi phát triển

Lần chạy đầu tiên đạt 58/64 case và tìm ra các lỗi thật của agent, nay đã sửa:

- "tập 5 ngày **thôi**" bị hiểu là **hủy**, vì từ "thôi" đứng một mình nằm trong danh sách cụm hủy.
- "tuần trước chưa tập, tuần này tập 4 ngày" bị lưu thành **0 ngày** mà không hỏi lại.
- "thấy **đỡ đau hơn**" bị hiểu là xấu đi, vì cụm này chứa "đau hơn".
- "cần chuẩn bị gì trước buổi **tái khám**" bị chuyển sang luồng đặt lịch.
- Tool do LLM gọi trong `rag_answer` không được ghi lại. Ngoài ra, câu trả lời chỉ được dẫn `doc_id` mà tool thực sự đã trả về.
- Checkpoint LangGraph giải mã các kiểu dữ liệu chưa đăng ký; bản LangGraph sau sẽ chặn việc này. Nay checkpointer dùng allowlist tự suy ra từ `AgentState` (`checkpoint_types()`).

`tests/integration/test_evaluation.py` kiểm tra chính bộ eval:

- Metric phát hiện được báo thành công giả, slot ảo, đặt trùng, FP/FN của validator.
- Một mutation test: tắt validator báo cáo thì eval phải báo FN ≥ 2 và có case thất bại.

## Chưa có

- **STT WER** và độ đúng các trường sau STT trên audio thật: cần bộ audio tiếng Việt và cài `.[stt]`.
- Latency theo từng node: hiện chỉ đo theo case. `tool_calls` có ghi latency của từng tool.
- Chi phí token khi chạy với Bedrock (FR-08.7).
