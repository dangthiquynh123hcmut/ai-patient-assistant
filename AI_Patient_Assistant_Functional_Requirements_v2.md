# AI Patient Assistant — Agent-Focused Functional Requirements

**Phiên bản:** 0.3  
**Ngày cập nhật:** 2026-10-28  
**Trạng thái:** Draft — phạm vi đã thu gọn để tập trung xây dựng AI Agent và đánh giá độ tin cậy.  
**Thay đổi so với 0.2:** đưa đặt lịch khám bằng text/giọng nói trên dữ liệu giả lập vào phạm vi MVP (§2, FR-02.1, FR-03, FR-09).

---

## 1. Mục tiêu dự án

Xây dựng một ứng dụng portfolio để thực hành các thành phần cốt lõi của **AI Agent** trong bối cảnh giả lập hỗ trợ bệnh nhân. Trọng tâm không phải là làm một ứng dụng bệnh viện có thật đầy đủ chức năng, mà là xây dựng một Agent có thể:

1. Hiểu yêu cầu của người dùng và chuyển yêu cầu thành dữ liệu có cấu trúc.
2. Chọn và gọi đúng function/tool khi cần lấy dữ liệu hoặc thực hiện hành động.
3. Kiểm tra dữ liệu AI tạo ra bằng schema, quy tắc nghiệp vụ và dữ liệu thực tế từ backend.
4. Giải thích kết quả bằng căn cứ có thể kiểm chứng, thay vì chỉ đưa ra một kết luận.
5. Tự sửa kết quả khi validator phát hiện lỗi, trong giới hạn số lần thử được cấu hình.
6. Biết khi nào cần hỏi lại người dùng, từ chối hành động hoặc chuyển cho con người thay vì đoán.
7. Nhận đầu vào bằng văn bản và **Speech-to-Text (STT)**; cả hai đi vào cùng một Agent workflow.

Ứng dụng sử dụng dữ liệu giả lập. Đây là dự án học tập về AI Agent, không phải hệ thống chẩn đoán hoặc ra quyết định điều trị.

## 2. Phạm vi MVP

### Trong phạm vi

- Chat bằng văn bản.
- Chuyển giọng nói thành văn bản (STT), hiển thị transcript để người dùng kiểm tra.
- Đặt lịch khám bằng văn bản hoặc giọng nói trên **dữ liệu lịch giả lập** (khoa, bác sĩ, slot), có xác nhận trước khi đặt; xem các lịch hẹn sắp tới của mình.
- Một Agent workflow có state và các bước rõ ràng.
- Function calling với một số tool có đầu vào/đầu ra được định nghĩa bằng schema.
- Validation nhiều lớp: schema, nghiệp vụ, nguồn dữ liệu và đánh giá chất lượng AI.
- Cơ chế sửa lỗi có giới hạn: validate → phản hồi lỗi cụ thể → thử tạo lại → validate lại.
- Yêu cầu AI cung cấp **lý do ngắn gọn, căn cứ và nguồn dữ liệu** cho kết quả.
- Ghi log phục vụ debug và đánh giá, tránh lưu dữ liệu nhạy cảm không cần thiết.
- Bộ test giả lập để đo độ chính xác, lỗi function calling và hiệu quả self-correction.

### Ngoài phạm vi ở giai đoạn đầu

- Đặt/đổi/hủy lịch khám **thật**: tích hợp HIS/EMR, hệ thống đặt lịch hoặc bất kỳ hệ thống bệnh viện nào. Booking trong MVP chỉ chạy trên dữ liệu giả lập.
- Đổi/hủy lịch hẹn trên dữ liệu giả lập (P1, FR-09.13) và giữ chỗ slot tạm thời trong lúc chờ xác nhận.
- Gợi ý khoa/chuyên khoa dựa trên triệu chứng (thuộc phân loại bệnh, ngoài phạm vi).
- Nhắc lịch, quản lý tài khoản nhiều vai trò, dashboard bác sĩ hoàn chỉnh.
- TTS (đọc câu trả lời thành tiếng); có thể bổ sung sau nếu cần.
- Chẩn đoán bệnh, đề xuất điều trị, khuyến nghị thay đổi thuốc hoặc xử lý cấp cứu tự động.
- Huấn luyện/fine-tune mô hình lớn để cải thiện chất lượng.
- Xây dựng nhiều Agent độc lập ngay từ đầu. MVP nên dùng một workflow Agent có các node/tool riêng để dễ kiểm thử.


## 4. Functional Requirements (FR)

Mức ưu tiên: **P0** = bắt buộc cho MVP; **P1** = bổ sung sau MVP.

### FR-01. Nhận đầu vào văn bản và giọng nói

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-01.1 | Người dùng có thể gửi yêu cầu bằng văn bản. | P0 |
| FR-01.2 | Người dùng có thể ghi âm và hệ thống chuyển âm thanh thành văn bản bằng STT. | P0 |
| FR-01.3 | Transcript được hiển thị để người dùng kiểm tra/sửa trước khi xử lý các thông tin quan trọng. | P0 |
| FR-01.4 | Nếu transcript không rõ hoặc có khả năng nhận sai số, ngày, giờ, mức độ đau, tên hoạt động, tên khoa hay tên bác sĩ, Agent phải hỏi lại thay vì tự đoán. | P0 |
| FR-01.5 | Đầu vào văn bản và transcript sau STT dùng chung một Agent workflow. | P0 |
| FR-01.6 | MVP dùng STT chạy local trên CPU; cấu hình model phải thay đổi được mà không viết lại Agent. | P0 |

### FR-02. Agent orchestration và state

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-02.1 | Agent phân loại yêu cầu thành các intent được hỗ trợ: `create_report`, `get_report_history`, `book_appointment`, `list_appointments`, `answer_faq` hoặc `unknown`. Intent P1: `reschedule_appointment`, `cancel_appointment`. Mỗi lượt xử lý một intent chính; nếu người dùng nêu nhiều việc, Agent làm lần lượt và hỏi trước khi chuyển sang việc tiếp theo. | P0 |
| FR-02.2 | Agent duy trì state của lượt xử lý: đầu vào gốc, transcript, intent, dữ liệu trích xuất, lỗi validation, kết quả tool và trạng thái xác nhận. | P0 |
| FR-02.3 | Agent chọn workflow/tool dựa trên intent và điều kiện đã định nghĩa; không gọi tool tùy tiện khi không cần. | P0 |
| FR-02.4 | Khi thiếu thông tin bắt buộc, Agent hỏi người dùng và tạm dừng hành động ghi dữ liệu. | P0 |
| FR-02.5 | Có giới hạn số bước/lần gọi tool để tránh vòng lặp vô hạn. | P0 |
| FR-02.6 | Lỗi từ model, STT, tool hoặc database phải được xử lý tường minh và không biến thành kết quả thành công giả. | P0 |

### FR-03. Function calling / tools

Mỗi tool phải có tên, mục đích, schema input/output, điều kiện được gọi và cách xử lý lỗi rõ ràng.

| Tool đề xuất | Mục đích | Quy tắc |
|---|---|---|
| `get_patient_plan` | Lấy kế hoạch theo dõi giả lập | Chỉ lấy dữ liệu qua backend; không để LLM tự tạo kế hoạch hoặc ID. |
| `get_report_history` | Lấy các báo cáo trước đó | Kiểm tra quyền và định danh bệnh nhân ở backend. |
| `validate_report` | Kiểm tra cấu trúc và quy tắc nghiệp vụ của báo cáo | Trả về lỗi cụ thể theo trường, không chỉ trả `valid=false`. |
| `save_patient_report` | Lưu báo cáo đã được xác nhận | Chỉ gọi sau khi validation đạt và người dùng xác nhận; backend kiểm tra lại. |
| `search_approved_knowledge` | Tìm thông tin trong tập tài liệu giả lập được duyệt | Trả về đoạn nguồn/ID tài liệu để có thể dẫn chứng. |
| `list_departments` | Lấy danh mục khoa/bác sĩ giả lập | Danh mục là nguồn duy nhất cho giá trị khoa/bác sĩ hợp lệ; LLM không tự tạo khoa. |
| `get_available_slots` | Tìm slot trống theo khoa, khoảng ngày, bác sĩ (tùy chọn) | Chỉ nhận ngày cụ thể do code quy đổi; trả `slot_id` do backend sinh, giới hạn số kết quả. |
| `list_my_appointments` | Lấy lịch hẹn của bệnh nhân | Định danh bệnh nhân lấy từ phiên đăng nhập ở backend, không lấy từ tham số LLM. |
| `create_appointment` | Đặt lịch cho slot đã chọn | Chỉ gọi sau khi validation đạt và người dùng xác nhận; `slot_id` phải thuộc danh sách vừa trả cho người dùng; backend chống đặt trùng và kiểm tra lại quy tắc nghiệp vụ. |
| `reschedule_appointment`, `cancel_appointment` (P1) | Đổi/hủy lịch hẹn | Cùng quy tắc với `create_appointment`. |

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-03.1 | LLM phải tạo lời gọi tool với tham số đúng schema. | P0 |
| FR-03.2 | Backend xác thực tham số, quyền truy cập và quy tắc nghiệp vụ độc lập với LLM. | P0 |
| FR-03.3 | Tool trả về kết quả có cấu trúc, gồm trạng thái, dữ liệu hoặc mã lỗi. | P0 |
| FR-03.4 | Tool đọc dữ liệu không được giả lập kết quả bằng trí nhớ của LLM. | P0 |
| FR-03.5 | Tool có tác dụng ghi/thay đổi dữ liệu (`save_patient_report`, `create_appointment` và các tool đổi/hủy lịch) phải yêu cầu xác nhận người dùng và có kiểm soát chống ghi trùng. Các tool này không được expose cho LLM tự gọi; workflow gọi chúng sau bước xác nhận. | P0 |
| FR-03.6 | Agent phải đọc kết quả tool trước khi phản hồi; không được coi việc gọi tool là bằng chứng hành động thành công. | P0 |
| FR-03.7 | Tool có lỗi phải trả lỗi chuẩn hóa để Agent có thể xử lý hoặc thông báo đúng. | P0 |

### FR-04. Validation đầu ra AI

Hệ thống phải xác minh kết quả theo nhiều lớp; không chỉ yêu cầu LLM tự chấm điểm cho câu trả lời của chính nó.

| Lớp validation | Nội dung | Ví dụ |
|---|---|---|
| 1. Schema validation | Kiểm tra kiểu dữ liệu, trường bắt buộc, enum và định dạng | `pain_level` là số; `exercise_days` là số nguyên. |
| 2. Business validation | Kiểm tra giới hạn và quy tắc nghiệp vụ | `pain_level` nằm trong 0–10; `exercise_days` nằm trong 0–7. |
| 3. Source-grounding validation | Đối chiếu giá trị trích xuất với transcript hoặc dữ liệu nguồn | Transcript nói “bốn ngày” thì không được lưu thành 5 ngày. |
| 4. Consistency validation | Phát hiện dữ liệu mâu thuẫn trong cùng báo cáo hoặc với ngữ cảnh đã xác nhận | Người dùng nói “chưa tập lần nào” nhưng `exercise_days=4`. |
| 5. Tool/result validation | Kiểm tra phản hồi tool/API trước khi khẳng định kết quả | Chỉ thông báo lưu thành công khi backend trả `success` và report ID hợp lệ. |
| 6. Safety validation | Chặn hành vi ngoài phạm vi hoặc nội dung y tế rủi ro | Không tạo chẩn đoán hay thay đổi điều trị. |

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-04.1 | Đầu ra trích xuất phải tuân theo Pydantic model/JSON Schema đã định nghĩa. | P0 |
| FR-04.2 | Validator trả danh sách lỗi có trường, loại lỗi, mô tả và gợi ý sửa. | P0 |
| FR-04.3 | Trường không có bằng chứng phải là `null`/`unknown` hoặc được đưa vào `missing_fields`, không tự điền. | P0 |
| FR-04.4 | Hệ thống giữ transcript/nguồn gốc để đối chiếu kết quả trích xuất. | P0 |
| FR-04.5 | Khi validation không đạt, không gọi tool ghi dữ liệu. | P0 |
| FR-04.6 | Người dùng có thể xem và xác nhận bản nháp trước khi lưu. | P0 |
| FR-04.7 | Các quy tắc quan trọng phải được thực thi ở backend, không chỉ nằm trong prompt. | P0 |

### FR-05. Lý do, bằng chứng và khả năng giải thích

“Bắt AI đưa ra lý do” trong dự án này có nghĩa là yêu cầu AI trình bày **giải thích ngắn gọn, có thể kiểm chứng**: dữ liệu nào hỗ trợ kết quả, quy tắc nào được áp dụng, và còn điểm nào chưa chắc chắn. Không yêu cầu hiển thị toàn bộ suy luận nội bộ dài của mô hình.

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-05.1 | Với mỗi trường quan trọng được trích xuất, Agent có thể chỉ ra đoạn transcript hỗ trợ trường đó hoặc ghi rõ không có bằng chứng. | P0 |
| FR-05.2 | Với kết quả lấy từ tool/RAG, Agent nêu nguồn dữ liệu hoặc ID tài liệu/bản ghi khi có. | P0 |
| FR-05.3 | Agent giải thích ngắn gọn vì sao kết quả được chấp nhận hoặc bị từ chối, dựa trên validator và quy tắc đã định nghĩa. | P0 |
| FR-05.4 | Agent phân biệt rõ dữ kiện từ người dùng, dữ liệu lấy từ backend và suy luận/tóm tắt do AI tạo. | P0 |
| FR-05.5 | Agent nêu trường chưa chắc chắn, thiếu dữ liệu hoặc mâu thuẫn; không tự thể hiện mức độ chắc chắn cao khi không có căn cứ. | P0 |
| FR-05.6 | Lý do giải thích không được dùng thay cho validation thực tế; việc AI nói “tôi chắc chắn” không làm kết quả hợp lệ. | P0 |

Ví dụ cấu trúc giải thích:

```json
{
  "field": "pain_level",
  "value": 5,
  "evidence": "Tôi đau mức 5 trên 10",
  "validation": "passed",
  "explanation": "Giá trị 5 được nêu trực tiếp trong transcript và nằm trong phạm vi 0–10."
}
```

### FR-06. Self-correction có kiểm soát

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-06.1 | Khi validator phát hiện lỗi, hệ thống gửi lỗi cụ thể trở lại bước tạo/sửa đầu ra để Agent thử sửa. | P0 |
| FR-06.2 | Sau mỗi lần sửa, kết quả phải chạy lại toàn bộ validator liên quan. | P0 |
| FR-06.3 | Số lần sửa tự động phải có giới hạn cấu hình, ví dụ tối đa 2 lần sau lần tạo đầu tiên. | P0 |
| FR-06.4 | Nếu vẫn không hợp lệ, Agent dừng và hỏi người dùng, trả trạng thái cần xem xét hoặc chuyển cho con người; không lặp vô hạn. | P0 |
| FR-06.5 | Không được tự sửa bằng cách bịa dữ liệu hoặc bỏ qua lỗi validator. | P0 |
| FR-06.6 | Hệ thống lưu metadata cho mỗi lần thử: phiên bản đầu ra, lỗi phát hiện, hành động sửa, kết quả validation cuối. | P0 |
| FR-06.7 | Có thể dùng một bước evaluator độc lập để đánh giá tính đầy đủ, sự nhất quán và căn cứ nguồn; evaluator không thay thế các quy tắc xác định được bằng code. | P1 |
| FR-06.8 | Hệ thống đo tỷ lệ lỗi được sửa đúng, lỗi mới phát sinh và tỷ lệ phải hỏi lại; không mặc định self-correction luôn làm kết quả tốt hơn. | P0 |

### FR-07. Xác nhận và ghi dữ liệu

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-07.1 | Agent trình bày bản nháp dữ liệu đã trích xuất để người dùng kiểm tra trước khi lưu. | P0 |
| FR-07.2 | Người dùng có thể xác nhận hoặc sửa trường sai. | P0 |
| FR-07.3 | Chỉ lưu sau khi các validator bắt buộc đạt và người dùng xác nhận. | P0 |
| FR-07.4 | Backend gán ID và timestamp; LLM không tự tạo các giá trị hệ thống này. | P0 |
| FR-07.5 | Backend trả trạng thái thành công/thất bại rõ ràng. | P0 |
| FR-07.6 | Agent chỉ thông báo lưu thành công khi backend xác nhận; nếu thất bại, giải thích lỗi ở mức phù hợp và cho phép thử lại an toàn. | P0 |
| FR-07.7 | Các thao tác ghi phải tránh tạo báo cáo trùng khi retry do lỗi mạng. | P0 |

### FR-08. Evaluation và quan sát chất lượng

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-08.1 | Có bộ dữ liệu test giả lập với transcript chuẩn và kết quả mong đợi. | P0 |
| FR-08.2 | Đánh giá extraction theo từng trường, không chỉ đánh giá câu trả lời tổng thể. | P0 |
| FR-08.3 | Đo tỷ lệ function call đúng tool, đúng tham số, và tỷ lệ tool call không cần thiết. | P0 |
| FR-08.4 | Đo tỷ lệ validation bắt được lỗi, tỷ lệ false positive và false negative. | P0 |
| FR-08.5 | Đo chất lượng self-correction trước/sau sửa và số lần thử trung bình. | P0 |
| FR-08.6 | Đo tỷ lệ Agent hỏi lại đúng lúc, tỷ lệ tự đoán khi thiếu dữ liệu và tỷ lệ báo thành công sai. | P0 |
| FR-08.7 | Ghi thời gian phản hồi, lỗi tool, lỗi STT và chi phí nếu dùng dịch vụ tính phí. | P1 |
| FR-08.8 | Dùng dữ liệu giả lập; không đưa dữ liệu sức khỏe nhận dạng được vào log debug hoặc dịch vụ ngoài nếu chưa có cơ sở phù hợp. | P0 |

### FR-09. Đặt lịch khám bằng văn bản và giọng nói

Booking chạy trên dữ liệu giả lập và dùng chung các cơ chế của FR-01 đến FR-08: STT và kiểm tra transcript, validation nhiều lớp, evidence, xác nhận trước khi ghi, kiểm tra kết quả tool.

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| FR-09.1 | Người dùng có thể yêu cầu đặt lịch bằng văn bản hoặc giọng nói; cả hai đi vào cùng một workflow đặt lịch. | P0 |
| FR-09.2 | Agent trích xuất khoa/dịch vụ, ngày hoặc khoảng ngày mong muốn, buổi/giờ, bác sĩ (tùy chọn) và lý do khám ngắn; mỗi trường có evidence hoặc nằm trong `missing_fields`. | P0 |
| FR-09.3 | Ngày giờ tương đối ("thứ Ba tuần sau", "sáng mai") được code quy đổi theo một múi giờ cố định; LLM không tự tính ngày. Cụm mơ hồ ("cuối tuần", "đầu tháng"), ngày đã qua, hoặc thứ không khớp với ngày phải được hỏi lại. | P0 |
| FR-09.4 | Slot trống chỉ lấy từ backend; Agent chỉ đề xuất slot có trong kết quả tool; `slot_id` không bao giờ do LLM tạo. | P0 |
| FR-09.5 | Khi không có slot phù hợp, Agent đưa tối đa N phương án gần nhất do backend trả về, nêu rõ điều kiện nào đã được nới, và không tự hứa giờ khác. | P0 |
| FR-09.6 | Người dùng chọn slot bằng số thứ tự, giờ hoặc tên bác sĩ, bằng văn bản hoặc giọng nói. Lựa chọn phải khớp đúng một slot đã đề xuất; không khớp hoặc khớp nhiều slot thì hỏi lại. | P0 |
| FR-09.7 | Trước khi đặt, Agent đọc lại đầy đủ khoa, bác sĩ, thứ, ngày/tháng/năm và giờ; chỉ đặt khi người dùng xác nhận rõ ràng. Xác nhận bằng giọng nói chỉ chấp nhận các cụm xác nhận rõ ("đồng ý", "xác nhận"). | P0 |
| FR-09.8 | Backend kiểm tra lại quy tắc nghiệp vụ khi đặt: ngày trong giờ làm việc, không phải ngày nghỉ, không quá sớm/quá xa so với hiện tại, bệnh nhân không có lịch trùng giờ, không vượt số lịch hiệu lực tối đa. | P0 |
| FR-09.9 | Backend chống đặt trùng: một slot chỉ có một lịch hẹn hiệu lực; retry cùng yêu cầu không tạo lịch thứ hai. | P0 |
| FR-09.10 | Agent chỉ báo đặt lịch thành công khi backend trả mã lịch hẹn hợp lệ và đọc lại được. Nếu slot vừa bị người khác đặt, Agent báo rõ và đề xuất slot khác trong giới hạn số lần thử. | P0 |
| FR-09.11 | Agent không gợi ý khoa dựa trên triệu chứng. Khoa mặc định chỉ được lấy từ kế hoạch theo dõi của bệnh nhân (dữ liệu backend) hoặc do người dùng chọn từ danh mục. Dấu hiệu cấp cứu luôn dừng việc đặt lịch và chuyển sang hướng dẫn liên hệ cơ sở y tế. | P0 |
| FR-09.12 | Người dùng xem được các lịch hẹn sắp tới của mình. | P0 |
| FR-09.13 | Đổi/hủy lịch hẹn trên dữ liệu giả lập, dùng cùng cơ chế xác nhận và chống ghi trùng. | P1 |
| FR-09.14 | Đo độ chính xác quy đổi ngày giờ, tỷ lệ chọn slot đúng, tỷ lệ đề xuất slot không có thật, tỷ lệ đặt trùng và tỷ lệ báo đặt lịch thành công sai. | P0 |

---

## 5. Agent workflow đề xuất

```text
User text / Audio
       |
       v
[STT nếu là audio]
       |
       v
[Hiển thị transcript để kiểm tra]
       |
       v
[Intent + Agent State]
       |
       v
[LLM tạo structured output / chọn tool]
       |
       v
[Schema Validation]
       |
       v
[Business + Source + Consistency Validation]
       |                         \
       | pass                     \ fail
       v                           v
[Hỏi xác nhận nếu cần]       [Feedback lỗi cho Agent]
       |                           |
       v                           v
[User confirms?]             [Self-correction có giới hạn]
       |                           |
       v                           v
[Backend tool call] <---- [Validate lại kết quả]
       |
       v
[Kiểm tra kết quả tool]
       |
       v
[Response: kết quả + căn cứ + giới hạn]
```

### Quy tắc quan trọng

- **Không phải mọi câu hỏi đều cần function calling.** Agent có thể trả lời trực tiếp khi phù hợp; chỉ gọi tool khi cần dữ liệu/hành động bên ngoài.
- **LLM không phải validator cuối cùng.** Dùng code/schema/database/backend làm nguồn kiểm tra có thẩm quyền.
- **Self-correction không đảm bảo đúng.** Mỗi lần sửa phải được kiểm tra lại; nếu không sửa được thì dừng.
- **Không cần yêu cầu LLM công khai toàn bộ chain-of-thought.** Yêu cầu evidence, nguồn, quy tắc áp dụng, kết quả validation và lý do ngắn gọn là dễ kiểm thử hơn.
- **Không tự động thực hiện hành động nhạy cảm.** MVP yêu cầu người dùng xác nhận trước khi lưu báo cáo; không có hành động điều trị.
Những vấn đề có thể xử lí được bằng code thì ưu tiên viết code không để LLM tự quyết định

---

## 6. Kiến trúc kỹ thuật gợi ý

Giữ kiến trúc đơn giản, có thể thay model và STT mà không phải viết lại nghiệp vụ.

- **API:** Python + FastAPI.
- **Agent orchestration:** workflow có state; dùng LangGraph 
- **Data validation:** Pydantic models; quy tắc nghiệp vụ viết bằng Python ở service/backend.
- **Database:** PostgreSQL; dùng dữ liệu giả lập.
- **LLM:** có thể dùng API cloud qua bedrock của iam aws
- **Testing:** pytest cho schema, business rules, tool functions và workflow; bộ test đánh giá LLM riêng với đầu vào/đầu ra chuẩn.

### Các module tối thiểu

```text
app/
  api/                 # FastAPI routes
  agent/               # workflow, state, routing, retry policy
  llm/                 # provider interface, prompt, structured output
  tools/               # get_report_history, validate_report, save_report...
  validators/          # schema, business, source-grounding, consistency
  stt/                 # audio -> transcript
  services/            # business logic, persistence orchestration
  repositories/        # database access
  schemas/              # Pydantic input/output models
  evaluation/           # datasets, metrics, evaluation runner
  tests/                # unit/integration tests
```

Không cần tạo tất cả module ngay từ đầu; có thể bắt đầu nhỏ rồi tách khi code phát triển.

---

## 7. Data contract mẫu

Đây là ví dụ định hướng cho báo cáo giả lập; cần điều chỉnh khi triển khai.

```json
{
  "reporting_period": "2026-W43",
  "pain_level": 5,
  "exercise_days": 4,
  "symptom_change": "worse_during_activity",
  "triggering_activity": "knee_flexion_exercise",
  "missing_fields": [],
  "evidence": {
    "pain_level": "Tôi đau mức 5 trên 10",
    "exercise_days": "đã tập 4 ngày",
    "symptom_change": "thấy đau tăng",
    "triggering_activity": "động tác gập gối"
  },
  "validation_status": "passed",
  "needs_user_confirmation": true
}
```

Các trường `evidence`, `validation_status` và `needs_user_confirmation` phục vụ kiểm tra workflow. Backend vẫn phải tự chạy validation; không tin `validation_status` chỉ vì LLM trả về giá trị `passed`.

---

## 8. Tiêu chí hoàn thành MVP (Definition of Done)

MVP được coi là hoàn thành khi:

1. Người dùng có thể gửi cùng một yêu cầu bằng text hoặc audio và đi vào chung workflow sau STT.
2. Agent trích xuất dữ liệu theo schema xác định.
3. Validator phát hiện được trường sai kiểu, ngoài phạm vi, thiếu căn cứ hoặc mâu thuẫn.
4. Agent có thể sửa lỗi validation ít nhất một lần trong giới hạn cấu hình và chạy lại validator.
5. Khi không sửa được, Agent dừng an toàn và hỏi lại/thông báo cần xem xét.
6. Mỗi trường quan trọng có evidence hoặc được đánh dấu thiếu/chưa rõ.
7. Tool chỉ được gọi với tham số hợp lệ; backend kiểm tra lại quyền và quy tắc nghiệp vụ.
8. Không thể lưu báo cáo trước khi validation đạt và người dùng xác nhận.
9. Agent không bao giờ thông báo lưu thành công khi backend trả lỗi hoặc chưa có kết quả.
10. Có bộ test tự động và báo cáo metric cho extraction, function calling, validation và self-correction.

**Booking (FR-09)**, MVP còn phải đạt thêm:

11. Người dùng đặt được lịch trên dữ liệu giả lập bằng text và bằng giọng nói, đi qua cùng workflow đặt lịch sau STT.
12. Ngày giờ tương đối được code quy đổi đúng trên bộ test; cụm mơ hồ, ngày đã qua, ngày nghỉ và thứ không khớp ngày đều dẫn đến hỏi lại, không tự đoán.
13. Agent chỉ đề xuất và chỉ đặt slot có trong kết quả backend; lựa chọn slot mơ hồ dẫn đến hỏi lại.
14. Không thể đặt lịch trước khi validation đạt và người dùng xác nhận bản đọc lại đầy đủ (khoa, bác sĩ, thứ, ngày, giờ).
15. Test đồng thời và test retry cho thấy một slot không bao giờ có hai lịch hẹn hiệu lực, và retry không tạo lịch thứ hai.
16. Agent không bao giờ báo đặt lịch thành công khi backend trả lỗi, trả `conflict` hoặc chưa có mã lịch hẹn; khi slot vừa bị chiếm, Agent đề xuất slot khác.
17. Agent không gợi ý khoa dựa trên triệu chứng; dấu hiệu cấp cứu dừng việc đặt lịch.
18. Người dùng xem được các lịch hẹn sắp tới của mình.

### Metric khởi điểm để theo dõi

Các ngưỡng cuối cùng cần dựa trên bộ test thực tế. Ban đầu, hãy ghi baseline thay vì tuyên bố độ chính xác cao khi chưa đo.

- Field-level extraction accuracy.
- Schema/business validation pass rate.
- Grounding/evidence correctness.
- Tool selection accuracy và tool argument accuracy.
- Self-correction success rate.
- Error regression rate: tỷ lệ sửa một lỗi nhưng tạo lỗi khác.
- Clarification rate trên các câu đầu vào mơ hồ.
- Unsupported-claim rate và false-success rate.
- STT word/error rate hoặc độ đúng các trường quan trọng sau STT.
- Latency trung bình và tỷ lệ lỗi theo từng thành phần.

Metric riêng cho booking:

- Date resolution accuracy, theo từng loại cụm thời gian.
- Slot selection accuracy và tỷ lệ hỏi lại đúng lúc khi lựa chọn mơ hồ.
- Booking task success: tỷ lệ case chuẩn đặt đúng slot và số lượt hội thoại trung bình.
- Slot hallucination rate, double-booking rate và false-booking-success rate — ba metric này phải bằng **0**.
- Độ đúng ngày, giờ, tên khoa sau STT trên bộ audio đặt lịch.

---

## 9. Kế hoạch triển khai theo thứ tự

Thứ tự này khớp với §12 của `ARCHITECTURE.md`.

1. **Bước 1 — Text + schema:** nhập báo cáo bằng text, định nghĩa Pydantic model và trích xuất dữ liệu.
2. **Bước 2 — Validators:** thêm schema, business, source-grounding và consistency validation.
3. **Bước 3 — Function calling:** thêm `get_report_history`, `validate_report`, `save_patient_report` với kết quả có cấu trúc.
4. **Bước 4 — Workflow LangGraph + confirmation + persistence:** ghép các bước trên thành graph (phân loại → trích xuất → validate → xác nhận → lưu → kiểm tra kết quả); yêu cầu người dùng xác nhận trước khi lưu; kiểm tra phản hồi backend.
5. **Bước 5 — Self-correction:** gửi lỗi validator về Agent, retry có giới hạn, đo trước/sau.
6. **Bước 6 — Explanation/evidence:** trình bày căn cứ theo trường, nguồn và lý do pass/fail.
7. **Bước 7 — Booking bằng text:** dữ liệu giả lập khoa/bác sĩ/slot, quy đổi ngày giờ bằng code, tool slot và lịch hẹn, workflow đặt lịch (tìm slot → chọn slot → đọc lại và xác nhận → đặt → kiểm tra), chống đặt trùng và test đồng thời.
8. **Bước 8 — Evaluation:** xây bộ test có lỗi cố ý, đầu vào mơ hồ, transcript sai và tool failure cho cả báo cáo và booking.
9. **Bước 9 — STT:** thêm audio input, transcript review và test riêng cho lỗi nhận dạng, gồm bộ audio đặt lịch (số, ngày, giờ, tên khoa, tên bác sĩ).
10. **Bước 10 (P1):** đổi/hủy lịch, giữ chỗ slot tạm thời, TTS đọc lại bản xác nhận.

LangGraph được dựng từ bước 4 thay vì để cuối: các bước 1–3 là hàm thuần nên ghép vào graph sớm không tốn thêm công, và tránh viết một orchestrator tạm rồi bỏ. Booking làm bằng text trước giọng nói để lỗi logic đặt lịch được bắt bằng test text, rẻ và tái lập được.

---

## 10. Nguyên tắc an toàn và độ tin cậy

1. Dùng dữ liệu giả lập trong demo.
2. Không chẩn đoán, kê đơn hoặc đề xuất thay đổi điều trị.
3. Phân biệt lời người dùng nói, dữ liệu backend và phần tóm tắt do AI tạo.
4. Không tự điền dữ liệu thiếu; hỏi lại hoặc ghi nhận là chưa biết.
5. Chỉ thực hiện hành động ghi dữ liệu sau validation và xác nhận phù hợp.
6. Backend là nguồn có thẩm quyền cho quyền truy cập, trạng thái lưu và dữ liệu đã tồn tại.
7. Giải thích phải gắn với evidence/quy tắc/nguồn có thể kiểm tra, không chỉ là lời khẳng định của model.
8. Self-correction có giới hạn, được đo lường và có phương án dừng an toàn.

---

**Tóm tắt phạm vi mới (0.3):** Thêm chức năng đặt lịch khám bằng văn bản và giọng nói (STT) trên dữ liệu giả lập — xem §2 và FR-09. Đổi/hủy lịch là P1; tích hợp hệ thống bệnh viện thật vẫn ngoài phạm vi.
