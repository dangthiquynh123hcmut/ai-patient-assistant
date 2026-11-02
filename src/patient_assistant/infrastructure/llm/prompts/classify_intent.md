Phân loại yêu cầu của người dùng trong ứng dụng hỗ trợ bệnh nhân (dữ liệu giả lập).

Các intent:
- create_report: gửi báo cáo theo dõi (mức đau, số ngày tập, thay đổi triệu chứng).
- get_report_history: xem các báo cáo đã gửi trước đó.
- book_appointment: đặt lịch khám mới.
- list_appointments: xem các lịch hẹn đã đặt.
- get_patient_info: xem thông tin cá nhân, hồ sơ, kế hoạch theo dõi/điều trị, tình hình tổng quan của mình.
- answer_faq: hỏi thông tin chung về bài tập, chuẩn bị tái khám, chăm sóc sau tập.
- small_talk: chào hỏi, cảm ơn, tạm biệt, hỏi trợ lý là ai / làm được gì, tán gẫu, hoặc câu ngoài phạm vi phòng khám
  (thời tiết, lập trình...). Mọi ngôn ngữ, ví dụ "hello", "hi there", "cảm ơn nhé", "bạn là ai".
- unknown: có vẻ là một yêu cầu công việc nhưng không rõ là việc nào.

Hiểu ý định thật của người dùng kể cả khi viết sai chính tả, không dấu, tiếng lóng hay tiếng Anh
(ví dụ "dat lich kham" = book_appointment, "show my appointments" = list_appointments).
Trả về danh sách intent theo thứ tự người dùng nhắc tới. Lời chào đi kèm yêu cầu thì chỉ trả intent của yêu cầu.
Nếu không chắc, trả ["unknown"].
