Phân loại yêu cầu của người dùng trong ứng dụng hỗ trợ bệnh nhân (dữ liệu giả lập).

Các intent:
- create_report: gửi báo cáo theo dõi (mức đau, số ngày tập, thay đổi triệu chứng).
- get_report_history: xem các báo cáo đã gửi trước đó.
- book_appointment: đặt lịch khám mới.
- list_appointments: xem các lịch hẹn đã đặt.
- answer_faq: hỏi thông tin chung về bài tập, chuẩn bị tái khám, chăm sóc sau tập.
- unknown: không thuộc các loại trên hoặc không rõ.

Trả về danh sách intent theo thứ tự người dùng nhắc tới. Nếu không chắc, trả ["unknown"].
